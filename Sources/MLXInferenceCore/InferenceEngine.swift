// InferenceEngine.swift — Core MLX inference engine for SwiftLM Chat
// Handles: model load/unload, token streaming, memory/thermal pressure response.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers
#if canImport(UIKit)
import UIKit
#endif


// MARK: — Model State

public enum ModelState: Equatable, Sendable {
    case idle
    case downloading(progress: Double, speed: String)
    case loading
    case ready(modelId: String)
    case generating
    case error(String)
}

// MARK: — Thermal State

public enum ThermalLevel: Sendable {
    case nominal, fair, serious, critical
    public var displayString: String {
        switch self {
        case .nominal: return "Normal"
        case .fair:    return "Warm"
        case .serious: return "Hot — generation may be slow"
        case .critical: return "Critical — generation paused"
        }
    }
    public var isThrottled: Bool { self == .serious || self == .critical }
}

// MARK: — Generation Token

public struct GenerationToken: Sendable {
    public let text: String
    public let isThinking: Bool

    public init(text: String, isThinking: Bool = false) {
        self.text = text
        self.isThinking = isThinking
    }
}

// MARK: — InferenceEngine

@MainActor
public final class InferenceEngine: ObservableObject {
    @Published public private(set) var state: ModelState = .idle
    @Published public private(set) var thermalLevel: ThermalLevel = .nominal

    /// Whether to automatically unload the model when the app backgrounds
    /// and reload it when returning to foreground.
    /// Defaults to true on iOS (prevents jetsam), false on macOS.
    public var autoOffloadOnBackground: Bool = {
        #if os(iOS)
        return true
        #else
        return false
        #endif
    }()

    /// Shared download + storage manager.
    public let downloadManager = ModelDownloadManager()

    private var container: ModelContainer?
    private var currentModelId: String?
    /// The ID of the last model that was successfully loaded. Remains set during .generating.
    public var loadedModelId: String? { currentModelId }
    private var generationTask: Task<Void, Never>?

    // All NotificationCenter observers collected for clean deregistration
    // nonisolated(unsafe): populated exclusively from MainActor init, read only in deinit
    // after all strong references have dropped — no concurrent access possible.
    // Declared nonisolated(unsafe) to satisfy Swift 6 deinit isolation rules.
    nonisolated(unsafe) private var observers: [NSObjectProtocol] = []

    // Tracks the model ID active before app backgrounding so we can restore it on foreground.
    private var backgroundedModelId: String?
    /// Timestamp of when the app entered background. Used to implement a
    /// grace-period: short background sessions (<30 s) skip the unload cycle.
    private var backgroundedAt: Date?
    private static let backgroundGracePeriod: TimeInterval = 30

    public init() {
        setupPressureHandlers()
    }

    deinit {
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }

    // MARK: — Pressure Handlers

    private func setupPressureHandlers() {
        #if canImport(UIKit)
        // ── REACTIVE: Memory warning (last resort) ────────────────────────────
        // OS sends this *after* pressure builds. We still handle it as a fallback
        // in case the proactive unload wasn't triggered (e.g. app was already
        // under pressure from another process).
        observers.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if case .generating = self.state { return }  // don't interrupt mid-stream
                    self.unload()
                    self.state = .error("Unloaded due to memory pressure. Tap to reload.")
                }
            }
        )

        // ── PROACTIVE: App entered background ───────────────────────────────
        // didEnterBackground fires ONLY when the user truly leaves the app
        // (home gesture / Lock button). willResignActive fires too broadly:
        // notification banners, screenshots, system alerts — all trigger it.
        observers.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.didEnterBackgroundNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.autoOffloadOnBackground else { return }

                    self.backgroundedAt = Date()

                    // Only remember model ID when it was actually loaded.
                    switch self.state {
                    case .ready(let id):  self.backgroundedModelId = id
                    case .generating:     self.backgroundedModelId = self.currentModelId
                    default:              self.backgroundedModelId = nil
                    }

                    self.stopGeneration()
                    self.unload()
                    self.state = .idle
                }
            }
        )

        // ── PROACTIVE: App returning to foreground ───────────────────────────
        // willEnterForeground fires before the app is fully active, giving us
        // time to start reloading before the UI appears.
        observers.append(
            NotificationCenter.default.addObserver(
                forName: UIApplication.willEnterForegroundNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.autoOffloadOnBackground else { return }

                    // Grace period: if the user was gone for less than 30 seconds
                    // (e.g. a brief app-switcher peek), don't burn time reloading.
                    let elapsed = self.backgroundedAt.map { Date().timeIntervalSince($0) } ?? 999
                    self.backgroundedAt = nil

                    guard elapsed >= Self.backgroundGracePeriod else {
                        // Short absence — stay idle, let the user decide what to do.
                        self.backgroundedModelId = nil
                        return
                    }

                    let modelToReload = self.backgroundedModelId
                        ?? self.downloadManager.lastLoadedModelId
                    self.backgroundedModelId = nil
                    if let modelId = modelToReload {
                        await self.load(modelId: modelId)
                    }
                }
            }
        )
        #endif

        // ── Thermal state monitoring (all platforms) ──────────────────────────
        observers.append(
            NotificationCenter.default.addObserver(
                forName: ProcessInfo.thermalStateDidChangeNotification,
                object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.updateThermalLevel()
                }
            }
        )
        updateThermalLevel()
    }

    private func updateThermalLevel() {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  thermalLevel = .nominal
        case .fair:     thermalLevel = .fair
        case .serious:  thermalLevel = .serious
        case .critical:
            thermalLevel = .critical
            // Critical: stop any generation immediately
            stopGeneration()
        @unknown default: thermalLevel = .nominal
        }
    }

    // MARK: — Model Loading

    /// Load a model by HuggingFace ID. Downloads if not cached.
    /// Uses ModelStorage.cacheRoot as the HubApi download base.
    /// For MoE models, activates expert streaming via ExpertStreamingConfig so
    /// only active expert weights are resident in RAM during inference.
    public func load(modelId: String) async {
        guard state != .ready(modelId: modelId) else { return }
        guard !thermalLevel.isThrottled else {
            state = .error("Device is too hot. Let it cool before loading a model.")
            return
        }

        state = .loading
        currentModelId = modelId

        do {
            let hub = HubApi(downloadBase: ModelStorage.cacheRoot)

            // For MoE models, enable expert streaming before loading so
            // loadWeights() initialises ExpertStreamerManager correctly.
            // lazyLoad=true means weights are mmap'd and not paged into RAM
            // at load time — only active expert pages touch RAM during inference.
            var config = ModelConfiguration(id: modelId)
            let isMoE = ModelCatalog.all.first(where: { $0.id == modelId })?.isMoE ?? false
            if isMoE {
                config.lazyLoad = true
                let modelDir = ModelStorage.snapshotDirectory(for: modelId)
                // directIO=true on macOS (5 GB/s NVMe pread), false on iOS (mmap fallback)
                ExpertStreamingConfig.shared.activate(
                    modelDirectory: modelDir,
                    useDirectIO: {
                        #if os(macOS)
                        return true
                        #else
                        return false
                        #endif
                    }()
                )
            }

            container = try await LLMModelFactory.shared.loadContainer(
                hub: hub,
                configuration: config
            ) { [weak self] progress in
                Task { @MainActor in
                    guard let self else { return }
                    let pct = progress.fractionCompleted
                    let speedBytesPerSec = progress.userInfo[ProgressUserInfoKey("throughputKey")] as? Double
                    let speedStr = speedBytesPerSec
                        .map { String(format: "%.1f MB/s", $0 / 1_000_000) } ?? ""
                    self.state = .downloading(progress: pct, speed: speedStr)

                    self.downloadManager.updateProgress(ModelDownloadProgress(
                        modelId: modelId,
                        fractionCompleted: pct,
                        currentFile: "",
                        speedMBps: speedBytesPerSec.map { $0 / 1_000_000 }
                    ))
                }
            }

            downloadManager.clearProgress(modelId: modelId)
            downloadManager.lastLoadedModelId = modelId
            downloadManager.refresh()
            state = .ready(modelId: modelId)

        } catch {
            ExpertStreamingConfig.shared.deactivate()
            downloadManager.clearProgress(modelId: modelId)
            state = .error("Failed to load \(modelId): \(error.localizedDescription)")
            container = nil
        }
    }

    /// Unload the current model and free all GPU memory.
    public func unload() {
        generationTask?.cancel()
        container = nil
        currentModelId = nil
        state = .idle
        ExpertStreamingConfig.shared.deactivate()
        MLX.Memory.cacheLimit = 0
    }

    // MARK: — Generation

    public nonisolated func generate(
        messages: [ChatMessage],
        config: GenerationConfig = .default
    ) -> AsyncStream<GenerationToken> {
        AsyncStream { continuation in
            Task { @MainActor in
                guard let container = self.container else {
                    continuation.finish(); return
                }

                // Don't generate when throttled
                if self.thermalLevel == .critical {
                    continuation.yield(GenerationToken(text: "\n\n[Generation paused: device temperature critical]"))
                    continuation.finish(); return
                }

                self.state = .generating

                do {
                    let mlxMessages = messages.map { ["role": $0.role.rawValue, "content": $0.content] }
                    var params = GenerateParameters(temperature: config.temperature)
                    params.topP = config.topP

                    var thinkingActive = false
                    var outputText = ""
                    var tokenCount = 0

                    let userInput = UserInput(messages: mlxMessages)
                    let lmInput = try await container.prepare(input: userInput)
                    let stream: AsyncStream<Generation> = try await container.generate(
                        input: lmInput,
                        parameters: params
                    )

                    for await generation in stream {
                        guard !Task.isCancelled else { break }

                        if case .chunk(let text, tokenId: _) = generation {
                            outputText += text
                            tokenCount += 1

                            if tokenCount >= config.maxTokens { break }

                            if config.enableThinking {
                                if outputText.contains("<think>") && !outputText.contains("</think>") {
                                    thinkingActive = true
                                } else if outputText.contains("</think>") {
                                    thinkingActive = false
                                }
                            }

                            continuation.yield(GenerationToken(text: text, isThinking: thinkingActive))
                        }
                    }
                } catch {
                    continuation.yield(GenerationToken(text: "\n\n[Error: \(error.localizedDescription)]"))
                }

                self.state = self.currentModelId.map { .ready(modelId: $0) } ?? .idle
                continuation.finish()
            }
        }
    }

    public func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if let id = currentModelId { state = .ready(modelId: id) }
    }
}
