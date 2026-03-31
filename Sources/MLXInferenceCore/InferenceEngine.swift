// InferenceEngine.swift — Core MLX inference engine for SwiftLM Chat
// Handles: model load/unload, token streaming, memory/thermal pressure response.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Hub
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

    /// Shared download + storage manager.
    public let downloadManager = ModelDownloadManager()

    private var container: ModelContainer?
    private var currentModelId: String?
    private var generationTask: Task<Void, Never>?
    private var pressureObserver: NSObjectProtocol?
    private var thermalObserver: NSObjectProtocol?

    public init() {
        setupPressureHandlers()
    }

    deinit {
        if let o = pressureObserver { NotificationCenter.default.removeObserver(o) }
        if let o = thermalObserver  { NotificationCenter.default.removeObserver(o) }
    }

    // MARK: — Pressure Handlers

    private func setupPressureHandlers() {
        // iOS memory pressure → unload model weights immediately
        #if canImport(UIKit)
        pressureObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Only unload if not actively generating
                if case .generating = self.state { return }
                self.unload()
                self.state = .error("Unloaded due to memory pressure. Tap to reload.")
            }
        }
        #endif

        // Thermal state monitoring (all platforms)
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.updateThermalLevel()
            }
        }
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
    public func load(modelId: String) async {
        guard state != .ready(modelId: modelId) else { return }
        guard !thermalLevel.isThrottled else {
            state = .error("Device is too hot. Let it cool before loading a model.")
            return
        }

        state = .loading
        currentModelId = modelId

        do {
            // Point HubApi at ModelStorage.cacheRoot so downloads land in the right
            // place on both platforms (macOS: ~/.cache/HF, iOS: Application Support)
            let hub = HubApi(downloadBase: ModelStorage.cacheRoot)
            let config = ModelConfiguration(id: modelId)

            container = try await LLMModelFactory.shared.loadContainer(
                hub: hub,
                configuration: config
            ) { [weak self] progress in
                Task { @MainActor in
                    guard let self else { return }
                    let pct = progress.fractionCompleted
                    let speedBytesPerSec = progress.userInfo[.throughputKey] as? Double
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
        MLX.GPU.set(cacheLimit: 0)
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
