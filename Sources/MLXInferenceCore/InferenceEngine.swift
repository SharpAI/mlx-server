// InferenceEngine.swift — Core MLX inference engine for SwiftLM Chat
// Handles: model load/unload, token streaming, memory/thermal pressure response.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Hub
import Tokenizers
#if canImport(UIKit)
import UIKit
#endif

// MARK: — Hub Downloader bridge (Downloader protocol conformance over HubApi)

private struct HubDownloader: Downloader, Sendable {
    let hub: HubApi
    func download(
        id: String, revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try await hub.snapshot(
            from: id,
            matching: patterns,
            progressHandler: progressHandler)
    }
}

// MARK: — swift-transformers TokenizerLoader bridge

private struct TransformersTokenizerLoader: TokenizerLoader, Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream)
    }
}

private struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer, Sendable {
    let upstream: any Tokenizers.Tokenizer
    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}


// MARK: — Model State

public enum ModelState: Equatable, Sendable {
    case idle
    case downloading(progress: Double, speed: String)
    case loading(progress: Double, stage: String)
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
    @Published public private(set) var activeContextTokens: Int = 0
    @Published public private(set) var maxContextWindow: Int = 0

    /// Set when a corrupted/truncated model is detected during inference.
    /// The UI should observe this and offer to delete & re-download.
    @Published public var corruptedModelId: String? = nil

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
        corruptedModelId = nil

        guard ModelStorage.verifyModelIntegrity(for: modelId) else {
            await downloadThenLoad(modelId: modelId)
            return
        }

        await loadVerifiedModel(modelId: modelId)
    }

    private func downloadThenLoad(modelId: String) async {
        print("[InferenceEngine] Model \(modelId) is missing or incomplete. Starting download before load.")
        releaseLoadedModelResources()
        state = .downloading(progress: 0.0, speed: "Preparing...")

        let task = downloadManager.startDownload(modelId: modelId)

        do {
            try await task.value
            state = .downloading(progress: 1.0, speed: "Verifying...")

            guard ModelStorage.verifyModelIntegrity(for: modelId) else {
                markModelCorrupted(
                    modelId: modelId,
                    message: "Model files are incomplete after download. Choose a recovery option."
                )
                return
            }

            await loadVerifiedModel(modelId: modelId)
        } catch is CancellationError {
            state = .idle
        } catch {
            state = .error("Failed to download \(modelId): \(error.localizedDescription)")
        }
    }

    private func loadVerifiedModel(modelId: String) async {
        setLoadingState(progress: 0.05, stage: "Preparing model configuration")
        currentModelId = modelId

        do {
            let hub = HubApi(downloadBase: ModelStorage.cacheRoot)

            // For MoE models, enable expert streaming before loading so
            // loadWeights() initialises ExpertStreamerManager correctly.
            // lazyLoad=true means weights are mmap'd and not paged into RAM
            // at load time — only active expert pages touch RAM during inference.
            var config = ModelConfiguration(id: modelId)
            let isMoE = ModelCatalog.all.first(where: { $0.id == modelId })?.isMoE ?? false
            let generationConfig = GenerationConfig.load()
            // SSD expert streaming defaults ON for MoE until the user saves a preference.
            // Once persisted, the saved toggle becomes authoritative for all models.
            let shouldStream = generationConfig.effectiveStreamExperts(defaultingTo: isMoE)
            if shouldStream {
                config.lazyLoad = true
                let modelDir = ModelStorage.snapshotDirectory(for: modelId)
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
                print("[InferenceEngine] SSD expert streaming: enabled (isMoE=\(isMoE), persisted=\(GenerationConfig.hasPersistedConfig), setting=\(generationConfig.streamExperts))")
            } else {
                print("[InferenceEngine] SSD expert streaming: disabled")
            }

            setLoadingState(progress: 0.15, stage: "Inspecting model architecture")
            let downloader = HubDownloader(hub: hub)
            let architecture = try await ModelArchitectureProbe.inspect(
                configuration: config,
                downloader: downloader
            )

            let loadingStage = architecture.supportsVision
                ? "Loading multimodal model"
                : "Loading language model"

            setLoadingState(progress: 0.22, stage: loadingStage)

            if architecture.supportsVision {
                container = try await VLMModelFactory.shared.loadContainer(
                    from: downloader,
                    using: TransformersTokenizerLoader(),
                    configuration: config
                ) { [weak self] progress in
                    Task { @MainActor in
                        guard let self else { return }
                        let pct = progress.fractionCompleted
                        self.setLoadingState(progress: 0.22 + (pct * 0.68), stage: loadingStage)
                    }
                }
            } else {
                container = try await LLMModelFactory.shared.loadContainer(
                    from: downloader,
                    using: TransformersTokenizerLoader(),
                    configuration: config
                ) { [weak self] progress in
                    Task { @MainActor in
                        guard let self else { return }
                        let pct = progress.fractionCompleted
                        self.setLoadingState(progress: 0.22 + (pct * 0.68), stage: loadingStage)
                    }
                }
            }

            downloadManager.clearProgress(modelId: modelId)
            downloadManager.lastLoadedModelId = modelId
            downloadManager.refresh()

            // Verify integrity to catch incomplete downloads before marking as ready
            setLoadingState(progress: 0.94, stage: "Verifying model files")
            guard ModelStorage.verifyModelIntegrity(for: modelId) else {
                throw NSError(domain: "InferenceEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model safetensors files are incomplete. Please delete and re-download."])
            }

            // Read the model's actual max context length from config.json
            setLoadingState(progress: 0.98, stage: "Reading model limits")
            if let ctxLen = ModelStorage.readMaxContextLength(for: modelId) {
                self.maxContextWindow = ctxLen
                print("[InferenceEngine] Model context window: \(ctxLen) tokens")
            } else {
                self.maxContextWindow = 8192  // conservative fallback for models without explicit limits
                print("[InferenceEngine] No explicit context limit found in config.json, defaulting to 8192")
            }

            state = .ready(modelId: modelId)

        } catch {
            ExpertStreamingConfig.shared.deactivate()
            downloadManager.clearProgress(modelId: modelId)
            state = .error("Failed to load \(modelId): \(error.localizedDescription)")

            // If the model is incomplete/corrupted, flag it so the UI shows the "Delete & Re-download" button
            let nsError = error as NSError
            if nsError.domain == "InferenceEngine" && nsError.code == 1 || Self.isModelCorruptionError(error) {
                markModelCorrupted(
                    modelId: modelId,
                    message: "Model weights are corrupted or incomplete. Choose a recovery option."
                )
                return
            }

            container = nil
            self.maxContextWindow = 0
            self.activeContextTokens = 0
        }
    }

    /// Unload the current model and free all GPU memory.
    public func unload() {
        releaseLoadedModelResources()
        corruptedModelId = nil
        state = .idle
    }

    private func releaseLoadedModelResources() {
        generationTask?.cancel()
        generationTask = nil
        container = nil
        currentModelId = nil
        maxContextWindow = 0
        activeContextTokens = 0
        ExpertStreamingConfig.shared.deactivate()
        MLX.Memory.cacheLimit = 0
    }

    private func setLoadingState(progress: Double, stage: String) {
        state = .loading(progress: min(max(progress, 0), 1), stage: stage)
    }

    private func markModelCorrupted(modelId: String?, message: String) {
        let failedModelId = modelId ?? currentModelId
        releaseLoadedModelResources()
        state = .error(message)
        corruptedModelId = failedModelId
    }

    private static func isModelCorruptionError(_ error: Error) -> Bool {
        let description = error.localizedDescription.lowercased()
        return description.contains("ssd streaming")
            || description.contains("pread")
            || description.contains("safetensors")
            || description.contains("corrupt")
            || description.contains("incomplete")
    }

    public func clearCorruptionRecovery() {
        corruptedModelId = nil
        if case .error = state {
            state = .idle
        }
    }

    // MARK: — Generation
}

// MARK: — Helpers

/// Removes all `<think>…</think>` spans from `text`, including the closing tag's
/// trailing newline when present.  Used to sanitise assistant history messages
/// before they are re-submitted to the Jinja chat-template renderer on subsequent
/// turns — Qwen3 (and similar "thinking" models) raise TemplateException error 1
/// when prior assistant turns contain raw thinking tags.
///
/// Trimming is applied only when at least one tag span was actually removed so
/// that assistant messages without thinking content are returned byte-for-byte
/// (preserving leading spaces, code-block indentation, etc.).
func stripThinkingTags(from text: String) -> String {
    var result = text
    var stripped = false
    while let openRange = result.range(of: "<think>") {
        stripped = true
        if let closeRange = result.range(of: "</think>", range: openRange.lowerBound..<result.endIndex) {
            // Include the optional newline that immediately follows </think>
            var endIdx = closeRange.upperBound
            if endIdx < result.endIndex && result[endIdx] == "\n" {
                endIdx = result.index(after: endIdx)
            }
            result.removeSubrange(openRange.lowerBound..<endIdx)
        } else {
            // Unclosed <think> — strip from opening tag to end of string
            result.removeSubrange(openRange.lowerBound...)
            break
        }
    }
    // Only trim surrounding whitespace that was introduced by stripping;
    // leave untouched messages that contained no think tags.
    return stripped ? result.trimmingCharacters(in: .whitespacesAndNewlines) : result
}

extension InferenceEngine {
    // MARK: — Generation (continued)

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
                    var finalMessages: [[String: String]] = []
                    var pendingSystemContext = ""
                    
                    for msg in messages {
                        if msg.role == .system {
                            pendingSystemContext += msg.content + "\n\n"
                        } else {
                            // Use the canonical role name — Qwen3 (and most models) use
                            // "assistant", not "model". The "model" alias is Gemma-specific
                            // and breaks Qwen3's Jinja chat template on multi-turn history.
                            let roleRaw = msg.role.rawValue  // "user" | "assistant" | "tool"
                            var content = msg.content
                            
                            // Strip <think>…</think> blocks from prior assistant turns.
                            // If the model generated thinking content on a previous turn and
                            // it was not already split into thinkingContent, the raw tags will
                            // be present in `content`. Feeding them back into the Jinja template
                            // on the next request causes TemplateException error 1 on Qwen3.
                            if msg.role == .assistant {
                                content = stripThinkingTags(from: content)
                            }
                            
                            if roleRaw == "user" && !pendingSystemContext.isEmpty {
                                content = "[SYSTEM CONTEXT / PERSONA DATA]\n" + pendingSystemContext + "\n[END CONTEXT]\n\n" + content
                                pendingSystemContext = "" // Clear after injecting
                            }
                            finalMessages.append(["role": roleRaw, "content": content])
                        }
                    }
                    
                    let mlxMessages = finalMessages
                    var params = GenerateParameters(
                        maxTokens: config.maxTokens,
                        kvBits: config.kvBits,
                        kvGroupSize: config.kvGroupSize,
                        temperature: config.temperature,
                        topP: config.topP,
                        topK: config.topK,
                        minP: config.minP,
                        repetitionPenalty: config.repetitionPenalty,
                        prefillStepSize: config.prefillSize
                    )
                    params.repetitionContextSize = 20

                    var thinkingActive = false
                    var outputText = ""
                    var tokenCount = 0

                    // Set RNG seed for reproducible output when requested.
                    if let seed = config.seed {
                        MLX.seed(seed)
                    }

                    // Pass enable_thinking to the Jinja chat template so the model
                    // actually generates <think> blocks when thinking mode is ON.
                    // Without this kwarg, Qwen3's template defaults to thinking=false
                    // regardless of what the UI toggle shows.
                    let additionalContext: [String: any Sendable]? = config.enableThinking
                        ? ["enable_thinking": true]
                        : ["enable_thinking": false]
                    let userInput = UserInput(
                        messages: mlxMessages,
                        additionalContext: additionalContext
                    )
                    let lmInput = try await container.prepare(input: userInput)
                    
                    // Use the real token count from the prepared LMInput rather than
                    // a character-length heuristic (which was consistently off by 2–3×
                    // for CJK and code content).
                    let baseTokens = lmInput.text.tokens.size
                    self.activeContextTokens = baseTokens
                    
                    // maxContextWindow is already set during loadModel() from config.json

                    // TurboKV: enable 3-bit PolarQuant+QJL on every KVCacheSimple layer
                    // before generation. Must be set on the model (not the cache) so the
                    // cache inherits the flag when newCache() is called inside generate().
                    if config.turboKV {
                        await container.perform { ctx in
                            for module in ctx.model.modules() {
                                if let simple = module as? KVCacheSimple {
                                    simple.turboQuantEnabled = true
                                }
                            }
                        }
                        print("[InferenceEngine] TurboKV enabled for this request")
                    }

                    let stream: AsyncStream<Generation> = try await container.generate(
                        input: lmInput,
                        parameters: params
                    )

                    for await generation in stream {
                        guard !Task.isCancelled else { break }

                        if case .chunk(let text, tokenId: _) = generation {
                            outputText += text
                            tokenCount += 1
                            
                            // Update the UI token counter periodically to save CPU
                            if tokenCount % 10 == 0 {
                                self.activeContextTokens = baseTokens + tokenCount
                            }

                            if tokenCount >= config.maxTokens { break }
                            
                            // Hard-stop constraint for Gemma 2/3 and DeepSeek MoE bounds since MLX fails to parse multi-array JSON eos_token_id manifests.
                            if outputText.contains("<end_of_turn>") || outputText.contains("<|im_end|>") || outputText.contains("<|eot_id|>") {
                                let clamped = text.replacingOccurrences(of: "<end_of_turn>", with: "")
                                                  .replacingOccurrences(of: "<|im_end|>", with: "")
                                                  .replacingOccurrences(of: "<|eot_id|>", with: "")
                                continuation.yield(GenerationToken(text: clamped, isThinking: thinkingActive))
                                break
                            }

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
                } catch let ssdError as SSDStreamingError {
                    // Corrupted/truncated safetensors — surface a clear, actionable error
                    let msg = "Model weights are corrupted or incomplete. Please re-download the model."
                    print("[InferenceEngine] SSD Streaming Error: \(ssdError.localizedDescription)")
                    continuation.yield(GenerationToken(text: "\n\n[Error: \(msg)]"))
                    self.markModelCorrupted(modelId: self.currentModelId, message: msg)
                } catch {
                    // Check if the generic error is also an SSD streaming issue
                    if Self.isModelCorruptionError(error) {
                        let msg = "Model weights are corrupted or incomplete. Please re-download the model."
                        self.markModelCorrupted(modelId: self.currentModelId, message: msg)
                    }
                    continuation.yield(GenerationToken(text: "\n\n[Error: \(error.localizedDescription)]"))
                }

                if let latchedError = SSDStreamingErrorLatch.shared.consume() {
                    let msg = "Model weights are corrupted or incomplete. Please re-download the model."
                    print("[InferenceEngine] Latched SSD error after generation: \(latchedError.localizedDescription)")
                    self.markModelCorrupted(modelId: self.currentModelId, message: msg)
                } else if case .error = self.state {
                    // Already in error state from catch block above
                } else {
                    self.state = self.currentModelId.map { .ready(modelId: $0) } ?? .idle
                }
                continuation.finish()
            }
        }
    }

    public func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if let id = currentModelId { state = .ready(modelId: id) }
    }

    /// Delete corrupted model files and start a fresh download.
    /// Called from the UI when the user confirms re-download after corruption is detected.
    public func deleteCorruptedAndRedownload() {
        guard let modelId = corruptedModelId else { return }

        releaseLoadedModelResources()
        state = .downloading(progress: 0.0, speed: "Deleting corrupted files...")

        do {
            try ModelStorage.delete(modelId)
            print("[InferenceEngine] Successfully deleted corrupted cache directory for \(modelId).")
        } catch {
            print("[InferenceEngine] FAILED to delete corrupted cache: \(error.localizedDescription)")
            state = .error("Failed to delete corrupted model: \(error.localizedDescription)")
            return
        }
        downloadManager.refresh()
        corruptedModelId = nil

        print("[InferenceEngine] Deleted corrupted files for \(modelId), starting fresh download")
        Task { @MainActor in
            await downloadThenLoad(modelId: modelId)
        }
    }
}
