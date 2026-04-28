// SwiftLM — Native Swift OpenAI-compatible HTTP server backed by Apple MLX Swift
//
// Endpoints:
//   GET  /health                    → { "status": "ok", "model": "<id>" }
//   GET  /v1/models                 → OpenAI-style model list
//   POST /v1/chat/completions       → OpenAI Chat Completions (streaming + non-streaming)
//   POST /v1/completions            → OpenAI Text Completions (streaming + non-streaming)
//
// Usage:
//   SwiftLM --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 5413

import ArgumentParser
import CoreImage
import DFlash
import Foundation
import HTTPTypes
import Hummingbird
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import MLXInferenceCore
import Tokenizers

extension LMInput: @retroactive @unchecked Sendable {}
extension MLXLMCommon.LMInput.Text: @retroactive @unchecked Sendable {}
extension MLXLMCommon.LMInput.ProcessedImage: @retroactive @unchecked Sendable {}

// ── Hub/Tokenizer bridges (Downloader + TokenizerLoader conformances) ─────────

private struct HubDownloader: Downloader, Sendable {
    let hub: HubApi
    func download(
        id: String, revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try await hub.snapshot(from: id, matching: patterns, progressHandler: progressHandler)
    }
}

private struct TransformersTokenizerLoader: TokenizerLoader, Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let t = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(t)
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

// ── CLI ──────────────────────────────────────────────────────────────────────

final class ProgressTracker {
    var isDone = false
    var spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    var frameIndex = 0
    let modelId: String
    private var trackingTask: Task<Void, Never>?
    private var lastUpdate: TimeInterval = 0
    private var lastBytes: Int64 = 0
    private var speedStr = "0.0 MB/s"
    
    init(modelId: String) {
        self.modelId = modelId
    }
    
    func getDownloadedBytes() -> Int64 {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
        let modelHubDir = home.appendingPathComponent(".cache/huggingface/hub/\(folderName)")
        let downloadDir = home.appendingPathComponent(".cache/huggingface/download")
        
        func sumDir(_ dir: URL) -> Int64 {
            var total: Int64 = 0
            if let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) {
                for case let file as URL in enumerator {
                    if let attr = try? file.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey]),
                       let size = attr.fileSize,
                       attr.isSymbolicLink != true {
                        total += Int64(size)
                    }
                }
            }
            return total
        }
        
        return sumDir(modelHubDir) + sumDir(downloadDir)
    }
    
    func printProgress(_ progress: Progress) {
        if trackingTask == nil {
            lastUpdate = Date().timeIntervalSince1970
            lastBytes = getDownloadedBytes()
            
            trackingTask = Task {
                while !self.isDone && !Task.isCancelled {
                    let now = Date().timeIntervalSince1970
                    let fraction = progress.fractionCompleted
                    let pct = Int(fraction * 100)
                    
                    let interval = now - self.lastUpdate
                    if interval >= 0.25 {
                        self.frameIndex = (self.frameIndex + 1) % self.spinnerFrames.count
                        
                        let currentBytes = self.getDownloadedBytes()
                        let diff = Double(currentBytes - self.lastBytes)
                        if diff >= 0 {
                            let speedMBps = (diff / interval) / 1_048_576.0
                            self.speedStr = String(format: "%.1f MB/s", speedMBps)
                        } else {
                            // File moved/cleaned up cache, omit negative speed
                        }
                        
                        self.lastBytes = currentBytes
                        self.lastUpdate = now
                    }
                    
                    var completedMB = String(format: "%.1f", Double(self.lastBytes) / 1_048_576)
                    var totalMB = "???"
                    if fraction > 0.001 {
                        let extrapolated = (Double(self.lastBytes) / fraction) / 1_048_576.0
                        totalMB = String(format: "%.1f", extrapolated)
                    } else if fraction == 0.0 {
                         completedMB = "0.0"
                    }
                    
                    let barLength = 20
                    let completedBars = min(barLength, Int(fraction * Double(barLength)))
                    let emptyBars = max(0, barLength - completedBars)
                    
                    var bars = ""
                    if completedBars > 0 {
                        bars += String(repeating: "=", count: completedBars - 1) + ">"
                    }
                    bars += String(repeating: " ", count: emptyBars)
                    
                    let pctStr = String(format: "%3d%%", pct)
                    let spinner = self.spinnerFrames[self.frameIndex]
                    let speedText = "| Speed: \(self.speedStr)"
                    
                    let msg = String(format: "\r[SwiftLM] Download: [%@] %@ %@ (%@ MB / %@ MB) %@", bars, pctStr, spinner, completedMB, totalMB, speedText)
                    
                    print(msg.padding(toLength: 100, withPad: " ", startingAt: 0), terminator: "")
                    fflush(stdout)
                    
                    if fraction >= 1.0 {
                        print("")
                        self.isDone = true
                        break
                    }
                    
                    do {
                        try await Task.sleep(nanoseconds: 100_000_000) // 100ms
                    } catch {
                        break
                    }
                }
            }
        }
    }
}

@main
struct MLXServer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "SwiftLM",
        abstract: "OpenAI-compatible LLM server powered by Apple MLX"
    )

    @Option(name: .long, help: "HuggingFace model ID or local path")
    var model: String

    @Option(name: .long, help: "Port to listen on")
    var port: Int = 5413

    @Option(name: .long, help: "Host to bind")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Max tokens to generate per request (default)")
    var maxTokens: Int = 2048

    @Option(name: .long, help: "Context window size (KV cache). When set, uses sliding window cache")
    var ctxSize: Int?

    @Option(name: .long, help: "Default sampling temperature (0 = greedy, overridable per-request)")
    var temp: Float = 0.6

    @Option(name: .long, help: "Default top-p nucleus sampling (overridable per-request)")
    var topP: Float = 1.0

    @Option(name: .long, help: "Default top-k sampling (overridable per-request)")
    var topK: Int?

    @Option(name: .long, help: "Default min-p sampling (overridable per-request)")
    var minP: Float?

    @Option(name: .long, help: "Repetition penalty factor (overridable per-request)")
    var repeatPenalty: Float?

    @Option(name: .long, help: "Number of parallel request slots")
    var parallel: Int = 1

    @Flag(name: .long, help: "Enable thinking/reasoning mode (Qwen3.5 etc). Default: disabled")
    var thinking: Bool = false

    @Flag(name: .long, help: "Enable VLM (vision-language model) mode for image inputs")
    var vision: Bool = false

    @Flag(name: .long, help: "Enable ALM (audio-language model) mode for audio inputs")
    var audio: Bool = false

    @Option(name: .long, help: "GPU memory limit in MB (default: system limit)")
    var memLimit: Int?

    @Option(name: .long, help: "API key for bearer token authentication")
    var apiKey: String?

    @Flag(name: .long, help: "Profile model memory requirements and exit (dry-run)")
    var info: Bool = false

    @Option(name: .long, help: "Number of layers to run on GPU (\"auto\" or integer, default: auto)")
    var gpuLayers: String?

    @Option(name: .long, help: "Allowed CORS origin (* for all, or a specific origin URL)")
    var cors: String?

    @Flag(name: .long, help: "Force re-calibration of optimal memory settings (normally auto-cached)")
    var calibrate: Bool = false

    @Flag(name: .long, help: "Enable SSD expert streaming for MoE models (Flash-MoE style memory-mapping)")
    var streamExperts: Bool = false

    @Flag(name: .long, help: "Enable 16-worker background SSD thread pool queue (PAPPS). Requires --stream-experts.")
    var ssdPrefetch: Bool = false

    @Flag(name: .long, help: "Enable TurboQuant KV-cache compression (3-bit PolarQuant+QJL). Compresses KV history > 8192 tokens to ~3.5 bits/token — recommended for 100k+ context. Default: disabled")
    var turboKV: Bool = false

    @Option(name: .long, help: "Chunk size for prefill evaluation (default: 512, lower to prevent GPU timeout on large models)")
    var prefillSize: Int = 512

    @Option(name: .long, help: "Draft model for speculative decoding (local path or HuggingFace ID). Must share tokenizer with main model.")
    var draftModel: String?

    @Option(name: .long, help: "Number of draft tokens per speculation round (default: 4)")
    var numDraftTokens: Int = 4

    @Flag(name: .long, help: "Enable DFlash block-diffusion speculative decoding. Requires a DFlash draft model (auto-resolved or specified via --draft-model).")
    var dflash: Bool = false

    @Option(name: .long, help: "DFlash block size (number of tokens per draft block). Default: use draft model's configured block_size.")
    var dflashBlockSize: Int?

    mutating func run() async throws {
        // Raise the open-file limit: large sharded models (e.g. Kimi K2.5, 182 safetensor
        // shards) + draft model + metallib + dylibs can exhaust the default macOS FD limit of 256.
        var rl = rlimit()
        getrlimit(RLIMIT_NOFILE, &rl)
        if rl.rlim_cur < 4096 {
            rl.rlim_cur = min(4096, rl.rlim_max)
            setrlimit(RLIMIT_NOFILE, &rl)
        }

        // Cap Metal command buffer size BEFORE any MLX operation to prevent the
        // 5-second Apple GPU Watchdog from killing processes under swap pressure.
        // This env var must be set before MLX's Metal backend initializes.
        // Value 50 splits large computation graphs into ~1-layer chunks so macOS
        // can page in weights incrementally without exceeding the watchdog timeout.
        if self.draftModel != nil || self.streamExperts {
            setenv("MLX_MAX_OPS_PER_BUFFER", "50", 1)
        }

        // Register SwiftLM-owned DFlash model types before any model loading.
        await registerDFlashModelTypes()

        print("[SwiftLM] Loading model: \(model)")
        let modelId = model

        // ── Load model ──
        var modelConfig: ModelConfiguration
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: modelId) {
            var isDir: ObjCBool = false
            fileManager.fileExists(atPath: modelId, isDirectory: &isDir)
            if isDir.boolValue {
                print("[SwiftLM] Loading from local directory: \(modelId)")
                modelConfig = ModelConfiguration(directory: URL(filePath: modelId))
            } else {
                modelConfig = ModelConfiguration(id: modelId)
            }
        } else {
            modelConfig = ModelConfiguration(id: modelId)
        }
        
        // Inject streaming flag into config to bypass eval(model) if requested
        if self.streamExperts {
            modelConfig.lazyLoad = true
        }

        // ── Strategy: --stream-experts + --draft-model ───────────────────────────
        // README.md notes speculative decoding is "counterproductive" for SSD-streaming
        // MoE at the default 4 draft tokens: the verify pass sends N+1 positions each
        // routing to *different* experts, scaling SSD I/O by the union of all expert
        // selections across every position simultaneously.
        //
        // However, with numDraftTokens = 1, the verify pass sends only 2 positions —
        // minimal fan-out. If the draft acceptance rate is ≥ 50%, the draft model's
        // speed advantage (~73 tok/s) still yields net positive throughput despite the
        // 2× SSD I/O overhead, especially on models where the draft hit rate is high.
        //
        // Strategy: auto-cap numDraftTokens to 1 and print a performance advisory.
        // This keeps the combination functional while minimising the fan-out penalty.
        // Users who understand the tradeoff can still benefit from the draft model.
        if self.streamExperts, self.draftModel != nil {
            if self.numDraftTokens > 1 {
                print("[SwiftLM] ⚠️  SSD streaming + draft model: auto-capping --num-draft-tokens to 1")
                print("[SwiftLM]    With N>1 draft tokens the verify pass fans expert I/O across N+1 SSD")
                print("[SwiftLM]    positions simultaneously, which regresses throughput vs no draft model.")
                print("[SwiftLM]    At 1 draft token (2 positions) the fan-out is minimal and net positive")
                print("[SwiftLM]    if draft acceptance rate ≥ 50%.")
                print("[SwiftLM]    ℹ️  For best throughput: use --stream-experts alone (no draft model).")
                self.numDraftTokens = 1
            } else {
                print("[SwiftLM] ℹ️  SSD streaming + draft model (1 token/round): minimal fan-out mode active.")
            }
        }

        // ── Pre-load profiling ──
        // Resolve model directory for profiling (checks HuggingFace cache)
        let modelDirectory = resolveModelDirectory(modelId: modelId)
        
        // ── Fix #72: Compute draft model footprint ONCE (Copilot review) ──────
        // Resolved before the streamExperts block so the exact byte count can be
        // reused for the early cap, both strategy branches, and logging without
        // repeating the filesystem walk.  Use weightFileSizeBytes (exact bytes)
        // instead of weightMemoryGB * 1_073_741_824 to avoid the ~7% GiB/GB
        // mismatch flagged in Copilot review (weightMemoryGB = bytes / 1e9, not /2^30).
        let draftFootprintBytes: Int
        if let draftPath = self.draftModel,
           let draftDir = resolveModelDirectory(modelId: draftPath),
           let draftProfile = ModelProfiler.profile(modelDirectory: draftDir, modelId: draftPath) {
            draftFootprintBytes = draftProfile.weightFileSizeBytes
        } else {
            draftFootprintBytes = 0
        }

        var mainModelProfile: ModelProfile? = nil

        if self.streamExperts, let modelDir = modelDirectory {
            setenv("EXPERIMENTAL_SSD_STREAM", modelDir.path, 1)
            // Activate the modern Swift ExpertStreamingConfig so Load.swift can:
            //  1. Initialize ExpertStreamerManager (shard index for getFile())
            //  2. Assign tensorName on every QuantizedSwitchLinear after quantize()
            // Without this, the streamedGatherMM direct-NVMe path never fires.
            ExpertStreamingConfig.shared.activate(
                modelDirectory: modelDir,
                useDirectIO: true  // macOS: 5 GB/s pread() via moe_stream_op.cpp
            )
            // Cap Metal command buffer size to avoid the 5s Apple GPU Watchdog.
            setenv("MLX_MAX_OPS_PER_BUFFER", "50", 1)
            print("[SwiftLM] Enabled Async SSD Streaming on directory: \(modelDir.lastPathComponent)")

            // ── Fix #72 (inference-time): Context-aware memoryLimit ────────────
            // The 200 GB sentinel bypasses MLX eval_impl's spin-wait loop and is
            // safe for SSD streaming alone, because only one model's expert pages
            // are demanded at a time.
            //
            // With --draft-model, speculative decoding alternates between the draft
            // model and the main model in tight succession.  If combined weights
            // exceed physical RAM, both models' pages thrash the SSD page cache
            // simultaneously, and the 200 GB sentinel lets MLX demand 40+ GB
            // without any back-pressure — swapping out to disk aggressively.
            //
            // Fix: when the combined footprint exceeds 70% of physical RAM, lower
            // memoryLimit to physicalRAM × 1.1.  MLX will then hit its hard limit
            // sooner and begin evicting old expert pages more aggressively instead
            // of extending into swap.
            let system = ModelProfiler.systemProfile()
            if draftFootprintBytes > 0 {
                print("[SwiftLM] 📦 Draft model footprint: \(String(format: "%.2f", Double(draftFootprintBytes) / 1e9))GB reserved from SSD budget")
            }
            Memory.cacheLimit = computeSSDMemoryBudget(totalRAMBytes: system.totalRAMBytes, draftWeightBytes: draftFootprintBytes)

            // Determine safe memoryLimit sentinel
            mainModelProfile = ModelProfiler.profile(modelDirectory: modelDir, modelId: modelId)
            let mainFootprintBytes = mainModelProfile?.weightFileSizeBytes ?? 0
            let combinedFootprint = mainFootprintBytes + draftFootprintBytes
            let physicalRAM = Int(system.totalRAMBytes)
            let combinedExceedsRAM = combinedFootprint > Int(Double(physicalRAM) * 0.70)

            if combinedExceedsRAM && draftFootprintBytes > 0 {
                // Combined model weights exceed 70% of physical RAM.
                // Speculative decoding causes both models' pages to be demanded
                // simultaneously during draft+verify cycles, which will thrash
                // the SSD page cache and trigger heavy swap.
                // Use a tight memoryLimit so MLX evicts pages rather than swapping.
                let tightLimit = Int(Double(physicalRAM) * 1.1)
                Memory.memoryLimit = tightLimit
                print("[SwiftLM] ⚠️  SSD + draft-model RAM pressure warning:")
                print("[SwiftLM]    Main model: \(String(format: "%.1f", Double(mainFootprintBytes) / 1e9))GB  Draft: \(String(format: "%.1f", Double(draftFootprintBytes) / 1e9))GB  Combined: \(String(format: "%.1f", Double(combinedFootprint) / 1e9))GB  Physical RAM: \(String(format: "%.1f", Double(physicalRAM) / 1e9))GB")
                print("[SwiftLM]    Speculative decoding alternates both models' forward passes.")
                print("[SwiftLM]    On this machine the combined weight exceeds physical RAM,")
                print("[SwiftLM]    causing page-cache thrashing and swap during inference.")
                print("[SwiftLM]    → Recommendation: remove --draft-model on this machine,")
                print("[SwiftLM]      or use a smaller draft model whose weights fit in")
                print("[SwiftLM]      remaining RAM after the main model's page budget (\(Memory.cacheLimit / (1024*1024*1024))GB).")
                print("[SwiftLM]    Memory limit set to \(tightLimit / (1024*1024*1024))GB (tight cap for MLX eviction pressure)")
            } else {
                // No draft model, or combined fits in RAM — use the standard sentinel
                // to bypass MLX eval_impl's spin-wait loop safely.
                Memory.memoryLimit = 200 * 1024 * 1024 * 1024 // 200 GB sentinel
            }
        } else if self.streamExperts {
            // modelDirectory is nil — model not yet downloaded (first-run).
            // Still apply the SSD memory cap so the download itself is bounded.
            let system = ModelProfiler.systemProfile()
            Memory.cacheLimit = computeSSDMemoryBudget(totalRAMBytes: system.totalRAMBytes, draftWeightBytes: draftFootprintBytes)
            Memory.memoryLimit = 200 * 1024 * 1024 * 1024 // 200 GB sentinel
        }
        
        var partitionPlan: PartitionPlan?
        if let modelDir = modelDirectory {
           let profile = mainModelProfile ?? ModelProfiler.profile(modelDirectory: modelDir, modelId: modelId)
           if let profile = profile {
            let system = ModelProfiler.systemProfile()
            let contextSize = self.ctxSize ?? 4096
            let plan = ModelProfiler.plan(model: profile, system: system, contextSize: contextSize, draftWeightBytes: draftFootprintBytes)
            partitionPlan = plan

            // --info mode: print report and exit
            if self.info {
                ModelProfiler.printReport(plan: plan, model: profile, system: system)
                return
            }

            // Apply memory strategy
            switch plan.strategy {
            case .fullGPU:
                print("[SwiftLM] \(plan.strategy.emoji) Memory strategy: FULL GPU (\(String(format: "%.1f", plan.weightMemoryGB))GB model, \(String(format: "%.1f", system.availableRAMGB))GB available)")
            case .swapAssisted:
                if self.streamExperts {
                    // SSD Streaming: expert weights are mmap'd from SSD via the OS page cache.
                    // No swap involved — the page cache evicts stale expert pages cleanly.
                    // draftFootprintBytes pre-computed once above (Copilot review).
                    let physicalBudget = computeSSDMemoryBudget(totalRAMBytes: system.totalRAMBytes, draftWeightBytes: draftFootprintBytes)
                    Memory.cacheLimit = physicalBudget
                    print("[SwiftLM] 💾 Memory strategy: SSD STREAMING (page-cache managed, \(physicalBudget / (1024*1024*1024))GB RAM budget, no swap)")
                } else {
                    Memory.cacheLimit = plan.recommendedCacheLimit
                    print("[SwiftLM] \(plan.strategy.emoji) Memory strategy: SWAP-ASSISTED (\(String(format: "%.1f", plan.overcommitRatio))× overcommit, cache limited to \(plan.recommendedCacheLimit / (1024*1024))MB)")
                    for w in plan.warnings { print("[SwiftLM]    \(w)") }
                }
            case .layerPartitioned:
                if self.streamExperts {
                    // draftFootprintBytes pre-computed once above (Copilot review).
                    let physicalBudget = computeSSDMemoryBudget(totalRAMBytes: system.totalRAMBytes, draftWeightBytes: draftFootprintBytes)
                    Memory.cacheLimit = physicalBudget
                    print("[SwiftLM] 💾 Memory strategy: SSD STREAMING (page-cache managed, \(physicalBudget / (1024*1024*1024))GB RAM budget, no swap)")
                } else {
                    Memory.cacheLimit = plan.recommendedCacheLimit
                    print("[SwiftLM] \(plan.strategy.emoji) Memory strategy: LAYER PARTITIONED (\(plan.recommendedGPULayers)/\(plan.totalLayers) GPU layers, cache limited to \(plan.recommendedCacheLimit / (1024*1024))MB)")
                    for w in plan.warnings { print("[SwiftLM]    \(w)") }
                }
            case .tooLarge:
                Memory.cacheLimit = plan.recommendedCacheLimit
                print("[SwiftLM] \(plan.strategy.emoji) WARNING: Model is \(String(format: "%.1f", plan.overcommitRatio))× system RAM. Loading will be extremely slow.")
                for w in plan.warnings { print("[SwiftLM]    \(w)") }
            }
           }
        } else if self.info {
            print("[SwiftLM] Model not yet downloaded. Run without --info to download first, or provide a local path.")
            return
        }

        // ── Determine GPU layer count ──
        // Priority: 1) explicit --gpu-layers flag, 2) partition plan auto, 3) nil (all GPU)
        var requestedGPULayers: Int? = nil
        if let gpuLayersArg = self.gpuLayers {
            if gpuLayersArg == "auto" {
                // Use partition plan recommendation if available
                requestedGPULayers = partitionPlan?.recommendedGPULayers
                print("[SwiftLM] --gpu-layers auto → \(requestedGPULayers.map(String.init) ?? "all") layers on GPU")
            } else if let n = Int(gpuLayersArg) {
                requestedGPULayers = n
                print("[SwiftLM] --gpu-layers \(n) → \(n) layers on GPU")
            } else {
                print("[SwiftLM] Warning: --gpu-layers must be 'auto' or an integer, got '\(gpuLayersArg)'. Using all GPU.")
            }
        } else if let plan = partitionPlan,
                  (plan.strategy == .layerPartitioned || plan.strategy == .swapAssisted),
                  plan.overcommitRatio > 1.0 {
            if self.streamExperts {
                print("[SwiftLM] SSD Streaming active: Bypassing CPU auto-partitioning (forcing all layers to GPU)")
                partitionPlan?.gpuLayers = plan.totalLayers
                // Keep requestedGPULayers = nil (all GPU)
            } else {
                // Auto-partition when model exceeds available RAM (no flag needed)
                requestedGPULayers = plan.recommendedGPULayers
                print("[SwiftLM] Auto-partitioning: \(plan.recommendedGPULayers)/\(plan.totalLayers) layers on GPU")
            }
        }

        let cacheRoot = URL.applicationSupportDirectory
            .appendingPathComponent("MLX", isDirectory: true)
            .appendingPathComponent("HuggingFace", isDirectory: true)
        let hub = HubApi(downloadBase: cacheRoot)
        let downloader = HubDownloader(hub: hub)
        let architecture = try await ModelArchitectureProbe.inspect(
            configuration: modelConfig,
            downloader: downloader
        )
        let isVision = self.vision
        let container: ModelContainer
        
        // Handle getting the simple model ID string for the tracker
        let resolvedModelId: String = {
            if case .id(let idStr, _) = modelConfig.id { return idStr }
            return self.model
        }()
        let tracker = ProgressTracker(modelId: resolvedModelId)
        
        let isAudio = self.audio
        if isVision && isAudio {
            print("[SwiftLM] Loading Omni-Language Model (Text + Vision + Audio)...")
            container = try await OmniModelFactory.shared.loadContainer(
                from: downloader,
                using: TransformersTokenizerLoader(),
                configuration: modelConfig
            ) { progress in
                tracker.printProgress(progress)
            }
        } else if isVision {
            print("[SwiftLM] Loading VLM (vision-language model)...")
            container = try await VLMModelFactory.shared.loadContainer(
                from: downloader,
                using: TransformersTokenizerLoader(),
                configuration: modelConfig
            ) { progress in
                tracker.printProgress(progress)
            }
        } else if isAudio {
            print("[SwiftLM] Loading ALM (audio-language model)...")
            // Use OmniModelFactory (VLM-backed) so Gemma4's audio tower is loaded
            // and the native prepareForMultimodal path extracts real mel features.
            container = try await OmniModelFactory.shared.loadContainer(
                from: downloader,
                using: TransformersTokenizerLoader(),
                configuration: modelConfig
            ) { progress in
                tracker.printProgress(progress)
            }
        } else {
            print("[SwiftLM] Loading LLM (large language model)...")
            container = try await LLMModelFactory.shared.loadContainer(
                from: downloader,
                using: TransformersTokenizerLoader(),
                configuration: modelConfig
            ) { progress in
                tracker.printProgress(progress)
            }
        }

        print("[SwiftLM] Loaded model configuration. Inferred tool call format: \(String(describing: await container.configuration.toolCallFormat))")

        // ── Check if target model supports DFlash ──
        let dflashTargetModel: (any DFlashTargetModel)? = await container.perform { context -> (any DFlashTargetModel)? in
            context.model as? any DFlashTargetModel
        }
        if self.dflash {
            if dflashTargetModel != nil {
                print("[SwiftLM] DFlash: target model supports DFlashTargetModel")
            } else {
                print("[SwiftLM] ⚠️  DFlash enabled but target model does NOT conform to DFlashTargetModel")
            }
        }

        // ── Load draft model for speculative decoding ──
        let draftModelRef: DraftModelRef?
        let numDraftTokensConfig = self.numDraftTokens
        if let draftModelPath = self.draftModel, !self.dflash {
            print("[SwiftLM] Loading draft model for speculative decoding: \(draftModelPath)")
            var draftConfig: ModelConfiguration
            let draftFM = FileManager.default
            if draftFM.fileExists(atPath: draftModelPath) {
                var isDir: ObjCBool = false
                draftFM.fileExists(atPath: draftModelPath, isDirectory: &isDir)
                if isDir.boolValue {
                    draftConfig = ModelConfiguration(directory: URL(filePath: draftModelPath))
                } else {
                    draftConfig = ModelConfiguration(id: draftModelPath)
                }
            } else {
                draftConfig = ModelConfiguration(id: draftModelPath)
            }
            // Fix #72: mirror lazyLoad so the draft model's weights are mmap'd
            // (not eagerly paged into unified RAM) when SSD streaming is active.
            if self.streamExperts {
                draftConfig.lazyLoad = true
            }
            let draftDownloader = HubDownloader(hub: HubApi(downloadBase: cacheRoot))
            let draftContainer = try await LLMModelFactory.shared.loadContainer(
                from: draftDownloader,
                using: TransformersTokenizerLoader(),
                configuration: draftConfig
            ) { progress in
                // Silent loading for draft model
            }
            draftModelRef = await draftContainer.extractDraftModel()
            print("[SwiftLM] Draft model loaded successfully (\(numDraftTokensConfig) tokens/round)")
            print("[SwiftLM] Using speculative decoding: \(draftModelPath) → \(modelId) (\(numDraftTokensConfig) draft tokens/round)")
        } else {
            draftModelRef = nil
        }

        // ── Load DFlash draft model for block-diffusion speculative decoding ──
        let dflashModel: DFlashDraftModel?
        let dflashBlockSizeConfig = self.dflashBlockSize
        let dflashConfig = DFlashDraftConfiguration.self
        if self.dflash {
            // Resolve draft model reference
            let resolvedDraftRef: String
            if let explicit = self.draftModel {
                resolvedDraftRef = explicit
            } else if let autoRef = DFlashDraftRegistry.resolveDraftRef(modelRef: modelId) {
                resolvedDraftRef = autoRef
                print("[SwiftLM] DFlash: auto-resolved draft model → \(autoRef)")
            } else {
                print("[SwiftLM] ⚠️  DFlash enabled but no draft model found for '\(modelId)'. Use --draft-model to specify one.")
                resolvedDraftRef = ""
            }

            if !resolvedDraftRef.isEmpty {
                print("[SwiftLM] Loading DFlash draft model: \(resolvedDraftRef)")
                let draftDir = resolveModelDirectory(modelId: resolvedDraftRef)
                if let dir = draftDir {
                    do {
                        let configURL = dir.appendingPathComponent("config.json")
                        let data = try Data(contentsOf: configURL)
                        let config = try JSONDecoder().decode(dflashConfig, from: data)
                        let model = DFlashDraftModel(config)

                        // Load weights
                        let weightURL = dir.appendingPathComponent("weights.safetensors")
                        let ntURL = dir.appendingPathComponent("model.safetensors")
                        let actualWeightURL = FileManager.default.fileExists(atPath: weightURL.path) ? weightURL : ntURL

                        let weights = try loadArrays(url: actualWeightURL)
                        let sanitized = model.sanitize(weights: weights)
                        let parameters = ModuleParameters.unflattened(sanitized)
                        try model.update(parameters: parameters, verify: .none)

                        dflashModel = model
                        // Register DFlashKernels as the global provider
                        // so Qwen35GatedDeltaNet can use tape-recording forward
                        DFlashKernelRegistry.provider = DFlashKernels.shared
                        DFlashDumper.setup()
                        print("[SwiftLM] DFlash draft model loaded (block_size=\(model.blockSize), \(model.targetLayerIDs.count) target layers, mask_token=\(model.maskTokenID))")
                        print("[SwiftLM] Draft model loaded successfully (\(model.blockSize) block size, DFlash mode)")
                        print("[SwiftLM] Using speculative decoding: \(resolvedDraftRef) → \(modelId) (DFlash block-diffusion)")
                    } catch {
                        print("[SwiftLM] ⚠️  Failed to load DFlash draft model: \(error)")
                        dflashModel = nil
                    }
                } else {
                    print("[SwiftLM] ⚠️  DFlash draft model not found locally: \(resolvedDraftRef). Download it first with: hf download \(resolvedDraftRef)")
                    dflashModel = nil
                }
            } else {
                dflashModel = nil
            }
        } else {
            dflashModel = nil
        }


        // ── Apply GPU/CPU layer partitioning ──
        if let gpuCount = requestedGPULayers {
            let actual = await container.setGPULayers(gpuCount)
            if let actual {
                let total = partitionPlan?.totalLayers ?? actual
                let cpuCount = total - actual
                print("[SwiftLM] 🔀 Layer split active: \(actual) GPU / \(cpuCount) CPU")
                // Update the partition plan to reflect actual split
                partitionPlan?.gpuLayers = actual
            } else {
                print("[SwiftLM] ⚠️  Model does not support layer partitioning (architecture not yet adapted)")
            }
        }

        // ── Apply SSD Expert Streaming ──
        if self.streamExperts {
            let streamingEnabled = await container.setStreamExperts(true)
            if streamingEnabled {
                print("[SwiftLM] 💾 SSD Expert Streaming enabled (lazy load + layer-sync)")
                if self.ssdPrefetch {
                    MLXFast.setPrefetchEnabled(true)
                    print("[SwiftLM] 🚀 PAPPS 16-Worker Thread Pool prefetcher enabled!")
                }
            } else {
                print("[SwiftLM] ⚠️  Model does not support SSD expert streaming")
            }
        }

        // ── Auto-calibration (Wisdom system) ──
        if let plan = partitionPlan, !self.streamExperts {
            if self.calibrate {
                // Force re-calibration
                if let wisdom = try? await Calibrator.calibrate(
                    container: container, plan: plan, modelId: modelId,
                    contextSize: self.ctxSize ?? 4096
                ) {
                    Memory.cacheLimit = wisdom.cacheLimit
                }
            } else if let wisdom = Calibrator.loadWisdom(modelId: modelId) {
                // Load cached wisdom
                if wisdom.cacheLimit > 0 {
                    Memory.cacheLimit = wisdom.cacheLimit
                }
                print("[SwiftLM] 📊 Loaded wisdom: \(String(format: "%.1f", wisdom.tokPerSec)) tok/s, cache=\(wisdom.cacheLimit / (1024*1024))MB (calibrated \(wisdom.calibratedAt.formatted(.relative(presentation: .named))))")
            }
        } else if self.streamExperts {
            print("[SwiftLM] 🧠 Auto-calibration (Wisdom) bypassed for SSD Streaming")
        }

        print("[SwiftLM] Model loaded. Starting HTTP server on \(host):\(port)")

        // ── Capture CLI defaults into a shared config ──
        let config = ServerConfig(
            modelId: modelId,
            maxTokens: self.maxTokens,
            ctxSize: self.ctxSize,
            temp: self.temp,
            topP: self.topP,
            topK: self.topK,
            minP: self.minP,
            repeatPenalty: self.repeatPenalty,
            thinking: self.thinking,
            isVision: isVision,
            prefillSize: self.prefillSize,
            turboKV: self.turboKV
        )

        let parallelSlots = self.parallel
        let corsOrigin = self.cors
        let apiKeyValue = self.apiKey

        // ── Memory limit enforcement (overrides wisdom) ──
        if let memLimitMB = self.memLimit {
            let bytes = memLimitMB * 1024 * 1024
            Memory.memoryLimit = bytes
            Memory.cacheLimit = bytes
            print("[SwiftLM] Memory limit set to \(memLimitMB)MB (overrides wisdom)")
        }

        // ── Concurrency limiter ──
        let semaphore = AsyncSemaphore(limit: parallelSlots)

        // ── Server stats tracker ──
        let stats = ServerStats()

        let ctxSizeStr = config.ctxSize.map { String($0) } ?? "model_default"
        let topKStr = config.topK.map { String($0) } ?? "disabled"
        let minPStr = config.minP.map { String($0) } ?? "disabled"
        let penaltyStr = config.repeatPenalty.map { String($0) } ?? "disabled"
        let corsStr = corsOrigin ?? "disabled"
        let memLimitStr = self.memLimit.map { "\($0)MB" } ?? "system_default"
        let authStr = apiKeyValue != nil ? "enabled" : "disabled"
        let thinkingStr = config.thinking ? "enabled" : "disabled"
        let ssdStr = self.streamExperts ? "enabled" : "disabled"
        let turboKVStr = config.turboKV ? "enabled" : "disabled"
        print("[SwiftLM] Config: ctx_size=\(ctxSizeStr), temp=\(config.temp), top_p=\(config.topP), top_k=\(topKStr), min_p=\(minPStr), repeat_penalty=\(penaltyStr), parallel=\(parallelSlots), cors=\(corsStr), mem_limit=\(memLimitStr), auth=\(authStr), thinking=\(thinkingStr), ssd_stream=\(ssdStr), turbo_kv=\(turboKVStr)")

        // ── Build Hummingbird router ──
        let router = Router()

        // ── CORS middleware ──
        if let origin = corsOrigin {
            router.add(middleware: CORSMiddleware(allowedOrigin: origin))
        }

        // ── API key authentication middleware ──
        if let key = apiKeyValue {
            router.add(middleware: ApiKeyMiddleware(apiKey: key))
        }

        // Health (enhanced v3 with memory + stats + partition plan)
        let isSSDStream = self.streamExperts  // capture before escaping closure
        router.get("/health") { [partitionPlan] _, _ -> Response in
            let activeMemMB = Memory.activeMemory / (1024 * 1024)
            let peakMemMB = Memory.peakMemory / (1024 * 1024)
            let cacheMemMB = Memory.cacheMemory / (1024 * 1024)
            let deviceInfo = GPU.deviceInfo()
            let totalMemMB = deviceInfo.memorySize / (1024 * 1024)
            let snapshot = await stats.snapshot()
            // Build partition info string
            var partitionJson = ""
            if let plan = partitionPlan {
                let isSSD = isSSDStream
                let pData: [String: Any] = [
                    "strategy": isSSD ? "ssd_streaming" : plan.strategy.rawValue,
                    "overcommit_ratio": round(plan.overcommitRatio * 100) / 100,
                    "model_weight_gb": round(plan.weightMemoryGB * 10) / 10,
                    "kv_cache_gb": round(plan.kvCacheMemoryGB * 10) / 10,
                    "total_required_gb": round(plan.totalRequiredGB * 10) / 10,
                    "gpu_layers": isSSD ? plan.totalLayers : plan.gpuLayers,
                    "cpu_layers": isSSD ? 0 : (plan.totalLayers - plan.gpuLayers),
                    "total_layers": plan.totalLayers,
                    "estimated_tok_s": isSSD
                        ? round(max(plan.estimatedTokensPerSec, plan.estimatedTokensPerSec * plan.overcommitRatio) * 10) / 10
                        : round(plan.estimatedTokensPerSec * 10) / 10,
                    "ssd_stream": isSSD
                ]
                if let pJson = try? JSONSerialization.data(withJSONObject: pData),
                   let pStr = String(data: pJson, encoding: .utf8) {
                    partitionJson = ",\"partition\":\(pStr)"
                }
            }
            let payload = """
{"status":"ok","model":"\(modelId)","vision":\(isVision),"memory":{"active_mb":\(activeMemMB),"peak_mb":\(peakMemMB),"cache_mb":\(cacheMemMB),"total_system_mb":\(totalMemMB),"gpu_architecture":"\(deviceInfo.architecture)"},"stats":{"requests_total":\(snapshot.requestsTotal),"requests_active":\(snapshot.requestsActive),"tokens_generated":\(snapshot.tokensGenerated),"avg_tokens_per_sec":\(String(format: "%.2f", snapshot.avgTokensPerSec))}\(partitionJson)}
"""
            return Response(
                status: .ok,
                headers: jsonHeaders(),
                body: .init(byteBuffer: ByteBuffer(string: payload))
            )
        }

        // Models list
        router.get("/v1/models") { _, _ -> Response in
            let payload = """
            {"object":"list","data":[{"id":"\(modelId)","object":"model","created":\(Int(Date().timeIntervalSince1970)),"owned_by":"mlx-community"}]}
            """
            return Response(
                status: .ok,
                headers: jsonHeaders(),
                body: .init(byteBuffer: ByteBuffer(string: payload))
            )
        }

        // Chat completions — handler extracted to avoid type-checker timeout
        let promptCache = PromptCache()
        router.post("/v1/chat/completions") { request, _ -> Response in
            do {
                let bodyData = try await collectBody(request)
                return try await handleChatCompletion(
                    request: request, bodyData: bodyData, config: config, container: container, semaphore: semaphore, stats: stats, promptCache: promptCache,
                    draftModelRef: draftModelRef, numDraftTokens: numDraftTokensConfig,
                    dflashModel: dflashModel, dflashBlockSize: dflashBlockSizeConfig,
                    dflashTargetModel: dflashTargetModel
                )
            } catch {
                let errMsg = String(describing: error).replacingOccurrences(of: "\"", with: "'")
                let payload = """
                {"error":{"message":"\(errMsg)","type":"server_error","code":"internal_error"}}
                """
                return Response(
                    status: .internalServerError,
                    headers: jsonHeaders(),
                    body: .init(byteBuffer: ByteBuffer(string: payload))
                )
            }
        }

        // Text completions — handler extracted to avoid type-checker timeout
        router.post("/v1/completions") { request, _ -> Response in
            do {
                let bodyData = try await collectBody(request)
                return try await handleTextCompletion(
                    request: request, bodyData: bodyData, config: config, container: container, semaphore: semaphore, stats: stats
                )
            } catch {
                let errMsg = String(describing: error).replacingOccurrences(of: "\"", with: "'")
                let payload = """
                {"error":{"message":"\(errMsg)","type":"server_error","code":"internal_error"}}
                """
                return Response(
                    status: .internalServerError,
                    headers: jsonHeaders(),
                    body: .init(byteBuffer: ByteBuffer(string: payload))
                )
            }
        }

        // Prometheus-compatible metrics endpoint
        router.get("/metrics") { _, _ -> Response in
            let activeMemBytes = Memory.activeMemory
            let peakMemBytes = Memory.peakMemory
            let cacheMemBytes = Memory.cacheMemory
            let snapshot = await stats.snapshot()
            let uptime = snapshot.uptimeSeconds
            var lines: [String] = []
            lines.append("# HELP swiftlm_requests_total Total requests processed")
            lines.append("# TYPE swiftlm_requests_total counter")
            lines.append("swiftlm_requests_total \(snapshot.requestsTotal)")
            lines.append("# HELP swiftlm_requests_active Currently active requests")
            lines.append("# TYPE swiftlm_requests_active gauge")
            lines.append("swiftlm_requests_active \(snapshot.requestsActive)")
            lines.append("# HELP swiftlm_tokens_generated_total Total tokens generated")
            lines.append("# TYPE swiftlm_tokens_generated_total counter")
            lines.append("swiftlm_tokens_generated_total \(snapshot.tokensGenerated)")
            lines.append("# HELP swiftlm_tokens_per_second Average token generation rate")
            lines.append("# TYPE swiftlm_tokens_per_second gauge")
            lines.append("swiftlm_tokens_per_second \(String(format: "%.2f", snapshot.avgTokensPerSec))")
            lines.append("# HELP swiftlm_memory_active_bytes Active GPU memory usage")
            lines.append("# TYPE swiftlm_memory_active_bytes gauge")
            lines.append("swiftlm_memory_active_bytes \(activeMemBytes)")
            lines.append("# HELP swiftlm_memory_peak_bytes Peak GPU memory usage")
            lines.append("# TYPE swiftlm_memory_peak_bytes gauge")
            lines.append("swiftlm_memory_peak_bytes \(peakMemBytes)")
            lines.append("# HELP swiftlm_memory_cache_bytes Cached GPU memory")
            lines.append("# TYPE swiftlm_memory_cache_bytes gauge")
            lines.append("swiftlm_memory_cache_bytes \(cacheMemBytes)")
            lines.append("# HELP swiftlm_uptime_seconds Server uptime")
            lines.append("# TYPE swiftlm_uptime_seconds gauge")
            lines.append("swiftlm_uptime_seconds \(String(format: "%.0f", uptime))")

            // ── SSD Flash-Stream metrics (only emitted when --stream-experts is active) ──
            if isSSDStream {
                let ssd = MLXFast.ssdMetricsSnapshot()
                lines.append("# HELP swiftlm_ssd_throughput_mbps NVMe read throughput (10 s rolling average, MB/s)")
                lines.append("# TYPE swiftlm_ssd_throughput_mbps gauge")
                lines.append("swiftlm_ssd_throughput_mbps \(String(format: "%.1f", ssd.throughputMBperS))")
                lines.append("# HELP swiftlm_ssd_bytes_read_total Lifetime bytes read from SSD for expert weights")
                lines.append("# TYPE swiftlm_ssd_bytes_read_total counter")
                lines.append("swiftlm_ssd_bytes_read_total \(ssd.totalBytesRead)")
                lines.append("# HELP swiftlm_ssd_chunks_total Lifetime expert chunks loaded from SSD")
                lines.append("# TYPE swiftlm_ssd_chunks_total counter")
                lines.append("swiftlm_ssd_chunks_total \(ssd.totalChunks)")
                lines.append("# HELP swiftlm_ssd_chunk_latency_ms Average per-chunk SSD read latency (ms, lifetime)")
                lines.append("# TYPE swiftlm_ssd_chunk_latency_ms gauge")
                lines.append("swiftlm_ssd_chunk_latency_ms \(String(format: "%.4f", ssd.avgChunkLatencyMS))")
            }

            lines.append("")
            let metrics = lines.joined(separator: "\n")
            return Response(
                status: .ok,
                headers: HTTPFields([HTTPField(name: .contentType, value: "text/plain; version=0.0.4; charset=utf-8")]),
                body: .init(byteBuffer: ByteBuffer(string: metrics))
            )
        }

        // ── Start server ──
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        print("[SwiftLM] ✅ Ready. Listening on http://\(host):\(port)")

        // ── Emit machine-readable ready event for Aegis integration ──
        var readyEvent: [String: Any] = [
            "event": "ready",
            "port": port,
            "model": modelId,
            "engine": "mlx",
            "vision": isVision
        ]
        if let plan = partitionPlan {
            var info = plan.healthInfo
            if self.streamExperts {
                // SSD streaming bypasses swap — report accurate strategy and suppress swap estimate
                info["strategy"] = "ssd_streaming"
                info["ssd_stream"] = true
                // Measured 3.81 tok/s on 122B MoE; use a reasonable SSD-streaming estimate
                // (swap estimate is artificially divided by overcommit — not applicable here)
                let ssdEstimate = max(plan.estimatedTokensPerSec, plan.estimatedTokensPerSec * plan.overcommitRatio)
                info["estimated_tok_s"] = round(ssdEstimate * 10) / 10
                // All layers on GPU when SSD streaming is active
                info["gpu_layers"] = plan.totalLayers
                info["cpu_layers"] = 0
            }
            readyEvent["partition"] = info
        }
        if let data = try? JSONSerialization.data(withJSONObject: readyEvent),
           let json = String(data: data, encoding: .utf8) {
            print(json)
            fflush(stdout)
        }

        // ── Graceful shutdown on SIGTERM/SIGINT ──
        let shutdownSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        let interruptSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        signal(SIGTERM, SIG_IGN)
        signal(SIGINT, SIG_IGN)

        shutdownSource.setEventHandler {
            print("\n[SwiftLM] Received SIGTERM, shutting down gracefully...")
            Darwin.exit(0)
        }
        interruptSource.setEventHandler {
            print("\n[SwiftLM] Received SIGINT, shutting down gracefully...")
            Darwin.exit(0)
        }
        shutdownSource.resume()
        interruptSource.resume()

        try await app.runService()
    }
}

// ── Server Config ────────────────────────────────────────────────────────────

struct ServerConfig: Sendable {
    let modelId: String
    let maxTokens: Int
    let ctxSize: Int?
    let temp: Float
    let topP: Float
    let topK: Int?
    let minP: Float?
    let repeatPenalty: Float?
    let thinking: Bool
    let isVision: Bool
    let prefillSize: Int
    /// When true, each KVCacheSimple layer compresses history > 8192 tokens to 3-bit PolarQuant.
    let turboKV: Bool
}

// ── SSD Memory Budget ────────────────────────────────────────────────────────

/// Compute the page-cache budget (bytes) for SSD streaming mode.
///
/// Formula: `totalRAM × 0.85 − osHeadroom − draftWeightBytes`, floored at 2 GB.
///
/// - Parameters:
///   - totalRAMBytes: Physical RAM reported by the OS (e.g. `system.totalRAMBytes`).
///   - draftWeightBytes: Weight size (bytes) of the draft model, or 0 if none.
///     Subtracted so the draft model's resident pages don't push the main model's
///     page cache over the physical limit and trigger swap (Issue #72).
/// - Returns: The recommended `Memory.cacheLimit` value in bytes.
func computeSSDMemoryBudget(totalRAMBytes: UInt64, draftWeightBytes: Int = 0) -> Int {
    let osHeadroom = 4 * 1024 * 1024 * 1024  // 4 GB for OS + system processes
    let raw = Int(Double(totalRAMBytes) * 0.85) - osHeadroom - draftWeightBytes
    return max(raw, 2 * 1024 * 1024 * 1024)  // floor at 2 GB
}

// ── Model Directory Resolution ───────────────────────────────────────────────

/// Resolve a model ID to its local directory (if already downloaded).
/// Checks: 1) local path, 2) HuggingFace Hub cache.
/// Returns nil if the model hasn't been downloaded yet.
func resolveModelDirectory(modelId: String) -> URL? {
    let fm = FileManager.default

    // Direct local path
    var isDir: ObjCBool = false
    if fm.fileExists(atPath: modelId, isDirectory: &isDir), isDir.boolValue {
        let url = URL(filePath: modelId)
        // Verify config.json exists
        if fm.fileExists(atPath: url.appendingPathComponent("config.json").path) {
            return url
        }
    }

    // HuggingFace Hub cache: ~/Library/Caches/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
    // Also check: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
    let hubModelDir = modelId.replacingOccurrences(of: "/", with: "--")

    let cacheDirs: [URL] = [
        // macOS standard: ~/Library/Caches/huggingface
        fm.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("huggingface/hub/models--\(hubModelDir)"),
        // Unix standard: ~/.cache/huggingface
        fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--\(hubModelDir)")
    ].compactMap { $0 }

    for cacheDir in cacheDirs {
        let snapshotsDir = cacheDir.appendingPathComponent("snapshots")
        guard let snapshots = try? fm.contentsOfDirectory(at: snapshotsDir, includingPropertiesForKeys: [.isDirectoryKey]) else {
            continue
        }
        // Use the most recently modified snapshot
        let sorted = snapshots
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { a, b in
                let aDate = (try? fm.attributesOfItem(atPath: a.path)[.modificationDate] as? Date) ?? .distantPast
                let bDate = (try? fm.attributesOfItem(atPath: b.path)[.modificationDate] as? Date) ?? .distantPast
                return aDate > bDate
            }
        if let latest = sorted.first {
            if fm.fileExists(atPath: latest.appendingPathComponent("config.json").path) {
                return latest
            }
        }
    }

    return nil
}

// ── Server Stats Tracker ───────────────────────────────────────────────────────

actor ServerStats {
    private var requestsTotal: Int = 0
    private var requestsActive: Int = 0
    private var tokensGenerated: Int = 0
    private var totalGenerationTimeSeconds: Double = 0
    private let startTime = Date()

    struct Snapshot: Sendable {
        let requestsTotal: Int
        let requestsActive: Int
        let tokensGenerated: Int
        let avgTokensPerSec: Double
        let uptimeSeconds: TimeInterval
    }

    func requestStarted() {
        requestsTotal += 1
        requestsActive += 1
    }

    func requestFinished(tokens: Int, duration: TimeInterval) {
        requestsActive -= 1
        tokensGenerated += tokens
        totalGenerationTimeSeconds += duration
    }

    func snapshot() -> Snapshot {
        let tps = totalGenerationTimeSeconds > 0 ? Double(tokensGenerated) / totalGenerationTimeSeconds : 0
        return Snapshot(
            requestsTotal: requestsTotal,
            requestsActive: requestsActive,
            tokensGenerated: tokensGenerated,
            avgTokensPerSec: tps,
            uptimeSeconds: Date().timeIntervalSince(startTime)
        )
    }
}



actor PromptCache {
    struct CachedState {
        let tokens: [Int]            // Full token sequence that generated this KV state
        let states: [[MLXArray]]     // Per-layer KV state arrays
        let metaStates: [[String]]   // Per-layer metadata
    }

    private var cached: CachedState?
    private var hits: Int = 0
    private var misses: Int = 0

    /// Save the full prompt token sequence and its KV state.
    /// IMPORTANT: We must eval() the state arrays immediately. The state getter may
    /// produce lazy computation graphs (e.g. TurboKV decode → reshape → concatenate).
    /// If not materialized now, those lazy references point to the live cache tensors
    /// which get overwritten by subsequent requests, causing stale data / SIGTRAP on restore.
    func save(tokens: [Int], cache: [KVCache]) {
        if cache.contains(where: { $0 is MambaCache }) {
            return
        }
        let P = tokens.count
        // For attention KVCacheSimple layers, the state tensor is [B, H, T, D] with a
        // pre-allocated T that can exceed the actual prompt length P. If we store the
        // full over-sized buffer, restore()'s trim() by (cached.tokens.count - matchLen)
        // still leaves T - P slots of garbage beyond the valid prefix. Slice T to P at
        // save time so cached.tokens.count === cached state's T.
        let states: [[MLXArray]] = cache.map { layer -> [MLXArray] in
            let s = layer.state
            if layer is KVCacheSimple {
                return s.map { arr -> MLXArray in
                    guard arr.ndim >= 3 else { return arr }
                    let T = arr.dim(2)
                    if T > P { return arr[.ellipsis, ..<P, 0...] }
                    return arr
                }
            }
            return s
        }
        let metaStates = cache.map { $0.metaState }
        // Materialize all lazy MLX arrays so they survive cache mutations
        let allArrays = states.flatMap { $0 }
        if !allArrays.isEmpty {
            eval(allArrays)
        }
        cached = CachedState(tokens: tokens, states: states, metaStates: metaStates)
    }

    /// Find the longest common prefix between `newTokens` and the cached sequence.
    /// Restores matched KV state, trims any excess — mirrors llama-server behaviour.
    /// Returns the number of matched tokens, or nil on a complete miss.
    func restore(newTokens: [Int], into cache: [KVCache]) -> Int? {
        // MambaCache/RNN states cannot be arbitrarily rolled back or safely saved
        // after the fact without exact sequence-boundary synchronization.
        // Disable prompt caching entirely for hybrid models (e.g. Qwen3Next).
        if cache.contains(where: { $0 is MambaCache }) {
            misses += 1
            return nil
        }

        guard let cached, !cached.tokens.isEmpty else {
            misses += 1
            return nil
        }
        // ── Recurrent-layer safety gate ──
        // MambaCache (and other recurrent caches) store a 2-D hidden state with no
        // T dimension, so the dim(2) read below would crash. Hybrid Mamba/attention
        // models (Qwen-Next, Mamba-2, etc.) can't be safely prefix-restored because
        // the recurrent hidden state was computed over the WHOLE previous sequence
        // and there is no trim(excess) operator for it. Treat any cache containing
        // a recurrent layer as a miss before we touch anything.
        let hasRecurrentLayer = cache.contains { layer in
            !(layer is KVCacheSimple) && !(String(describing: type(of: layer)).contains("Rotating"))
        }
        if hasRecurrentLayer {
            misses += 1
            return nil
        }
        // Token-by-token longest common prefix scan
        var matchLen = 0
        for (a, b) in zip(cached.tokens, newTokens) {
            guard a == b else { break }
            matchLen += 1
        }
        guard matchLen > 0 else {
            misses += 1
            return nil
        }
        // Pre-flight safety check: compute the minimum sequence length across
        // all cached layers. Sliding-window layers (RotatingKVCache) store far
        // fewer tokens than the full prompt (e.g. 1440 vs 5537). If the trim
        // would zero-out any layer, bail BEFORE touching the live cache.
        let excess = cached.tokens.count - matchLen
        if excess > 0 {
            // The state getter stores keys as the first element: [B, H, T, D]
            // dim(2) = T = the number of cached tokens for that layer.
            let minCachedSeqLen = cached.states.map { arrays -> Int in
                guard let firstArray = arrays.first else { return 0 }
                guard firstArray.ndim >= 3 else { return 0 }
                return firstArray.dim(2)  // T dimension
            }.min() ?? 0
            if excess >= minCachedSeqLen {
                // Trim would empty or corrupt at least one layer → treat as miss
                misses += 1
                return nil
            }
        }
        // Safe to restore: trim won't corrupt any layer
        for i in 0..<min(cache.count, cached.states.count) {
            var layer = cache[i]
            layer.state = cached.states[i]
            layer.metaState = cached.metaStates[i]
        }
        if excess > 0 {
            for layer in cache { layer.trim(excess) }
        }
        hits += 1
        print("[SwiftLM] \u{1F5C2} Prompt cache HIT: \(matchLen)/\(newTokens.count) tokens reused (\(excess > 0 ? "partial" : "full") match)")
        return matchLen
    }

    func stats() -> (hits: Int, misses: Int) { (hits, misses) }
}

// ── Request Body Extraction ──────────────────────────────────────────────────

func collectBody(_ request: Request) async throws -> Data {
    var bodyBuffer = try await request.body.collect(upTo: 100 * 1024 * 1024)
    let bodyBytes = bodyBuffer.readBytes(length: bodyBuffer.readableBytes) ?? []
    return Data(bodyBytes)
}

// ── Chat Completions Handler ─────────────────────────────────────────────────

func handleChatCompletion(
    request: Request,
    bodyData: Data,
    config: ServerConfig,
    container: ModelContainer,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    promptCache: PromptCache,
    draftModelRef: DraftModelRef? = nil,
    numDraftTokens: Int = 4,
    dflashModel: DFlashDraftModel? = nil,
    dflashBlockSize: Int? = nil,
    dflashTargetModel: (any DFlashTargetModel)? = nil
) async throws -> Response {
    let chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: bodyData)
    let isStream = chatReq.stream ?? false
    let jsonMode = chatReq.responseFormat?.type == "json_object"
    let emitPrefillProgress = prefillProgressEnabled(in: request)

    // ── Merge per-request overrides with CLI defaults ──
    let tokenLimit = chatReq.maxTokens ?? config.maxTokens
    let temperature = chatReq.temperature.map(Float.init) ?? config.temp
    let topP = chatReq.topP.map(Float.init) ?? config.topP
    let topK = chatReq.topK ?? config.topK ?? 50
    let minP = chatReq.minP.map(Float.init) ?? config.minP ?? 0.0
    let repeatPenalty = chatReq.repetitionPenalty.map(Float.init) ?? config.repeatPenalty
    let stopSequences = (chatReq.stop ?? []) + ["<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<turn|>", "<|tool_response|>"]
    let includeUsage = chatReq.streamOptions?.includeUsage ?? false

    // Log extra sampling params if provided (accepted for API compat, not all are used)
    if chatReq.frequencyPenalty != nil || chatReq.presencePenalty != nil {
        // These are accepted but may not affect generation if MLX doesn't support them
    }

    // ── Validate kv_bits: only nil, 4, and 8 are supported ──
    if let kb = chatReq.kvBits, kb != 4 && kb != 8 {
        let errBody = "{\"error\":{\"message\":\"Invalid kv_bits value \(kb). Supported values are 4 and 8.\",\"type\":\"invalid_request_error\",\"code\":\"invalid_kv_bits\"}}"
        return Response(
            status: .badRequest,
            headers: jsonHeaders(),
            body: .init(byteBuffer: ByteBuffer(string: errBody))
        )
    }

    let params = GenerateParameters(
        maxTokens: tokenLimit,
        maxKVSize: config.ctxSize,
        kvBits: chatReq.kvBits,
        temperature: temperature,
        topP: topP,
        topK: topK,
        minP: minP,
        repetitionPenalty: repeatPenalty,
        prefillStepSize: config.prefillSize
    )

    // ── Seed for deterministic generation ──
    if let seed = chatReq.seed {
        MLXRandom.seed(UInt64(seed))
    }

    // ── Parse messages with multipart content support (for VLM images) ──
    var chatMessages: [Chat.Message] = []
    var systemPromptText = ""
    for msg in chatReq.messages {
        let chatMsg = msg.toChatMessage()
        chatMessages.append(chatMsg)
        if chatMsg.role == .system {
            systemPromptText += chatMsg.content
        }
    }

    // ── JSON mode: inject system prompt for JSON output ──
    if jsonMode {
        let jsonStr = "You must respond with valid JSON only. No markdown code fences, no explanation text, no preamble. Output raw JSON."
        if !chatMessages.isEmpty && chatMessages[0].role == .system {
            chatMessages[0].content += "\n\n" + jsonStr
        } else {
            chatMessages.insert(.system(jsonStr), at: 0)
        }
        systemPromptText = "JSON_MODE:" + systemPromptText
    }

    // Convert OpenAI tools format → [String: any Sendable] for UserInput
    let toolSpecs: [[String: any Sendable]]? = chatReq.tools?.map { tool in
        var spec: [String: any Sendable] = ["type": tool.type]
        var fn: [String: any Sendable] = ["name": tool.function.name]
        if let desc = tool.function.description { fn["description"] = desc }
        if let params = tool.function.parameters {
            fn["parameters"] = params.mapValues { $0.value }
        }
        spec["function"] = fn
        return spec
    }

    // ── Acquire slot (concurrency limiter) ──
    await semaphore.wait()
    await stats.requestStarted()
    let genStart = Date()

    // Pass enable_thinking to the Jinja chat template via additionalContext.
    // Precedence: top-level request > per-request chat_template_kwargs > server --thinking flag
    var enableThinking: Bool
    if let explicitTopLevel = chatReq.enableThinking {
        enableThinking = explicitTopLevel
    } else if let kwargs = chatReq.chatTemplateKwargs, let perRequest = kwargs["enable_thinking"] {
        enableThinking = perRequest  // per-request override wins
    } else {
        enableThinking = config.thinking  // fall back to server --thinking flag
    }

    // Workaround for Gemma-4 Tool-Call bug (Resolves https://github.com/SharpAI/SwiftLM/issues/69)
    // If tools are present, the Gemma-4 Jinja template appends an anti-thinking prefix
    // (`<|channel>thought\n<channel|>`) when enable_thinking=false. This forcibly suppresses
    // the reasoning channel, flattening the first-token output distribution at the `<|tool_call>`
    // vs `text` decision point, resulting in complete failure (garbage tokens, Korean repeats,
    // or ignoring tools entirely) on vague requests.
    //
    // Fix: Unconditionally enable the thinking channel when tools are provided, giving the
    // Gemma-4 router time to process the system prompt before deciding to emit a tool_call.
    //
    // Coverage details:
    // - Tested Model: `mlx-community/gemma-4-26b-a4b-it-4bit`
    // - Verification: Verified via `run_benchmark.sh` (Test 8) using dynamic `tool_call` regression mapping.
    //                 The test covers vague query fallback (graceful TEXT handling bypassing degeneration)
    //                 and explicit query execution (driven via structured System Prompt conditioning).
    // - Known Limitations: While this logic repairs expected 4-bit decoding structures, evaluating at
    //                    zero-temperature (`temp=0.0`) without active repetition penalties can inherently 
    //                    induce repeating loop failure vectors beyond the purview of this fix.
    if chatReq.enableThinking == nil,
       chatReq.chatTemplateKwargs?["enable_thinking"] == nil,
       toolSpecs?.isEmpty == false,
       await container.configuration.toolCallFormat == .gemma4
    {
        enableThinking = true
    }

    // The Jinja template evaluates `not enable_thinking | default(false)`. If we pass nil instead of
    // true, it evaluates to false and still breaks. We MUST explicitly pass the boolean.
    let templateContext: [String: any Sendable] = ["enable_thinking": enableThinking]
    let userInput = UserInput(chat: chatMessages, tools: toolSpecs, additionalContext: templateContext)
    print("[Server Debug] Created UserInput with \(userInput.images.count) images and \(userInput.audio.count) audio inputs.")
    let lmInput = try await container.prepare(input: userInput)

    // ── Prompt caching: full token sequence for prefix matching ──
    let promptTokenCount = lmInput.text.tokens.size
    let promptTokens = lmInput.text.tokens.asArray(Int.self)

    // llama-server style: announce prefill start
    print("srv  slot_launch: id 0 | prompt=\(promptTokenCount)t | thinking=\(enableThinking) | prefilling...")
    fflush(stdout)
    let prefillStart = Date()

    // ── DFlash block-diffusion speculative decoding ──
    // When --dflash is enabled and both DFlash draft model and target model conform
    // to DFlashTargetModel, we use DFlashRuntime.generate instead of the standard path.
    if let dflashDraft = dflashModel, let targetModel = dflashTargetModel {
        print("[SwiftLM] ⚡ DFlash block-diffusion speculative decoding active")
        print("[SwiftLM] Using speculative decoding: DFlash block-diffusion mode active")
        fflush(stdout)
        // Convert DFlashEvent stream to Generation stream with proper streaming detokenizer
        let dflashTokenizer = await container.tokenizer
        let dflashStream = DFlashRuntime.generate(
            targetModel: targetModel,
            draftModel: dflashDraft,
            promptTokens: promptTokens,
            maxNewTokens: tokenLimit,
            blockTokens: dflashBlockSize
        )

        // Use a class wrapper so the detokenizer can be mutated inside the closure
        final class DetokenizerBox: @unchecked Sendable {
            var detokenizer: NaiveStreamingDetokenizer
            init(_ d: NaiveStreamingDetokenizer) { self.detokenizer = d }
        }
        let box = DetokenizerBox(NaiveStreamingDetokenizer(tokenizer: dflashTokenizer))

        let genStream = AsyncStream<Generation> { continuation in
            Task {
                for await event in dflashStream {
                    switch event {
                    case .token(let tokenID, _, _, _):
                        box.detokenizer.append(token: tokenID)
                        if let chunk = box.detokenizer.next() {
                            continuation.yield(.chunk(chunk, tokenId: tokenID))
                        }
                    case .prefill, .prefillProgress:
                        break
                    case .summary(let summary):
                        print("[SwiftLM] DFlash summary: \(summary.generationTokens) tokens, \(String(format: "%.1f", summary.tokensPerSecond)) tok/s, acceptance=\(String(format: "%.1f%%", summary.acceptanceRatio * 100)), \(summary.cyclesCompleted) cycles")
                    }
                }
                continuation.finish()
            }
        }

        let modelId = config.modelId
        if isStream {
            return handleChatStreaming(
                stream: genStream, modelId: modelId, stopSequences: stopSequences,
                includeUsage: includeUsage, promptTokenCount: promptTokenCount,
                enableThinking: enableThinking, jsonMode: jsonMode, semaphore: semaphore,
                stats: stats, genStart: genStart, prefillStart: prefillStart,
                emitPrefillProgress: false, onPrefillDone: nil
            )
        } else {
            return try await handleChatNonStreaming(
                stream: genStream, modelId: modelId, stopSequences: stopSequences,
                promptTokenCount: promptTokenCount, enableThinking: enableThinking,
                jsonMode: jsonMode, semaphore: semaphore,
                stats: stats, genStart: genStart, prefillStart: prefillStart, onPrefillDone: nil
            )
        }
    }

    // ── Cache-aware generation (standard path) ──
    let (stream, onPrefillDone) = try await container.perform { context -> (AsyncStream<Generation>, (() async -> Void)?) in
        let cache = context.model.newCache(parameters: params)

        // ── TurboQuant: enable 3-bit KV compression on every KVCacheSimple layer ──
        // This compresses cache history older than 8192 tokens into 3.5-bit Polar+QJL
        // form, halving KV RAM for long-context (100k+) requests.
        if config.turboKV {
            for layer in cache {
                if let simple = layer as? KVCacheSimple {
                    simple.turboQuantEnabled = true
                }
            }
        }

        // ── Prompt cache: bypass for multimodal inputs ──
        // The prompt cache only stores KV state for text token sequences. For multimodal
        // requests (image/audio), prepare() must inject the vision/audio feature embeddings
        // before the language model runs. A cache hit would skip that injection, feeding
        // raw <|image|>/<|audio|> token embeddings instead of the projected features.
        let isMultimodalRequest = lmInput.image != nil || lmInput.audio != nil

        // ── Decision branch ──
        // Speculative decoding is CHECKED FIRST because a cache-hit rollback
        // corrupts the draft model's KV state (draft and main model cycle tokens
        // in lock-step). We'd rather pay the prefill than emit garbage.
        //
        // Skip prompt cache for quantized-KV requests: the prompt cache stores KV state
        // produced with KVCacheSimple; restoring it into a QuantizedKVCache (or vice-versa)
        // is unsafe and produces incorrect results or runtime failures.
        let skipPromptCache = isMultimodalRequest || params.kvBits != nil
        var stream: AsyncStream<Generation>
        if let draftRef = draftModelRef {
            // Speculative decoding path: draft model generates candidates, main model verifies.
            // Bypass prompt cache to avoid draft/main KV drift on partial-match restores.
            print("[SwiftLM] Using speculative decoding (\(numDraftTokens) draft tokens/round)")
            stream = try MLXLMCommon.generate(
                input: lmInput, cache: cache, parameters: params, context: context,
                draftModel: draftRef.model, numDraftTokens: numDraftTokens
            )
        } else if !skipPromptCache, let cachedCount = await promptCache.restore(newTokens: promptTokens, into: cache) {
            // Cache hit: KV state is pre-populated up to cachedCount tokens.
            // Only compute the remaining (new) tokens.
            var startIndex = cachedCount
            if startIndex >= lmInput.text.tokens.count {
                // Full match: all tokens are cached. We still need to feed at least
                // the last token so the model can produce next-token logits.
                startIndex = lmInput.text.tokens.count - 1
                // Trim the KV cache back by 1 to avoid double-counting the replayed token.
                for layer in cache { layer.trim(1) }
            }
            let remainingTokens = lmInput.text.tokens[startIndex...]
            let trimmedInput = LMInput(tokens: remainingTokens)
            stream = try MLXLMCommon.generate(
                input: trimmedInput, cache: cache, parameters: params, context: context
            )
        } else {
            // Cache miss: process the full prompt.
            stream = try MLXLMCommon.generate(
                input: lmInput, cache: cache, parameters: params, context: context
            )
        }
        
        // Return a closure that will save the cache state synchronously AFTER
        // the generator stream has evaluated the prefill (on its very first token).
        //
        // ⚠️ TurboQuant guard: when TurboQuant has compressed data, cache.state
        // decodes ALL polar buffers back to full fp16 to create a restorable snapshot.
        // At 100K context this creates a ~37 GB allocation that completely negates
        // the compression savings (52 GB → should be ~20 GB).
        // Skip prompt cache save when any layer has active TurboQuant compression.
        // Short contexts (< turboMinActivationTokens) are unaffected — TurboQuant
        // hasn't compressed anything yet so the state getter is a zero-copy view.
        let turboHasCompressed = cache.contains { layer in
            if let simple = layer as? KVCacheSimple {
                return simple.turboQuantEnabled && simple.compressedOffset > 0
            }
            return false
        }
        let onPrefillDone: (() async -> Void)? = {
            if turboHasCompressed {
                print("[SwiftLM] 🧠 Skipping prompt cache save — TurboQuant has compressed \(cache.compactMap { ($0 as? KVCacheSimple)?.compressedOffset }.max() ?? 0) tokens. Saving would decode ~37 GB back to fp16.")
            } else if params.kvBits != nil {
                // kv_bits is set: the cache contains QuantizedKVCache layers whose token
                // format is incompatible with the FP16 KVCacheSimple format expected by
                // promptCache.save. Skip saving to prevent unsafe mixed-format restores.
            } else {
                await promptCache.save(tokens: promptTokens, cache: cache)
            }
        }
        return (stream, onPrefillDone)
    }

    let modelId = config.modelId

    if isStream {
        return handleChatStreaming(
            stream: stream, modelId: modelId, stopSequences: stopSequences,
            includeUsage: includeUsage, promptTokenCount: promptTokenCount,
            enableThinking: enableThinking, jsonMode: jsonMode, semaphore: semaphore,
            stats: stats, genStart: genStart, prefillStart: prefillStart,
            emitPrefillProgress: emitPrefillProgress, onPrefillDone: onPrefillDone
        )
    } else {
        return try await handleChatNonStreaming(
            stream: stream, modelId: modelId, stopSequences: stopSequences,
            promptTokenCount: promptTokenCount, enableThinking: enableThinking,
            jsonMode: jsonMode, semaphore: semaphore,
            stats: stats, genStart: genStart, prefillStart: prefillStart, onPrefillDone: onPrefillDone
        )
    }
}

// ── Thinking State Tracker ────────────────────────────────────────────────────

/// Parses the raw token stream from a thinking-capable model and separates
/// <think>…</think> content from the final response content.
/// Matches llama-server's behaviour: thinking tokens → delta.reasoning_content,
/// response tokens → delta.content (content is nil while thinking).
struct ThinkingStateTracker {
    enum Phase { case thinking, responding }
    private(set) var phase: Phase = .responding
    private var buffer = ""  // accumulates chars looking for tag boundaries

    /// Feed the next text fragment. Returns (reasoningContent, responseContent)
    /// where either value may be empty but never both non-empty simultaneously.
    mutating func process(_ text: String) -> (reasoning: String, content: String) {
        buffer += text
        var reasoning = ""
        var content = ""

        while !buffer.isEmpty {
            switch phase {
            case .responding:
                let startRange = buffer.range(of: "<thinking>") ?? buffer.range(of: "<think>") ?? buffer.range(of: "<|channel>thought\n") ?? buffer.range(of: "<|channel>thought")
                if let range = startRange {
                    // Flush text before the tag as response content
                    content += String(buffer[buffer.startIndex..<range.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                    phase = .thinking
                } else if isSuffixOfTag(buffer, tags: ["<think>", "<thinking>", "<|channel>thought\n", "<|channel>thought"]) {
                    // Partial tag — hold in buffer until we know more
                    return (reasoning, content)
                } else {
                    content += buffer
                    buffer = ""
                }
            case .thinking:
                let endRange = buffer.range(of: "</thinking>") ?? buffer.range(of: "</think>") ?? buffer.range(of: "<channel|>")
                if let range = endRange {
                    // Flush reasoning before the closing tag
                    reasoning += String(buffer[buffer.startIndex..<range.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                    phase = .responding
                } else if isSuffixOfTag(buffer, tags: ["</think>", "</thinking>", "<channel|>"]) {
                    // Partial closing tag — hold in buffer
                    return (reasoning, content)
                } else {
                    reasoning += buffer
                    buffer = ""
                }
            }
        }
        return (reasoning, content)
    }

    private func isSuffixOfTag(_ s: String, tags: [String]) -> Bool {
        for tag in tags {
            for len in stride(from: min(s.count, tag.count), through: 1, by: -1) {
                let tagPrefix = String(tag.prefix(len))
                if s.hasSuffix(tagPrefix) { return true }
            }
        }
        return false
    }
}

// ── Chat Streaming ───────────────────────────────────────────────────────────

/// Tracks prefill progress: whether it is done, and how many tokens have been processed.
/// n_past is updated by activePrefillProgressHook (called from LLMModel.prepare after each chunk)
/// and read by the SSE heartbeat task every 2 s.
actor PrefillState {
    private(set) var done: Bool = false
    private(set) var nPast: Int = 0
    func finish() { done = true }
    func update(nPast: Int) { self.nPast = nPast }
}

func handleChatStreaming(
    stream: AsyncStream<Generation>,
    modelId: String,
    stopSequences: [String],
    includeUsage: Bool,
    promptTokenCount: Int,
    enableThinking: Bool = false,
    jsonMode: Bool = false,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date,
    prefillStart: Date,
    emitPrefillProgress: Bool,
    onPrefillDone: (() async -> Void)? = nil
) -> Response {
    let (sseStream, cont) = AsyncStream<String>.makeStream()

    let prefillState = PrefillState()
    // ── Prefill heartbeat (opt-in via X-SwiftLM-Prefill-Progress: true) ──
    // We capture the hook in a local variable so that concurrent requests
    // cannot clobber each other's hook via the global. The global is still
    // written here because LLMModel.prepare() reads it, but the semaphore
    // ensures only one generation runs at a time.
    var heartbeatTask: Task<Void, Never>? = nil
    activePrefillProgressHook = nil
    if emitPrefillProgress {
        // Hook is scoped to this request: the local prefillState is the only
        // shared state, and it is actor-isolated.
        activePrefillProgressHook = { nPast, _ in
            Task { await prefillState.update(nPast: nPast) }
        }
        heartbeatTask = Task {
            var elapsed = 0
            while await !prefillState.done {
                try? await Task.sleep(for: .seconds(2))
                // Guard against Task cancellation on client disconnect.
                guard !Task.isCancelled else { break }
                if await !prefillState.done {
                    elapsed += 2
                    let nPast = await prefillState.nPast
                    _ = cont.yield(ssePrefillChunk(
                        nPast: nPast,
                        promptTokens: promptTokenCount,
                        elapsedSeconds: elapsed))
                }
            }
        }
    }

    Task {
        var hasToolCalls = false
        var toolCallIndex = 0
        var completionTokenCount = 0
        var fullText = ""
        var stopped = false
        var firstToken = true
        var tracker = ThinkingStateTracker()
        // Unconditional cleanup: guarantees heartbeat is cancelled on ALL exit paths
        // (normal completion, client disconnect, or task cancellation during prefill).
        defer {
            heartbeatTask?.cancel()
            heartbeatTask = nil
            activePrefillProgressHook = nil
        }
        
        // ── JSON mode streaming: buffer early tokens to strip hallucinated prefixes ──
        var jsonBuffering = jsonMode
        var jsonBuffer = ""

        for await generation in stream {
            if stopped { break }
            switch generation {
            case .chunk(let text, _):
                completionTokenCount += 1
                fullText += text
                // GPU yield: prevent Metal from starving macOS WindowServer
                if completionTokenCount % 8 == 0 {
                    try? await Task.sleep(for: .microseconds(50))
                }
                // Signal first token — stops the prefill heartbeat task
                if firstToken {
                    // First decode token: cancel heartbeat and clear the prefill progress hook.
                    heartbeatTask?.cancel()
                    heartbeatTask = nil
                    activePrefillProgressHook = nil
                    await prefillState.finish()
                    let prefillDur = Date().timeIntervalSince(prefillStart)
                    let prefillTokPerSec = prefillDur > 0 ? Double(promptTokenCount) / prefillDur : 0
                    let memSnap = MemoryUtils.snapshot()
                    print("srv  slot update: id 0 | prefill done | n_tokens=\(promptTokenCount), t=\(String(format: "%.2f", prefillDur))s, \(String(format: "%.1f", prefillTokPerSec))t/s | OS_RAM=\(String(format: "%.1f", memSnap.os))GB | MEM_DEMAND=\(String(format: "%.1f", memSnap.demand))GB | GPU_MEM=\(String(format: "%.1f", memSnap.gpu))GB")
                    print("srv  generate: id 0 | ", terminator: "")
                    if let onPrefillDone { await onPrefillDone() }
                    firstToken = false
                }
                print(text, terminator: "")
                fflush(stdout)

                // ── JSON mode buffering: accumulate early tokens, strip prefix, then flush ──
                if jsonBuffering {
                    jsonBuffer += text
                    let trimmed = jsonBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
                    let enoughTokens = completionTokenCount >= 3
                    let hitMax = completionTokenCount >= 32
                    let hasDoubleBrace = enoughTokens && trimmed.hasPrefix("{") && trimmed.dropFirst().contains("{")

                    if hitMax || hasDoubleBrace {
                        var cleaned = trimmed
                        if hasDoubleBrace {
                            if let firstBrace = cleaned.firstIndex(of: "{") {
                                let afterFirst = cleaned.index(after: firstBrace)
                                if let secondBrace = cleaned[afterFirst...].firstIndex(of: "{") {
                                    cleaned = String(cleaned[secondBrace...])
                                }
                            }
                        }
                        let (rText, cText) = enableThinking ? tracker.process(cleaned) : ("", cleaned)
                        if !rText.isEmpty || !cText.isEmpty {
                            cont.yield(sseChunk(
                                modelId: modelId,
                                reasoningContent: rText.isEmpty ? nil : rText,
                                content: cText.isEmpty ? nil : cText,
                                finishReason: nil
                            ))
                        }
                        jsonBuffering = false
                    }
                    continue  // skip normal emit while buffering or just flushed
                }

                // ── Route text through thinking state machine ──
                let (reasoningText, contentText) = enableThinking
                    ? tracker.process(text)
                    : ("", text)

                // ── Stop sequence check (operate on full accumulated text) ──
                if let (trimmedFull, _) = checkStopSequences(fullText, stopSequences: stopSequences) {
                    // Emit any final partial content that hasn't been sent yet
                    let emittedSoFar = fullText.count - text.count
                    if trimmedFull.count > emittedSoFar {
                        let partialText = String(trimmedFull.suffix(trimmedFull.count - emittedSoFar))
                        let (r, c) = enableThinking ? tracker.process(partialText) : ("", partialText)
                        cont.yield(sseChunk(modelId: modelId, reasoningContent: r.isEmpty ? nil : r,
                                            content: c.isEmpty ? nil : c, finishReason: nil))
                    }
                    cont.yield(sseChunk(modelId: modelId, reasoningContent: nil, content: nil, finishReason: "stop"))
                    let genDur = Date().timeIntervalSince(genStart)
                    let genTokPerSec = genDur > 0 ? Double(completionTokenCount) / genDur : 0
                    if includeUsage {
                        cont.yield(sseUsageChunk(modelId: modelId, promptTokens: promptTokenCount, completionTokens: completionTokenCount, tokPerSec: genTokPerSec, durationMs: genDur * 1000))
                    }
                    cont.yield("data: [DONE]\r\n\r\n")
                    cont.finish()
                    stopped = true
                } else {
                    // Emit the chunk — reasoning_content and/or content as appropriate
                    let hasReasoning = !reasoningText.isEmpty
                    let hasContent = !contentText.isEmpty
                    if hasReasoning || hasContent {
                        cont.yield(sseChunk(
                            modelId: modelId,
                            reasoningContent: hasReasoning ? reasoningText : nil,
                            content: hasContent ? contentText : nil,
                            finishReason: nil
                        ))
                    }
                    // If tracker buffer is holding a partial tag, nothing to emit yet — that's fine.
                }

            case .toolCall(let tc):
                hasToolCalls = true
                let argsJson = serializeToolCallArgs(tc.function.arguments)
                cont.yield(sseToolCallChunk(modelId: modelId, index: toolCallIndex, name: tc.function.name, arguments: argsJson))
                toolCallIndex += 1

            case .info(let info):
                heartbeatTask?.cancel()
                heartbeatTask = nil
                activePrefillProgressHook = nil
                await prefillState.finish()
                if !stopped {
                    var reason: String
                    switch info.stopReason {
                    case .length:
                        reason = "length"
                    case .error:
                        reason = "error"
                    case .cancelled, .stop:
                        reason = hasToolCalls ? "tool_calls" : "stop"
                    }
                    cont.yield(sseChunk(modelId: modelId, reasoningContent: nil, content: nil, finishReason: reason))
                    let genDur = Date().timeIntervalSince(genStart)
                    let genTokPerSec = genDur > 0 ? Double(completionTokenCount) / genDur : 0
                    if includeUsage {
                        cont.yield(sseUsageChunk(modelId: modelId, promptTokens: promptTokenCount, completionTokens: completionTokenCount, tokPerSec: genTokPerSec, durationMs: genDur * 1000))
                    }
                    cont.yield("data: [DONE]\r\n\r\n")
                    cont.finish()
                    // llama-server style: print newline then full response JSON
                    print("")  // end the real-time token stream line
                    let postMemSnap = MemoryUtils.snapshot()
                    print("srv  slot done: id 0 | gen_tokens=\(completionTokenCount) | OS_RAM=\(String(format: "%.1f", postMemSnap.os))GB | MEM_DEMAND=\(String(format: "%.1f", postMemSnap.demand))GB | GPU_MEM=\(String(format: "%.1f", postMemSnap.gpu))GB")
                    let dur = genDur
                    let tokPerSec = genTokPerSec
                    let logContent: Any = hasToolCalls ? NSNull() : fullText
                    let logResp: [String: Any] = [
                        "choices": [[
                            "index": 0,
                            "message": ["role": "assistant", "content": logContent],
                            "finish_reason": reason
                        ]],
                        "usage": [
                            "prompt_tokens": promptTokenCount,
                            "completion_tokens": completionTokenCount,
                            "total_tokens": promptTokenCount + completionTokenCount
                        ],
                        "timings": ["predicted_per_second": tokPerSec]
                    ]
                    if let logData = try? JSONSerialization.data(withJSONObject: logResp),
                       let logStr = String(data: logData, encoding: .utf8) {
                        print("srv  log_server_r: response: \(logStr)")
                        fflush(stdout)
                    }
                }
            }
        }
        cont.finish()
        let duration = Date().timeIntervalSince(genStart)
        await stats.requestFinished(tokens: completionTokenCount, duration: duration)
        await semaphore.signal()
    }
    return Response(
        status: .ok,
        headers: sseHeaders(),
        body: .init(asyncSequence: sseStream.map { ByteBuffer(string: $0) })
    )
}

// ── Chat Non-Streaming ───────────────────────────────────────────────────────

func handleChatNonStreaming(
    stream: AsyncStream<Generation>,
    modelId: String,
    stopSequences: [String],
    promptTokenCount: Int,
    enableThinking: Bool = false,
    jsonMode: Bool = false,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date,
    prefillStart: Date,
    onPrefillDone: (() async -> Void)? = nil
) async throws -> Response {
    var fullText = ""
    var completionTokenCount = 0
    var collectedToolCalls: [ToolCallResponse] = []
    var tcIndex = 0
    var generationStopReason: GenerateStopReason = .stop
    var firstToken = true
    for await generation in stream {
        switch generation {
        case .chunk(let text, _):
            fullText += text
            completionTokenCount += 1
            // GPU yield: prevent Metal from starving macOS WindowServer
            if completionTokenCount % 8 == 0 {
                try? await Task.sleep(for: .microseconds(50))
            }
            // Real-time stdout: on first token, log prefill completion + start generate line
            if firstToken {
                let prefillDur = Date().timeIntervalSince(prefillStart)
                let prefillTokPerSec = prefillDur > 0 ? Double(promptTokenCount) / prefillDur : 0
                let memSnap = MemoryUtils.snapshot()
                print("srv  slot update: id 0 | prefill done | n_tokens=\(promptTokenCount), t=\(String(format: "%.2f", prefillDur))s, \(String(format: "%.1f", prefillTokPerSec))t/s | OS_RAM=\(String(format: "%.1f", memSnap.os))GB | MEM_DEMAND=\(String(format: "%.1f", memSnap.demand))GB | GPU_MEM=\(String(format: "%.1f", memSnap.gpu))GB")
                print("srv  generate: id 0 | ", terminator: "")
                if let onPrefillDone { await onPrefillDone() }
                firstToken = false
            }
            print(text, terminator: "")
            fflush(stdout)
        case .toolCall(let tc):
            let argsJson = serializeToolCallArgs(tc.function.arguments)
            collectedToolCalls.append(ToolCallResponse(
                id: "call_\(UUID().uuidString.prefix(8))",
                type: "function",
                function: ToolCallFunction(name: tc.function.name, arguments: argsJson)
            ))
            tcIndex += 1
        case .info(let info):
            generationStopReason = info.stopReason
        }
    }
    print("")  // end the real-time token stream line
    let postMemSnap = MemoryUtils.snapshot()
    print("srv  slot done: id 0 | gen_tokens=\(completionTokenCount) | OS_RAM=\(String(format: "%.1f", postMemSnap.os))GB | MEM_DEMAND=\(String(format: "%.1f", postMemSnap.demand))GB | GPU_MEM=\(String(format: "%.1f", postMemSnap.gpu))GB")
    let duration = Date().timeIntervalSince(genStart)
    await stats.requestFinished(tokens: completionTokenCount, duration: duration)
    await semaphore.signal()

    // ── Apply stop sequences to final text ──
    var finishReason: String
    switch generationStopReason {
    case .length:
        finishReason = "length"
    default:
        finishReason = "stop"
    }
    if checkStopSequences(fullText, stopSequences: stopSequences) != nil {
        fullText = checkStopSequences(fullText, stopSequences: stopSequences)!.0
        finishReason = "stop"
    }

    // ── Thinking: extract <think>…</think> into reasoning_content ──
    var reasoningContent: String? = nil
    var responseContent = fullText
    if enableThinking {
        print("srv debug: pre-extract fullText=\(fullText.prefix(40).debugDescription)")
        let (extracted, remaining) = extractThinkingBlock(from: fullText)
        print("srv debug: extracted=\(extracted != nil ? "true" : "false"), remaining_len=\(remaining.count)")
        if let extracted {
            reasoningContent = extracted
            responseContent = remaining
        }
    }

    // ── JSON mode validation ──
    if jsonMode {
        let stripped = responseContent
            .replacingOccurrences(of: "```json\n", with: "")
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```\n", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        responseContent = stripped
    }

    let totalTokens = promptTokenCount + completionTokenCount
    let hasToolCalls = !collectedToolCalls.isEmpty

    let resp = ChatCompletionResponse(
        id: "chatcmpl-\(UUID().uuidString)",
        model: modelId,
        created: Int(Date().timeIntervalSince1970),
        choices: [
            Choice(
                index: 0,
                message: AssistantMessage(
                    role: "assistant",
                    content: responseContent.isEmpty && hasToolCalls ? nil : responseContent,
                    reasoningContent: reasoningContent,
                    toolCalls: hasToolCalls ? collectedToolCalls : nil
                ),
                finishReason: hasToolCalls ? "tool_calls" : finishReason
            )
        ],
        usage: TokenUsage(promptTokens: promptTokenCount, completionTokens: completionTokenCount, totalTokens: totalTokens),
        timings: ChatCompletionResponse.Timings(
            predictedPerSecond: duration > 0 ? Double(completionTokenCount) / duration : 0,
            predictedN: completionTokenCount,
            predictedMs: duration * 1000
        )
    )
    let encoded = try JSONEncoder().encode(resp)
    // llama-server style: log full response JSON on one line
    if let responseStr = String(data: encoded, encoding: .utf8) {
        print("srv  log_server_r: response: \(responseStr)")
        fflush(stdout)
    }
    return Response(
        status: .ok,
        headers: jsonHeaders(),
        body: .init(byteBuffer: ByteBuffer(data: encoded))
    )
}

/// Returns (thinkingContent, remainingContent) or (nil, original) if no block found.
func extractThinkingBlock(from text: String) -> (String?, String) {
    let startTag = text.range(of: "<thinking>") ?? text.range(of: "<think>") ?? text.range(of: "<|channel>thought\n") ?? text.range(of: "<|channel>thought") ?? (text.hasPrefix("thought\n") ? text.range(of: "thought\n") : nil)
    let endTag = text.range(of: "</thinking>") ?? text.range(of: "</think>") ?? text.range(of: "<channel|>")
    
    guard let startRange = startTag, let endRange = endTag else {
        // If there's an unclosed thinking block (still thinking when stopped)
        if let startRange = startTag {
            let thinking = String(text[startRange.upperBound...])
            return (thinking.isEmpty ? nil : thinking, "")
        }
        return (nil, text)
    }
    let thinking = String(text[startRange.upperBound..<endRange.lowerBound])
    let remaining = String(text[endRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    return (thinking.isEmpty ? nil : thinking, remaining)
}

// ── Text Completions Handler ─────────────────────────────────────────────────

func handleTextCompletion(
    request: Request,
    bodyData: Data,
    config: ServerConfig,
    container: ModelContainer,
    semaphore: AsyncSemaphore,
    stats: ServerStats
) async throws -> Response {
    let compReq = try JSONDecoder().decode(TextCompletionRequest.self, from: bodyData)
    let isStream = compReq.stream ?? false
    let emitPrefillProgress = prefillProgressEnabled(in: request)

    let tokenLimit = compReq.maxTokens ?? config.maxTokens
    let temperature = compReq.temperature.map(Float.init) ?? config.temp
    let topP = compReq.topP.map(Float.init) ?? config.topP
    let topK = compReq.topK ?? config.topK ?? 50
    let minP = compReq.minP.map(Float.init) ?? config.minP ?? 0.0
    let repeatPenalty = compReq.repetitionPenalty.map(Float.init) ?? config.repeatPenalty
    let stopSequences = compReq.stop ?? []

    let params = GenerateParameters(
        maxTokens: tokenLimit,
        maxKVSize: config.ctxSize,
        temperature: temperature,
        topP: topP,
        topK: topK,
        minP: minP,
        repetitionPenalty: repeatPenalty,
        prefillStepSize: config.prefillSize
    )

    if let seed = compReq.seed {
        MLXRandom.seed(UInt64(seed))
    }

    await semaphore.wait()
    await stats.requestStarted()
    let genStart = Date()

    let userInput = UserInput(prompt: compReq.prompt)
    let lmInput = try await container.prepare(input: userInput)

    // ── Get actual prompt token count before generate() to avoid data race ──
    let promptTokenCount = lmInput.text.tokens.size

    let stream = try await container.generate(input: lmInput, parameters: params)
    let modelId = config.modelId

    if isStream {
        return handleTextStreaming(
            stream: stream, modelId: modelId, stopSequences: stopSequences,
            promptTokenCount: promptTokenCount, semaphore: semaphore, stats: stats,
            genStart: genStart, emitPrefillProgress: emitPrefillProgress
        )
    } else {
        return try await handleTextNonStreaming(
            stream: stream, modelId: modelId, stopSequences: stopSequences,
            promptTokenCount: promptTokenCount, semaphore: semaphore, stats: stats, genStart: genStart
        )
    }
}

// ── Text Streaming ───────────────────────────────────────────────────────────

func handleTextStreaming(
    stream: AsyncStream<Generation>,
    modelId: String,
    stopSequences: [String],
    promptTokenCount: Int,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date,
    emitPrefillProgress: Bool
) -> Response {
    let (sseStream, cont) = AsyncStream<String>.makeStream()
    let prefillState = PrefillState()
    var heartbeatTask: Task<Void, Never>? = nil
    activePrefillProgressHook = nil
    if emitPrefillProgress {
        activePrefillProgressHook = { nPast, _ in
            Task { await prefillState.update(nPast: nPast) }
        }
        heartbeatTask = Task {
            var elapsed = 0
            while await !prefillState.done {
                try? await Task.sleep(for: .seconds(2))
                guard !Task.isCancelled else { break }
                if await !prefillState.done {
                    elapsed += 2
                    let nPast = await prefillState.nPast
                    _ = cont.yield(ssePrefillChunk(
                        nPast: nPast,
                        promptTokens: promptTokenCount,
                        elapsedSeconds: elapsed))
                }
            }
        }
    }
    Task {
        var completionTokenCount = 0
        var fullText = ""
        var stopped = false
        var firstToken = true
        // Unconditional cleanup: guarantees heartbeat is cancelled on ALL exit paths
        // (normal completion, client disconnect, or task cancellation during prefill).
        defer {
            heartbeatTask?.cancel()
            heartbeatTask = nil
            activePrefillProgressHook = nil
        }
        for await generation in stream {
            if stopped { break }
            switch generation {
            case .chunk(let text, _):
                if firstToken {
                    heartbeatTask?.cancel()
                    heartbeatTask = nil
                    activePrefillProgressHook = nil
                    await prefillState.finish()
                    firstToken = false
                }
                completionTokenCount += 1
                fullText += text
                // GPU yield: prevent Metal from starving macOS WindowServer
                if completionTokenCount % 8 == 0 {
                    try? await Task.sleep(for: .microseconds(50))
                }
                if let (trimmedText, _) = checkStopSequences(fullText, stopSequences: stopSequences) {
                    let emittedSoFar = fullText.count - text.count
                    if trimmedText.count > emittedSoFar {
                        let partialText = String(trimmedText.suffix(trimmedText.count - emittedSoFar))
                        cont.yield(sseTextChunk(modelId: modelId, text: partialText, finishReason: nil))
                    }
                    cont.yield(sseTextChunk(modelId: modelId, text: "", finishReason: "stop"))
                    cont.yield("data: [DONE]\n\n")
                    cont.finish()
                    stopped = true
                } else {
                    cont.yield(sseTextChunk(modelId: modelId, text: text, finishReason: nil))
                }
            case .toolCall:
                break
            case .info(let info):
                heartbeatTask?.cancel()
                heartbeatTask = nil
                activePrefillProgressHook = nil
                await prefillState.finish()
                if !stopped {
                    var reason: String
                    switch info.stopReason {
                    case .length:
                        reason = "length"
                    case .error:
                        reason = "error"
                    case .cancelled, .stop:
                        reason = "stop"
                    }
                    cont.yield(sseTextChunk(modelId: modelId, text: "", finishReason: reason))
                    cont.yield("data: [DONE]\n\n")
                    cont.finish()
                }
            }
        }
        cont.finish()
        let duration = Date().timeIntervalSince(genStart)
        await stats.requestFinished(tokens: completionTokenCount, duration: duration)
        await semaphore.signal()
    }
    return Response(
        status: .ok,
        headers: sseHeaders(),
        body: .init(asyncSequence: sseStream.map { ByteBuffer(string: $0) })
    )
}

// ── Text Non-Streaming ───────────────────────────────────────────────────────

func handleTextNonStreaming(
    stream: AsyncStream<Generation>,
    modelId: String,
    stopSequences: [String],
    promptTokenCount: Int,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date
) async throws -> Response {
    var fullText = ""
    var completionTokenCount = 0
    for await generation in stream {
        switch generation {
        case .chunk(let text, _):
            fullText += text
            completionTokenCount += 1
            // GPU yield: prevent Metal from starving macOS WindowServer
            if completionTokenCount % 8 == 0 {
                try? await Task.sleep(for: .microseconds(50))
            }
        case .toolCall, .info:
            break
        }
    }
    let duration = Date().timeIntervalSince(genStart)
    await stats.requestFinished(tokens: completionTokenCount, duration: duration)
    await semaphore.signal()

    var finishReason = "stop"
    if let (trimmedText, _) = checkStopSequences(fullText, stopSequences: stopSequences) {
        fullText = trimmedText
        finishReason = "stop"
    }

    let totalTokens = promptTokenCount + completionTokenCount

    let resp = TextCompletionResponse(
        id: "cmpl-\(UUID().uuidString)",
        model: modelId,
        created: Int(Date().timeIntervalSince1970),
        choices: [
            TextChoice(index: 0, text: fullText, finishReason: finishReason)
        ],
        usage: TokenUsage(promptTokens: promptTokenCount, completionTokens: completionTokenCount, totalTokens: totalTokens),
        timings: ChatCompletionResponse.Timings(
            predictedPerSecond: duration > 0 ? Double(completionTokenCount) / duration : 0,
            predictedN: completionTokenCount,
            predictedMs: duration * 1000
        )
    )
    let encoded = try JSONEncoder().encode(resp)
    return Response(
        status: .ok,
        headers: jsonHeaders(),
        body: .init(byteBuffer: ByteBuffer(data: encoded))
    )
}

// ── AsyncSemaphore — lightweight concurrency limiter ─────────────────────────

actor AsyncSemaphore {
    private let limit: Int
    private var count: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(limit: Int) {
        self.limit = limit
        self.count = limit
    }

    func wait() async {
        if count > 0 {
            count -= 1
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func signal() {
        if waiters.isEmpty {
            count = min(count + 1, limit)
        } else {
            let waiter = waiters.removeFirst()
            waiter.resume()
        }
    }
}

// ── CORS Middleware ───────────────────────────────────────────────────────────

struct CORSMiddleware<Context: RequestContext>: RouterMiddleware {
    let allowedOrigin: String

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        if request.method == .options {
            return Response(
                status: .noContent,
                headers: corsHeaders(for: request)
            )
        }
        var response = try await next(request, context)
        let headers = corsHeaders(for: request)
        for field in headers {
            response.headers.append(field)
        }
        return response
    }

    private func corsHeaders(for request: Request) -> HTTPFields {
        var fields: [HTTPField] = []
        if allowedOrigin == "*" {
            fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Origin")!, value: "*"))
        } else {
            let requestOrigin = request.headers[values: HTTPField.Name("Origin")!].first ?? ""
            if requestOrigin == allowedOrigin {
                fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Origin")!, value: allowedOrigin))
                fields.append(HTTPField(name: HTTPField.Name("Vary")!, value: "Origin"))
            }
        }
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Methods")!, value: "GET, POST, OPTIONS"))
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Headers")!, value: "Content-Type, Authorization, X-SwiftLM-Prefill-Progress"))
        return HTTPFields(fields)
    }
}

// ── API Key Authentication Middleware ────────────────────────────────────────

struct ApiKeyMiddleware<Context: RequestContext>: RouterMiddleware {
    let apiKey: String

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        // Exempt health and metrics endpoints from auth
        let path = request.uri.path
        if path == "/health" || path == "/metrics" {
            return try await next(request, context)
        }

        // Check Authorization header: "Bearer <key>"
        let authHeader = request.headers[values: .authorization].first ?? ""
        let expectedHeader = "Bearer \(apiKey)"

        if authHeader == expectedHeader || authHeader == apiKey {
            return try await next(request, context)
        }

        // Unauthorized
        let errorPayload = "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}"
        return Response(
            status: .unauthorized,
            headers: jsonHeaders(),
            body: .init(byteBuffer: ByteBuffer(string: errorPayload))
        )
    }
}

// ── Stop Sequence Detection ──────────────────────────────────────────────────

func checkStopSequences(_ text: String, stopSequences: [String]) -> (String, String)? {
    for stop in stopSequences {
        if let range = text.range(of: stop) {
            let trimmed = String(text[text.startIndex..<range.lowerBound])
            return (trimmed, stop)
        }
    }
    return nil
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func jsonHeaders() -> HTTPFields {
    HTTPFields([HTTPField(name: .contentType, value: "application/json")])
}

let prefillProgressHeaderName = HTTPField.Name("X-SwiftLM-Prefill-Progress")!

func parseTruthyHeaderValue(_ value: String?) -> Bool {
    guard let value else { return false }
    switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "1", "on", "true", "yes":
        return true
    default:
        return false
    }
}

func prefillProgressEnabled(in request: Request) -> Bool {
    parseTruthyHeaderValue(request.headers[values: prefillProgressHeaderName].first)
}

func sseHeaders() -> HTTPFields {
    HTTPFields([
        HTTPField(name: .contentType, value: "text/event-stream"),
        HTTPField(name: .cacheControl, value: "no-cache"),
        HTTPField(name: HTTPField.Name("X-Accel-Buffering")!, value: "no"),
    ])
}

/// Build a chat.completion.chunk SSE event.
/// - reasoningContent: if non-nil, added to delta as "reasoning_content" (llama-server thinking style)
/// - content: if non-nil, added to delta as "content" (standard response text)
/// Both may be nil simultaneously (used for the final finish_reason chunk).
func sseChunk(modelId: String, reasoningContent: String?, content: String?, finishReason: String?) -> String {
    var deltaObj: [String: Any] = [:]
    // Always include role on the very first chunk when we have content
    if reasoningContent != nil || content != nil {
        deltaObj["role"] = "assistant"
    }
    if let rc = reasoningContent {
        deltaObj["reasoning_content"] = rc
    }
    if let c = content {
        deltaObj["content"] = c
    }
    var choiceObj: [String: Any] = [
        "index": 0,
        "delta": deltaObj,
    ]
    if let finishReason {
        choiceObj["finish_reason"] = finishReason
    }
    let chunk: [String: Any] = [
        "id": "chatcmpl-\(UUID().uuidString)",
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelId,
        "choices": [choiceObj]
    ]
    let data = try! JSONSerialization.data(withJSONObject: chunk)
    return "data: \(String(data: data, encoding: .utf8)!)\r\n\r\n"
}

/// Prefill-progress heartbeat chunk — emitted every 2s while the server is processing the prompt
/// when explicitly enabled via `X-SwiftLM-Prefill-Progress: true`.
/// It is sent as a named SSE event (`event: prefill_progress`) to avoid breaking strict
/// OpenAI-compatible clients (e.g. OpenCode), which reject unknown `data:` objects.
/// Format mirrors llama-server's slot_update event:
///   n_past          : tokens evaluated so far (real value from chunked prefill, or 0 for single-chunk)
///   n_prompt_tokens : total prompt token count
///   fraction        : n_past / n_prompt_tokens (0.0–1.0), useful for progress bars
///   elapsed_seconds : wall-clock time since the request started
/// Note: `model` is intentionally omitted — clients can correlate from preceding stream chunks.
/// Note: `on` is accepted as a truthy header value for parity with common reverse proxy conventions.
func ssePrefillChunk(nPast: Int = 0, promptTokens: Int, elapsedSeconds: Int) -> String {
    let fraction = promptTokens > 0 ? Double(nPast) / Double(promptTokens) : 0.0
    let chunk: [String: Any] = [
        "status": "processing",
        "n_past": nPast,
        "n_prompt_tokens": promptTokens,
        "fraction": fraction,
        "elapsed_seconds": elapsedSeconds
    ]
    let data = try! JSONSerialization.data(withJSONObject: chunk)
    return "event: prefill_progress\r\ndata: \(String(data: data, encoding: .utf8)!)\r\n\r\n"
}

func sseUsageChunk(modelId: String, promptTokens: Int, completionTokens: Int, tokPerSec: Double? = nil, durationMs: Double? = nil) -> String {
    var usage: [String: Any] = [
        "prompt_tokens": promptTokens,
        "completion_tokens": completionTokens,
        "total_tokens": promptTokens + completionTokens
    ]
    if let tokPerSec, let durationMs {
        usage["timings"] = [
            "predicted_per_second": tokPerSec,
            "predicted_n": completionTokens,
            "predicted_ms": durationMs
        ]
    }
    let chunk: [String: Any] = [
        "id": "chatcmpl-\(UUID().uuidString)",
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelId,
        "choices": [] as [[String: Any]],
        "usage": usage
    ]
    let data = try! JSONSerialization.data(withJSONObject: chunk)
    return "data: \(String(data: data, encoding: .utf8)!)\r\n\r\n"
}

func sseToolCallChunk(modelId: String, index: Int, name: String, arguments: String) -> String {
    let chunk: [String: Any] = [
        "id": "chatcmpl-\(UUID().uuidString)",
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelId,
        "choices": [[
            "index": 0,
            "delta": [
                "role": "assistant",
                "tool_calls": [[
                    "index": index,
                    "id": "call_\(UUID().uuidString.prefix(8))",
                    "type": "function",
                    "function": [
                        "name": name,
                        "arguments": arguments,
                    ] as [String: Any],
                ] as [String: Any]],
            ] as [String: Any],
        ] as [String: Any]]
    ]
    let data = try! JSONSerialization.data(withJSONObject: chunk)
    return "data: \(String(data: data, encoding: .utf8)!)\r\n\r\n"
}

func sseTextChunk(modelId: String, text: String, finishReason: String?) -> String {
    var choiceObj: [String: Any] = [
        "index": 0,
        "text": text,
    ]
    if let finishReason {
        choiceObj["finish_reason"] = finishReason
    }
    let chunk: [String: Any] = [
        "id": "cmpl-\(UUID().uuidString)",
        "object": "text_completion",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelId,
        "choices": [choiceObj]
    ]
    let data = try! JSONSerialization.data(withJSONObject: chunk)
    return "data: \(String(data: data, encoding: .utf8)!)\r\n\r\n"
}

func serializeToolCallArgs(_ args: [String: JSONValue]) -> String {
    let anyDict = args.mapValues { $0.anyValue }
    guard let data = try? JSONSerialization.data(withJSONObject: anyDict) else {
        return "{}"
    }
    return String(data: data, encoding: .utf8) ?? "{}"
}

// ── OpenAI-compatible types ───────────────────────────────────────────────────

struct StreamOptions: Decodable {
    let includeUsage: Bool?
    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

struct ResponseFormat: Decodable {
    let type: String
}

struct ChatCompletionRequest: Decodable {
    /// Message content can be a plain string or an array of content parts (text + image_url)
    struct Message: Decodable {
        let role: String
        let content: MessageContent?
        let tool_calls: [ToolCallResponse]?
        let tool_call_id: String?

        /// Extract plain text from content (handles both string and multipart)
        var textContent: String {
            guard let content = content else { return "" }
            switch content {
            case .string(let s): return s
            case .parts(let parts):
                return parts.compactMap { part in
                    if part.type == "text" { return part.text }
                    return nil
                }.joined(separator: "\n")
            }
        }

        /// Extract images from multipart content (base64 data URIs and HTTP URLs)
        func extractImages() -> [UserInput.Image] {
            guard let content = content, case .parts(let parts) = content else { return [] }
            return parts.compactMap { part -> UserInput.Image? in
                guard part.type == "image_url", let imageUrl = part.imageUrl else { return nil }
                let urlStr = imageUrl.url
                // Handle base64 data URIs: data:image/png;base64,...
                if urlStr.hasPrefix("data:") {
                    guard let commaIdx = urlStr.firstIndex(of: ",") else { return nil }
                    let base64Str = String(urlStr[urlStr.index(after: commaIdx)...])
                    guard let data = Data(base64Encoded: base64Str),
                          let ciImage = CIImage(data: data) else { return nil }
                    return .ciImage(ciImage)
                }
                // Handle HTTP/HTTPS URLs
                if let url = URL(string: urlStr),
                   (url.scheme == "http" || url.scheme == "https") {
                    return .url(url)
                }
                // Handle file URLs
                if let url = URL(string: urlStr) {
                    return .url(url)
                }
                return nil
            }
        }

        /// Extract audio from multipart content
        func extractAudio() -> [UserInput.Audio] {
            guard let content = content, case .parts(let parts) = content else { return [] }
            return parts.compactMap { part -> UserInput.Audio? in
                guard part.type == "input_audio", let audio = part.inputAudio else { return nil }
                
                // Be tolerant of optional data URI prefixes like "data:audio/wav;base64,"
                var base64Str = audio.data
                if base64Str.hasPrefix("data:") {
                    if let commaIdx = base64Str.firstIndex(of: ",") {
                        base64Str = String(base64Str[base64Str.index(after: commaIdx)...])
                    }
                }
                
                if let data = Data(base64Encoded: base64Str, options: .ignoreUnknownCharacters) {
                    return .data(data, format: audio.format)
                } else {
                    print("[Server] Fatal Base64 parse error for audio data!")
                }
                return nil
            }
        }

        /// Convert OpenAI JSON spec message to internal Chat.Message
        func toChatMessage() -> Chat.Message {
            let text = textContent
            let imgs = extractImages()
            let aud = extractAudio()
            
            switch role {
            case "system", "developer":
                return .system(text, images: imgs, audio: aud)
            case "assistant":
                var formattedToolCalls: [[String: any Sendable]]? = nil
                if let tc = tool_calls, !tc.isEmpty {
                    formattedToolCalls = tc.enumerated().map { (index, call) in
                        [
                            "index": index,
                            "id": call.id,
                            "type": call.type,
                            "function": [
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            ] as [String: any Sendable]
                        ] as [String: any Sendable]
                    }
                }
                return .assistant(text, images: imgs, audio: aud, toolCalls: formattedToolCalls)
            case "tool":
                return .tool(text, toolCallId: tool_call_id)
            default:
                return .user(text, images: imgs, audio: aud)
            }
        }
    }

    /// Message content: either a plain string or structured multipart content
    enum MessageContent: Decodable {
        case string(String)
        case parts([ContentPart])

        init(from decoder: Swift.Decoder) throws {
            let svc = try decoder.singleValueContainer()
            if let str = try? svc.decode(String.self) {
                self = .string(str)
            } else if let parts = try? svc.decode([ContentPart].self) {
                self = .parts(parts)
            } else {
                self = .string("")
            }
        }
    }

    struct ContentPart: Decodable {
        let type: String
        let text: String?
        let imageUrl: ImageUrlContent?
        let inputAudio: InputAudioContent?

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageUrl = "image_url"
            case inputAudio = "input_audio"
        }
    }

    struct ImageUrlContent: Decodable {
        let url: String
        let detail: String?
    }

    struct InputAudioContent: Decodable {
        let data: String
        let format: String
    }

    struct ToolDef: Decodable {
        let type: String
        let function: ToolFuncDef
    }
    struct ToolFuncDef: Decodable {
        let name: String
        let description: String?
        let parameters: [String: AnyCodable]?
    }
    let model: String?
    let messages: [Message]
    let stream: Bool?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let minP: Double?
    let repetitionPenalty: Double?
    let frequencyPenalty: Double?
    let presencePenalty: Double?
    let tools: [ToolDef]?
    let stop: [String]?
    let seed: Int?
    let streamOptions: StreamOptions?
    let responseFormat: ResponseFormat?
    /// Per-request Jinja template kwargs (e.g. {"enable_thinking": false} for Qwen3/Qwen3.5)
    let chatTemplateKwargs: [String: Bool]?
    /// Top-level thinking override emitted by Aegis-AI gateway
    let enableThinking: Bool?
    /// Number of bits for native MLX quantized KV cache (nil = no quantization).
    /// Only 4 and 8 are supported by the underlying MLX QuantizedKVCache.
    /// Enables `QuantizedKVCache` instead of `KVCacheSimple`.  Separate from `--turbo-kv`.
    let kvBits: Int?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools, stop, seed
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case streamOptions = "stream_options"
        case responseFormat = "response_format"
        case chatTemplateKwargs = "chat_template_kwargs"
        case enableThinking = "enable_thinking"
        case kvBits = "kv_bits"
    }
}

struct TextCompletionRequest: Decodable {
    let model: String?
    let prompt: String
    let stream: Bool?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let minP: Double?
    let repetitionPenalty: Double?
    let stop: [String]?
    let seed: Int?

    enum CodingKeys: String, CodingKey {
        case model, prompt, stream, temperature, stop, seed
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
    }
}

struct ChatCompletionResponse: Encodable {
    let id: String
    let object: String = "chat.completion"
    let model: String
    let created: Int
    let choices: [Choice]
    let usage: TokenUsage
    let timings: Timings?

    struct Timings: Encodable {
        let predictedPerSecond: Double
        let predictedN: Int
        let predictedMs: Double

        enum CodingKeys: String, CodingKey {
            case predictedPerSecond = "predicted_per_second"
            case predictedN = "predicted_n"
            case predictedMs = "predicted_ms"
        }
    }
}

struct Choice: Encodable {
    let index: Int
    let message: AssistantMessage
    let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
    }
}

struct AssistantMessage: Encodable {
    let role: String
    let content: String?
    /// Separated reasoning/thinking content (llama-server compatible).
    /// Only present when the model produced a <think>…</think> block.
    let reasoningContent: String?
    let toolCalls: [ToolCallResponse]?

    init(role: String, content: String?, reasoningContent: String? = nil, toolCalls: [ToolCallResponse]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }

    enum CodingKeys: String, CodingKey {
        case role, content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }
}

struct ToolCallResponse: Codable {
    let id: String
    let type: String
    let function: ToolCallFunction
}

struct ToolCallFunction: Codable {
    let name: String
    let arguments: String
}

struct TextCompletionResponse: Encodable {
    let id: String
    let object: String = "text_completion"
    let model: String
    let created: Int
    let choices: [TextChoice]
    let usage: TokenUsage
    let timings: ChatCompletionResponse.Timings?
}

struct TextChoice: Encodable {
    let index: Int
    let text: String
    let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }
}

// AnyCodable: a Sendable-compatible type-erased JSON value.
// `value` stores only Sendable-compatible Foundation types: Bool, Int, Double,
// String, NSNull, [AnyCodable.SendableValue], [String: AnyCodable.SendableValue]
// AnyCodable: a type-erased JSON value that bridges to Foundation types.
// Marked @unchecked Sendable: all stored types (Bool/Int/Double/String/NSNull/
// recursive AnyCodable) are in fact Sendable; `Any` is used for ergonomic storage.
// AnyCodable: type-erased Decodable wrapper over JSON scalars/arrays/objects.
// `value` holds Sendable-safe Foundation types (Bool/Int/Double/String/NSNull + collections).
struct AnyCodable: @unchecked Sendable {
    let value: Any

    static func toSendable(_ dict: [String: AnyCodable]?) -> [String: any Sendable]? {
        guard let dict else { return nil }
        return dict.mapValues { $0.value as any Sendable }
    }
}

extension AnyCodable: Decodable {
    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { value = NSNull(); return }
        if let b = try? c.decode(Bool.self)   { value = b; return }
        if let i = try? c.decode(Int.self)    { value = i; return }
        if let d = try? c.decode(Double.self) { value = d; return }
        if let s = try? c.decode(String.self) { value = s; return }
        if let a = try? c.decode([AnyCodable].self) { value = a.map { $0.value }; return }
        if let o = try? c.decode([String: AnyCodable].self) { value = o.mapValues { $0.value }; return }
        value = NSNull()
    }
}

struct TokenUsage: Encodable {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

// ── ALM Factory & Tokenizer Bridging ──────────────────────────────────────────

public struct ALMUserInputProcessor: UserInputProcessor, @unchecked Sendable {
    let tokenizer: MLXLMCommon.Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator
    let fusionProcessor: MultimodalFusionProcessor
    let numAudioEmbeddings: Int

    public init(
        tokenizer: any MLXLMCommon.Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator,
        boaToken: Int = 255010, eoaToken: Int = 255011,
        numAudioEmbeddings: Int = 128
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
        self.fusionProcessor = MultimodalFusionProcessor(boaToken: boaToken, eoaToken: eoaToken)
        self.numAudioEmbeddings = numAudioEmbeddings
    }

    public func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            print("Messages:", messages); let promptTokensInt = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            
            // Check if there is audio to interleave
            if !input.audio.isEmpty {
                print("[ALM] Interleaving Audio Tokens into prompt.")
                // Mock num audio embeddings for now - typically derived from the model or audio lengths
                let rawSequence = fusionProcessor.interleave(
                    textTokens: promptTokensInt,
                    numAudioEmbeddings: numAudioEmbeddings,
                    audioFirst: true
                )
                return LMInput(tokens: MLXArray(rawSequence))
            }
            
            return LMInput(tokens: MLXArray(promptTokensInt))
        } catch MLXLMCommon.TokenizerError.missingChatTemplate {
            let prompt = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

public final class ALMModelFactory: ModelFactory, @unchecked Sendable {
    public static let shared = ALMModelFactory()
    public let typeRegistry: ModelTypeRegistry = LLMTypeRegistry.shared
    public let modelRegistry: AbstractModelRegistry = LLMRegistry.shared
    
    public init() {}

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        let context = try await LLMModelFactory.shared._load(configuration: configuration, tokenizerLoader: tokenizerLoader)
        
        let numAudioEmbeddings = OmniModelFactory.extractNumAudioEmbeddings(configuration: configuration)
        let messageGenerator = DefaultMessageGenerator()
        let processor = ALMUserInputProcessor(
            tokenizer: context.tokenizer,
            configuration: context.configuration,
            messageGenerator: messageGenerator,
            boaToken: 255010,
            eoaToken: 255011,
            numAudioEmbeddings: numAudioEmbeddings
        )
        
        return .init(
            configuration: context.configuration,
            model: context.model,
            processor: processor,
            tokenizer: context.tokenizer
        )
    }
}

public struct OmniUserInputProcessor: UserInputProcessor, @unchecked Sendable {
    let vlmProcessor: any UserInputProcessor
    let fusionProcessor: MultimodalFusionProcessor
    let numAudioEmbeddings: Int
    
    public init(vlmProcessor: any UserInputProcessor, boaToken: Int = 255010, eoaToken: Int = 255011, numAudioEmbeddings: Int = 128) {
        self.vlmProcessor = vlmProcessor
        self.fusionProcessor = MultimodalFusionProcessor(boaToken: boaToken, eoaToken: eoaToken)
        self.numAudioEmbeddings = numAudioEmbeddings
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Run standard VLM image substitution & image array processing
        let vlmInput = try await vlmProcessor.prepare(input: input)
        
        let tokens = vlmInput.text.tokens.asArray(Int.self)
        
        // If the VLM processor already natively extracted and processed the audio, do NOT mangle its layout with dummy interleaving!
        if vlmInput.audio != nil {
            return vlmInput
        }
        
        if !input.audio.isEmpty && !tokens.isEmpty {
            print("[Omni] Interleaving Audio Tokens into VLM prompt structure.")
            let rawSequence = fusionProcessor.interleave(
                textTokens: tokens,
                numAudioEmbeddings: numAudioEmbeddings,
                audioFirst: false // Append audio after vision context typically
            )
            return LMInput(text: .init(tokens: MLXArray(rawSequence)), image: vlmInput.image, audio: vlmInput.audio)
        }
        
        return vlmInput
    }
}

public final class OmniModelFactory: ModelFactory, @unchecked Sendable {
    public static let shared = OmniModelFactory()
    public let typeRegistry: ModelTypeRegistry = VLMTypeRegistry.shared
    public let modelRegistry: AbstractModelRegistry = VLMRegistry.shared
    
    public init() {}

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        let vlmContext = try await VLMModelFactory.shared._load(configuration: configuration, tokenizerLoader: tokenizerLoader)
        let numAudioEmbeddings = OmniModelFactory.extractNumAudioEmbeddings(configuration: configuration)
        let omniProcessor = OmniUserInputProcessor(
            vlmProcessor: vlmContext.processor,
            numAudioEmbeddings: numAudioEmbeddings
        )
        
        return .init(
            configuration: vlmContext.configuration,
            model: vlmContext.model,
            processor: omniProcessor,
            tokenizer: vlmContext.tokenizer
        )
    }

    public static func extractNumAudioEmbeddings(configuration: ResolvedModelConfiguration) -> Int {
        let configurationURL = configuration.modelDirectory.appending(component: "config.json")
        if let data = try? Data(contentsOf: configurationURL),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            
            if let subsampling = dict["subsampling_conv_channels"] as? [Int] {
                return subsampling.first ?? 128
            }
            if let audioConfig = dict["audio_config"] as? [String: Any],
               let embeddings = audioConfig["num_audio_embeddings"] as? Int {
                return embeddings
            }
        }
        return 128
    }
}
