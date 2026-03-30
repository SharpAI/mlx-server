// mlx-server — Minimal OpenAI-compatible HTTP server backed by Apple MLX Swift
//
// Endpoints:
//   GET  /health                    → { "status": "ok", "model": "<id>" }
//   GET  /v1/models                 → OpenAI-style model list
//   POST /v1/chat/completions       → OpenAI Chat Completions (streaming + non-streaming)
//   POST /v1/completions            → OpenAI Text Completions (streaming + non-streaming)
//
// Usage:
//   mlx-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 5413

import ArgumentParser
import CoreImage
import Foundation
import HTTPTypes
import Hummingbird
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

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
                    
                    let msg = String(format: "\r[mlx-server] Download: [%@] %@ %@ (%@ MB / %@ MB) %@", bars, pctStr, spinner, completedMB, totalMB, speedText)
                    
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
        commandName: "mlx-server",
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

    @Option(name: .long, help: "Repetition penalty factor (overridable per-request)")
    var repeatPenalty: Float?

    @Option(name: .long, help: "Number of parallel request slots")
    var parallel: Int = 1

    @Flag(name: .long, help: "Enable thinking/reasoning mode (Qwen3.5 etc). Default: disabled")
    var thinking: Bool = false

    @Flag(name: .long, help: "Enable VLM (vision-language model) mode for image inputs")
    var vision: Bool = false

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

    @Option(name: .long, help: "Chunk size for prefill evaluation (default: 512, lower to prevent GPU timeout on large models)")
    var prefillSize: Int = 512

    mutating func run() async throws {
        print("[mlx-server] Loading model: \(model)")
        let modelId = model

        // ── Load model ──
        var modelConfig: ModelConfiguration
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: modelId) {
            var isDir: ObjCBool = false
            fileManager.fileExists(atPath: modelId, isDirectory: &isDir)
            if isDir.boolValue {
                print("[mlx-server] Loading from local directory: \(modelId)")
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

        // ── Pre-load profiling ──
        // Resolve model directory for profiling (checks HuggingFace cache)
        let modelDirectory = resolveModelDirectory(modelId: modelId)
        
        if self.streamExperts, let modelDir = modelDirectory {
            setenv("EXPERIMENTAL_SSD_STREAM", modelDir.path, 1)
            // Cap Metal command buffer size to avoid the 5s Apple GPU Watchdog.
            // Each GatedDeltaNet kernel + attention + MLP = ~40 ops/layer max.
            // With 3 linear-attention layers before each full-attention layer,
            // we need a buffer budget of ~120-150 ops before the outer eval+sync
            // in partitionedLayerCall fires. Set a conservative limit.
            setenv("MLX_MAX_OPS_PER_BUFFER", "50", 1)
            print("[mlx-server] Enabled Async SSD Streaming on directory: \(modelDir.lastPathComponent)")
        }
        
        var partitionPlan: PartitionPlan?
        if let modelDir = modelDirectory,
           let profile = ModelProfiler.profile(modelDirectory: modelDir, modelId: modelId) {
            let system = ModelProfiler.systemProfile()
            let contextSize = self.ctxSize ?? 4096
            let plan = ModelProfiler.plan(model: profile, system: system, contextSize: contextSize)
            partitionPlan = plan

            // --info mode: print report and exit
            if self.info {
                ModelProfiler.printReport(plan: plan, model: profile, system: system)
                return
            }

            // Apply memory strategy
            switch plan.strategy {
            case .fullGPU:
                print("[mlx-server] \(plan.strategy.emoji) Memory strategy: FULL GPU (\(String(format: "%.1f", plan.weightMemoryGB))GB model, \(String(format: "%.1f", system.availableRAMGB))GB available)")
            case .swapAssisted:
                if self.streamExperts {
                    // SSD Streaming: bypass Metal's capped working set and use physical RAM budget
                    // (85% of total RAM minus 4GB OS reservation)
                    let physicalBudget = Int(Double(system.totalRAMBytes) * 0.85) - (4 * 1024 * 1024 * 1024)
                    Memory.cacheLimit = physicalBudget
                    Memory.memoryLimit = 200 * 1024 * 1024 * 1024 // 200GB sentinel to bypass MLX eval_impl spin loop
                    print("[mlx-server] \(plan.strategy.emoji) Memory strategy: SWAP-ASSISTED + SSD Streaming. Cache limit: \(physicalBudget / (1024*1024*1024))GB (physical RAM budget).")
                } else {
                    Memory.cacheLimit = plan.recommendedCacheLimit
                    print("[mlx-server] \(plan.strategy.emoji) Memory strategy: SWAP-ASSISTED (\(String(format: "%.1f", plan.overcommitRatio))× overcommit, cache limited to \(plan.recommendedCacheLimit / (1024*1024))MB)")
                }
                for w in plan.warnings { print("[mlx-server]    \(w)") }
            case .layerPartitioned:
                if self.streamExperts {
                    // SSD Streaming: bypass Metal's capped working set and use physical RAM budget
                    let physicalBudget = Int(Double(system.totalRAMBytes) * 0.85) - (4 * 1024 * 1024 * 1024)
                    Memory.cacheLimit = physicalBudget
                    Memory.memoryLimit = 200 * 1024 * 1024 * 1024 // 200GB sentinel to bypass MLX eval_impl spin loop
                    print("[mlx-server] \(plan.strategy.emoji) Memory strategy: LAYER PARTITIONED + SSD Streaming. Cache limit: \(physicalBudget / (1024*1024*1024))GB (physical RAM budget).")
                } else {
                    Memory.cacheLimit = plan.recommendedCacheLimit
                    print("[mlx-server] \(plan.strategy.emoji) Memory strategy: LAYER PARTITIONED (\(plan.recommendedGPULayers)/\(plan.totalLayers) GPU layers, cache limited to \(plan.recommendedCacheLimit / (1024*1024))MB)")
                }
                for w in plan.warnings { print("[mlx-server]    \(w)") }
            case .tooLarge:
                Memory.cacheLimit = plan.recommendedCacheLimit
                print("[mlx-server] \(plan.strategy.emoji) WARNING: Model is \(String(format: "%.1f", plan.overcommitRatio))× system RAM. Loading will be extremely slow.")
                for w in plan.warnings { print("[mlx-server]    \(w)") }
            }
        } else if self.info {
            print("[mlx-server] Model not yet downloaded. Run without --info to download first, or provide a local path.")
            return
        }

        // ── Determine GPU layer count ──
        // Priority: 1) explicit --gpu-layers flag, 2) partition plan auto, 3) nil (all GPU)
        var requestedGPULayers: Int? = nil
        if let gpuLayersArg = self.gpuLayers {
            if gpuLayersArg == "auto" {
                // Use partition plan recommendation if available
                requestedGPULayers = partitionPlan?.recommendedGPULayers
                print("[mlx-server] --gpu-layers auto → \(requestedGPULayers.map(String.init) ?? "all") layers on GPU")
            } else if let n = Int(gpuLayersArg) {
                requestedGPULayers = n
                print("[mlx-server] --gpu-layers \(n) → \(n) layers on GPU")
            } else {
                print("[mlx-server] Warning: --gpu-layers must be 'auto' or an integer, got '\(gpuLayersArg)'. Using all GPU.")
            }
        } else if let plan = partitionPlan,
                  (plan.strategy == .layerPartitioned || plan.strategy == .swapAssisted),
                  plan.overcommitRatio > 1.0 {
            if self.streamExperts {
                print("[mlx-server] SSD Streaming active: Bypassing CPU auto-partitioning (forcing all layers to GPU)")
                partitionPlan?.gpuLayers = plan.totalLayers
                // Keep requestedGPULayers = nil (all GPU)
            } else {
                // Auto-partition when model exceeds available RAM (no flag needed)
                requestedGPULayers = plan.recommendedGPULayers
                print("[mlx-server] Auto-partitioning: \(plan.recommendedGPULayers)/\(plan.totalLayers) layers on GPU")
            }
        }

        let isVision = self.vision
        let container: ModelContainer
        
        // Handle getting the simple model ID string for the tracker
        let resolvedModelId: String = {
            if case .id(let idStr, _) = modelConfig.id { return idStr }
            return self.model
        }()
        let tracker = ProgressTracker(modelId: resolvedModelId)
        
        if isVision {
            print("[mlx-server] Loading VLM (vision-language model)...")
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { progress in
                tracker.printProgress(progress)
            }
        } else {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { progress in
                tracker.printProgress(progress)
            }
        }

        // ── Apply GPU/CPU layer partitioning ──
        if let gpuCount = requestedGPULayers {
            let actual = await container.setGPULayers(gpuCount)
            if let actual {
                let total = partitionPlan?.totalLayers ?? actual
                let cpuCount = total - actual
                print("[mlx-server] 🔀 Layer split active: \(actual) GPU / \(cpuCount) CPU")
                // Update the partition plan to reflect actual split
                partitionPlan?.gpuLayers = actual
            } else {
                print("[mlx-server] ⚠️  Model does not support layer partitioning (architecture not yet adapted)")
            }
        }

        // ── Apply SSD Expert Streaming ──
        if self.streamExperts {
            let streamingEnabled = await container.setStreamExperts(true)
            if streamingEnabled {
                print("[mlx-server] 💾 SSD Expert Streaming enabled (lazy load + layer-sync)")
            } else {
                print("[mlx-server] ⚠️  Model does not support SSD expert streaming")
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
                print("[mlx-server] 📊 Loaded wisdom: \(String(format: "%.1f", wisdom.tokPerSec)) tok/s, cache=\(wisdom.cacheLimit / (1024*1024))MB (calibrated \(wisdom.calibratedAt.formatted(.relative(presentation: .named))))")
            }
        } else if self.streamExperts {
            print("[mlx-server] 🧠 Auto-calibration (Wisdom) bypassed for SSD Streaming")
        }

        print("[mlx-server] Model loaded. Starting HTTP server on \(host):\(port)")

        // ── Capture CLI defaults into a shared config ──
        let config = ServerConfig(
            modelId: modelId,
            maxTokens: self.maxTokens,
            ctxSize: self.ctxSize,
            temp: self.temp,
            topP: self.topP,
            repeatPenalty: self.repeatPenalty,
            thinking: self.thinking,
            isVision: isVision,
            prefillSize: self.prefillSize
        )

        let parallelSlots = self.parallel
        let corsOrigin = self.cors
        let apiKeyValue = self.apiKey

        // ── Memory limit enforcement (overrides wisdom) ──
        if let memLimitMB = self.memLimit {
            let bytes = memLimitMB * 1024 * 1024
            Memory.memoryLimit = bytes
            Memory.cacheLimit = bytes
            print("[mlx-server] Memory limit set to \(memLimitMB)MB (overrides wisdom)")
        }

        // ── Concurrency limiter ──
        let semaphore = AsyncSemaphore(limit: parallelSlots)

        // ── Server stats tracker ──
        let stats = ServerStats()

        let ctxSizeStr = config.ctxSize.map { String($0) } ?? "model_default"
        let penaltyStr = config.repeatPenalty.map { String($0) } ?? "disabled"
        let corsStr = corsOrigin ?? "disabled"
        let memLimitStr = self.memLimit.map { "\($0)MB" } ?? "system_default"
        let authStr = apiKeyValue != nil ? "enabled" : "disabled"
        print("[mlx-server] Config: ctx_size=\(ctxSizeStr), temp=\(config.temp), top_p=\(config.topP), repeat_penalty=\(penaltyStr), parallel=\(parallelSlots), cors=\(corsStr), mem_limit=\(memLimitStr), auth=\(authStr)")

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
        router.get("/health") { _, _ -> Response in
            let activeMemMB = Memory.activeMemory / (1024 * 1024)
            let peakMemMB = Memory.peakMemory / (1024 * 1024)
            let cacheMemMB = Memory.cacheMemory / (1024 * 1024)
            let deviceInfo = GPU.deviceInfo()
            let totalMemMB = deviceInfo.memorySize / (1024 * 1024)
            let snapshot = await stats.snapshot()
            // Build partition info string
            var partitionJson = ""
            if let plan = partitionPlan {
                let pData: [String: Any] = [
                    "strategy": plan.strategy.rawValue,
                    "overcommit_ratio": round(plan.overcommitRatio * 100) / 100,
                    "model_weight_gb": round(plan.weightMemoryGB * 10) / 10,
                    "kv_cache_gb": round(plan.kvCacheMemoryGB * 10) / 10,
                    "total_required_gb": round(plan.totalRequiredGB * 10) / 10,
                    "gpu_layers": plan.recommendedGPULayers,
                    "total_layers": plan.totalLayers,
                    "estimated_tok_s": round(plan.estimatedTokensPerSec * 10) / 10,
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
                    bodyData: bodyData, config: config, container: container, semaphore: semaphore, stats: stats, promptCache: promptCache
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
                    bodyData: bodyData, config: config, container: container, semaphore: semaphore, stats: stats
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
            lines.append("# HELP mlx_server_requests_total Total requests processed")
            lines.append("# TYPE mlx_server_requests_total counter")
            lines.append("mlx_server_requests_total \(snapshot.requestsTotal)")
            lines.append("# HELP mlx_server_requests_active Currently active requests")
            lines.append("# TYPE mlx_server_requests_active gauge")
            lines.append("mlx_server_requests_active \(snapshot.requestsActive)")
            lines.append("# HELP mlx_server_tokens_generated_total Total tokens generated")
            lines.append("# TYPE mlx_server_tokens_generated_total counter")
            lines.append("mlx_server_tokens_generated_total \(snapshot.tokensGenerated)")
            lines.append("# HELP mlx_server_tokens_per_second Average token generation rate")
            lines.append("# TYPE mlx_server_tokens_per_second gauge")
            lines.append("mlx_server_tokens_per_second \(String(format: "%.2f", snapshot.avgTokensPerSec))")
            lines.append("# HELP mlx_server_memory_active_bytes Active GPU memory usage")
            lines.append("# TYPE mlx_server_memory_active_bytes gauge")
            lines.append("mlx_server_memory_active_bytes \(activeMemBytes)")
            lines.append("# HELP mlx_server_memory_peak_bytes Peak GPU memory usage")
            lines.append("# TYPE mlx_server_memory_peak_bytes gauge")
            lines.append("mlx_server_memory_peak_bytes \(peakMemBytes)")
            lines.append("# HELP mlx_server_memory_cache_bytes Cached GPU memory")
            lines.append("# TYPE mlx_server_memory_cache_bytes gauge")
            lines.append("mlx_server_memory_cache_bytes \(cacheMemBytes)")
            lines.append("# HELP mlx_server_uptime_seconds Server uptime")
            lines.append("# TYPE mlx_server_uptime_seconds gauge")
            lines.append("mlx_server_uptime_seconds \(String(format: "%.0f", uptime))")
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

        print("[mlx-server] ✅ Ready. Listening on http://\(host):\(port)")

        // ── Emit machine-readable ready event for Aegis integration ──
        var readyEvent: [String: Any] = [
            "event": "ready",
            "port": port,
            "model": modelId,
            "engine": "mlx",
            "vision": isVision
        ]
        if let plan = partitionPlan {
            readyEvent["partition"] = plan.healthInfo
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
            print("\n[mlx-server] Received SIGTERM, shutting down gracefully...")
            Darwin.exit(0)
        }
        interruptSource.setEventHandler {
            print("\n[mlx-server] Received SIGINT, shutting down gracefully...")
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
    let repeatPenalty: Float?
    let thinking: Bool
    let isVision: Bool
    let prefillSize: Int
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

// ── Prompt Cache ─────────────────────────────────────────────────────────────

actor PromptCache {
    struct CachedState {
        let states: [[MLXArray]]     // Per-layer KV state arrays
        let metaStates: [[String]]   // Per-layer metadata
        let tokenCount: Int          // Number of cached tokens
    }

    private var cached: CachedState?
    private var cachedTokenHash: Int?
    private var hits: Int = 0
    private var misses: Int = 0

    func save(tokenHash: Int, cache: [KVCache], tokenCount: Int) {
        let states = cache.map { $0.state }
        let metaStates = cache.map { $0.metaState }
        cached = CachedState(states: states, metaStates: metaStates, tokenCount: tokenCount)
        cachedTokenHash = tokenHash
    }

    func restore(tokenHash: Int, into cache: [KVCache]) -> Int? {
        guard let cached, cachedTokenHash == tokenHash else {
            misses += 1
            return nil
        }
        for i in 0..<min(cache.count, cached.states.count) {
            var layer = cache[i]
            layer.state = cached.states[i]
            layer.metaState = cached.metaStates[i]
        }
        hits += 1
        return cached.tokenCount
    }

    func stats() -> (hits: Int, misses: Int) { (hits, misses) }
}

// ── Request Body Extraction ──────────────────────────────────────────────────

func collectBody(_ request: Request) async throws -> Data {
    var bodyBuffer = try await request.body.collect(upTo: 10 * 1024 * 1024)
    let bodyBytes = bodyBuffer.readBytes(length: bodyBuffer.readableBytes) ?? []
    return Data(bodyBytes)
}

// ── Chat Completions Handler ─────────────────────────────────────────────────

func handleChatCompletion(
    bodyData: Data,
    config: ServerConfig,
    container: ModelContainer,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    promptCache: PromptCache
) async throws -> Response {
    let chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: bodyData)
    let isStream = chatReq.stream ?? false
    let jsonMode = chatReq.responseFormat?.type == "json_object"

    // ── Merge per-request overrides with CLI defaults ──
    let tokenLimit = chatReq.maxTokens ?? config.maxTokens
    let temperature = chatReq.temperature.map(Float.init) ?? config.temp
    let topP = chatReq.topP.map(Float.init) ?? config.topP
    let repeatPenalty = chatReq.repetitionPenalty.map(Float.init) ?? config.repeatPenalty
    let stopSequences = chatReq.stop ?? []
    let includeUsage = chatReq.streamOptions?.includeUsage ?? false

    // Log extra sampling params if provided (accepted for API compat, not all are used)
    if chatReq.topK != nil || chatReq.frequencyPenalty != nil || chatReq.presencePenalty != nil {
        // These are accepted but may not affect generation if MLX doesn't support them
    }

    let params = GenerateParameters(
        maxTokens: tokenLimit,
        maxKVSize: config.ctxSize,
        temperature: temperature,
        topP: topP,
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
        let textContent = msg.textContent
        let images = msg.extractImages()
        switch msg.role {
        case "system":
            chatMessages.append(.system(textContent, images: images))
            systemPromptText += textContent
        case "assistant":
            chatMessages.append(.assistant(textContent, images: images))
        default:
            chatMessages.append(.user(textContent, images: images))
        }
    }

    // ── JSON mode: inject system prompt for JSON output ──
    if jsonMode {
        let jsonSystemMsg = Chat.Message.system("You must respond with valid JSON only. No markdown code fences, no explanation text, no preamble. Output raw JSON.")
        chatMessages.insert(jsonSystemMsg, at: 0)
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

    // Pass enable_thinking to the Jinja chat template via additionalContext
    let templateContext: [String: any Sendable]? = config.thinking ? nil : ["enable_thinking": false]
    let userInput = UserInput(chat: chatMessages, tools: toolSpecs, additionalContext: templateContext)
    let lmInput = try await container.prepare(input: userInput)

    // ── Prompt caching: compute hash for system prompt ──
    let promptTokenCount = lmInput.text.tokens.size
    let systemHash = systemPromptText.hashValue

    // ── Cache-aware generation ──
    let stream: AsyncStream<Generation> = try await container.perform { context in
        let cache = context.model.newCache(parameters: params)

        // Try to restore cached system prompt KV state
        if let cachedCount = await promptCache.restore(tokenHash: systemHash, into: cache) {
            // Cache hit: skip the cached prefix tokens, process only the rest
            let remainingTokens = lmInput.text.tokens[cachedCount...]
            let trimmedInput = LMInput(tokens: remainingTokens)
            return try MLXLMCommon.generate(
                input: trimmedInput, cache: cache, parameters: params, context: context
            )
        } else {
            // Cache miss: process everything, then save system prompt state
            // Count system prompt tokens using the tokenizer
            var systemTokenCount = 0
            if !systemPromptText.isEmpty {
                // Approximate system token count from the tokenizer
                let sysTokens = context.tokenizer.encode(text: systemPromptText)
                // Add overhead for chat template tokens (BOS, role markers, etc.)
                systemTokenCount = sysTokens.count + 4
            }

            let stream = try MLXLMCommon.generate(
                input: lmInput, cache: cache, parameters: params, context: context
            )

            // Save cache state after prefill (cache now contains all prompt tokens)
            if systemTokenCount > 0 {
                // Save the full prompt cache, but record the system token count
                // so future requests with different user messages can still benefit
                // (they'll have the system prefix cached)
                //
                // Note: We save after generate() starts, which means the cache
                // has been populated by the TokenIterator's prepare() call.
                // We use Task to save asynchronously after the first token.
                Task {
                    // Small delay to let prefill complete and populate the cache
                    try? await Task.sleep(for: .milliseconds(100))
                    await promptCache.save(tokenHash: systemHash, cache: cache, tokenCount: systemTokenCount)
                }
            }

            return stream
        }
    }

    let modelId = config.modelId

    if isStream {
        return handleChatStreaming(
            stream: stream, modelId: modelId, stopSequences: stopSequences,
            includeUsage: includeUsage, promptTokenCount: promptTokenCount,
            jsonMode: jsonMode, semaphore: semaphore, stats: stats, genStart: genStart
        )
    } else {
        return try await handleChatNonStreaming(
            stream: stream, modelId: modelId, stopSequences: stopSequences,
            promptTokenCount: promptTokenCount, jsonMode: jsonMode, semaphore: semaphore, stats: stats, genStart: genStart
        )
    }
}

// ── Chat Streaming ───────────────────────────────────────────────────────────

func handleChatStreaming(
    stream: AsyncStream<Generation>,
    modelId: String,
    stopSequences: [String],
    includeUsage: Bool,
    promptTokenCount: Int,
    jsonMode: Bool = false,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date
) -> Response {
    let (sseStream, cont) = AsyncStream<String>.makeStream()
    Task {
        var hasToolCalls = false
        var toolCallIndex = 0
        var completionTokenCount = 0
        var fullText = ""
        var stopped = false
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
                // ── Stop sequence check ──
                if let (trimmedText, _) = checkStopSequences(fullText, stopSequences: stopSequences) {
                    let emittedSoFar = fullText.count - text.count
                    if trimmedText.count > emittedSoFar {
                        let partialText = String(trimmedText.suffix(trimmedText.count - emittedSoFar))
                        cont.yield(sseChunk(modelId: modelId, delta: partialText, finishReason: nil))
                    }
                    cont.yield(sseChunk(modelId: modelId, delta: "", finishReason: "stop"))
                    if includeUsage {
                        cont.yield(sseUsageChunk(modelId: modelId, promptTokens: promptTokenCount, completionTokens: completionTokenCount))
                    }
                    cont.yield("data: [DONE]\n\n")
                    cont.finish()
                    stopped = true
                } else {
                    cont.yield(sseChunk(modelId: modelId, delta: text, finishReason: nil))
                }
            case .toolCall(let tc):
                hasToolCalls = true
                let argsJson = serializeToolCallArgs(tc.function.arguments)
                cont.yield(sseToolCallChunk(modelId: modelId, index: toolCallIndex, name: tc.function.name, arguments: argsJson))
                toolCallIndex += 1
            case .info(let info):
                if !stopped {
                    var reason: String
                    switch info.stopReason {
                    case .length:
                        reason = "length"
                    case .cancelled, .stop:
                        reason = hasToolCalls ? "tool_calls" : "stop"
                    }
                    cont.yield(sseChunk(modelId: modelId, delta: "", finishReason: reason))
                    if includeUsage {
                        cont.yield(sseUsageChunk(modelId: modelId, promptTokens: promptTokenCount, completionTokens: completionTokenCount))
                    }
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

// ── Chat Non-Streaming ───────────────────────────────────────────────────────

func handleChatNonStreaming(
    stream: AsyncStream<Generation>,
    modelId: String,
    stopSequences: [String],
    promptTokenCount: Int,
    jsonMode: Bool = false,
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date
) async throws -> Response {
    var fullText = ""
    var completionTokenCount = 0
    var collectedToolCalls: [ToolCallResponse] = []
    var tcIndex = 0
    var generationStopReason: GenerateStopReason = .stop
    for await generation in stream {
        switch generation {
        case .chunk(let text, _):
            fullText += text
            completionTokenCount += 1
            // GPU yield: prevent Metal from starving macOS WindowServer
            if completionTokenCount % 8 == 0 {
                try? await Task.sleep(for: .microseconds(50))
            }
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

    // ── JSON mode validation ──
    if jsonMode {
        // Strip markdown code fences if model wrapped response
        let stripped = fullText
            .replacingOccurrences(of: "```json\n", with: "")
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```\n", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        fullText = stripped
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
                    content: fullText.isEmpty && hasToolCalls ? nil : fullText,
                    toolCalls: hasToolCalls ? collectedToolCalls : nil
                ),
                finishReason: hasToolCalls ? "tool_calls" : finishReason
            )
        ],
        usage: TokenUsage(promptTokens: promptTokenCount, completionTokens: completionTokenCount, totalTokens: totalTokens)
    )
    let encoded = try JSONEncoder().encode(resp)
    return Response(
        status: .ok,
        headers: jsonHeaders(),
        body: .init(byteBuffer: ByteBuffer(data: encoded))
    )
}

// ── Text Completions Handler ─────────────────────────────────────────────────

func handleTextCompletion(
    bodyData: Data,
    config: ServerConfig,
    container: ModelContainer,
    semaphore: AsyncSemaphore,
    stats: ServerStats
) async throws -> Response {
    let compReq = try JSONDecoder().decode(TextCompletionRequest.self, from: bodyData)
    let isStream = compReq.stream ?? false

    let tokenLimit = compReq.maxTokens ?? config.maxTokens
    let temperature = compReq.temperature.map(Float.init) ?? config.temp
    let topP = compReq.topP.map(Float.init) ?? config.topP
    let repeatPenalty = compReq.repetitionPenalty.map(Float.init) ?? config.repeatPenalty
    let stopSequences = compReq.stop ?? []

    let params = GenerateParameters(
        maxTokens: tokenLimit,
        maxKVSize: config.ctxSize,
        temperature: temperature,
        topP: topP,
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
            semaphore: semaphore, stats: stats, genStart: genStart
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
    semaphore: AsyncSemaphore,
    stats: ServerStats,
    genStart: Date
) -> Response {
    let (sseStream, cont) = AsyncStream<String>.makeStream()
    Task {
        var completionTokenCount = 0
        var fullText = ""
        var stopped = false
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
                if !stopped {
                    var reason: String
                    switch info.stopReason {
                    case .length:
                        reason = "length"
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
        usage: TokenUsage(promptTokens: promptTokenCount, completionTokens: completionTokenCount, totalTokens: totalTokens)
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
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Headers")!, value: "Content-Type, Authorization"))
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

func sseHeaders() -> HTTPFields {
    HTTPFields([
        HTTPField(name: .contentType, value: "text/event-stream"),
        HTTPField(name: .cacheControl, value: "no-cache"),
        HTTPField(name: HTTPField.Name("X-Accel-Buffering")!, value: "no"),
    ])
}

func sseChunk(modelId: String, delta: String, finishReason: String?) -> String {
    var deltaObj: [String: Any] = [:]
    if !delta.isEmpty {
        deltaObj = ["role": "assistant", "content": delta]
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
    return "data: \(String(data: data, encoding: .utf8)!)\n\n"
}

func sseUsageChunk(modelId: String, promptTokens: Int, completionTokens: Int) -> String {
    let chunk: [String: Any] = [
        "id": "chatcmpl-\(UUID().uuidString)",
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelId,
        "choices": [] as [[String: Any]],
        "usage": [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens
        ]
    ]
    let data = try! JSONSerialization.data(withJSONObject: chunk)
    return "data: \(String(data: data, encoding: .utf8)!)\n\n"
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
    return "data: \(String(data: data, encoding: .utf8)!)\n\n"
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
    return "data: \(String(data: data, encoding: .utf8)!)\n\n"
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
    }

    /// Message content: either a plain string or structured multipart content
    enum MessageContent: Decodable {
        case string(String)
        case parts([ContentPart])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .string(str)
            } else if let parts = try? container.decode([ContentPart].self) {
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

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageUrl = "image_url"
        }
    }

    struct ImageUrlContent: Decodable {
        let url: String
        let detail: String?
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
    let repetitionPenalty: Double?
    let frequencyPenalty: Double?
    let presencePenalty: Double?
    let tools: [ToolDef]?
    let stop: [String]?
    let seed: Int?
    let streamOptions: StreamOptions?
    let responseFormat: ResponseFormat?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools, stop, seed
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case streamOptions = "stream_options"
        case responseFormat = "response_format"
    }
}

struct TextCompletionRequest: Decodable {
    let model: String?
    let prompt: String
    let stream: Bool?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let repetitionPenalty: Double?
    let stop: [String]?
    let seed: Int?

    enum CodingKeys: String, CodingKey {
        case model, prompt, stream, temperature, stop, seed
        case maxTokens = "max_tokens"
        case topP = "top_p"
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
    let toolCalls: [ToolCallResponse]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
    }
}

struct ToolCallResponse: Encodable {
    let id: String
    let type: String
    let function: ToolCallFunction
}

struct ToolCallFunction: Encodable {
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

struct AnyCodable: Decodable, Sendable {
    let value: Any
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { value = NSNull() }
        else if let b = try? c.decode(Bool.self) { value = b }
        else if let i = try? c.decode(Int.self) { value = i }
        else if let d = try? c.decode(Double.self) { value = d }
        else if let s = try? c.decode(String.self) { value = s }
        else if let a = try? c.decode([AnyCodable].self) { value = a.map { $0.value } }
        else if let d = try? c.decode([String: AnyCodable].self) { value = d.mapValues { $0.value } }
        else { value = NSNull() }
    }
    static func toSendable(_ dict: [String: AnyCodable]?) -> [String: any Sendable]? {
        guard let dict else { return nil }
        return dict.mapValues { $0.value as! any Sendable }
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
