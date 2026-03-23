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

    @Option(name: .long, help: "Allowed CORS origin (* for all, or a specific origin URL)")
    var cors: String?

    mutating func run() async throws {
        print("[mlx-server] Loading model: \(model)")
        let modelId = model

        // ── Load model ──
        let modelConfig: ModelConfiguration
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

        let isVision = self.vision
        let container: ModelContainer
        if isVision {
            print("[mlx-server] Loading VLM (vision-language model)...")
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                print("[mlx-server] Download: \(pct)%")
            }
        } else {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                print("[mlx-server] Download: \(pct)%")
            }
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
            isVision: isVision
        )

        let parallelSlots = self.parallel
        let corsOrigin = self.cors

        // ── Memory limit enforcement ──
        if let memLimitMB = self.memLimit {
            let bytes = memLimitMB * 1024 * 1024
            Memory.memoryLimit = bytes
            Memory.cacheLimit = bytes
            print("[mlx-server] Memory limit set to \(memLimitMB)MB")
        }

        // ── Concurrency limiter ──
        let semaphore = AsyncSemaphore(limit: parallelSlots)

        // ── Server stats tracker ──
        let stats = ServerStats()

        let ctxSizeStr = config.ctxSize.map { String($0) } ?? "model_default"
        let penaltyStr = config.repeatPenalty.map { String($0) } ?? "disabled"
        let corsStr = corsOrigin ?? "disabled"
        let memLimitStr = self.memLimit.map { "\($0)MB" } ?? "system_default"
        print("[mlx-server] Config: ctx_size=\(ctxSizeStr), temp=\(config.temp), top_p=\(config.topP), repeat_penalty=\(penaltyStr), parallel=\(parallelSlots), cors=\(corsStr), mem_limit=\(memLimitStr)")

        // ── Build Hummingbird router ──
        let router = Router()

        // ── CORS middleware ──
        if let origin = corsOrigin {
            router.add(middleware: CORSMiddleware(allowedOrigin: origin))
        }

        // Health (enhanced v2 with memory + stats)
        router.get("/health") { _, _ -> Response in
            let activeMemMB = Memory.activeMemory / (1024 * 1024)
            let peakMemMB = Memory.peakMemory / (1024 * 1024)
            let cacheMemMB = Memory.cacheMemory / (1024 * 1024)
            let deviceInfo = GPU.deviceInfo()
            let totalMemMB = deviceInfo.memorySize / (1024 * 1024)
            let snapshot = await stats.snapshot()
            let payload = """
{"status":"ok","model":"\(modelId)","vision":\(isVision),"memory":{"active_mb":\(activeMemMB),"peak_mb":\(peakMemMB),"cache_mb":\(cacheMemMB),"total_system_mb":\(totalMemMB),"gpu_architecture":"\(deviceInfo.architecture)"},"stats":{"requests_total":\(snapshot.requestsTotal),"requests_active":\(snapshot.requestsActive),"tokens_generated":\(snapshot.tokensGenerated),"avg_tokens_per_sec":\(String(format: "%.2f", snapshot.avgTokensPerSec))}}
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
        router.post("/v1/chat/completions") { request, _ -> Response in
            let bodyData = try await collectBody(request)
            return try await handleChatCompletion(
                bodyData: bodyData, config: config, container: container, semaphore: semaphore, stats: stats
            )
        }

        // Text completions — handler extracted to avoid type-checker timeout
        router.post("/v1/completions") { request, _ -> Response in
            let bodyData = try await collectBody(request)
            return try await handleTextCompletion(
                bodyData: bodyData, config: config, container: container, semaphore: semaphore, stats: stats
            )
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
        let readyEvent: [String: Any] = [
            "event": "ready",
            "port": port,
            "model": modelId,
            "engine": "mlx",
            "vision": isVision
        ]
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
    stats: ServerStats
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
        repetitionPenalty: repeatPenalty
    )

    // ── Seed for deterministic generation ──
    if let seed = chatReq.seed {
        MLXRandom.seed(UInt64(seed))
    }

    // ── Parse messages with multipart content support (for VLM images) ──
    var chatMessages: [Chat.Message] = []
    for msg in chatReq.messages {
        let textContent = msg.textContent
        let images = msg.extractImages()
        switch msg.role {
        case "system":
            chatMessages.append(.system(textContent, images: images))
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

    // ── Get actual prompt token count before generate() to avoid data race ──
    let promptTokenCount = lmInput.text.tokens.size

    let stream = try await container.generate(input: lmInput, parameters: params)

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
            case .chunk(let text):
                completionTokenCount += 1
                fullText += text
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
            case .info:
                if !stopped {
                    let reason = hasToolCalls ? "tool_calls" : "stop"
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
    for await generation in stream {
        switch generation {
        case .chunk(let text):
            fullText += text
            completionTokenCount += 1
        case .toolCall(let tc):
            let argsJson = serializeToolCallArgs(tc.function.arguments)
            collectedToolCalls.append(ToolCallResponse(
                id: "call_\(UUID().uuidString.prefix(8))",
                type: "function",
                function: ToolCallFunction(name: tc.function.name, arguments: argsJson)
            ))
            tcIndex += 1
        case .info:
            break
        }
    }
    let duration = Date().timeIntervalSince(genStart)
    await stats.requestFinished(tokens: completionTokenCount, duration: duration)
    await semaphore.signal()

    // ── Apply stop sequences to final text ──
    var finishReason = "stop"
    if let (trimmedText, _) = checkStopSequences(fullText, stopSequences: stopSequences) {
        fullText = trimmedText
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
        repetitionPenalty: repeatPenalty
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
            case .chunk(let text):
                completionTokenCount += 1
                fullText += text
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
            case .info:
                if !stopped {
                    cont.yield(sseTextChunk(modelId: modelId, text: "", finishReason: "stop"))
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
        case .chunk(let text):
            fullText += text
            completionTokenCount += 1
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
        let content: MessageContent

        /// Extract plain text from content (handles both string and multipart)
        var textContent: String {
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
            guard case .parts(let parts) = content else { return [] }
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
