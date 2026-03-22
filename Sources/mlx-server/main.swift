// mlx-server — Minimal OpenAI-compatible HTTP server backed by Apple MLX Swift
//
// Endpoints:
//   GET  /health                    → { "status": "ok", "model": "<id>" }
//   GET  /v1/models                 → OpenAI-style model list
//   POST /v1/chat/completions       → OpenAI Chat Completions (streaming + non-streaming)
//
// Usage:
//   mlx-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 5413

import ArgumentParser
import Foundation
import HTTPTypes
import Hummingbird
import MLXLLM
import MLXLMCommon

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

    mutating func run() async throws {
        print("[mlx-server] Loading model: \(model)")
        let modelId = model

        // ── Load model ──
        let modelConfig = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: modelConfig
        ) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            print("[mlx-server] Download: \(pct)%")
        }

        print("[mlx-server] Model loaded. Starting HTTP server on \(host):\(port)")

        // ── Capture CLI defaults ──
        let defaultMaxTokens = self.maxTokens
        let defaultCtxSize = self.ctxSize
        let defaultTemp = self.temp
        let defaultTopP = self.topP
        let defaultRepeatPenalty = self.repeatPenalty
        let thinkingEnabled = self.thinking
        let parallelSlots = self.parallel

        // ── Concurrency limiter ──
        let semaphore = AsyncSemaphore(limit: parallelSlots)

        let ctxSizeStr = defaultCtxSize.map { String($0) } ?? "model_default"
        let penaltyStr = defaultRepeatPenalty.map { String($0) } ?? "disabled"
        print("[mlx-server] Config: ctx_size=\(ctxSizeStr), temp=\(defaultTemp), top_p=\(defaultTopP), repeat_penalty=\(penaltyStr), parallel=\(parallelSlots)")

        // ── Build Hummingbird router ──
        let router = Router()

        // Health
        router.get("/health") { _, _ -> Response in
            let payload = "{\"status\":\"ok\",\"model\":\"\(modelId)\"}"
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

        // Chat completions
        router.post("/v1/chat/completions") { request, _ -> Response in
            var bodyBuffer = try await request.body.collect(upTo: 10 * 1024 * 1024)
            let bodyBytes = bodyBuffer.readBytes(length: bodyBuffer.readableBytes) ?? []
            let bodyData = Data(bodyBytes)

            let chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: bodyData)
            let isStream = chatReq.stream ?? false

            // ── Merge per-request overrides with CLI defaults ──
            let tokenLimit = chatReq.maxTokens ?? defaultMaxTokens
            let temperature = chatReq.temperature.map(Float.init) ?? defaultTemp
            let topP = chatReq.topP.map(Float.init) ?? defaultTopP
            let repeatPenalty = chatReq.repetitionPenalty.map(Float.init) ?? defaultRepeatPenalty

            let params = GenerateParameters(
                maxTokens: tokenLimit,
                maxKVSize: defaultCtxSize,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repeatPenalty
            )

            // Convert request messages → Chat.Message
            let chatMessages: [Chat.Message] = chatReq.messages.compactMap { msg in
                switch msg.role {
                case "system":    return .system(msg.content)
                case "assistant": return .assistant(msg.content)
                default:          return .user(msg.content)
                }
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

            // Pass enable_thinking to the Jinja chat template via additionalContext
            // (mirrors llama-server's --chat-template-kwargs '{"enable_thinking":false}')
            let templateContext: [String: any Sendable]? = thinkingEnabled ? nil : ["enable_thinking": false]
            let userInput = UserInput(chat: chatMessages, tools: toolSpecs, additionalContext: templateContext)
            let lmInput = try await container.prepare(input: userInput)
            let stream = try await container.generate(input: lmInput, parameters: params)

            if isStream {
                // SSE streaming
                let (sseStream, cont) = AsyncStream<String>.makeStream()
                Task {
                    var hasToolCalls = false
                    var toolCallIndex = 0
                    for await generation in stream {
                        switch generation {
                        case .chunk(let text):
                            cont.yield(sseChunk(modelId: modelId, delta: text, finishReason: nil))
                        case .toolCall(let tc):
                            hasToolCalls = true
                            let argsJson = serializeToolCallArgs(tc.function.arguments)
                            cont.yield(sseToolCallChunk(modelId: modelId, index: toolCallIndex, name: tc.function.name, arguments: argsJson))
                            toolCallIndex += 1
                        case .info:
                            let reason = hasToolCalls ? "tool_calls" : "stop"
                            cont.yield(sseChunk(modelId: modelId, delta: "", finishReason: reason))
                            cont.yield("data: [DONE]\n\n")
                            cont.finish()
                        }
                    }
                    cont.finish()
                    await semaphore.signal()
                }
                return Response(
                    status: .ok,
                    headers: sseHeaders(),
                    body: .init(asyncSequence: sseStream.map { ByteBuffer(string: $0) })
                )
            } else {
                // Non-streaming: collect all chunks and tool calls
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
                await semaphore.signal()

                // Approximate prompt tokens (chars / 4 is a reasonable heuristic for most tokenizers)
                let promptText = chatReq.messages.map { $0.content }.joined(separator: " ")
                let estimatedPromptTokens = max(1, promptText.count / 4)
                let totalTokens = estimatedPromptTokens + completionTokenCount

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
                            finishReason: hasToolCalls ? "tool_calls" : "stop"
                        )
                    ],
                    usage: TokenUsage(promptTokens: estimatedPromptTokens, completionTokens: completionTokenCount, totalTokens: totalTokens)
                )
                let encoded = try JSONEncoder().encode(resp)
                return Response(
                    status: .ok,
                    headers: jsonHeaders(),
                    body: .init(byteBuffer: ByteBuffer(data: encoded))
                )
            }
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
            "vision": false
        ]
        if let data = try? JSONSerialization.data(withJSONObject: readyEvent),
           let json = String(data: data, encoding: .utf8) {
            print(json)
            fflush(stdout)
        }

        // ── Handle SIGTERM/SIGINT for graceful shutdown ──
        // Note: C signal handlers can't safely do complex I/O.
        // Aegis detects process exit and fires onEngineStopped() automatically.

        try await app.runService()
    }
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
    var chunk: [String: Any] = [
        "id": "chatcmpl-\(UUID().uuidString)",
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelId,
        "choices": [[
            "index": 0,
            "delta": deltaObj,
        ] as [String: Any]]
    ]
    if let finishReason {
        if var choices = chunk["choices"] as? [[String: Any]], !choices.isEmpty {
            choices[0]["finish_reason"] = finishReason
            chunk["choices"] = choices
        }
    }
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

/// Serialize ToolCall arguments ([String: JSONValue]) to a JSON string
func serializeToolCallArgs(_ args: [String: JSONValue]) -> String {
    let anyDict = args.mapValues { $0.anyValue }
    guard let data = try? JSONSerialization.data(withJSONObject: anyDict) else {
        return "{}"
    }
    return String(data: data, encoding: .utf8) ?? "{}"
}

// ── OpenAI-compatible types ───────────────────────────────────────────────────

struct ChatCompletionRequest: Decodable {
    struct Message: Decodable {
        let role: String
        let content: String
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
    let repetitionPenalty: Double?
    let tools: [ToolDef]?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools
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

/// AnyCodable: decode arbitrary JSON for tool parameters pass-through
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
    // Convert back to [String: any Sendable] for ToolSpec usage
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
