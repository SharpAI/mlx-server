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

    @Option(name: .long, help: "Max tokens to generate per request")
    var maxTokens: Int = 2048

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

        let maxTok = self.maxTokens

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
            let tokenLimit = chatReq.maxTokens ?? maxTok

            // Convert request messages → Chat.Message
            let chatMessages: [Chat.Message] = chatReq.messages.compactMap { msg in
                switch msg.role {
                case "system":    return .system(msg.content)
                case "assistant": return .assistant(msg.content)
                default:          return .user(msg.content)
                }
            }

            let userInput = UserInput(chat: chatMessages)
            let lmInput = try await container.prepare(input: userInput)
            let params = GenerateParameters(
                maxTokens: tokenLimit,
                temperature: Float(chatReq.temperature ?? 0.7)
            )
            let stream = try await container.generate(input: lmInput, parameters: params)

            if isStream {
                // SSE streaming
                let (sseStream, cont) = AsyncStream<String>.makeStream()
                Task {
                    for await generation in stream {
                        switch generation {
                        case .chunk(let text):
                            cont.yield(sseChunk(modelId: modelId, delta: text, finishReason: nil))
                        case .info:
                            cont.yield(sseChunk(modelId: modelId, delta: "", finishReason: "stop"))
                            cont.yield("data: [DONE]\n\n")
                            cont.finish()
                        case .toolCall:
                            break
                        }
                    }
                    cont.finish()
                }
                return Response(
                    status: .ok,
                    headers: sseHeaders(),
                    body: .init(asyncSequence: sseStream.map { ByteBuffer(string: $0) })
                )
            } else {
                // Non-streaming: collect all chunks
                var fullText = ""
                for await generation in stream {
                    switch generation {
                    case .chunk(let text):
                        fullText += text
                    case .info, .toolCall:
                        break
                    }
                }

                let resp = ChatCompletionResponse(
                    id: "chatcmpl-\(UUID().uuidString)",
                    model: modelId,
                    created: Int(Date().timeIntervalSince1970),
                    choices: [
                        Choice(
                            index: 0,
                            message: AssistantMessage(role: "assistant", content: fullText),
                            finishReason: "stop"
                        )
                    ],
                    usage: TokenUsage(promptTokens: 0, completionTokens: 0, totalTokens: 0)
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
        try await app.runService()
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

// ── OpenAI-compatible types ───────────────────────────────────────────────────

struct ChatCompletionRequest: Decodable {
    struct Message: Decodable {
        let role: String
        let content: String
    }
    let model: String?
    let messages: [Message]
    let stream: Bool?
    let maxTokens: Int?
    let temperature: Double?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream
        case maxTokens = "max_tokens"
        case temperature
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
    let content: String
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
