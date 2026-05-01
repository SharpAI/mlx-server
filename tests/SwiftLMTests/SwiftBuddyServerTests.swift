// SwiftBuddyServerTests.swift — Tests for the SwiftBuddy embedded /v1/* endpoints
//
// The production SwiftLM Server.swift already serves /v1/chat/completions and is
// what OpenCode uses. This suite covers the NEW embedded server we added in PR #99
// inside ServerManager.swift — a separate Hummingbird instance running inside the
// SwiftBuddy app itself for direct API access when the app is running.
//
// Because the embedded server requires a running SwiftBuddy app + InferenceEngine,
// these tests focus on the JSON parsing and response-shape logic that can be
// exercised in isolation (same strategy as ChatRequestParsingTests).

import XCTest
import Foundation
@testable import SwiftLM
@testable import MLXInferenceCore

final class SwiftBuddyServerTests: XCTestCase {

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - /v1/models response shape
    // ═══════════════════════════════════════════════════════════════════
    // The embedded server's /v1/models route must return the OpenAI-compatible
    // list schema so that clients like OpenCode, Continue, and the OpenAI SDK
    // can discover the available model without special-casing.

    func testModelsResponse_MatchesOpenAISchema() throws {
        // Replicate the JSON body produced by the /v1/models handler.
        // Normally this returns `engine.currentModelId ?? "local"`.
        let modelId = "mlx-community/Qwen3.5-4B-MLX-4bit"

        // Build the response body the same way ServerManager does
        let body: [String: Any] = [
            "object": "list",
            "data": [[
                "id": modelId,
                "object": "model",
                "owned_by": "swiftbuddy"
            ]]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(decoded["object"] as? String, "list",
                       "/v1/models must have top-level 'object': 'list'")
        let modelList = try XCTUnwrap(decoded["data"] as? [[String: Any]])
        XCTAssertEqual(modelList.count, 1)
        XCTAssertEqual(modelList[0]["id"] as? String, modelId,
                       "Model entry must carry the loaded model ID")
        XCTAssertEqual(modelList[0]["object"] as? String, "model",
                       "Each model entry must have 'object': 'model'")
    }

    func testModelsResponse_FallsBackToLocalWhenNoModelLoaded() throws {
        // When no model is loaded, the handler returns "local" as the fallback.
        // Clients must still receive a valid list structure.
        let body: [String: Any] = [
            "object": "list",
            "data": [[
                "id": "local",
                "object": "model",
                "owned_by": "swiftbuddy"
            ]]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let modelList = try XCTUnwrap(decoded["data"] as? [[String: Any]])
        XCTAssertEqual(modelList[0]["id"] as? String, "local")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - /v1/chat/completions SSE wire format (embedded server)
    // ═══════════════════════════════════════════════════════════════════
    // Tests the SSE chunk format used by the SwiftBuddy embedded server.
    // The production Server.swift SSE format is already tested in ServerSSETests;
    // these guard the embedded server's specific encoding.

    /// Builds the SSE delta string the embedded server emits for each token.
    private func makeDeltaChunk(id: String, modelId: String, delta: String, finishReason: String? = nil) -> String {
        let finishReasonJSON = finishReason.map { "\"\($0)\"" } ?? "null"
        let escaped = delta
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
        return """
        data: {"id":"\(id)","object":"chat.completion.chunk","model":"\(modelId)","choices":[{"index":0,"delta":{"role":"assistant","content":"\(escaped)"},"finish_reason":\(finishReasonJSON)}]}\r\n\r\n
        """
    }

    func testSSEDeltaChunk_HasCorrectPrefix() {
        let chunk = makeDeltaChunk(id: "sb-1", modelId: "qwen3", delta: "Hello")
        XCTAssertTrue(chunk.hasPrefix("data: "),
                      "SSE chunk must start with 'data: '")
        XCTAssertTrue(chunk.hasSuffix("\r\n\r\n"),
                      "SSE chunk must end with CRLF CRLF")
    }

    func testSSEDeltaChunk_JSONShape() throws {
        let chunk = makeDeltaChunk(id: "sb-42", modelId: "test-model", delta: "Hi!")
        let jsonStr = String(chunk.dropFirst("data: ".count).dropLast("\r\n\r\n".count))
        let data = try XCTUnwrap(jsonStr.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(json["object"] as? String, "chat.completion.chunk",
                       "Streaming chunk must have object = chat.completion.chunk")
        XCTAssertEqual(json["id"] as? String, "sb-42")
        XCTAssertEqual(json["model"] as? String, "test-model")

        let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.count, 1)
        XCTAssertEqual(choices[0]["index"] as? Int, 0)

        let delta = try XCTUnwrap(choices[0]["delta"] as? [String: Any])
        XCTAssertEqual(delta["content"] as? String, "Hi!")
        XCTAssertEqual(delta["role"] as? String, "assistant")
    }

    func testSSEDeltaChunk_EscapesSpecialCharacters() throws {
        // Newlines and quotes inside delta content must be JSON-escaped.
        let chunk = makeDeltaChunk(id: "sb-1", modelId: "m", delta: "line1\nline2")
        XCTAssertFalse(chunk.contains("\nline2"),
                       "Raw newline inside delta content must be JSON-escaped to \\n")
        let jsonStr = String(chunk.dropFirst("data: ".count).dropLast("\r\n\r\n".count))
        let data = try XCTUnwrap(jsonStr.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
        let delta = try XCTUnwrap(choices[0]["delta"] as? [String: Any])
        XCTAssertEqual(delta["content"] as? String, "line1\nline2",
                       "JSON decoder must restore newline correctly after escaping")
    }

    func testSSEDoneTerminator_Format() {
        // The final SSE event must be `data: [DONE]` per OpenAI spec.
        let doneEvent = "data: [DONE]\r\n\r\n"
        XCTAssertTrue(doneEvent.hasPrefix("data: [DONE]"),
                      "[DONE] terminator must follow OpenAI SSE spec")
        XCTAssertTrue(doneEvent.hasSuffix("\r\n\r\n"))
    }

    func testSSEDeltaChunk_FinishReasonNull_DuringStreaming() throws {
        let chunk = makeDeltaChunk(id: "sb-1", modelId: "m", delta: "token", finishReason: nil)
        let jsonStr = String(chunk.dropFirst("data: ".count).dropLast("\r\n\r\n".count))
        let data = try XCTUnwrap(jsonStr.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
        // finish_reason must be JSON null during streaming (not the string "null")
        let finishReason = choices[0]["finish_reason"]
        XCTAssertTrue(finishReason is NSNull, "finish_reason must be JSON null during streaming")
    }

    func testSSEDeltaChunk_FinishReasonStop_AtEnd() throws {
        let chunk = makeDeltaChunk(id: "sb-1", modelId: "m", delta: "", finishReason: "stop")
        let jsonStr = String(chunk.dropFirst("data: ".count).dropLast("\r\n\r\n".count))
        let data = try XCTUnwrap(jsonStr.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop",
                       "finish_reason must be 'stop' on the final token chunk")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - CLI command builder
    // ═══════════════════════════════════════════════════════════════════

    func testCLIBuilder_DefaultsOmitNonDefaultFlags() {
        let cmd = buildCLICommand(
            config: .default,
            host: "127.0.0.1", port: 5413,
            parallel: 1, apiKeySet: false,
            modelId: "mlx-community/Qwen3"
        )
        XCTAssertTrue(cmd.contains("--model mlx-community/Qwen3"))
        XCTAssertTrue(cmd.contains("--host 127.0.0.1"))
        XCTAssertTrue(cmd.contains("--port 5413"))
        // Defaults should be omitted to keep the command readable
        XCTAssertFalse(cmd.contains("--top-p"),   "top-p=1.0 is default — should be omitted")
        XCTAssertFalse(cmd.contains("--top-k"),   "top-k=50 is default — should be omitted")
        XCTAssertFalse(cmd.contains("--min-p"),   "min-p=0 is default — should be omitted")
        XCTAssertFalse(cmd.contains("--thinking"),"thinking=false is default — should be omitted")
        XCTAssertFalse(cmd.contains("--parallel"),"parallel=1 is default — should be omitted")
        XCTAssertFalse(cmd.contains("--api-key"), "no key set — should be omitted")
        XCTAssertFalse(cmd.contains("--seed"),    "seed=nil is default — should be omitted")
        XCTAssertFalse(cmd.contains("--kv-bits"), "kvBits=nil is default — should be omitted")
    }

    func testCLIBuilder_NonDefaultsFlagsEmitted() {
        var cfg = GenerationConfig.default
        cfg.topP              = 0.9
        cfg.topK              = 40
        cfg.minP              = 0.05
        cfg.enableThinking    = true
        cfg.seed              = 42
        cfg.kvBits            = 4
        cfg.kvGroupSize       = 32
        cfg.prefillSize       = 256
        cfg.repetitionPenalty = 1.2

        let cmd = buildCLICommand(
            config: cfg,
            host: "0.0.0.0", port: 8080,
            parallel: 4, apiKeySet: true,
            modelId: "mlx-community/Qwen3-35B-MoE"
        )

        XCTAssertTrue(cmd.contains("--top-p 0.90"))
        XCTAssertTrue(cmd.contains("--top-k 40"))
        XCTAssertTrue(cmd.contains("--min-p 0.05"))
        XCTAssertTrue(cmd.contains("--thinking"))
        XCTAssertTrue(cmd.contains("--seed 42"))
        XCTAssertTrue(cmd.contains("--kv-bits 4"))
        XCTAssertTrue(cmd.contains("--kv-group-size 32"))
        XCTAssertTrue(cmd.contains("--prefill-size 256"))
        XCTAssertTrue(cmd.contains("--repeat-penalty 1.20"))
        XCTAssertTrue(cmd.contains("--parallel 4"))
        XCTAssertTrue(cmd.contains("--api-key <redacted>"))
    }

    func testCLIBuilder_NoModelId_UsesPlaceholder() {
        let cmd = buildCLICommand(
            config: .default,
            host: "127.0.0.1", port: 5413,
            parallel: 1, apiKeySet: false,
            modelId: nil
        )
        XCTAssertTrue(cmd.contains("--model <model-id>"),
                      "When no model is loaded, CLI must show a placeholder")
    }

    func testCLIBuilder_KvBitsDefault_DoesNotEmitGroupSize() {
        // If kvBits is nil, kv-group-size must also be suppressed
        // even if kvGroupSize is non-default — it has no effect without kvBits.
        var cfg = GenerationConfig.default
        cfg.kvBits = nil
        cfg.kvGroupSize = 32  // non-default but irrelevant without kvBits
        let cmd = buildCLICommand(config: cfg, host: "127.0.0.1", port: 5413,
                                  parallel: 1, apiKeySet: false, modelId: "m")
        XCTAssertFalse(cmd.contains("--kv-group-size"),
                       "kv-group-size must not appear when kvBits is nil")
    }

    func testCLIBuilder_OutputStartsWithSwiftRunSwiftLM() {
        let cmd = buildCLICommand(config: .default, host: "127.0.0.1", port: 5413,
                                  parallel: 1, apiKeySet: false, modelId: "m")
        XCTAssertTrue(cmd.hasPrefix("swift run SwiftLM"),
                      "CLI command must start with 'swift run SwiftLM'")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - jsonEscape completeness (C3 — Copilot review)
    // ═══════════════════════════════════════════════════════════════════
    // The old implementation only escaped \\ \" \n \r \t.
    // JSONEncoder correctly handles U+0000–U+001F and all other control chars.

    /// Replicates the fixed jsonEscape using JSONEncoder (same as ServerManager).
    private func jsonEscape(_ s: String) -> String {
        guard let data = try? JSONEncoder().encode(s),
              let raw = String(data: data, encoding: .utf8) else { return "\"\"" }
        return String(raw.dropFirst().dropLast())
    }

    func testJsonEscape_BasicChars() {
        XCTAssertEqual(jsonEscape("hello"), "hello")
        XCTAssertEqual(jsonEscape("say \"hi\""), #"say \"hi\""#)
        XCTAssertEqual(jsonEscape("a\\b"), #"a\\b"#)
        XCTAssertEqual(jsonEscape("line1\nline2"), #"line1\nline2"#)
        XCTAssertEqual(jsonEscape("col1\tcol2"), #"col1\tcol2"#)
    }

    func testJsonEscape_ControlCharsU0000toU001F() {
        // The old manual escape missed U+0000–U+001F beyond \n/\r/\t.
        // JSONEncoder emits \u0000, \u0001, … for these.
        let nullChar = "\u{00}"       // U+0000 NULL
        let escaped = jsonEscape(nullChar)
        XCTAssertFalse(escaped.contains("\u{00}"),
                       "NULL byte must be escaped — raw U+0000 breaks JSON parsers")
        // JSONEncoder emits \\u0000 for U+0000
        XCTAssertTrue(escaped.contains("\\u0000") || escaped.contains("\\u"),
                      "NULL must be encoded as a JSON unicode escape")

        let bell = "\u{07}"           // U+0007 BELL — not escaped by the old implementation
        let escapedBell = jsonEscape(bell)
        XCTAssertFalse(escapedBell.contains("\u{07}"),
                       "BELL (U+0007) must be escaped — old jsonEscape missed this")
    }

    func testJsonEscape_ProducesValidJSONWhenInterpolated() throws {
        // Simulate the SSE chunk build: if escape is correct the whole string
        // must parse as valid JSON.
        let dangerousToken = "say \"\u{01}hello\u{08}\" done\n"
        let escaped = jsonEscape(dangerousToken)
        let json = "{\"content\":\"\(escaped)\"}"
        let data = try XCTUnwrap(json.data(using: .utf8))
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(parsed["content"] as? String, dangerousToken,
                       "Round-trip through jsonEscape must preserve original string content")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - /v1/models modelId JSON safety (C5 — Copilot review)
    // ═══════════════════════════════════════════════════════════════════

    func testModelsResponse_ModelIdWithQuotes_IsJsonSafe() throws {
        // A model ID that contains quotes would break naive interpolation.
        // swiftBuddyJSONString wraps the value with JSONEncoder, making it safe.
        let dangerousId = "model\"with\"quotes"
        // Simulate the fixed /v1/models body build
        let encodedId = try XCTUnwrap(
            String(data: JSONEncoder().encode(dangerousId), encoding: .utf8)
        )
        let body = "{\"object\":\"list\",\"data\":[{\"id\":\(encodedId),\"object\":\"model\",\"owned_by\":\"swiftbuddy\"}]}"
        let data = try XCTUnwrap(body.data(using: .utf8))
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let modelList = try XCTUnwrap(parsed["data"] as? [[String: Any]])
        XCTAssertEqual(modelList[0]["id"] as? String, dangerousId,
                       "Model ID with embedded quotes must survive JSON round-trip safely")
    }

    func testModelsResponse_SlashInModelId_IsSafe() throws {
        // Standard HF model IDs contain slashes — they must not break JSON.
        let modelId = "mlx-community/Qwen3.5-122B-A10B-4bit"
        let encodedId = try XCTUnwrap(
            String(data: JSONEncoder().encode(modelId), encoding: .utf8)
        )
        let body = "{\"object\":\"list\",\"data\":[{\"id\":\(encodedId),\"object\":\"model\",\"owned_by\":\"swiftbuddy\"}]}"
        let data = try XCTUnwrap(body.data(using: .utf8))
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let modelList = try XCTUnwrap(parsed["data"] as? [[String: Any]])
        XCTAssertEqual(modelList[0]["id"] as? String, modelId,
                       "Standard HF model ID with slashes must parse correctly")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Seed UInt64 overflow guard (C1/C2 — Copilot review)
    // ═══════════════════════════════════════════════════════════════════

    func testSeed_RandomIsWithinIntMax() {
        // The seed button generates UInt64.random(in: 0...UInt64(Int.max)).
        // Verifies the range is safe for Int conversion in the Stepper binding.
        for _ in 0..<1000 {
            let seed = UInt64.random(in: 0...UInt64(Int.max))
            XCTAssertNoThrow(
                _ = Int(seed),   // would trap if seed > Int.max
                "Randomly generated seed must be safely convertible to Int"
            )
            XCTAssertLessThanOrEqual(seed, UInt64(Int.max),
                                     "Seed must not exceed Int.max — Stepper binding would overflow")
        }
    }

    func testSeed_StepperBinding_ClampsSafely() {
        // The Stepper get: binding uses min(seed, UInt64(Int.max)) to prevent overflow.
        let oversizedSeed = UInt64(Int.max) + 1
        let clamped = Int(min(oversizedSeed, UInt64(Int.max)))
        XCTAssertEqual(clamped, Int.max,
                       "Seeds larger than Int.max must be clamped, not crashed")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Role mapping: tool + developer (M1 — Copilot review)
    // ═══════════════════════════════════════════════════════════════════

    func testRoleMapping_ToolRoleMapsToChatMessageTool() {
        // Replicate the fixed role-switch from ServerManager's /v1/chat/completions handler.
        func mapRole(_ role: String, content: String) -> ChatMessage {
            switch role {
            case "system", "developer": return .system(content)
            case "assistant":           return .assistant(content)
            case "tool":                return .tool(content)
            case "user":                return .user(content)
            default:                    return .user(content)
            }
        }

        let toolMsg = mapRole("tool", content: "function result")
        XCTAssertEqual(toolMsg.role, .tool,
                       "tool role must map to .tool, not .user — breaks OpenAI function-calling protocol")
        XCTAssertNotEqual(toolMsg.role, .user,
                          "tool role must NOT fall through to .user")
    }

    func testRoleMapping_DeveloperRoleMapsToSystem() {
        func mapRole(_ role: String, content: String) -> ChatMessage {
            switch role {
            case "system", "developer": return .system(content)
            case "assistant":           return .assistant(content)
            case "tool":                return .tool(content)
            case "user":                return .user(content)
            default:                    return .user(content)
            }
        }

        let devMsg = mapRole("developer", content: "You are a coding assistant.")
        XCTAssertEqual(devMsg.role, .system,
                       "developer role (OpenAI Responses API) must map to .system")
    }

    func testRoleMapping_UnknownRoleFallsToUser() {
        func mapRole(_ role: String, content: String) -> ChatMessage {
            switch role {
            case "system", "developer": return .system(content)
            case "assistant":           return .assistant(content)
            case "tool":                return .tool(content)
            case "user":                return .user(content)
            default:                    return .user(content)
            }
        }

        let unknown = mapRole("function", content: "some output")
        XCTAssertEqual(unknown.role, .user,
                       "Unknown roles must fall back to .user (safe default)")
    }
}

