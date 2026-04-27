import XCTest
import Foundation
@testable import SwiftLM

final class ChatRequestParsingTests: XCTestCase {

    // MARK: - Helper: decode a ChatCompletionRequest from a JSON string

    private func decode(_ json: String) throws -> ChatCompletionRequest {
        let data = try XCTUnwrap(json.data(using: .utf8))
        return try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
    }

    // MARK: - Helper: replicate the exact mapping logic from handleChatCompletion
    // This mirrors the production code so the test locks down current behavior.

    private func mapAssistantToolCalls(_ msg: ChatCompletionRequest.Message) -> [[String: any Sendable]]? {
        guard let tc = msg.tool_calls, !tc.isEmpty else { return nil }
        return tc.enumerated().map { (index, call) in
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

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 1. Tool calls with index field (PR #92)
    // ═══════════════════════════════════════════════════════════════════

    func testToolCallsMappingIncludesIndex() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "I'll search for that.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\\"city\\": \\"Tokyo\\"}"
                            }
                        }
                    ]
                }
            ]
        }
        """

        let req = try decode(json)
        let msg = req.messages[0]
        let mapped = try XCTUnwrap(mapAssistantToolCalls(msg))

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0]["index"] as? Int, 0, "First tool call must have index 0")
        XCTAssertEqual(mapped[0]["id"] as? String, "call_abc123")
        XCTAssertEqual(mapped[0]["type"] as? String, "function")

        let fn = try XCTUnwrap(mapped[0]["function"] as? [String: any Sendable])
        XCTAssertEqual(fn["name"] as? String, "get_weather")
        XCTAssertEqual(fn["arguments"] as? String, "{\"city\": \"Tokyo\"}")
    }

    func testMultipleToolCallsHaveCorrectIndices() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": { "name": "search", "arguments": "{\\"q\\": \\"a\\"}" }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": { "name": "lookup", "arguments": "{\\"id\\": 42}" }
                        },
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": { "name": "save", "arguments": "{}" }
                        }
                    ]
                }
            ]
        }
        """

        let req = try decode(json)
        let mapped = try XCTUnwrap(mapAssistantToolCalls(req.messages[0]))

        XCTAssertEqual(mapped.count, 3)
        for i in 0..<3 {
            XCTAssertEqual(mapped[i]["index"] as? Int, i, "Tool call at position \(i) must have index \(i)")
        }
        XCTAssertEqual(mapped[0]["id"] as? String, "call_1")
        XCTAssertEqual(mapped[1]["id"] as? String, "call_2")
        XCTAssertEqual(mapped[2]["id"] as? String, "call_3")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 2. Assistant message without tool_calls
    // ═══════════════════════════════════════════════════════════════════

    func testAssistantWithoutToolCalls() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hello, how can I help you?"
                }
            ]
        }
        """

        let req = try decode(json)
        let mapped = mapAssistantToolCalls(req.messages[0])
        XCTAssertNil(mapped, "Assistant message without tool_calls should map to nil")
    }

    func testAssistantWithEmptyToolCalls() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Done.",
                    "tool_calls": []
                }
            ]
        }
        """

        let req = try decode(json)
        let mapped = mapAssistantToolCalls(req.messages[0])
        XCTAssertNil(mapped, "Assistant message with empty tool_calls array should map to nil")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 3. Tool role message (tool_call_id)
    // ═══════════════════════════════════════════════════════════════════

    func testToolRoleMessage() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "tool",
                    "content": "{\\"temp\\": 22}",
                    "tool_call_id": "call_abc123"
                }
            ]
        }
        """

        let req = try decode(json)
        let msg = req.messages[0]
        XCTAssertEqual(msg.role, "tool")
        XCTAssertEqual(msg.tool_call_id, "call_abc123")
        XCTAssertEqual(msg.textContent, "{\"temp\": 22}")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 4. Multi-turn conversation with tool round-trip
    // ═══════════════════════════════════════════════════════════════════

    func testFullToolRoundTrip() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": "What's the weather in Tokyo?" },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_w1",
                            "type": "function",
                            "function": { "name": "get_weather", "arguments": "{\\"city\\":\\"Tokyo\\"}" }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "content": "{\\"temp\\":22,\\"condition\\":\\"sunny\\"}",
                    "tool_call_id": "call_w1"
                },
                { "role": "assistant", "content": "It's 22°C and sunny in Tokyo." }
            ]
        }
        """

        let req = try decode(json)
        XCTAssertEqual(req.messages.count, 5)

        // Message 0: system
        XCTAssertEqual(req.messages[0].role, "system")

        // Message 1: user
        XCTAssertEqual(req.messages[1].role, "user")

        // Message 2: assistant with tool_calls
        let assistantToolMsg = req.messages[2]
        XCTAssertEqual(assistantToolMsg.role, "assistant")
        let mapped = try XCTUnwrap(mapAssistantToolCalls(assistantToolMsg))
        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0]["index"] as? Int, 0)
        XCTAssertEqual(mapped[0]["id"] as? String, "call_w1")

        // Message 3: tool response
        XCTAssertEqual(req.messages[3].role, "tool")
        XCTAssertEqual(req.messages[3].tool_call_id, "call_w1")

        // Message 4: final assistant
        XCTAssertEqual(req.messages[4].role, "assistant")
        XCTAssertNil(req.messages[4].tool_calls)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 5. MessageContent decoding (string vs multipart)
    // ═══════════════════════════════════════════════════════════════════

    func testTextContentFromPlainString() throws {
        let json = """
        {
            "model": "m",
            "messages": [{ "role": "user", "content": "Hello world" }]
        }
        """
        let req = try decode(json)
        XCTAssertEqual(req.messages[0].textContent, "Hello world")
    }

    func testTextContentFromMultipartParts() throws {
        let json = """
        {
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "Describe this image:" },
                    { "type": "image_url", "image_url": { "url": "https://example.com/cat.jpg" } }
                ]
            }]
        }
        """
        let req = try decode(json)
        XCTAssertEqual(req.messages[0].textContent, "Describe this image:")
    }

    func testNullContentDecodesToEmptyString() throws {
        let json = """
        {
            "model": "m",
            "messages": [{ "role": "assistant", "content": null }]
        }
        """
        let req = try decode(json)
        XCTAssertEqual(req.messages[0].textContent, "")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 6. Tools definition parsing
    // ═══════════════════════════════════════════════════════════════════

    func testToolsDefinitionParsing() throws {
        let json = """
        {
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": { "type": "string" }
                            },
                            "required": ["city"]
                        }
                    }
                }
            ]
        }
        """
        let req = try decode(json)
        XCTAssertNotNil(req.tools)
        XCTAssertEqual(req.tools?.count, 1)
        XCTAssertEqual(req.tools?[0].function.name, "get_weather")
        XCTAssertEqual(req.tools?[0].function.description, "Get current weather for a city")
    }
}
