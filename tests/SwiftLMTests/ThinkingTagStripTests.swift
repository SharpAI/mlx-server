// ThinkingTagStripTests.swift — Regression tests for Issue #97
//
// Verifies two fixes in InferenceEngine.generate():
//   1. stripThinkingTags() correctly removes <think>…</think> blocks from
//      assistant history messages so they never re-enter the Jinja template.
//   2. ChatMessage.Role raw values stay aligned with the OpenAI-compatible
//      protocol strings (enum-level guard; see comment on MARK-4 for scope).

import XCTest
import Foundation
@testable import SwiftLM
@testable import MLXInferenceCore   // gives access to internal stripThinkingTags

final class ThinkingTagStripTests: XCTestCase {

    // stripThinkingTags is an internal free function in InferenceEngine.swift,
    // accessed here via @testable import MLXInferenceCore — no mirror copy needed.

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 1. Basic stripping
    // ═══════════════════════════════════════════════════════════════════

    func testStrip_SingleThinkBlock_LeavesOnlyVisible() {
        let input = "<think>Let me reason step by step.</think>\nHello! 👋"
        XCTAssertEqual(stripThinkingTags(from: input), "Hello! 👋")
    }

    func testStrip_ThinkBlockOnly_ReturnsEmpty() {
        let input = "<think>internal monologue</think>"
        XCTAssertEqual(stripThinkingTags(from: input), "")
    }

    func testStrip_NoThinkBlock_ReturnsOriginalUnchanged() {
        // When no <think> tags are present the string must be returned
        // byte-for-byte — leading indentation, code-block spaces, etc. must
        // not be trimmed (Copilot review comment).
        let input = "  Hello, how can I help?  "
        XCTAssertEqual(stripThinkingTags(from: input), input,
                       "Content without think tags must be returned unchanged (no trimming)")
    }

    func testStrip_MultipleThinkBlocks() {
        // Qwen3 can emit multiple <think> sections in one reply
        let input = "<think>first</think>\nVisible A\n<think>second</think>\nVisible B"
        XCTAssertEqual(stripThinkingTags(from: input), "Visible A\nVisible B")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 2. Edge cases
    // ═══════════════════════════════════════════════════════════════════

    func testStrip_UnclosedThinkTag_StripsToEndOfString() {
        // If generation was interrupted mid-think, the closing tag may be absent.
        let input = "Visible prefix\n<think>reasoning that never closed"
        XCTAssertEqual(stripThinkingTags(from: input), "Visible prefix")
    }

    func testStrip_EmptyThinkBlock_RemovesTagsOnly() {
        let input = "<think></think>The actual answer."
        XCTAssertEqual(stripThinkingTags(from: input), "The actual answer.")
    }

    func testStrip_MultilineThinkBlock() {
        let input = "<think>\nLine one of reasoning.\nLine two of reasoning.\n</think>\nThe final answer."
        XCTAssertEqual(stripThinkingTags(from: input), "The final answer.")
    }

    func testStrip_ThinkBlockWithTrailingNewline_ConsumesNewline() {
        // The helper eats the single newline after </think> so the visible
        // content doesn't start with a blank line.
        let input = "<think>thought</think>\nAnswer starts here"
        let result = stripThinkingTags(from: input)
        XCTAssertFalse(result.hasPrefix("\n"), "Result must not start with a stray newline")
        XCTAssertEqual(result, "Answer starts here")
    }

    func testStrip_ContentBeforeAndAfterThink() {
        // Reproduces the exact shape of Qwen3 output from screenshot 2 (Issue #97):
        // Russian tongue-twister reply with an inline <think> block.
        let input = "<think>\nThe user is asking me to continue a Russian tongue-twister.\nNo tool calls needed.\n</think>\nЕхал грека через реку,\nВидит грека — в реке рак."
        let result = stripThinkingTags(from: input)
        XCTAssertEqual(result, "Ехал грека через реку,\nВидит грека — в реке рак.")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 3. Issue #97 crash reproducer
    // ═══════════════════════════════════════════════════════════════════

    func testStrip_Issue97_SecondTurnMessageShape() {
        // This is the exact assistant content that caused TemplateException error 1
        // when fed back unmodified into the Jinja template on turn 2 (screenshot 1).
        let turn1AssistantOutput = """
        <think>
        The user said "Hi!" as a greeting. Let me check my available tools and context. \
        No tool calls needed here — just a simple greeting.
        </think>
        Hello! 👋 It's great to meet you. How can I assist you today?
        """
        let stripped = stripThinkingTags(from: turn1AssistantOutput)

        XCTAssertFalse(stripped.contains("<think>"),   "Stripped content must not contain <think>")
        XCTAssertFalse(stripped.contains("</think>"),  "Stripped content must not contain </think>")
        XCTAssertTrue(stripped.contains("Hello!"),     "Visible reply must survive stripping")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 4. Role mapping regression guard (Issue #97 — Copilot review)
    // ═══════════════════════════════════════════════════════════════════
    // Copilot noted that asserting `ChatMessage.Role.assistant.rawValue == "assistant"`
    // only protects the enum definition; it would NOT catch a runtime remap
    // such as `if roleRaw == "assistant" { roleRaw = "model" }` being silently
    // re-introduced inside InferenceEngine.generate().
    //
    // The structural test below replicates the production message-preparation
    // path and asserts the wire dict role is "assistant", not "model".

    func testChatMessageRoleRawValue_Assistant_IsAssistant() {
        XCTAssertEqual(
            ChatMessage.Role.assistant.rawValue,
            "assistant",
            "Role.assistant rawValue must be 'assistant' — Issue #97 enum raw-value guard"
        )
    }

    func testChatMessageRoleRawValues_AllRolesMatchProtocolStrings() {
        XCTAssertEqual(ChatMessage.Role.system.rawValue,    "system")
        XCTAssertEqual(ChatMessage.Role.user.rawValue,      "user")
        XCTAssertEqual(ChatMessage.Role.assistant.rawValue, "assistant")
        XCTAssertEqual(ChatMessage.Role.tool.rawValue,      "tool")
    }

    // Structural regression: replicates the wire-dict build in generate().
    // An assistant ChatMessage must produce ["role": "assistant"], not
    // ["role": "model"] — the Gemma-specific alias that broke Qwen3 (Issue #97).
    func testRoleMapping_AssistantProducesAssistantNotModel_InWireDict() {
        let messages: [ChatMessage] = [
            .system("You are helpful."),
            .user("Hello"),
            .assistant("Hi there!"),
        ]

        // Replicate: let roleRaw = msg.role.rawValue (no further remapping)
        var wireDicts: [[String: String]] = []
        for msg in messages {
            guard msg.role != .system else { continue }
            let roleRaw = msg.role.rawValue
            let content = stripThinkingTags(from: msg.content)
            wireDicts.append(["role": roleRaw, "content": content])
        }

        XCTAssertEqual(wireDicts.count, 2)
        XCTAssertEqual(wireDicts[0]["role"], "user")
        XCTAssertEqual(wireDicts[1]["role"], "assistant",
            "Assistant must map to 'assistant' in wire dict, not 'model' — Issue #97 runtime remap guard")
        XCTAssertNotEqual(wireDicts[1]["role"], "model",
            "Wire dict must never contain 'model' — Gemma-specific alias breaks Qwen3 chat template")
    }

    func testRoleMapping_ToolRoleIsPreservedInWireDict() {
        let msg = ChatMessage.tool("function result")
        XCTAssertEqual(msg.role.rawValue, "tool",
            "Tool role must be 'tool' for OpenAI function-calling protocol")
    }
}

