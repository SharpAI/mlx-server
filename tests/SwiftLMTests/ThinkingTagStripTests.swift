// ThinkingTagStripTests.swift — Regression tests for Issue #97
//
// Verifies two fixes:
//   1. stripThinkingTags() correctly removes <think>…</think> blocks from
//      assistant history messages so they never re-enter the Jinja template.
//   2. The role mapping for "assistant" is NOT changed to "model" (Qwen3 fix).
//
// stripThinkingTags is private at file scope in InferenceEngine.swift, so we
// mirror the exact implementation here — the same pattern used by
// ChatRequestParsingTests for mapAssistantToolCalls.

import XCTest
import Foundation
@testable import SwiftLM
import MLXInferenceCore

final class ThinkingTagStripTests: XCTestCase {

    // ── Mirror of the production helper (InferenceEngine.swift) ───────────────
    // Keep in sync if the production implementation changes.

    private func stripThinkingTags(from text: String) -> String {
        var result = text
        while let openRange = result.range(of: "<think>") {
            if let closeRange = result.range(of: "</think>", range: openRange.lowerBound..<result.endIndex) {
                var endIdx = closeRange.upperBound
                if endIdx < result.endIndex && result[endIdx] == "\n" {
                    endIdx = result.index(after: endIdx)
                }
                result.removeSubrange(openRange.lowerBound..<endIdx)
            } else {
                result.removeSubrange(openRange.lowerBound...)
                break
            }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

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

    func testStrip_NoThinkBlock_ReturnsTrimmedOriginal() {
        let input = "  Hello, how can I help?  "
        XCTAssertEqual(stripThinkingTags(from: input), "Hello, how can I help?")
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
        let input = """
        <think>
        Line one of reasoning.
        Line two of reasoning.
        </think>
        The final answer.
        """
        XCTAssertEqual(stripThinkingTags(from: input), "The final answer.")
    }

    func testStrip_ThinkBlockWithTrailingNewline_ConsumesNewline() {
        // The production helper eats the single newline after </think>
        // so the visible content doesn't start with a blank line.
        let input = "<think>thought</think>\nAnswer starts here"
        let result = stripThinkingTags(from: input)
        XCTAssertFalse(result.hasPrefix("\n"), "Result must not start with a stray newline")
        XCTAssertEqual(result, "Answer starts here")
    }

    func testStrip_ContentBeforeAndAfterThink() {
        // Reproduces the exact shape of Qwen3 output with thinking ON:
        // the UI shows the <think> block inline and the answer follows.
        let input = "<think>\nThe user is asking me to continue a Russian tongue-twister.\nNo tool calls needed.\n</think>\nЕхал грека через реку,\nВидит грека — в реке рак."
        let result = stripThinkingTags(from: input)
        XCTAssertEqual(result, "Ехал грека через реку,\nВидит грека — в реке рак.")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 3. Issue #97 crash reproducer
    // ═══════════════════════════════════════════════════════════════════

    func testStrip_Issue97_SecondTurnMessageShape() {
        // This is the exact assistant content that caused TemplateException error 1
        // when fed back unmodified into the Jinja template on turn 2.
        let turn1AssistantOutput = """
        <think>
        The user said "Hi!" as a greeting. Let me check my available tools and context. \
        No tool calls needed here — just a simple greeting.
        </think>
        Hello! 👋 It's great to meet you. How can I assist you today?
        """
        let stripped = stripThinkingTags(from: turn1AssistantOutput)

        // After stripping, no <think> tag should remain
        XCTAssertFalse(stripped.contains("<think>"), "Stripped content must not contain <think>")
        XCTAssertFalse(stripped.contains("</think>"), "Stripped content must not contain </think>")

        // The visible reply must be preserved
        XCTAssertTrue(stripped.contains("Hello!"), "Visible reply must survive stripping")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 4. Role mapping regression guard (Issue #97)
    // ═══════════════════════════════════════════════════════════════════
    // The ChatCompletionRequest pipeline in Server.swift passes roles through
    // as-is. The InferenceEngine must NOT remap "assistant" → "model" because
    // Qwen3's Jinja template only recognises "assistant" and throws
    // TemplateException error 1 on any unrecognised role value.

    func testRoleMapping_AssistantRawValue_IsAssistant() {
        // ChatMessage.Role.assistant.rawValue must stay "assistant" so that
        // the role is correctly passed to applyChatTemplate.
        // If someone changes the enum rawValue, this test fails loudly.
        XCTAssertEqual(
            ChatMessage.Role.assistant.rawValue,
            "assistant",
            "Role.assistant rawValue must be 'assistant', not 'model' — Qwen3 Jinja template fix (Issue #97)"
        )
    }

    func testRoleMapping_AllRolesHaveExpectedRawValues() {
        // Canonical role strings for the OpenAI-compatible message protocol.
        XCTAssertEqual(ChatMessage.Role.system.rawValue,    "system")
        XCTAssertEqual(ChatMessage.Role.user.rawValue,      "user")
        XCTAssertEqual(ChatMessage.Role.assistant.rawValue, "assistant")
        XCTAssertEqual(ChatMessage.Role.tool.rawValue,      "tool")
    }
}
