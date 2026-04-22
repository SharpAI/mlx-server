import XCTest
import Foundation
@testable import SwiftLM

final class ServerSSETests: XCTestCase {

    // MARK: - Truthy header parser

    func testParseTruthyHeaderValue() {
        XCTAssertTrue(parseTruthyHeaderValue("true"))
        XCTAssertTrue(parseTruthyHeaderValue("TRUE"))
        XCTAssertTrue(parseTruthyHeaderValue(" yes "))
        XCTAssertTrue(parseTruthyHeaderValue("1"))
        XCTAssertFalse(parseTruthyHeaderValue(nil))
        XCTAssertFalse(parseTruthyHeaderValue("false"))
        XCTAssertFalse(parseTruthyHeaderValue("0"))
    }

    // MARK: - 1a: "on" is a documented truthy alias (HTML-form / reverse-proxy parity)

    func testParseTruthyHeaderValue_OnAlias() {
        // "on" is intentionally accepted for parity with common reverse-proxy conventions.
        // See ssePrefillChunk doc comment for the rationale.
        XCTAssertTrue(parseTruthyHeaderValue("on"))
        XCTAssertTrue(parseTruthyHeaderValue("ON"))
    }

    // MARK: - Named event + lean payload (existing test, Fix #4 applied)

    func testPrefillChunkUsesNamedEventAndLeanPayload() throws {
        let chunk = ssePrefillChunk(nPast: 32, promptTokens: 128, elapsedSeconds: 4)

        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        XCTAssertTrue(chunk.hasPrefix(prefix))
        XCTAssertTrue(chunk.hasSuffix(suffix))

        // Fix #4: use suffix.count not the literal 4, so multi-byte chars at boundary
        // don't silently corrupt the JSON slice.
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(json["status"] as? String, "processing")
        XCTAssertEqual(json["n_past"] as? Int, 32)
        XCTAssertEqual(json["n_prompt_tokens"] as? Int, 128)
        XCTAssertEqual(json["elapsed_seconds"] as? Int, 4)
        XCTAssertNil(json["object"])
        XCTAssertNil(json["choices"])
    }

    // MARK: - 1b: Zero-token boundary (no divide-by-zero crash)

    func testPrefillChunk_ZeroTokenBoundary() throws {
        let chunk = ssePrefillChunk(nPast: 0, promptTokens: 0, elapsedSeconds: 0)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        let fraction = try XCTUnwrap(json["fraction"] as? Double)
        XCTAssertEqual(fraction, 0.0, accuracy: 1e-9, "Division by zero must yield 0.0")
        XCTAssertFalse(fraction.isNaN, "fraction must not be NaN")
        XCTAssertFalse(fraction.isInfinite, "fraction must not be infinite")
    }

    // MARK: - 1c: dropLast correctness regression guard

    func testPrefillChunk_DropLastSafe() throws {
        // Confirms the suffix-count trim extracts parseable JSON for any content length.
        let chunk = ssePrefillChunk(nPast: 100, promptTokens: 400, elapsedSeconds: 6)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        XCTAssertTrue(chunk.hasSuffix(suffix), "SSE terminator must be \\r\\n\\r\\n")
        let trimmed = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(trimmed.data(using: .utf8))
        // Must parse — would crash if dropLast sliced inside a multi-byte char
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: data))
    }

    // MARK: - 1d: No OpenAI-specific fields bleed into prefill payload

    func testPrefillChunk_NoOpenAIFields() throws {
        let chunk = ssePrefillChunk(nPast: 1, promptTokens: 4, elapsedSeconds: 1)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        // Fields that would confuse strict OpenAI-SDK clients (e.g. OpenCode) must be absent
        XCTAssertNil(json["id"],      "prefill chunk must not carry an id field")
        XCTAssertNil(json["object"],  "prefill chunk must not carry an object field")
        XCTAssertNil(json["model"],   "prefill chunk must not carry a model field")
        XCTAssertNil(json["choices"], "prefill chunk must not carry a choices field")
    }

    // MARK: - 1e: PrefillState.finish() is idempotent (Issue #2 guard)

    func testPrefillState_FinishIsIdempotent() async {
        let state = PrefillState()
        await state.finish()
        await state.finish()  // second call must not throw or reset done
        let done = await state.done
        XCTAssertTrue(done, "PrefillState.done must remain true after double finish()")
    }

    // MARK: - 1f: PrefillState contract: update after finish (Issue #2 guard)

    func testPrefillState_UpdateAfterFinishContract() async {
        let state = PrefillState()
        await state.update(nPast: 50)
        await state.finish()
        await state.update(nPast: 999)  // post-done update
        let done = await state.done
        // Invariant: done must stay true — the heartbeat loop guards on this
        XCTAssertTrue(done, "PrefillState.done must remain true after post-finish update")
        // The heartbeat loop reads nPast only when !done, so its value after finish
        // is irrelevant to correctness. We capture the current contract here.
        // If a post-done guard is added later, add XCTAssertNotEqual(await state.nPast, 999).
    }
}
