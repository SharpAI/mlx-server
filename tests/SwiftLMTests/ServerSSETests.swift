import XCTest
@testable import SwiftLM

final class ServerSSETests: XCTestCase {
    func testParseTruthyHeaderValue() {
        XCTAssertTrue(parseTruthyHeaderValue("true"))
        XCTAssertTrue(parseTruthyHeaderValue("TRUE"))
        XCTAssertTrue(parseTruthyHeaderValue(" yes "))
        XCTAssertTrue(parseTruthyHeaderValue("1"))
        XCTAssertFalse(parseTruthyHeaderValue(nil))
        XCTAssertFalse(parseTruthyHeaderValue("false"))
        XCTAssertFalse(parseTruthyHeaderValue("0"))
    }

    func testPrefillChunkUsesNamedEventAndLeanPayload() throws {
        let chunk = ssePrefillChunk(nPast: 32, promptTokens: 128, elapsedSeconds: 4)

        XCTAssertTrue(chunk.hasPrefix("event: prefill_progress\r\ndata: "))
        XCTAssertTrue(chunk.hasSuffix("\r\n\r\n"))

        let prefix = "event: prefill_progress\r\ndata: "
        let payload = String(chunk.dropFirst(prefix.count).dropLast(4))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(json["status"] as? String, "processing")
        XCTAssertEqual(json["n_past"] as? Int, 32)
        XCTAssertEqual(json["n_prompt_tokens"] as? Int, 128)
        XCTAssertEqual(json["elapsed_seconds"] as? Int, 4)
        XCTAssertNil(json["object"])
        XCTAssertNil(json["choices"])
    }
}
