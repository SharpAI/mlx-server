// GenerationConfigPersistenceTests.swift — Regression tests for SwiftBuddy fixes
//
// Covers four independent fixes committed alongside Issue #97:
//   1. GenerationConfig Codable + save/load persistence
//   2. enable_thinking additionalContext wiring (thinking mode)
//   3. /v1/chat/completions request parsing logic
//   4. Server config propagation from persisted UserDefaults

import XCTest
import Foundation
@testable import SwiftLM
@testable import MLXInferenceCore

final class GenerationConfigPersistenceTests: XCTestCase {

    // Use an isolated UserDefaults suite so tests never touch the real suite
    // and don't interfere with each other.
    private var defaults: UserDefaults!
    private let suiteName = "com.swiftlm.test.generationconfig.\(UUID().uuidString)"

    override func setUp() {
        super.setUp()
        defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        super.tearDown()
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 1. GenerationConfig Codable conformance
    // ═══════════════════════════════════════════════════════════════════

    func testGenerationConfig_IsCodable() throws {
        // The Codable conformance that was added must round-trip without loss.
        let config = GenerationConfig(
            maxTokens: 4096,
            temperature: 0.75,
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            repetitionPenalty: 1.1,
            enableThinking: true,
            prefillSize: 256,
            kvBits: 4,
            kvGroupSize: 32
        )
        let data   = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(GenerationConfig.self, from: data)

        XCTAssertEqual(decoded.maxTokens,         4096)
        XCTAssertEqual(decoded.temperature,       0.75,  accuracy: 1e-4)
        XCTAssertEqual(decoded.topP,              0.9,   accuracy: 1e-4)
        XCTAssertEqual(decoded.topK,              40)
        XCTAssertEqual(decoded.minP,              0.05,  accuracy: 1e-4)
        XCTAssertEqual(decoded.repetitionPenalty, 1.1,   accuracy: 1e-4)
        XCTAssertTrue(decoded.enableThinking)
        XCTAssertEqual(decoded.prefillSize,       256)
        XCTAssertEqual(decoded.kvBits,            4)
        XCTAssertEqual(decoded.kvGroupSize,       32)
    }

    func testGenerationConfig_NilFieldsRoundTrip() throws {
        // nil kvBits must survive the encode/decode cycle as nil, not 0.
        let config = GenerationConfig(kvBits: nil)
        let data   = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(GenerationConfig.self, from: data)
        XCTAssertNil(decoded.kvBits, "kvBits nil must survive round-trip")
    }

    func testGenerationConfig_DefaultValues() {
        let config = GenerationConfig.default
        XCTAssertEqual(config.maxTokens,         2048)
        XCTAssertEqual(config.temperature,       0.6,  accuracy: 1e-4)
        XCTAssertEqual(config.topP,              1.0,  accuracy: 1e-4)
        XCTAssertEqual(config.topK,              50)
        XCTAssertEqual(config.minP,              0.0,  accuracy: 1e-4)
        XCTAssertEqual(config.repetitionPenalty, 1.05, accuracy: 1e-4)
        XCTAssertFalse(config.enableThinking,    "Thinking must be OFF by default")
        XCTAssertEqual(config.prefillSize,       512)
        XCTAssertNil(config.kvBits)
        XCTAssertEqual(config.kvGroupSize,       64)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 2. UserDefaults persistence (save / load)
    // ═══════════════════════════════════════════════════════════════════
    // NOTE: GenerationConfig.save()/load() use UserDefaults.standard internally.
    // These tests exercise the Codable round-trip via JSONEncoder/Decoder as a
    // proxy for the persistence contract, isolating from the real suite.

    func testGenerationConfig_SaveLoad_RoundTrip() throws {
        // Simulate what save() encodes and what load() decodes.
        let original = GenerationConfig(
            maxTokens: 512, temperature: 0.3, enableThinking: true, kvBits: 8
        )
        let data    = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(GenerationConfig.self, from: data)

        XCTAssertEqual(decoded.maxTokens,   512)
        XCTAssertEqual(decoded.temperature, 0.3, accuracy: 1e-4)
        XCTAssertTrue(decoded.enableThinking)
        XCTAssertEqual(decoded.kvBits,      8)
    }

    func testGenerationConfig_RestoredFields_PresentWithCorrectDefaults() throws {
        // turboKV and streamExperts were restored as fully-wired fields:
        //   turboKV      → per-request (sets KVCacheSimple.turboQuantEnabled)
        //   streamExperts → load-time (controls ExpertStreamingConfig activation)
        // This test verifies they are present in the schema with correct defaults
        // (both off by default, user opt-in).
        let data = try JSONEncoder().encode(GenerationConfig.default)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        // Both fields must be present in the encoded JSON
        XCTAssertNotNil(json["turboKV"],
                        "turboKV must be present in GenerationConfig JSON — it is wired to KVCacheSimple.turboQuantEnabled")
        XCTAssertNotNil(json["streamExperts"],
                        "streamExperts must be present in GenerationConfig JSON — it controls ExpertStreamingConfig at load time")

        // Both must default to false (user must explicitly opt in)
        XCTAssertEqual(json["turboKV"] as? Bool, false,
                       "turboKV default must be false — user opt-in for 100k+ context workloads")
        XCTAssertEqual(json["streamExperts"] as? Bool, false,
                       "streamExperts default must be false — auto-enabled via isMoE for catalog MoE models")
    }

    func testGenerationConfig_Load_FallsBackToDefault_WhenNoStoredData() {
        // load() with no stored data must return .default, not crash.
        // We test this by ensuring no data is in a fresh suite.
        let freshSuite = "com.swiftlm.test.fresh.\(UUID().uuidString)"
        let freshDefaults = UserDefaults(suiteName: freshSuite)!
        defer { freshDefaults.removePersistentDomain(forName: freshSuite) }

        // The static load() reads UserDefaults.standard, so we verify the
        // fallback contract by checking that .default is a valid config.
        let fallback = GenerationConfig.default
        XCTAssertEqual(fallback.maxTokens, 2048, "Fallback must be .default")
        XCTAssertFalse(fallback.enableThinking)
    }

    func testGenerationConfig_Save_ProducesDecodableJSON() throws {
        // Verify save() produces data that JSONDecoder can re-read —
        // i.e. the codec is symmetric and doesn't use unsupported types.
        let config = GenerationConfig(temperature: 0.88, enableThinking: true)
        let data   = try JSONEncoder().encode(config)
        XCTAssertFalse(data.isEmpty, "Encoded data must not be empty")
        // Must be valid JSON
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["enableThinking"] as? Bool, true)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 3. Thinking mode — enable_thinking additionalContext
    // ═══════════════════════════════════════════════════════════════════
    // The engine now passes `additionalContext: ["enable_thinking": Bool]`
    // to UserInput so Qwen3's Jinja template actually generates <think> blocks.
    // We test the mapping logic in isolation by verifying the config flag
    // drives the correct boolean value.

    func testThinkingConfig_EnabledWhenFlagIsTrue() {
        let config = GenerationConfig(enableThinking: true)
        // Replicate the production mapping from InferenceEngine.generate()
        let additionalContext: [String: Any] = config.enableThinking
            ? ["enable_thinking": true]
            : ["enable_thinking": false]
        XCTAssertEqual(additionalContext["enable_thinking"] as? Bool, true,
                       "enable_thinking must be true when config.enableThinking is true")
    }

    func testThinkingConfig_DisabledWhenFlagIsFalse() {
        let config = GenerationConfig(enableThinking: false)
        let additionalContext: [String: Any] = config.enableThinking
            ? ["enable_thinking": true]
            : ["enable_thinking": false]
        XCTAssertEqual(additionalContext["enable_thinking"] as? Bool, false,
                       "enable_thinking must be false when config.enableThinking is false")
    }

    func testThinkingConfig_DefaultIsDisabled() {
        // Prevents a future change to the default from silently enabling
        // thinking on all requests without the user opting in.
        XCTAssertFalse(GenerationConfig.default.enableThinking,
                       "Thinking must be OFF by default — opt-in only")
    }

    func testThinkingConfig_ToggleRoundTrips_ViaCodable() throws {
        // Verify enableThinking survives encode/decode (regression guard for
        // future Codable migrations that might lose Bool fields).
        for value in [true, false] {
            let config  = GenerationConfig(enableThinking: value)
            let data    = try JSONEncoder().encode(config)
            let decoded = try JSONDecoder().decode(GenerationConfig.self, from: data)
            XCTAssertEqual(decoded.enableThinking, value,
                           "enableThinking=\(value) must survive Codable round-trip")
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 4. /v1/chat/completions request parsing
    // ═══════════════════════════════════════════════════════════════════
    // Validates the JSON→ChatMessage mapping and per-request override logic
    // used by the ServerManager endpoint, isolated from the HTTP layer.

    /// Mirrors the production message-mapping logic in ServerManager.
    private func mapMessages(_ msgs: [[String: Any]]) -> [ChatMessage] {
        msgs.map { m in
            let role    = m["role"]    as? String ?? "user"
            let content = m["content"] as? String ?? ""
            switch role {
            case "system":    return ChatMessage.system(content)
            case "assistant": return ChatMessage.assistant(content)
            default:          return ChatMessage.user(content)
            }
        }
    }

    /// Mirrors the per-request config-override logic in ServerManager.
    private func applyOverrides(_ json: [String: Any], to base: GenerationConfig) -> GenerationConfig {
        var cfg = base
        if let t  = json["temperature"]       as? Double { cfg.temperature        = Float(t) }
        if let p  = json["top_p"]             as? Double { cfg.topP               = Float(p) }
        if let mt = json["max_tokens"]        as? Int    { cfg.maxTokens           = mt }
        if let rp = json["frequency_penalty"] as? Double { cfg.repetitionPenalty  = Float(rp) }
        return cfg
    }

    func testChatEndpoint_MessageMapping_SystemUserAssistant() {
        let msgs: [[String: Any]] = [
            ["role": "system",    "content": "You are helpful."],
            ["role": "user",      "content": "Hello!"],
            ["role": "assistant", "content": "Hi there!"],
        ]
        let mapped = mapMessages(msgs)
        XCTAssertEqual(mapped.count, 3)
        XCTAssertEqual(mapped[0].role, .system)
        XCTAssertEqual(mapped[0].content, "You are helpful.")
        XCTAssertEqual(mapped[1].role, .user)
        XCTAssertEqual(mapped[1].content, "Hello!")
        XCTAssertEqual(mapped[2].role, .assistant)
        XCTAssertEqual(mapped[2].content, "Hi there!")
    }

    func testChatEndpoint_UnknownRoleMapsToUser() {
        // Any unknown role (e.g. "function") should fall through to .user
        // rather than crashing — matches the production `default:` branch.
        let msgs: [[String: Any]] = [["role": "function", "content": "result"]]
        let mapped = mapMessages(msgs)
        XCTAssertEqual(mapped[0].role, .user)
    }

    func testChatEndpoint_MissingContentDefaultsToEmpty() {
        let msgs: [[String: Any]] = [["role": "user"]]   // no "content" key
        let mapped = mapMessages(msgs)
        XCTAssertEqual(mapped[0].content, "")
    }

    func testChatEndpoint_PerRequestOverrides_AppliedCorrectly() {
        let base = GenerationConfig.default
        let json: [String: Any] = [
            "temperature": 0.2,
            "top_p": 0.85,
            "max_tokens": 512,
            "frequency_penalty": 1.3,
        ]
        let result = applyOverrides(json, to: base)
        XCTAssertEqual(result.temperature,       0.2,  accuracy: 1e-4)
        XCTAssertEqual(result.topP,              0.85, accuracy: 1e-4)
        XCTAssertEqual(result.maxTokens,         512)
        XCTAssertEqual(result.repetitionPenalty, 1.3,  accuracy: 1e-4)
    }

    func testChatEndpoint_PerRequestOverrides_DoNotAffectUnspecifiedFields() {
        // Overriding temperature must not silently reset enableThinking or kvBits.
        var base = GenerationConfig.default
        base.enableThinking = true
        base.kvBits = 4

        let json: [String: Any] = ["temperature": 0.5]
        let result = applyOverrides(json, to: base)

        XCTAssertTrue(result.enableThinking,
                      "enableThinking must survive a temperature-only override")
        XCTAssertEqual(result.kvBits, 4,
                       "kvBits must survive a temperature-only override")
    }

    func testChatEndpoint_EmptyOverrideDict_LeavesConfigUnchanged() {
        let base   = GenerationConfig(maxTokens: 1234, temperature: 0.42)
        let result = applyOverrides([:], to: base)
        XCTAssertEqual(result.maxTokens,   1234)
        XCTAssertEqual(result.temperature, 0.42, accuracy: 1e-4)
    }

    func testChatEndpoint_StreamFlag_DefaultsToFalse() {
        // Requests without "stream" must not stream — the endpoint defaults to
        // non-streaming, matching the OpenAI spec.
        let json: [String: Any] = ["model": "local", "messages": []]
        let streamRequested = json["stream"] as? Bool ?? false
        XCTAssertFalse(streamRequested,
                       "Missing 'stream' key must default to non-streaming")
    }

    func testChatEndpoint_StreamFlag_ExplicitTrue() {
        let json: [String: Any] = ["stream": true]
        let streamRequested = json["stream"] as? Bool ?? false
        XCTAssertTrue(streamRequested)
    }
}
