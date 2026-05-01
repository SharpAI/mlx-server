// GenerationConfig.swift — SwiftLM inference parameters
import Foundation

/// Per-request generation parameters, persisted across app launches via UserDefaults.
///
/// ### Field classification
/// **Per-request** (applied on every `generate()` call — no reload needed):
///   temperature, topP, topK, minP, repetitionPenalty, seed, enableThinking,
///   prefillSize, kvBits, kvGroupSize, turboKV
///
/// **Load-time** (requires model reload to take effect):
///   streamExperts — controls SSD expert streaming for MoE and large models.
///   Stored here for persistence but applied by InferenceEngine at load time.
public struct GenerationConfig: Sendable, Codable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float

    /// Optional RNG seed for reproducible outputs.
    /// When non-nil, `MLX.seed(seed)` is called before each generation using this `UInt64` value.
    public var seed: UInt64?

    public var enableThinking: Bool

    /// Chunk size for prefill evaluation.
    /// Lower values prevent GPU timeout on large models.
    public var prefillSize: Int

    /// KV-cache quantization bits (nil = no quantization, 4 or 8 typical).
    public var kvBits: Int?

    /// KV-cache quantization group size (default 64).
    public var kvGroupSize: Int

    /// Enable 3-bit TurboQuant KV-cache compression (PolarQuant+QJL).
    /// Compresses KV history older than 8192 tokens to ~3.5 bits/token.
    /// Recommended for 100k+ context to halve KV RAM usage.
    /// Applied per-request — no model reload needed.
    public var turboKV: Bool

    /// Enable SSD expert streaming for MoE (and any large) models.
    /// When true, expert weights are mmap'd from NVMe and only active
    /// expert pages reside in RAM during inference (Flash-MoE style).
    /// ⚠️ LOAD-TIME flag: changes take effect on the next model load.
    /// MoE models (isMoE == true) default to true automatically;
    /// this flag lets users override that for non-catalog models or
    /// force-disable streaming even on MoE models.
    public var streamExperts: Bool

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 50,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.05,
        seed: UInt64? = nil,
        enableThinking: Bool = false,
        prefillSize: Int = 512,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        turboKV: Bool = false,
        streamExperts: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.enableThinking = enableThinking
        self.prefillSize = prefillSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.turboKV = turboKV
        self.streamExperts = streamExperts
    }

    public static let `default` = GenerationConfig()

    // MARK: — Persistence

    private static let storageKey = "swiftlm.generationConfig"

    /// True when the user has previously saved a GenerationConfig.
    /// Used to distinguish the first-run/default state from an explicit choice.
    public static var hasPersistedConfig: Bool {
        UserDefaults.standard.object(forKey: storageKey) != nil
    }

    /// Computes the effective SSD streaming setting.
    /// Before the user has saved settings, MoE models default to streaming on.
    /// After settings are persisted, the saved toggle becomes authoritative.
    public func effectiveStreamExperts(defaultingTo defaultValue: Bool) -> Bool {
        Self.hasPersistedConfig ? streamExperts : defaultValue
    }

    public func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }

    public static func load() -> GenerationConfig {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode(GenerationConfig.self, from: data)
        else { return .default }
        return decoded
    }
}
