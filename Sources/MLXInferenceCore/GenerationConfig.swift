// GenerationConfig.swift — SwiftLM inference parameters
import Foundation

/// Configuration for a single generation request.
public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var enableThinking: Bool

    // ── SwiftLM Engine Parameters ──────────────────────────────────────
    /// Enable TurboQuant KV-cache compression (3-bit PolarQuant+QJL).
    /// Compresses KV history > 8192 tokens to ~3.5 bits/token.
    public var turboKV: Bool

    /// Enable SSD expert streaming for MoE models.
    public var streamExperts: Bool

    /// Chunk size for prefill evaluation.
    /// Lower values prevent GPU timeout on large models.
    public var prefillSize: Int

    /// KV-cache quantization bits (nil = no quantization, 4 or 8 typical).
    public var kvBits: Int?

    /// KV-cache quantization group size (default 64).
    public var kvGroupSize: Int

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 50,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.05,
        seed: UInt64? = nil,
        enableThinking: Bool = false,
        turboKV: Bool = false,
        streamExperts: Bool = false,
        prefillSize: Int = 512,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.enableThinking = enableThinking
        self.turboKV = turboKV
        self.streamExperts = streamExperts
        self.prefillSize = prefillSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
    }

    public static let `default` = GenerationConfig()
}
