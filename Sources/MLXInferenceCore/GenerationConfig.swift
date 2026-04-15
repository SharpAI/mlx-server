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

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 50,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.05,
        seed: UInt64? = nil,
        enableThinking: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.enableThinking = enableThinking
    }

    public static let `default` = GenerationConfig()
}
