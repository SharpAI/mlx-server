// CLICommandBuilder.swift — Pure function for building the equivalent CLI command
// Lives in MLXInferenceCore so it can be unit-tested by SwiftLMTests without
// requiring the SwiftBuddy app target.
import Foundation

/// Builds the equivalent `swift run SwiftLM` command string from persisted settings.
/// Only emits flags that differ from the CLI defaults, keeping the command readable.
///
/// - Parameters:
///   - config:  The current `GenerationConfig`.
///   - host:    The server host string (e.g. "127.0.0.1").
///   - port:    The server port (e.g. 5413).
///   - parallel: Number of parallel request slots (default 1).
///   - apiKeySet: `true` if an API key is configured (key itself is redacted).
///   - modelId:  The currently loaded model ID, or `nil` when no model is loaded.
/// - Returns: A multi-line shell command string suitable for display and copy.
public func buildCLICommand(
    config: GenerationConfig,
    host: String,
    port: Int,
    parallel: Int,
    apiKeySet: Bool,
    modelId: String?
) -> String {
    var parts: [String] = []

    parts.append("--model \(modelId ?? "<model-id>")")
    parts.append("--host \(host)")
    parts.append("--port \(port)")
    parts.append("--max-tokens \(config.maxTokens)")
    parts.append("--temp \(String(format: "%.2f", config.temperature))")

    if config.topP < 1.0 {
        parts.append("--top-p \(String(format: "%.2f", config.topP))")
    }
    if config.topK != 50 {
        parts.append("--top-k \(config.topK)")
    }
    if config.minP > 0 {
        parts.append("--min-p \(String(format: "%.2f", config.minP))")
    }
    if config.repetitionPenalty != 1.05 {
        parts.append("--repeat-penalty \(String(format: "%.2f", config.repetitionPenalty))")
    }
    if config.prefillSize != 512 {
        parts.append("--prefill-size \(config.prefillSize)")
    }
    if let kvBits = config.kvBits {
        parts.append("--kv-bits \(kvBits)")
        if config.kvGroupSize != 64 {
            parts.append("--kv-group-size \(config.kvGroupSize)")
        }
    }
    if config.enableThinking {
        parts.append("--thinking")
    }
    if let seed = config.seed {
        parts.append("--seed \(seed)")
    }
    if parallel > 1 {
        parts.append("--parallel \(parallel)")
    }
    if apiKeySet {
        parts.append("--api-key <redacted>")
    }

    return "swift run SwiftLM " + parts.joined(separator: " \\\n  ")
}
