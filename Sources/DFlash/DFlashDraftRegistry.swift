// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Draft Model Registry

/// Registry mapping target model names to their DFlash draft models.
public enum DFlashDraftRegistry {

    /// Known target → draft model mappings.
    static let registry: [String: String] = [
        "Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
        "Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
        "Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
        "Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
        "Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
        "Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
        "Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
    ]

    /// Normalize a model reference by stripping the org prefix.
    private static func stripModelOrg(_ modelRef: String) -> String {
        modelRef.split(separator: "/").last.map(String.init) ?? modelRef
    }

    /// Resolve an optional draft model reference for the given target model.
    ///
    /// - Parameters:
    ///   - modelRef: The target model reference (org/name or local path)
    ///   - draftRef: An explicit draft model reference (takes priority)
    /// - Returns: The resolved draft model reference, or nil if none found
    public static func resolveDraftRef(modelRef: String, draftRef: String? = nil) -> String? {
        if let draftRef { return draftRef }

        let stripped = stripModelOrg(modelRef).lowercased()

        // Exact match
        for (key, value) in registry where key.lowercased() == stripped {
            return value
        }

        // Prefix match (e.g., "qwen3.5-4b-4bit" matches "qwen3.5-4b")
        var bestMatch: (key: String, value: String)?
        for (key, value) in registry {
            let lowered = key.lowercased()
            if stripped == lowered
                || stripped.hasPrefix(lowered + "-")
                || stripped.hasPrefix(lowered + "_")
            {
                if bestMatch == nil || key.count > bestMatch!.key.count {
                    bestMatch = (key, value)
                }
            }
        }

        return bestMatch?.value
    }

    /// List supported base model names.
    public static func supportedBaseModels() -> [String] {
        Array(registry.keys).sorted()
    }
}
