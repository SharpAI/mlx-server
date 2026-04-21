// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Engine Protocol

/// Protocol for DFlash verify/rollback engines.
///
/// Two concrete implementations exist:
/// - ``FullAttentionEngine`` — for pure-attention target models
/// - ``HybridGDNEngine`` — for hybrid GatedDeltaNet + attention target models
public protocol DFlashEngine: Sendable {
    /// Arm the target model's cache for rollback before verification.
    func armRollback(targetCache: [KVCache], prefixLen: Int)

    /// Roll back the target cache after partial acceptance.
    func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int
}

// MARK: - Full Attention Engine

/// Engine for pure-attention target models (no recurrent layers).
/// Rollback is just KV cache trimming.
public final class FullAttentionEngine: DFlashEngine, @unchecked Sendable {
    public init() {}

    public func armRollback(targetCache: [KVCache], prefixLen: Int) {
        // Pure attention: no arming needed
    }

    public func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        DFlashRuntime.restoreTargetCacheAfterAcceptance(
            targetCache,
            targetLen: targetLen,
            acceptanceLength: acceptanceLength,
            draftedTokens: draftedTokens
        )
    }
}

// MARK: - Hybrid GDN Engine

/// Engine for hybrid GatedDeltaNet + attention target models.
/// Uses RecurrentRollbackCache for recurrent layers with tape replay.
public final class HybridGDNEngine: DFlashEngine, @unchecked Sendable {
    public init() {}

    public func armRollback(targetCache: [KVCache], prefixLen: Int) {
        for cache in targetCache {
            if let rollbackCache = cache as? RecurrentRollbackCache {
                rollbackCache.armRollback(prefixLen: prefixLen)
            }
        }
    }

    public func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        DFlashRuntime.restoreTargetCacheAfterAcceptance(
            targetCache,
            targetLen: targetLen,
            acceptanceLength: acceptanceLength,
            draftedTokens: draftedTokens
        )
    }
}
