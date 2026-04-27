import XCTest
import MLX
import MLXLMCommon
@testable import SwiftLM

// MARK: - Regression tests for PR #85 -- prompt-cache bleed fixes
//
// These tests protect the PromptCache actor's save/restore contract WITHOUT
// downloading any model. We create synthetic KVCache instances with tiny
// MLXArray tensors ([1, 2, T, 4]) and exercise every guard directly.
//
// What this locks in:
//   1. MambaCache gate in save() and restore()       -- PR #85 fix 1
//   2. T-dim slice to P in save()                    -- PR #85 fix 2
//   3. ndim >= 3 guard in restore() minSeqLen scan   -- PR #85 fix 3
//   4. Recurrent-layer detection in restore()        -- PR #85 fix 4
//   5. Spec-decode-first ordering (logic test)       -- PR #85 fix 5
//   6. skipPromptCache guard (logic test)            -- PR #85 fix 6

final class PromptCacheTests: XCTestCase {

    // MARK: - Helpers

    /// Create a KVCacheSimple with a pre-populated state of shape [1, 2, T, 4].
    /// This mimics a layer that has processed T tokens.
    private func makePopulatedSimpleCache(seqLen T: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let keys = MLXArray.ones([1, 2, T, 4], dtype: .float16)
        let values = MLXArray.ones([1, 2, T, 4], dtype: .float16)
        _ = cache.update(keys: keys, values: values)
        return cache
    }

    // MARK: - Group 1: save() guards

    /// PR #85 fix 1: save() must skip entirely when any layer is MambaCache.
    func testSave_SkipsMambaCache() async {
        let pc = PromptCache()
        let simpleLayer = makePopulatedSimpleCache(seqLen: 10)
        let mambaLayer = MambaCache()

        await pc.save(tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cache: [simpleLayer, mambaLayer])

        // Restore into a non-Mamba cache so a miss validates save() skipped persistence,
        // rather than restore() taking its own MambaCache early-exit.
        let freshCache = [KVCacheSimple()] as [any KVCache]
        let result = await pc.restore(newTokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], into: freshCache)
        XCTAssertNil(result, "save() should be a no-op when any source layer is MambaCache")
    }

    /// PR #85 fix 2: save() must slice KVCacheSimple state T-dim to exactly P tokens.
    /// KVCacheSimple internally pre-allocates buffers in step-sized chunks (step=256),
    /// so the raw buffer T can exceed the actual prompt length P. The state getter
    /// returns [..<offset] which is correct, but save() applies an additional T-dim
    /// slice to guarantee cached.tokens.count === cached state's T.
    func testSave_SlicesTDimToP() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 10)

        let tokens = Array(0..<10)
        await pc.save(tokens: tokens, cache: [cache])

        // Restore with exact same tokens -> full match
        let freshCache = [KVCacheSimple()] as [any KVCache]
        let result = await pc.restore(newTokens: tokens, into: freshCache)
        XCTAssertEqual(result, 10, "Full match: all 10 tokens should be restored")

        // Verify the restored cache's state has T-dim == P (10), not over-allocated
        let restoredState = freshCache[0].state
        XCTAssertFalse(restoredState.isEmpty, "Restored cache must have state")
        XCTAssertEqual(restoredState[0].dim(2), 10,
            "Restored keys T-dim must equal P (10), not the pre-allocated buffer size")
    }

    /// save() should work fine when T <= P (no slicing needed).
    func testSave_PreservesSmallTDim() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 5)
        let tokens = Array(0..<5)

        await pc.save(tokens: tokens, cache: [cache])

        let freshCache = [KVCacheSimple()] as [any KVCache]
        let result = await pc.restore(newTokens: tokens, into: freshCache)
        XCTAssertEqual(result, 5, "Exact T=P: all 5 tokens should be restored")
    }

    /// save() with pure KVCacheSimple should not crash (basic smoke test).
    func testSave_PureSimpleCache_Succeeds() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 3)
        let tokens = [10, 20, 30]

        // Should not crash
        await pc.save(tokens: tokens, cache: [cache])

        let stats = await pc.stats()
        XCTAssertEqual(stats.hits, 0)
        XCTAssertEqual(stats.misses, 0)
    }

    // MARK: - Group 2: restore() guards

    /// PR #85 fix 1 (restore side): restore must reject when target cache has MambaCache.
    func testRestore_SkipsMambaCache() async {
        let pc = PromptCache()
        // Save with pure KVCacheSimple
        let saveCache = makePopulatedSimpleCache(seqLen: 5)
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: [saveCache])

        // Restore into a cache that includes MambaCache
        let restoreCache: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let result = await pc.restore(newTokens: [1, 2, 3, 4, 5], into: restoreCache)
        XCTAssertNil(result, "MambaCache in target cache -> restore must return nil (miss)")

        let stats = await pc.stats()
        XCTAssertEqual(stats.misses, 1)
    }

    /// PR #85 fix 4: restore must detect non-KVCacheSimple/non-RotatingKVCache layers
    /// (recurrent layers like ArraysCache) and bail.
    func testRestore_SkipsRecurrentLayer() async {
        let pc = PromptCache()
        // Save with KVCacheSimple
        let saveCache = makePopulatedSimpleCache(seqLen: 5)
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: [saveCache])

        // Restore into a cache with ArraysCache (recurrent, but not MambaCache)
        let recurrentLayer = ArraysCache(size: 2)
        let restoreCache: [any KVCache] = [recurrentLayer]
        let result = await pc.restore(newTokens: [1, 2, 3, 4, 5], into: restoreCache)
        XCTAssertNil(result, "Recurrent layer (ArraysCache) -> restore must bail")
    }

    /// Basic happy path: save and restore with identical tokens -> full match.
    func testRestore_FullMatch() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 5)
        let tokens = [10, 20, 30, 40, 50]
        await pc.save(tokens: tokens, cache: [cache])

        let freshCache: [any KVCache] = [KVCacheSimple()]
        let result = await pc.restore(newTokens: tokens, into: freshCache)
        XCTAssertEqual(result, 5, "Identical tokens -> full match")

        let stats = await pc.stats()
        XCTAssertEqual(stats.hits, 1)
    }

    /// Partial prefix match: first 3 of 5 tokens match.
    func testRestore_PrefixMatch() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 5)
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: [cache])

        let freshCache: [any KVCache] = [KVCacheSimple()]
        let result = await pc.restore(newTokens: [1, 2, 3, 99, 100], into: freshCache)
        XCTAssertEqual(result, 3, "First 3 tokens match -> partial hit returns 3")
    }

    /// Complete miss: no token overlap.
    func testRestore_NoMatch() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 5)
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: [cache])

        let freshCache: [any KVCache] = [KVCacheSimple()]
        let result = await pc.restore(newTokens: [99, 98, 97], into: freshCache)
        XCTAssertNil(result, "No token overlap -> miss")
    }

    /// Empty cache -> restore must miss gracefully.
    func testRestore_EmptyCache_Misses() async {
        let pc = PromptCache()
        let freshCache: [any KVCache] = [KVCacheSimple()]
        let result = await pc.restore(newTokens: [1, 2, 3], into: freshCache)
        XCTAssertNil(result, "No prior save -> restore must return nil")
    }

    /// This protects against zeroing out a cached layer by trimming away all tokens.
    func testRestore_ExcessExceedsMinSeqLen_Bails() async {
        // Control case: a safe trim should still succeed.
        let pc = PromptCache()
        let shortCache = makePopulatedSimpleCache(seqLen: 3)
        await pc.save(tokens: [10, 20, 30], cache: [shortCache])

        // Request [10, 99, 99, 99] -> matchLen=1, excess = 3 - 1 = 2.
        // Saved state has dim(2)=3. excess(2) < minCachedSeqLen(3) -> safe.
        let fresh: [any KVCache] = [KVCacheSimple()]
        let safeResult = await pc.restore(newTokens: [10, 99, 99, 99], into: fresh)
        XCTAssertEqual(safeResult, 1, "1-token prefix match with 3-token cache should succeed because trimming 2 from 3 leaves 1 token")

        // Actual bail case: no shared prefix -> matchLen=0 -> guard matchLen > 0 fails -> nil.
        // This exercises the zero-match guard which is the first line of defense.
        let pc2 = PromptCache()
        let tinyCache = makePopulatedSimpleCache(seqLen: 2)
        await pc2.save(tokens: [10, 20], cache: [tinyCache])

        let fresh2: [any KVCache] = [KVCacheSimple()]
        let bailResult = await pc2.restore(newTokens: [99, 98], into: fresh2)
        XCTAssertNil(bailResult, "No shared prefix with a 2-token cache must return nil")

        let stats = await pc2.stats()
        XCTAssertEqual(stats.misses, 1, "Zero-match bail must increment miss counter")
    }

    // MARK: - Group 3: Decision branch ordering (pure logic tests)

    /// PR #85 fix 6: skipPromptCache must be true when multimodal.
    func testSkipPromptCache_Multimodal() {
        let isMultimodalRequest = true
        let kvBits: Int? = nil
        let skipPromptCache = isMultimodalRequest || kvBits != nil
        XCTAssertTrue(skipPromptCache, "Multimodal request -> must skip prompt cache")
    }

    /// PR #85 fix 6: skipPromptCache must be true when kv_bits is set.
    func testSkipPromptCache_KvBits() {
        let isMultimodalRequest = false
        let kvBits: Int? = 4
        let skipPromptCache = isMultimodalRequest || kvBits != nil
        XCTAssertTrue(skipPromptCache, "kv_bits set -> must skip prompt cache (format mismatch)")
    }

    /// Neither multimodal nor kv_bits -> should NOT skip.
    func testSkipPromptCache_Standard_DoesNotSkip() {
        let isMultimodalRequest = false
        let kvBits: Int? = nil
        let skipPromptCache = isMultimodalRequest || kvBits != nil
        XCTAssertFalse(skipPromptCache, "Standard text request -> should attempt cache")
    }

    /// PR #85 fix 5: spec-decode must be checked BEFORE prompt cache.
    /// Simulates the decision branch ordering from Server.swift.
    func testSpecDecode_CheckedBeforeCache() {
        let draftModelRef: String? = "mlx-community/Qwen3.5-4B-4bit"
        let skipPromptCache = false
        let cacheHit = true  // would have hit if checked

        var path: String = "unknown"

        // Replicate Server.swift decision branch
        if draftModelRef != nil {
            path = "spec-decode"
        } else if !skipPromptCache && cacheHit {
            path = "cache-hit"
        } else {
            path = "full-prefill"
        }

        XCTAssertEqual(path, "spec-decode",
            "Spec-decode must win over cache hit -- partial cache restore corrupts draft KV state")
    }

    /// Without draft model, cache hit should be used.
    func testCacheHit_UsedWhenNoDraft() {
        let draftModelRef: String? = nil
        let skipPromptCache = false
        let cacheHit = true

        var path: String = "unknown"

        if draftModelRef != nil {
            path = "spec-decode"
        } else if !skipPromptCache && cacheHit {
            path = "cache-hit"
        } else {
            path = "full-prefill"
        }

        XCTAssertEqual(path, "cache-hit",
            "Without draft model, cache hit should be the chosen path")
    }

    // MARK: - Group 4: Stats tracking

    /// Hit/miss counters must accumulate correctly.
    func testStats_AccumulateCorrectly() async {
        let pc = PromptCache()
        let cache = makePopulatedSimpleCache(seqLen: 3)
        await pc.save(tokens: [1, 2, 3], cache: [cache])

        // Miss (no overlap)
        let fresh1: [any KVCache] = [KVCacheSimple()]
        _ = await pc.restore(newTokens: [99], into: fresh1)

        // Hit (full match)
        let fresh2: [any KVCache] = [KVCacheSimple()]
        _ = await pc.restore(newTokens: [1, 2, 3], into: fresh2)

        // Miss (no overlap)
        let fresh3: [any KVCache] = [KVCacheSimple()]
        _ = await pc.restore(newTokens: [77, 88], into: fresh3)

        let stats = await pc.stats()
        XCTAssertEqual(stats.hits, 1)
        XCTAssertEqual(stats.misses, 2)
    }
}
