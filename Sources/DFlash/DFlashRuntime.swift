// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Model Introspection Protocol

/// Protocol that target models can conform to in order to expose their
/// internal structure for DFlash speculative decoding.
///
/// The DFlash runtime needs to:
/// 1. Access the embedding layer for draft noise embeddings
/// 2. Access the lm_head for draft logits
/// 3. Run a custom forward pass that captures intermediate hidden states
/// 4. Determine if the model has hybrid GDN layers
public protocol DFlashTargetModel: LanguageModel {
    /// Embed token IDs and return the embedding vectors.
    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray

    /// Compute logits from hidden states (via lm_head or tied weights).
    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray

    /// Run a forward pass capturing hidden states at the specified layer indices.
    ///
    /// - Parameters:
    ///   - inputIDs: Input token IDs [1, seqLen]
    ///   - cache: The KV cache array
    ///   - captureLayerIDs: Set of 0-based layer indices whose output to capture
    /// - Returns: Tuple of (logits, captured hidden states keyed by layerID+1)
    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray])

    /// Whether the model contains hybrid GatedDeltaNet layers.
    var dflashIsHybridGDN: Bool { get }
}

// MARK: - DFlash Generation Event

/// Events emitted during DFlash generation.
public enum DFlashEvent: Sendable {
    /// Prefill completed
    case prefill(promptTokenCount: Int, prefillUs: Double)
    /// Prefill progress (chunked)
    case prefillProgress(tokensProcessed: Int, tokensTotal: Int)
    /// A token was generated
    case token(tokenID: Int, generatedTokens: Int, acceptanceRatio: Double, cyclesCompleted: Int)
    /// Generation summary
    case summary(DFlashSummary)
}

/// Summary statistics for a DFlash generation run.
public struct DFlashSummary: Sendable {
    public let elapsedUs: Double
    public let promptTokenCount: Int
    public let generatedTokenIDs: [Int]
    public let acceptedFromDraft: Int
    public let acceptanceRatio: Double
    public let blockTokens: Int
    public let cyclesCompleted: Int
    public let phaseTimingsUs: PhaseTimings

    public struct PhaseTimings: Sendable {
        public let prefill: Double
        public let draft: Double
        public let verify: Double
        public let replay: Double
    }

    public var generationTokens: Int { generatedTokenIDs.count }
    public var tokensPerSecond: Double {
        let genUs = elapsedUs - phaseTimingsUs.prefill
        return genUs > 0 ? Double(generationTokens) / (genUs / 1_000_000.0) : 0
    }
}

// MARK: - DFlash Runtime

/// The main DFlash speculative decoding runtime.
///
/// Orchestrates the block-diffusion draft → verify → accept/reject → rollback
/// cycle for lossless speculative decoding on Apple Silicon.
public enum DFlashRuntime {

    // MARK: - Token Utilities

    /// Build a suppress token mask from a list of token IDs.
    public static func buildSuppressTokenMask(
        vocabSize: Int,
        suppressTokenIDs: [Int]?
    ) -> MLXArray? {
        let ids = Set((suppressTokenIDs ?? []).map { Int($0) }.filter { $0 >= 0 && $0 < vocabSize })
        guard !ids.isEmpty else { return nil }
        let sorted = ids.sorted()
        let vocabIndices = MLXArray.arange(vocabSize, dtype: .int32)
        let tokenArray = MLXArray(sorted.map { Int32($0) })
        return MLX.any(
            MLX.equal(
                expandedDimensions(vocabIndices, axis: 1),
                expandedDimensions(tokenArray, axis: 0)
            ),
            axis: 1
        )
    }

    /// Greedy token selection with optional suppress mask.
    public static func greedyTokensWithMask(
        logits: MLXArray,
        suppressTokenMask: MLXArray? = nil
    ) -> MLXArray {
        if let mask = suppressTokenMask {
            let floor = MLXArray(-1e9, dtype: logits.dtype)
            let maskedLogits = MLX.where(mask, floor, logits)
            return argMax(maskedLogits, axis: -1).asType(.uint32)
        }
        return argMax(logits, axis: -1).asType(.uint32)
    }

    /// Match the acceptance length between drafted and posterior tokens.
    /// Returns the number of consecutive matches starting from position 0.
    /// E.g. if drafted=[1,2,3] and posterior=[1,2,5], returns 2.
    public static func matchAcceptanceLength(
        draftedTokens: MLXArray,
        posteriorTokens: MLXArray
    ) -> MLXArray {
        let count = draftedTokens.dim(0)
        guard count > 0 else { return MLXArray(0, dtype: .int32) }
        let matches = (draftedTokens .== posteriorTokens).asType(.int32)
        // cumprod: [1,1,0,...] for consecutive matches, then sum counts them
        return cumprod(matches, axis: 0).sum(axis: 0, keepDims: false)
    }

    // MARK: - Target Cache Management

    /// Create the appropriate cache entries for the target model.
    /// For hybrid GDN models, replaces MambaCache with RecurrentRollbackCache
    /// for GDN (linear attention) layers.
    public static func makeTargetCache(
        targetModel: any DFlashTargetModel
    ) -> [KVCache] {
        var cache = targetModel.newCache(parameters: nil)
        if targetModel.dflashIsHybridGDN {
            for i in 0 ..< cache.count {
                if cache[i] is MambaCache {
                    cache[i] = RecurrentRollbackCache()
                }
            }
        }
        return cache
    }

    /// Arm all rollback-capable caches in the target model.
    /// For DFlashRollbackCache (GDN layers), arms for tape recording.
    /// For MambaCache, checkpoints the state.
    public static func armTargetRollback(targetCache: [KVCache], prefixLen: Int) {
        for cache in targetCache {
            if let rollbackCache = cache as? DFlashRollbackCache {
                rollbackCache.armRollback(prefixLen: prefixLen)
            }
            // Note: Python only calls arm_rollback on caches that implement it.
            // Plain MambaCache instances are NOT checkpointed here.
        }
    }

    /// Restore the target cache after partial acceptance of draft tokens.
    ///
    /// For MambaCache: we don't have innovation-tape rollback (unlike the Python
    /// reference which uses RecurrentRollbackCache with speculative hooks). Instead,
    /// we clear the checkpoint. The GDN state will contain contributions from all
    /// verify tokens including rejected ones, but the attention layers' KV caches
    /// will be correctly trimmed. This is a known quality trade-off that slightly
    /// reduces acceptance rate for GDN layers.
    ///
    /// For KVCacheSimple: trim to remove rejected tokens' KV entries.
    ///
    /// - Returns: Time spent on replay in nanoseconds
    @discardableResult
    public static func restoreTargetCacheAfterAcceptance(
        _ cacheEntries: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        let fullyAccepted = draftedTokens > 0 && acceptanceLength == draftedTokens
        var replayNs: Int = 0

        for cache in cacheEntries {
            if let rollbackCache = cache as? DFlashRollbackCache {
                if fullyAccepted {
                    rollbackCache.clearTransients()
                    continue
                }
                let startNs = Int(DispatchTime.now().uptimeNanoseconds)
                rollbackCache.rollback(nAccepted: acceptanceLength)
                replayNs += Int(DispatchTime.now().uptimeNanoseconds) - startNs
            } else if let mambaCache = cache as? MambaCache {
                // Plain MambaCache (non-rollback): no checkpoint-based rollback available.
                // Python doesn't call checkpoint/trim on these. The state contains
                // contributions from all verify tokens but we can't undo them.
                // Only update the offset to reflect the accepted prefix.
                mambaCache.offset = targetLen
            } else if cache.isTrimmable {
                let offset = cache.offset
                if offset > targetLen {
                    let startNs = Int(DispatchTime.now().uptimeNanoseconds)
                    cache.trim(offset - targetLen)
                    replayNs += Int(DispatchTime.now().uptimeNanoseconds) - startNs
                }
            }
        }

        return replayNs
    }

    // MARK: - Main Generation Loop

    /// Generate tokens using DFlash speculative decoding.
    ///
    /// - Parameters:
    ///   - targetModel: The target (large) language model (must conform to DFlashTargetModel)
    ///   - draftModel: The DFlash block-diffusion draft model
    ///   - promptTokens: Pre-tokenized prompt token IDs
    ///   - maxNewTokens: Maximum number of new tokens to generate
    ///   - blockTokens: Number of tokens per draft block (default: draft model's block_size)
    ///   - stopTokenIDs: Token IDs that signal end of generation
    ///   - suppressTokenIDs: Token IDs to suppress during generation
    ///   - draftSinkSize: Sink tokens to keep in draft cache
    ///   - draftWindowSize: Sliding window size for draft cache
    /// - Returns: AsyncStream of DFlashEvent values
    public static func generate(
        targetModel: any DFlashTargetModel,
        draftModel: DFlashDraftModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        blockTokens: Int? = nil,
        stopTokenIDs: [Int] = [],
        suppressTokenIDs: [Int]? = nil,
        draftSinkSize: Int = 64,
        draftWindowSize: Int = 1024
    ) -> AsyncStream<DFlashEvent> {
        // Run generateSync once and buffer all events, then yield them one at a time
        let events = generateSync(
            targetModel: targetModel,
            draftModel: draftModel,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            blockTokens: blockTokens,
            stopTokenIDs: stopTokenIDs,
            suppressTokenIDs: suppressTokenIDs,
            draftSinkSize: draftSinkSize,
            draftWindowSize: draftWindowSize
        )
        var iterator = events.makeIterator()
        return AsyncStream(unfolding: {
            iterator.next()
        })
    }

    /// Synchronous generation that returns all events at once.
    /// Used internally by the async generator.
    public static func generateSync(
        targetModel: any DFlashTargetModel,
        draftModel: DFlashDraftModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        blockTokens: Int? = nil,
        stopTokenIDs: [Int] = [],
        suppressTokenIDs: [Int]? = nil,
        draftSinkSize: Int = 64,
        draftWindowSize: Int = 1024
    ) -> [DFlashEvent] {
        var events: [DFlashEvent] = []

        let promptLen = promptTokens.count
        guard promptLen > 0 && maxNewTokens > 0 else { return events }

        let promptArray = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, -1).asType(.uint32)

        // Detect engine and create caches
        let engine: any DFlashEngine = targetModel.dflashIsHybridGDN
            ? HybridGDNEngine()
            : FullAttentionEngine()

        let draftBackend = DFlashDraftBackend()

        var targetCache = makeTargetCache(targetModel: targetModel)

        let draftCache = draftBackend.makeCache(
            draftModel: draftModel,
            sinkSize: draftSinkSize,
            windowSize: draftWindowSize
        )

        let targetLayerIDList = draftModel.targetLayerIDs
        let captureLayerIDs = Set(targetLayerIDList.map { $0 + 1 })
        let maskTokenID = draftModel.maskTokenID

        let startNanos = DispatchTime.now().uptimeNanoseconds

        // ── Prefill ────────────────────────────────────────────────
        let prefillStepSize = 2048
        var targetHidden: MLXArray?
        var prefillLogits: MLXArray!

        for chunkStart in stride(from: 0, to: promptLen, by: prefillStepSize) {
            let chunkEnd = min(chunkStart + prefillStepSize, promptLen)
            let chunkIDs = promptArray[0..., chunkStart ..< chunkEnd]

            let (chunkLogits, chunkHidden) = targetModel.dflashForwardWithCapture(
                inputIDs: chunkIDs,
                cache: targetCache,
                captureLayerIDs: captureLayerIDs
            )

            eval(chunkLogits)
            for (_, v) in chunkHidden { eval(v) }

            let feat = extractContextFeatureFromDict(
                capturedDict: chunkHidden,
                targetLayerIDs: targetLayerIDList
            )

            if targetHidden == nil {
                targetHidden = MLXArray.zeros(
                    [feat.dim(0), promptLen, feat.dim(-1)],
                    dtype: feat.dtype
                )
            }
            targetHidden![0..., chunkStart ..< chunkEnd, 0...] = feat
            eval(targetHidden!)

            prefillLogits = chunkLogits

            DFlashDumper.save("swift_target_hidden", targetHidden!)
            DFlashDumper.save("swift_prefill_logits", chunkLogits)

            events.append(.prefillProgress(
                tokensProcessed: chunkEnd,
                tokensTotal: promptLen
            ))
        }

        MLX.Memory.clearCache()

        let prefillNanos = Int(DispatchTime.now().uptimeNanoseconds) - Int(startNanos)

        let suppressTokenMask = buildSuppressTokenMask(
            vocabSize: Int(prefillLogits.dim(-1)),
            suppressTokenIDs: suppressTokenIDs
        )

        var stagedFirst = greedyTokensWithMask(
            logits: prefillLogits[0..., -1, 0...],
            suppressTokenMask: suppressTokenMask
        ).reshaped(-1)

        events.append(.prefill(
            promptTokenCount: promptLen,
            prefillUs: Double(prefillNanos) / 1000.0
        ))

        // Yield the first token
        let firstTokenID = Int(stagedFirst.item(Int.self))
        events.append(.token(
            tokenID: firstTokenID,
            generatedTokens: 1,
            acceptanceRatio: 0.0,
            cyclesCompleted: 0
        ))

        // ── Generation Loop ───────────────────────────────────────
        let draftBlockSize = draftModel.blockSize
        let requestedBlockTokens = blockTokens ?? draftBlockSize
        let effectiveBlockTokens = max(1, min(requestedBlockTokens, draftBlockSize))
        let verifyLenCap = effectiveBlockTokens  // default; env var override not implemented

        var generatedTokenIDs: [Int] = []
        var acceptedFromDraft = 0
        var cyclesCompleted = 0
        var start = promptLen
        var firstTokenYielded = false

        // Add the first token (from prefill) to generated list
        generatedTokenIDs.append(firstTokenID)
        firstTokenYielded = true

        let maskTokenTail = MLXArray.full(
            [max(0, effectiveBlockTokens - 1)],
            values: MLXArray(Int32(maskTokenID), dtype: .uint32)
        )

        var verifyNsTotal: Int = 0
        var draftNsTotal: Int = 0
        var replayNsTotal: Int = 0

        while generatedTokenIDs.count < maxNewTokens {
            let remaining = maxNewTokens - generatedTokenIDs.count
            let blockLen = max(1, min(effectiveBlockTokens, remaining))

            // ── Draft Phase ──────────────────────────────────────
            var drafted: MLXArray?
            var currentStagedFirst = stagedFirst
            if blockLen > 1 {
                let draftStart = Int(DispatchTime.now().uptimeNanoseconds)
                drafted = draftBackend.draftGreedy(
                    targetModel: targetModel,
                    draftModel: draftModel,
                    draftCache: draftCache,
                    stagedFirst: stagedFirst,
                    targetHidden: targetHidden!,
                    blockLen: blockLen,
                    maskTokenTail: maskTokenTail,
                    suppressTokenMask: suppressTokenMask
                )
                DFlashDumper.save("swift_cycle_draft", drafted ?? MLXArray())
                draftNsTotal += Int(DispatchTime.now().uptimeNanoseconds) - draftStart
            }

            // ── Verify Phase ────────────────────────────────────
            // Construct verify token IDs per Python reference:
            //   verify_token_count = min(block_len, verify_len_cap)
            //   verify_token_ids = concat([staged_first[:1], drafted[:verify_token_count-1]])
            let verifyTokenCount = min(blockLen, verifyLenCap)
            let verifyTokenIDs: MLXArray
            if blockLen <= 1 {
                verifyTokenIDs = currentStagedFirst[..<1]
            } else if let drafted = drafted, verifyTokenCount > 1 {
                verifyTokenIDs = concatenated(
                    [currentStagedFirst[..<1], drafted[..<(verifyTokenCount - 1)]],
                    axis: 0
                )
            } else {
                verifyTokenIDs = currentStagedFirst[..<1]
            }
            let verifyIDs = verifyTokenIDs[.newAxis]

            armTargetRollback(targetCache: targetCache, prefixLen: start)

            let verifyStart = Int(DispatchTime.now().uptimeNanoseconds)
            let (verifyLogits, verifyHiddenStates) = targetModel.dflashForwardWithCapture(
                inputIDs: verifyIDs,
                cache: targetCache,
                captureLayerIDs: captureLayerIDs
            )
            eval(verifyLogits)
            for (_, v) in verifyHiddenStates { eval(v) }
            verifyNsTotal += Int(DispatchTime.now().uptimeNanoseconds) - verifyStart

            // ── Accept/Reject ──────────────────────────────────
            let posterior = greedyTokensWithMask(
                logits: verifyLogits[0],
                suppressTokenMask: suppressTokenMask
            )
            asyncEval(posterior)
            DFlashDumper.save("swift_cycle_posterior", posterior)
            DFlashDumper.saveInt("swift_cycle_verifyIDs", verifyTokenIDs)

            // Acceptance: compare drafted tokens (positions 1+) against
            // posterior tokens at positions 0..<n-1
            let acceptanceLen: Int
            if verifyTokenIDs.dim(0) > 1 {
                acceptanceLen = Int(
                    matchAcceptanceLength(
                        draftedTokens: verifyTokenIDs[1...],
                        posteriorTokens: posterior[..<(verifyTokenIDs.dim(0) - 1)]
                    ).item(Int.self)
                )
            } else {
                acceptanceLen = 0
            }
            print("[DFlash] Cycle \(cyclesCompleted + 1): blockLen=\(blockLen), verifyLen=\(verifyTokenIDs.dim(0)), acceptanceLen=\(acceptanceLen), commitCount=\(1 + acceptanceLen)")
            fflush(stdout)
            fflush(stdout)

            let committedHidden = extractContextFeatureFromDict(
                capturedDict: verifyHiddenStates,
                targetLayerIDs: targetLayerIDList
            )[0..., ..<(1 + acceptanceLen), 0...]
            eval(committedHidden)

            let commitCount = 1 + acceptanceLen
            let committedSegment = verifyTokenIDs[..<(commitCount)]

            // ── Rollback ───────────────────────────────────────
            start += commitCount
            targetHidden = committedHidden
            let replayNs = engine.rollback(
                targetCache: targetCache,
                targetLen: start,
                acceptanceLength: acceptanceLen,
                draftedTokens: blockLen - 1
            )
            replayNsTotal += replayNs
            cyclesCompleted += 1
            acceptedFromDraft += acceptanceLen

            let stagedFirstNext = posterior[acceptanceLen ..< (acceptanceLen + 1)]

            // ── Emit tokens ───────────────────────────────────
            let committedIDs = committedSegment.asArray(Int.self)
            for tokenID in committedIDs {
                guard generatedTokenIDs.count < maxNewTokens else { break }
                generatedTokenIDs.append(tokenID)

                // Skip the first token (already yielded during prefill)
                if firstTokenYielded {
                    firstTokenYielded = false
                    continue
                }

                let acceptanceRatio = generatedTokenIDs.count > 0
                    ? Double(acceptedFromDraft) / Double(generatedTokenIDs.count)
                    : 0.0
                events.append(.token(
                    tokenID: tokenID,
                    generatedTokens: generatedTokenIDs.count,
                    acceptanceRatio: acceptanceRatio,
                    cyclesCompleted: cyclesCompleted
                ))
            }

            // Check for stop tokens
            let hit = committedIDs.contains { id in
                stopTokenIDs.contains(id)
            }
            if hit { break }

            stagedFirst = stagedFirstNext
        }

        // ── Summary ────────────────────────────────────────────
        let elapsedNanos = Int(DispatchTime.now().uptimeNanoseconds) - Int(startNanos)
        let acceptanceRatio = generatedTokenIDs.count > 0
            ? Double(acceptedFromDraft) / Double(generatedTokenIDs.count)
            : 0.0

        events.append(.summary(DFlashSummary(
            elapsedUs: Double(elapsedNanos) / 1000.0,
            promptTokenCount: promptLen,
            generatedTokenIDs: generatedTokenIDs,
            acceptedFromDraft: acceptedFromDraft,
            acceptanceRatio: acceptanceRatio,
            blockTokens: effectiveBlockTokens,
            cyclesCompleted: cyclesCompleted,
            phaseTimingsUs: .init(
                prefill: Double(prefillNanos) / 1000.0,
                draft: Double(draftNsTotal) / 1000.0,
                verify: Double(verifyNsTotal) / 1000.0,
                replay: Double(replayNsTotal) / 1000.0
            )
        )))

        return events
    }
}
