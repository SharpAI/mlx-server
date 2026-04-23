import DFlash
import MLX
import MLXLLM
import MLXLMCommon

extension Qwen3NextModel: DFlashTargetModel {

    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let cacheOpt: [KVCache?] = cache.map { $0 }
        let (hiddenStates, captured) = model.callCapturing(
            inputIDs, cache: cacheOpt, captureLayerIDs: captureLayerIDs)
        return (dflashLmHeadLogits(hiddenStates), captured)
    }

    /// Qwen3Next has GDN-style linear attention layers, but any rollback scheme
    /// (tape or snapshot) degrades acceptance rate by leaving recurrent state stale.
    /// Without rollback, rejected-token contamination is empirically negligible
    /// (< 1 reject per accepted cycle at long context) and gives ~3x speedup.
    /// Python avoids this tradeoff via @mx.compile on the verify pass (free tape).
    public var dflashIsHybridGDN: Bool { false }
}
