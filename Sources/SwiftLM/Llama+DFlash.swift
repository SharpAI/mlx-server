// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Bridge: LlamaModel (and Mistral) conform to DFlashTargetModel

import DFlash
import MLX
import MLXLLM
import MLXLMCommon

extension LlamaModel: DFlashTargetModel {
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

    public var dflashIsHybridGDN: Bool { false }
}
