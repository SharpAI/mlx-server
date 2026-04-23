// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Bridge: Qwen35 models conform to DFlashTargetModel
//
// The dflash* methods are defined on Qwen35TextModel/Qwen35Model in the
// MLXLLM module. This file adds the DFlashTargetModel protocol conformance
// so the DFlash runtime can use them generically.

import DFlash
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Qwen35TextModel + DFlashTargetModel

extension Qwen35TextModel: DFlashTargetModel {
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

// MARK: - Qwen35Model + DFlashTargetModel

extension Qwen35Model: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        languageModel.dflashEmbedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        languageModel.dflashLmHeadLogits(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        languageModel.dflashForwardWithCapture(inputIDs: inputIDs, cache: cache, captureLayerIDs: captureLayerIDs)
    }

    public var dflashIsHybridGDN: Bool { languageModel.dflashIsHybridGDN }
}
