// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
//
// DeepSeek V3 model owned by SwiftLM with DFlash speculative decoding support.
//
// Port of mlx-lm/mlx_lm/models/deepseek_v3.py
// Also handles kimi_k25 model type (wraps the same architecture).
//
// Kept in SwiftLM to avoid upstream submodule changes:
// callCapturing and DFlashTargetModel conformance live here alongside
// the model implementation so no public API surface is needed in MLXLLM.

import DFlash
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

struct DSV3Config: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }
}

// MARK: - Helpers

private func clippedSilu(_ x: MLXArray) -> MLXArray {
    clip(x * sigmoid(x), min: -100, max: 100)
}

// MARK: - Attention

private class DSV3Attention: Module {
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(config: DSV3Config) {
        numHeads = config.numAttentionHeads
        qLoraRank = config.qLoraRank
        qkRopeHeadDim = config.qkRopeHeadDim
        kvLoraRank = config.kvLoraRank
        vHeadDim = config.vHeadDim
        qkNopeHeadDim = config.qkNopeHeadDim
        qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        scale = pow(Float(qHeadDim), -0.5)

        if let r = config.qLoraRank {
            _qAProj.wrappedValue = Linear(config.hiddenSize, r, bias: config.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: r)
            _qBProj.wrappedValue = Linear(r, numHeads * qHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * qHeadDim, bias: false)
        }

        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        _kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim), bias: false)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
                if scalingFactor > 1 {
                    let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                    scale = scale * s * s
                }
            }
        }

        rope = initializeRope(
            dims: qkRopeHeadDim, base: config.ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (splitQ[0], splitQ[1])

        var compressedKv = kvAProjWithMqa(x)
        let splitKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitKv[0]
        var kPe = splitKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKV2 = split(kv, indices: [qkNopeHeadDim], axis: -1)
        var (kNope, values) = (splitKV2[0], splitKV2[1])

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        var keys: MLXArray
        if let cache {
            (keys, values) = cache.update(
                keys: concatenated([kNope, kPe], axis: -1), values: values)
        } else {
            keys = concatenated([kNope, kPe], axis: -1)
        }

        let queries = concatenated([qNope, qPe], axis: -1)
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP

private class DSV3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DSV3Config, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let h = hiddenSize ?? config.hiddenSize
        let i = intermediateSize ?? config.intermediateSize
        _gateProj.wrappedValue = Linear(h, i, bias: false)
        _upProj.wrappedValue = Linear(h, i, bias: false)
        _downProj.wrappedValue = Linear(i, h, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Gate

private class DSV3MoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(config: DSV3Config) {
        topK = config.numExpertsPerTok ?? 1
        normTopkProb = config.normTopkProb
        nRoutedExperts = config.nRoutedExperts ?? 1
        routedScalingFactor = config.routedScalingFactor
        nGroup = config.nGroup ?? 1
        topkGroup = config.topkGroup ?? 1
        weight = zeros([nRoutedExperts, config.hiddenSize])
        e_score_correction_bias = zeros([nRoutedExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))
        let hiddenStates = x.matmul(weight.T)
        var scores = sigmoid(hiddenStates)
        let scoresForChoice = scores + e_score_correction_bias
        let groupScores = scoresForChoice.reshaped(bsz, seqLen, nGroup, -1)
        let topKGroup = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroup, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, nRoutedExperts / nGroup])
        scores = putAlong(groupScores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)
        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        scores = takeAlong(scores, inds, axis: -1)
        if topK > 1, normTopkProb {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20) * routedScalingFactor
        }
        return (inds, scores)
    }
}

// MARK: - MoE

private class DSV3MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DSV3MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DSV3MLP?

    init(config: DSV3Config) {
        numExpertsPerTok = config.numExpertsPerTok ?? 1
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1,
            activation: clippedSilu)
        gate = DSV3MoEGate(config: config)
        if let sharedCount = config.nSharedExperts {
            _sharedExperts.wrappedValue = DSV3MLP(
                config: config, intermediateSize: config.moeIntermediateSize * sharedCount)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts { y = y + shared(x) }
        return y
    }
}

// MARK: - Decoder Layer

private class DSV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DSV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DSV3Config, layerIdx: Int) {
        _selfAttn.wrappedValue = DSV3Attention(config: config)
        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            mlp = DSV3MoE(config: config)
        } else {
            mlp = DSV3MLP(config: config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Model Inner

private class DSV3ModelInner: Module, LayerPartitionable {
    var gpuLayerCount: Int? = nil
    var totalLayerCount: Int { layers.count }

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [DSV3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DSV3Config) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = (0 ..< config.numHiddenLayers).map {
            DSV3DecoderLayer(config: config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount) {
                layer(h, mask: mask, cache: cache?[i])
            }
        }
        return norm(h)
    }

    func callCapturing(
        _ x: MLXArray, cache: [KVCache?]? = nil, captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        var h = embedTokens(x)
        let kvCache: [KVCache?] = {
            guard let c = cache else { return Array(repeating: nil, count: layers.count) }
            var out = Array(repeating: nil as KVCache?, count: layers.count)
            for (i, v) in c.prefix(layers.count).enumerated() { out[i] = v }
            return out
        }()
        let mask = createAttentionMask(h: h, cache: kvCache.first ?? nil)
        var captured: [Int: MLXArray] = [:]
        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount) {
                layer(h, mask: mask, cache: kvCache[i])
            }
            if captureLayerIDs.contains(i) { captured[i] = h }
        }
        return (norm(h), captured)
    }
}

// MARK: - Public Model

/// DeepSeek V3 model owned by SwiftLM.
/// Registered for `deepseek_v3` and `kimi_k25` model types at DFlash setup time,
/// overriding the MLXLLM factory default so DFlash conformance is available.
public class DeepseekV3DFlashModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel,
    DFlashTargetModel
{
    public var kvHeads: [Int] = []

    private let args: DSV3Config
    private let inner: DSV3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DSV3Config) {
        self.args = args
        inner = DSV3ModelInner(config: args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(inner(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var p = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            p = p.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = p * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    w[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if w[key] == nil {
                w[key] = value
            }
        }

        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let joined = (0 ..< (args.nRoutedExperts ?? 1)).map {
                            weights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        w["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }

        return w.filter { key, _ in
            !key.starts(with: "model.layers.61") && !key.contains("rotary_emb.inv_freq")
        }
    }

    public var loraLayers: [Module] { inner.layers }

    // MARK: DFlashTargetModel

    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        inner.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let cacheOpt: [KVCache?] = cache.map { $0 }
        let (hidden, captured) = inner.callCapturing(
            inputIDs, cache: cacheOpt, captureLayerIDs: captureLayerIDs)
        return (dflashLmHeadLogits(hidden), captured)
    }

    public var dflashIsHybridGDN: Bool { false }
}
