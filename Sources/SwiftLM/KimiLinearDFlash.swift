// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
//
// Kimi linear (hybrid KDA/MLA) model owned by SwiftLM with DFlash support.
//
// Port of mlx-lm/mlx_lm/models/kimi_linear.py
// Handles model types: "kimi_linear"
//
// Kept in SwiftLM to avoid upstream submodule changes.
// DFlashTargetModel conformance and callCapturing live here with the model.

import DFlash
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private struct LinearAttnConfig: Codable, Sendable {
    var kdaLayers: [Int]  // 1-indexed layer indices that use KimiDeltaAttention
    var numHeads: Int
    var headDim: Int
    var shortConvKernelSize: Int

    enum CodingKeys: String, CodingKey {
        case kdaLayers = "kda_layers"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kdaLayers = try c.decode([Int].self, forKey: .kdaLayers)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        shortConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .shortConvKernelSize) ?? 4
    }
}

public struct KimiLinearConfiguration: Codable, Sendable {
    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var intermediateSize: Int
    var headDim: Int
    var rmsNormEps: Float
    fileprivate var linearAttnConfig: LinearAttnConfig
    var modelMaxLength: Int
    var numExperts: Int
    var moeIntermediateSize: Int
    var kvLoraRank: Int
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var qkNopeHeadDim: Int?
    var qkRopeHeadDim: Int?
    var vHeadDim: Int?
    var numExpertsPerToken: Int
    var numSharedExperts: Int
    var moeRouterActivationFunc: String
    var moeRenormalize: Bool
    var routedScalingFactor: Float
    var firstKDenseReplace: Int
    var moeLayerFreq: Int
    var numExpertGroup: Int
    var topkGroup: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case linearAttnConfig = "linear_attn_config"
        case modelMaxLength = "model_max_length"
        case numExperts = "num_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case kvLoraRank = "kv_lora_rank"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case numExpertsPerToken = "num_experts_per_token"
        case numSharedExperts = "num_shared_experts"
        case moeRouterActivationFunc = "moe_router_activation_func"
        case moeRenormalize = "moe_renormalize"
        case routedScalingFactor = "routed_scaling_factor"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case numExpertGroup = "num_expert_group"
        case topkGroup = "topk_group"
    }

    var resolvedQkNopeHeadDim: Int { qkNopeHeadDim ?? headDim }
    var resolvedQkRopeHeadDim: Int { qkRopeHeadDim ?? 0 }
    var resolvedVHeadDim: Int { vHeadDim ?? headDim }
    var qHeadDim: Int { resolvedQkNopeHeadDim + resolvedQkRopeHeadDim }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        linearAttnConfig = try c.decode(LinearAttnConfig.self, forKey: .linearAttnConfig)
        modelMaxLength = try c.decode(Int.self, forKey: .modelMaxLength)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim)
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 1
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        moeRouterActivationFunc =
            try c.decodeIfPresent(String.self, forKey: .moeRouterActivationFunc) ?? "sigmoid"
        moeRenormalize = try c.decodeIfPresent(Bool.self, forKey: .moeRenormalize) ?? true
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        numExpertGroup = try c.decodeIfPresent(Int.self, forKey: .numExpertGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
    }
}

// MARK: - KimiMLP

private class KimiMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(gate(x) * silu(up(x))) }
}

// MARK: - KimiMultiLinear

private class KimiMultiLinear: Module {
    var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        weight = MLXArray.zeros([numHeads, outputDims, inputDims])
    }

    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        transpose ? x.matmul(weight.transposed(-1, -2)) : x.matmul(weight)
    }
}

// MARK: - KimiMLAAttention

private class KimiMLAAttention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: KimiMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: KimiMultiLinear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ args: KimiLinearConfiguration) {
        numHeads = args.numAttentionHeads
        qkNopeHeadDim = args.resolvedQkNopeHeadDim
        qkRopeHeadDim = args.resolvedQkRopeHeadDim
        qHeadDim = args.qHeadDim
        vHeadDim = args.resolvedVHeadDim
        kvLoraRank = args.kvLoraRank
        scale = pow(Float(args.qHeadDim), -0.5)

        let h = args.hiddenSize
        _qProj.wrappedValue = Linear(h, numHeads * qHeadDim, bias: false)
        _kvAProj.wrappedValue = Linear(h, kvLoraRank + max(qkRopeHeadDim, 0), bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: args.rmsNormEps)
        _embedQ.wrappedValue = KimiMultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = KimiMultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, h, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: ArraysCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qNope = q[.ellipsis, ..<qkNopeHeadDim]

        let kvRaw = kvAProj(x)
        var kvLatent = kvALayerNorm(kvRaw[.ellipsis, ..<kvLoraRank])
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        if let cache {
            if let prev0 = cache[0], let prev1 = cache[1] {
                kvLatent = concatenated([prev0, kvLatent], axis: -2)
                let curKpe = qkRopeHeadDim > 0
                    ? kvRaw[.ellipsis, kvLoraRank...].reshaped(B, L, 1, qkRopeHeadDim)
                        .transposed(0, 2, 1, 3)
                    : MLXArray.zeros([B, 1, L, 0], dtype: kvLatent.dtype)
                cache[0] = kvLatent
                cache[1] = concatenated([prev1, curKpe], axis: -2)
            } else {
                cache[0] = kvLatent
                cache[1] = qkRopeHeadDim > 0
                    ? kvRaw[.ellipsis, kvLoraRank...].reshaped(B, L, 1, qkRopeHeadDim)
                        .transposed(0, 2, 1, 3)
                    : MLXArray.zeros([B, 1, L, 0], dtype: kvLatent.dtype)
            }
            cache.offset += L
        }
        let totalL = kvLatent.dim(-2)

        var peScores: MLXArray? = nil
        if qkRopeHeadDim > 0, let kPe = cache?[1] {
            let qPe = q[.ellipsis, qkNopeHeadDim...]
            peScores = (qPe * scale).matmul(kPe.transposed(-1, -2))
        }

        let output: MLXArray
        if L == 1 {
            let qMapped = embedQ(qNope)
            var scores = qMapped.matmul(kvLatent.transposed(-1, -2)) * scale
            if let pe = peScores { scores = scores + pe }
            let weights = softmax(scores, axis: -1)
            output = unembedOut(weights.matmul(kvLatent))
        } else {
            let k = embedQ(kvLatent, transpose: false)
            let v = unembedOut(kvLatent)
            var scores = qNope.matmul(k.transposed(-1, -2)) * scale
            scores = scores + makeCausalBias(L: L, totalL: totalL, dtype: scores.dtype)
            if let pe = peScores { scores = scores + pe }
            let weights = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
            output = weights.matmul(v)
        }

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    private func makeCausalBias(L: Int, totalL: Int, dtype: DType) -> MLXArray {
        let rows = MLXArray(Array(totalL - L ..< totalL)).reshaped(L, 1)
        let cols = MLXArray(Array(0 ..< totalL)).reshaped(1, totalL)
        return ((rows .< cols).asType(.float32) * Float(-1e9)).asType(dtype).reshaped(1, 1, L, totalL)
    }
}

// MARK: - ShortConv1d

private class ShortConv1d: Module {
    let kernelSize: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        _conv.wrappedValue = Conv1d(
            inputChannels: 1, outputChannels: channels, kernelSize: kernelSize,
            stride: 1, padding: 0, dilation: 1, groups: channels, bias: false)
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let nKeep = kernelSize - 1
        let prevState = state ?? MLXArray.zeros([B, nKeep, C], dtype: x.dtype)
        let convInput = concatenated([prevState, x], axis: 1)
        let out = silu(conv(convInput))
        return (out, convInput[0..., T...])
    }
}

// MARK: - KimiDeltaAttention

private class KimiDeltaAttention: Module {
    let numHeads: Int
    let headDim: Int
    let projDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv") var qConv: ShortConv1d
    @ModuleInfo(key: "k_conv") var kConv: ShortConv1d
    @ModuleInfo(key: "v_conv") var vConv: ShortConv1d
    @ModuleInfo(key: "f_a_proj") var faProj: Linear
    @ModuleInfo(key: "f_b_proj") var fbProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gaProj: Linear
    @ModuleInfo(key: "g_b_proj") var gbProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    var aLog: MLXArray
    var dtBias: MLXArray

    init(_ args: KimiLinearConfiguration, layerIdx: Int) {
        let cfg = args.linearAttnConfig
        numHeads = cfg.numHeads
        headDim = cfg.headDim
        projDim = numHeads * headDim
        scale = pow(Float(headDim), -0.5)

        let h = args.hiddenSize
        let K = cfg.shortConvKernelSize
        _qProj.wrappedValue = Linear(h, projDim, bias: false)
        _kProj.wrappedValue = Linear(h, projDim, bias: false)
        _vProj.wrappedValue = Linear(h, projDim, bias: false)
        _qConv.wrappedValue = ShortConv1d(channels: projDim, kernelSize: K)
        _kConv.wrappedValue = ShortConv1d(channels: projDim, kernelSize: K)
        _vConv.wrappedValue = ShortConv1d(channels: projDim, kernelSize: K)
        _faProj.wrappedValue = Linear(h, headDim, bias: false)
        _fbProj.wrappedValue = Linear(headDim, projDim, bias: false)
        _bProj.wrappedValue = Linear(h, numHeads, bias: false)
        _gaProj.wrappedValue = Linear(h, headDim, bias: false)
        _gbProj.wrappedValue = Linear(headDim, projDim, bias: false)
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _oProj.wrappedValue = Linear(projDim, h, bias: false)
        aLog = MLXArray.zeros([numHeads])
        dtBias = MLXArray.zeros([projDim])
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: ArraysCache?
    ) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        let (qConvOut, newQState) = qConv(qProj(x), state: cache?[0])
        let (kConvOut, newKState) = kConv(kProj(x), state: cache?[1])
        let (vConvOut, newVState) = vConv(vProj(x), state: cache?[2])
        if let cache {
            cache[0] = newQState
            cache[1] = newKState
            cache[2] = newVState
        }
        var q = qConvOut.reshaped(B, T, numHeads, headDim)
        var k = kConvOut.reshaped(B, T, numHeads, headDim)
        let v = vConvOut.reshaped(B, T, numHeads, headDim)
        q = (scale * scale) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = scale * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
        let aLogits = fbProj(faProj(x)).reshaped(B, T, numHeads, headDim)
        let bLogits = bProj(x).reshaped(B, T, numHeads)
        let (out, newSsmState) = kimiGatedDeltaUpdate(
            q: q, k: k, v: v,
            aLogits: aLogits, bLogits: bLogits,
            aLog: aLog.reshaped(numHeads, 1),
            dtBias: dtBias.reshaped(numHeads, headDim),
            state: cache?[3])
        if let cache {
            cache[3] = newSsmState
            cache.offset += T
        }
        let gate = gbProj(gaProj(x)).reshaped(B, T, numHeads, headDim)
        return oProj((oNorm(out) * sigmoid(gate)).reshaped(B, T, -1))
    }
}

// MARK: - Kimi Gated Delta Update

private func kimiGatedDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray,
    aLogits: MLXArray, bLogits: MLXArray,
    aLog: MLXArray, dtBias: MLXArray,
    state: MLXArray?
) -> (MLXArray, MLXArray) {
    let (B, T, H, Dv, Dk) = (q.dim(0), q.dim(1), q.dim(2), v.dim(3), q.dim(3))
    let g = exp(-exp(aLog) * softplus(aLogits + dtBias))
    let beta = sigmoid(bLogits)
    var s = state ?? MLXArray.zeros([B, H, Dv, Dk], dtype: q.dtype)
    var ys = [MLXArray]()
    ys.reserveCapacity(T)
    for t in 0 ..< T {
        let qt = q[0..., t]; let kt = k[0..., t]; let vt = v[0..., t]
        let gt = g[0..., t]; let betat = beta[0..., t]
        s = s * expandedDimensions(gt, axis: -2)
        let kvMem = (s * expandedDimensions(kt, axis: -2)).sum(axis: -1)
        let delta = (vt - kvMem) * expandedDimensions(betat, axis: -1)
        s = s + expandedDimensions(kt, axis: -2) * expandedDimensions(delta, axis: -1)
        ys.append((s * expandedDimensions(qt, axis: -2)).sum(axis: -1))
    }
    return (MLX.stacked(ys, axis: 1), s)
}

// MARK: - KimiSparseMoE

private class KimiSparseMoE: Module, UnaryLayer {
    let numExperts: Int
    let numExpertsPerToken: Int
    let numExpertGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let renormalize: Bool
    let scoreFunction: String

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var eScoreCorrectionBias: MLXArray

    @ModuleInfo(key: "shared_experts") var sharedExperts: KimiMLP?

    init(_ args: KimiLinearConfiguration) {
        numExperts = args.numExperts
        numExpertsPerToken = args.numExpertsPerToken
        numExpertGroup = args.numExpertGroup
        topkGroup = args.topkGroup
        routedScalingFactor = args.routedScalingFactor
        renormalize = args.moeRenormalize
        scoreFunction = args.moeRouterActivationFunc
        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize, numExperts: numExperts)
        eScoreCorrectionBias = MLXArray.zeros([numExperts])
        if args.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = KimiMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.moeIntermediateSize * args.numSharedExperts)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x)
        var scores = scoreFunction == "softmax"
            ? MLX.softmax(logits, axis: -1, precise: true)
            : sigmoid(logits)
        let origScores = scores
        scores = scores + eScoreCorrectionBias.asType(scores.dtype)
        if numExpertGroup > 1 {
            let grouped = scores.reshaped(scores.shape.dropLast() + [numExpertGroup, -1])
            let groupTop = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = numExpertGroup - topkGroup
            let groupIdx = argPartition(groupTop, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            scores = putAlong(grouped, stopGradient(groupIdx), values: MLXArray(Float(0)), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }
        let k = numExpertsPerToken
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var weights = takeAlong(origScores, inds, axis: -1)
        if k > 1 && renormalize {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor
        var out = (switchMLP(x, inds) * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts { out = out + shared(x) }
        return out
    }
}

// MARK: - KimiDecoderLayer

private class KimiDecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "self_attn") var deltaAttn: KimiDeltaAttention?
    @ModuleInfo(key: "self_attn") var mlaAttn: KimiMLAAttention?
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnLayerNorm: RMSNorm

    init(_ args: KimiLinearConfiguration, layerIdx: Int) {
        let kdaSet = Set(args.linearAttnConfig.kdaLayers)
        isLinear = kdaSet.contains(layerIdx + 1)
        if isLinear {
            _deltaAttn.wrappedValue = KimiDeltaAttention(args, layerIdx: layerIdx)
        } else {
            _mlaAttn.wrappedValue = KimiMLAAttention(args)
        }
        if args.numExperts > 0
            && layerIdx >= args.firstKDenseReplace
            && layerIdx % args.moeLayerFreq == 0
        {
            mlp = KimiSparseMoE(args)
        } else {
            mlp = KimiMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttnLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: ArraysCache?
    ) -> MLXArray {
        let attended = isLinear
            ? deltaAttn!(inputLayerNorm(x), mask: mask, cache: cache)
            : mlaAttn!(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + attended
        return h + mlp(postAttnLayerNorm(h))
    }
}

// MARK: - KimiLinearModelInner

private class KimiLinearModelInner: Module, LayerPartitionable {
    var gpuLayerCount: Int?
    var totalLayerCount: Int { layers.count }

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [KimiDecoderLayer]
    let norm: RMSNorm
    let attnLayerIdx: Int  // first MLA (full-attention) layer index

    init(_ args: KimiLinearConfiguration) {
        precondition(args.vocabSize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        layers = (0 ..< args.numHiddenLayers).map { KimiDecoderLayer(args, layerIdx: $0) }
        norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        let kdaSet = Set(args.linearAttnConfig.kdaLayers)
        attnLayerIdx = (0 ..< args.numHiddenLayers).first { !kdaSet.contains($0 + 1) } ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?[attnLayerIdx] as? ArraysCache)
        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount) {
                layer(h, mask: mask, cache: cache?[i] as? ArraysCache)
            }
        }
        return norm(h)
    }

    func callCapturing(
        _ inputs: MLXArray, cache: [KVCache?]? = nil, captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        var h = embedTokens(inputs)
        let kvCache: [KVCache?] = {
            guard let c = cache else { return Array(repeating: nil, count: layers.count) }
            var out = Array(repeating: nil as KVCache?, count: layers.count)
            for (i, v) in c.prefix(layers.count).enumerated() { out[i] = v }
            return out
        }()
        let mask = createAttentionMask(h: h, cache: kvCache[attnLayerIdx] as? ArraysCache)
        var captured: [Int: MLXArray] = [:]
        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount) {
                layer(h, mask: mask, cache: kvCache[i] as? ArraysCache)
            }
            if captureLayerIDs.contains(i) { captured[i] = h }
        }
        return (norm(h), captured)
    }
}

// MARK: - Public Model

/// Kimi linear (hybrid KDA/MLA) model owned by SwiftLM.
/// Registered for `kimi_linear` model type at DFlash setup time.
public class KimiLinearDFlashModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel,
    DFlashTargetModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let inner: KimiLinearModelInner
    private let configuration: KimiLinearConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: KimiLinearConfiguration) {
        configuration = args
        vocabularySize = args.vocabSize
        kvHeads = Array(repeating: 1, count: args.numHiddenLayers)
        inner = KimiLinearModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = inner(inputs, cache: cache)
        return lmHead.map { $0(out) } ?? inner.embedTokens.asLinear(out)
    }

    public func makeCache(parameters: GenerateParameters?) -> [any KVCache] {
        inner.layers.map { layer in
            layer.isLinear
                ? ArraysCache(size: 4)  // [q_state, k_state, v_state, ssm_state]
                : ArraysCache(size: 2)  // [kv_latent, k_pe]
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights.filter { !$0.key.hasPrefix("model.mtp") }
        if configuration.tieWordEmbeddings { w["lm_head.weight"] = nil }

        for (i, layer) in inner.layers.enumerated() {
            let prefix = "model.layers.\(i)"
            if layer.mlp is KimiSparseMoE {
                let src = "\(prefix).block_sparse_moe"
                let dst = "\(prefix).mlp"
                for (srcN, dstN) in [("w1","gate_proj"),("w2","down_proj"),("w3","up_proj")] {
                    let key0 = "\(src).experts.0.\(srcN).weight"
                    if w[key0] != nil {
                        let n = configuration.numExperts
                        let stacked = (0 ..< n).map {
                            w.removeValue(forKey: "\(src).experts.\($0).\(srcN).weight")!
                        }
                        w["\(dst).switch_mlp.\(dstN).weight"] = MLX.stacked(stacked)
                    }
                }
                for name in ["gate_proj","up_proj","down_proj"] {
                    if let v = w.removeValue(forKey: "\(src).shared_experts.\(name).weight") {
                        w["\(dst).shared_experts.\(name).weight"] = v
                    }
                }
                if let v = w.removeValue(forKey: "\(src).gate.weight") { w["\(dst).gate.weight"] = v }
                if let v = w.removeValue(forKey: "\(src).gate.e_score_correction_bias") {
                    w["\(dst).e_score_correction_bias"] = v
                }
            }
            let attnP = "\(prefix).self_attn"
            for (srcN, dstN) in [("q_conv1d","q_conv"),("k_conv1d","k_conv"),("v_conv1d","v_conv")] {
                if var convW = w.removeValue(forKey: "\(attnP).\(srcN).weight") {
                    if convW.ndim == 3 { convW = convW.transposed(0, 2, 1) }
                    w["\(attnP).\(dstN).conv.weight"] = convW
                }
            }
            if let dtW = w["\(attnP).dt_bias"], dtW.ndim > 1 {
                w["\(attnP).dt_bias"] = dtW.reshaped(-1)
            }
            if let kvB = w.removeValue(forKey: "\(attnP).kv_b_proj.weight") {
                let qkNope = configuration.resolvedQkNopeHeadDim
                let vHead = configuration.resolvedVHeadDim
                let heads = configuration.numAttentionHeads
                let r = kvB.reshaped(heads, qkNope + vHead, -1)
                w["\(attnP).embed_q.weight"] = MLX.contiguous(r[0..., ..<qkNope, 0...].transposed(-1, -2))
                w["\(attnP).unembed_out.weight"] = MLX.contiguous(r[0..., qkNope..., 0...])
            }
        }
        return w.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }

    public var loraLayers: [Module] { inner.layers }

    // MARK: DFlashTargetModel

    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        inner.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? inner.embedTokens.asLinear(hiddenStates)
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

    // Kimi linear uses ArraysCache-backed KDA + MLA layers (no GDN rollback needed).
    public var dflashIsHybridGDN: Bool { false }
}
