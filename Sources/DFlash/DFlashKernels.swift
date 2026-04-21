// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Metal kernels for DFlash speculative decoding.
///
/// Provides:
/// - **Tape replay kernel**: Replays accepted innovation steps through the
///   GatedDeltaNet recurrent state for efficient rollback.
/// - **GatedDelta kernel with tape**: Modified GatedDelta forward that records
///   the innovation tape alongside the normal output.
/// - **Batched SDPA 2-pass kernel**: Custom attention kernel for long-context
///   verify that stays numerically aligned with stock MLX attention.
public enum DFlashKernels {

    /// Shared instance for use as the global DFlashKernelProvider
    public static let shared = DFlashKernelsInstance()

    // MARK: - Tape Replay Kernel

    private static func makeTapeReplayKernel(
        hasMask: Bool = false,
        vectorized: Bool = false
    ) -> MLXFast.MLXFastKernel? {
        let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

        let (gComment, gSetup, gAccess, gAdvance): (String, String, String, String)
        if vectorized {
            gComment = "// g: [B, T, Hv, Dk]"
            gSetup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
            gAccess = "g_[s_idx]"
            gAdvance = "g_ += Hv * Dk;"
        } else {
            gComment = "// g: [B, T, Hv]"
            gSetup = "auto g_ = g + b_idx * T * Hv;"
            gAccess = "g_[hv_idx]"
            gAdvance = "g_ += Hv;"
        }

        let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // tape: [B, T, Hv, Dv]
            auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

            // k: [B, T, Hk, Dk]
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            \(gComment)
            \(gSetup)

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                auto delta = static_cast<float>(tape_[dv_idx]);
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * \(gAccess);
                  state[i] = state[i] + k_[s_idx] * delta;
                }
                for (int i = 0; i < n_per_t; ++i) {
                  state[i] = static_cast<float>(static_cast<InT>(state[i]));
                }
              }
              tape_ += Hv * Dv;
              k_ += Hk * Dk;
              \(gAdvance)
            }

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

        var inputNames = ["tape", "k", "g", "state_in", "T"]
        if hasMask { inputNames.append("mask") }

        var suffix = ""
        if vectorized { suffix += "_vec" }
        if hasMask { suffix += "_mask" }

        return MLXFast.metalKernel(
            name: "dflash_tape_replay\(suffix)",
            inputNames: inputNames,
            outputNames: ["state_out"],
            source: source
        )
    }

    // MARK: - GatedDelta with Tape Kernel

    private static func makeGatedDeltaTapeKernel(
        hasMask: Bool = false,
        vectorized: Bool = false
    ) -> MLXFast.MLXFastKernel? {
        let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

        let (gSetup, gAccess, gAdvance): (String, String, String)
        if vectorized {
            gSetup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
            gAccess = "g_[s_idx]"
            gAdvance = "g_ += Hv * Dk;"
        } else {
            gSetup = "auto g_ = g + b_idx * T * Hv;"
            gAccess = "g_[hv_idx]"
            gAdvance = "g_ += Hv;"
        }

        let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;
            auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            \(gSetup)
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float delta = 0.0f;
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * \(gAccess);
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);
                delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              if (thread_index_in_simdgroup == 0) {
                tape_[dv_idx] = delta;
              }
              for (int i = 0; i < n_per_t; ++i) {
                state[i] = static_cast<float>(static_cast<InT>(state[i]));
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              tape_ += Hv * Dv;
              \(gAdvance)
              beta_ += Hv;
            }

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

        var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
        if hasMask { inputNames.append("mask") }

        var suffix = ""
        if vectorized { suffix += "_vec" }
        if hasMask { suffix += "_mask" }

        return MLXFast.metalKernel(
            name: "dflash_gated_delta_tape\(suffix)",
            inputNames: inputNames,
            outputNames: ["y", "state_out", "innovation_tape"],
            source: source
        )
    }

    // MARK: - Lazy Kernel Singleton

    private final class KernelCache {
        static let shared = KernelCache()

        let tapeReplayKernel: MLXFast.MLXFastKernel?
        let tapeReplayKernelMasked: MLXFast.MLXFastKernel?
        let tapeReplayKernelVec: MLXFast.MLXFastKernel?
        let tapeReplayKernelVecMasked: MLXFast.MLXFastKernel?

        let gatedDeltaTapeKernel: MLXFast.MLXFastKernel?
        let gatedDeltaTapeKernelMasked: MLXFast.MLXFastKernel?
        let gatedDeltaTapeKernelVec: MLXFast.MLXFastKernel?
        let gatedDeltaTapeKernelVecMasked: MLXFast.MLXFastKernel?

        private init() {
            tapeReplayKernel = makeTapeReplayKernel()
            tapeReplayKernelMasked = makeTapeReplayKernel(hasMask: true)
            tapeReplayKernelVec = makeTapeReplayKernel(vectorized: true)
            tapeReplayKernelVecMasked = makeTapeReplayKernel(hasMask: true, vectorized: true)

            gatedDeltaTapeKernel = makeGatedDeltaTapeKernel()
            gatedDeltaTapeKernelMasked = makeGatedDeltaTapeKernel(hasMask: true)
            gatedDeltaTapeKernelVec = makeGatedDeltaTapeKernel(vectorized: true)
            gatedDeltaTapeKernelVecMasked = makeGatedDeltaTapeKernel(hasMask: true, vectorized: true)
        }
    }

    // MARK: - Public API: Tape Replay

    /// Replay the innovation tape through the GatedDeltaNet state.
    ///
    /// - Parameters:
    ///   - tape: Innovation tape [B, T, Hv, Dv]
    ///   - k: Keys [B, T, Hk, Dk]
    ///   - g: Gates (decay) — either [B, T, Hv] or [B, T, Hv, Dk]
    ///   - state: Current recurrent state [B, Hv, Dv, Dk]
    ///   - mask: Optional mask [B, T]
    /// - Returns: Replayed state [B, Hv, Dv, Dk]
    public static func tapeReplayKernel(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let forceFallback = ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil
        let isCPU = Device.defaultDevice().deviceType == .cpu
        if isCPU || forceFallback { return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask) }

        let B = k.dim(0)
        let steps = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = tape.dim(2)
        let Dv = tape.dim(3)
        let inputType = state.dtype

        if Dk < 32 || Dk % 32 != 0 {
            return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask)
        }

        let kernel: MLXFast.MLXFastKernel?
        var inputs: [MLXArray] = [tape, k, g, state, MLXArray(steps)]
        if g.ndim == 4 {
            if let mask {
                kernel = KernelCache.shared.tapeReplayKernelVecMasked
                inputs.append(mask)
            } else {
                kernel = KernelCache.shared.tapeReplayKernelVec
            }
        } else {
            if let mask {
                kernel = KernelCache.shared.tapeReplayKernelMasked
                inputs.append(mask)
            } else {
                kernel = KernelCache.shared.tapeReplayKernel
            }
        }

        guard let kernel else {
            return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask)
        }

        let outputs = kernel(
            inputs,
            template: [
                ("InT", inputType),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [state.shape],
            outputDTypes: [inputType]
        )
        return outputs[0]
    }

    // MARK: - Public API: GatedDelta with Tape

    /// Run GatedDelta forward while recording the innovation tape for rollback.
    ///
    /// - Parameters:
    ///   - q: Queries [B, T, Hk, Dk]
    ///   - k: Keys [B, T, Hk, Dk]
    ///   - v: Values [B, T, Hv, Dv]
    ///   - g: Gates (decay) — either [B, T, Hv] or [B, T, Hv, Dk]
    ///   - beta: Beta values [B, T, Hv]
    ///   - state: Recurrent state [B, Hv, Dv, Dk]
    ///   - mask: Optional mask [B, T]
    /// - Returns: Tuple of (output [B, T, Hv, Dv], new state, innovation tape [B, T, Hv, Dv])
    public static func gatedDeltaKernelWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let forceFallback = ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil
        let isCPU = Device.defaultDevice().deviceType == .cpu
        if isCPU || forceFallback { return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask) }

        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)

        if Dk < 32 || Dk % 32 != 0 {
            return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        }

        let inputType = q.dtype
        let kernel: MLXFast.MLXFastKernel?
        var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
        if g.ndim == 4 {
            if let mask {
                kernel = KernelCache.shared.gatedDeltaTapeKernelVecMasked
                inputs.append(mask)
            } else {
                kernel = KernelCache.shared.gatedDeltaTapeKernelVec
            }
        } else {
            if let mask {
                kernel = KernelCache.shared.gatedDeltaTapeKernelMasked
                inputs.append(mask)
            } else {
                kernel = KernelCache.shared.gatedDeltaTapeKernel
            }
        }

        guard let kernel else {
            return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        }

        let outputs = kernel(
            inputs,
            template: [
                ("InT", inputType),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv]],
            outputDTypes: [inputType, inputType, DType.float32]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    // MARK: - Fallback: Ops-based implementations

    private static func tapeReplayOps(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let Hk = k.dim(2)
        let Hv = tape.dim(2)
        let repeatFactor = Hv / Hk
        var k = k
        if repeatFactor > 1 {
            k = MLX.repeated(k, count: repeatFactor, axis: 2)
        }

        var state = state
        for t in 0 ..< tape.dim(1) {
            let prev = state
            let decay: MLXArray
            if g.ndim == 4 {
                decay = g[0..., t, 0..., .newAxis, 0...]
            } else {
                decay = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            }
            let delta = tape[0..., t, 0..., .newAxis]
            let kT = expandedDimensions(k[0..., t, 0...], axis: -2)
            state = state * decay
            state = state + delta * kT
            if let mask {
                let stepMask = mask[0..., t][.newAxis, .newAxis, .newAxis]
                state = MLX.where(stepMask, state, prev)
            }
        }
        return state
    }

    private static func gatedDeltaOpsWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = q.dim(0)
        let T = q.dim(1)
        let Hk = q.dim(2)
        let Dk = q.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        let repeatFactor = Hv / Hk
        var q = q
        var k = k
        if repeatFactor > 1 {
            q = MLX.repeated(q, count: repeatFactor, axis: 2)
            k = MLX.repeated(k, count: repeatFactor, axis: 2)
        }

        var state = state
        var outputs = [MLXArray]()
        var tapeEntries = [MLXArray]()

        for t in 0 ..< T {
            let oldState = state
            let decay: MLXArray
            if g.ndim == 4 {
                decay = g[0..., t, 0..., .newAxis, 0...]
            } else {
                decay = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            }
            let decayedState = state * decay
            let kvMem = (decayedState * expandedDimensions(k[0..., t, 0...], axis: -2)).sum(axis: -1)
            let delta = (v[0..., t, 0...] - kvMem) * expandedDimensions(beta[0..., t, 0...], axis: -1)
            let newState = decayedState + expandedDimensions(k[0..., t, 0...], axis: -2) * expandedDimensions(delta, axis: -1)
            let y = (newState * expandedDimensions(q[0..., t, 0...], axis: -2)).sum(axis: -1)

            if let mask {
                let stepMask = mask[0..., t][.newAxis, .newAxis, .newAxis]
                let yMask = mask[0..., t][.newAxis, .newAxis]
                state = MLX.where(stepMask, newState, oldState)
                let maskedDelta = MLX.where(yMask, delta, MLXArray.zeros(delta.shape, dtype: delta.dtype))
                let maskedY = MLX.where(yMask, y, MLXArray.zeros(y.shape, dtype: y.dtype))
                outputs.append(maskedY)
                tapeEntries.append(maskedDelta.asType(DType.float32))
            } else {
                state = newState
                outputs.append(y)
                tapeEntries.append(delta.asType(DType.float32))
            }
        }

        return (
            MLX.stacked(outputs, axis: 1),
            state,
            MLX.stacked(tapeEntries, axis: 1)
        )
    }
}

/// Concrete DFlashKernelProvider that delegates to DFlashKernels static methods.
public final class DFlashKernelsInstance: DFlashKernelProvider, @unchecked Sendable {
    public func gatedDeltaKernelWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray,
        state: MLXArray, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        DFlashKernels.gatedDeltaKernelWithTape(
            q: q, k: k, v: v, g: g, beta: beta,
            state: state, mask: mask
        )
    }
}
