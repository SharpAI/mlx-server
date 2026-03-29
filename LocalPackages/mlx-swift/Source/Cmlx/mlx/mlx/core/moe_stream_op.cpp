// Copyright © 2026 SharpAI
// moe_stream_op.cpp
// Custom MLX Operation that combines GatherMM with SSD Streaming

#include "mlx/core/moe_stream_op.h"
#include <iostream>
#include <chrono>
#include "mlx/primitives.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include <iostream>

namespace mlx::core {

class StreamedGatherMM : public Primitive {
public:
    StreamedGatherMM(
        Stream s,
        uint32_t active_expert,
        std::shared_ptr<fast::SSDStreamer> streamer,
        const std::vector<off_t>& expert_offsets
    ) : Primitive(s), active_expert_(active_expert), streamer_(streamer), expert_offsets_(expert_offsets) {}
    
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& x = inputs[0];
        auto& o = outputs[0];

        uint32_t active_expert = active_expert_;
        
        // Ensure within bounds
        if (active_expert + 1 >= expert_offsets_.size()) {
            throw std::runtime_error("[StreamedGatherMM] Expert index out of bounds.");
        }
        
        off_t block_offset = expert_offsets_[active_expert];
        
        // Determine bytes for this expert slab.
        // Use the delta between consecutive offsets when available; 
        // otherwise fall back to the output-dim × input-dim × 2 estimate.
        size_t matrix_bytes;
        if (active_expert + 1 < expert_offsets_.size()) {
            matrix_bytes = static_cast<size_t>(expert_offsets_[active_expert + 1] - block_offset);
        } else {
            matrix_bytes = (x.shape().back() * o.shape().back()) * 2;
        }

        // Allocate a raw CPU buffer for the expert weights.
        // We use MLX's allocator to get Metal-accessible (unified) memory.
        // This buffer is valid until `w` goes out of scope at the end of eval_gpu.
        array w({static_cast<int>(matrix_bytes / sizeof(uint32_t))}, uint32);
        w.set_data(allocator::malloc(matrix_bytes));

        // ─────────────────────────────────────────────────────────────────────
        // LOAD — synchronous SSD read into the CPU/GPU-accessible buffer.
        //
        // This is a blocking pread(). Since `partitionedLayerCall(stream: true)`
        // calls `eval(layer_output)` + `Stream.gpu.synchronize()` PER LAYER after
        // the full layer forward pass, the GPU command buffer never accumulates
        // more than ONE layer of ops. Per-layer GPU work for 10 tokens on a
        // 122B MoE model completes in well under the 5-second Watchdog limit.
        //
        // IMPORTANT: We must NOT call commit_command_buffer() or release() on
        // the current CB inside eval_gpu(). The outer gpu::eval() in eval.cpp
        // captures the command buffer pointer BEFORE calling eval_gpu(), and it
        // will call addCompletedHandler() on that pointer AFTER we return.
        // Committing+releasing the CB here would leave gpu::eval() with a
        // ─────────────────────────────────────────────────────────────────────
        auto start_read = std::chrono::high_resolution_clock::now();
        streamer_->load_sync(block_offset, matrix_bytes, w.data<void>());
        auto end_read = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> read_ms = end_read - start_read;
        std::cout << "[SSD] Read " << (matrix_bytes / 1024.0 / 1024.0) << " MB in " << read_ms.count() << " ms\n";

        auto& d = metal::device(mlx::core::Device::gpu);

        // ─────────────────────────────────────────────────────────────────────
        // Ensure supported type and determine Metal type string
        // ─────────────────────────────────────────────────────────────────────
        std::string type_str;
        if (x.dtype() == float32) {
            type_str = "float";
        } else if (x.dtype() == float16) {
            type_str = "half";
        } else {
            throw std::runtime_error("[StreamedGatherMM] Unsupported datatype. Inputs must be cast to float16 or float32 before streaming.");
        }

        // Inline Metal source — JIT compiled on first use via MLX's runtime compile path
        std::string moe_kernel_src = R"(
#include <metal_stdlib>
using namespace metal;

kernel void streamed_moe_gemm(
    const device TYPE* x         [[buffer(0)]],
    const device uint32_t* w     [[buffer(1)]],
    device TYPE* out             [[buffer(2)]],
    constant uint& M             [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    uint packed_K = (K + 7) / 8;
    for (uint block = 0; block < packed_K; ++block) {
        uint32_t packed = w[col * packed_K + block];
        for (uint b = 0; b < 8; ++b) {
            uint ki = block * 8 + b;
            if (ki >= K) break;
            int nibble = (int)((packed >> (b * 4)) & 0xF);
            if (nibble >= 8) nibble -= 16;
            float w_val = (float)nibble / 8.0f;
            float x_val = (float)x[row * K + ki];
            sum += x_val * w_val;
        }
    }
    out[row * N + col] = (TYPE)sum;
}
)";

        // Replace "TYPE" with the actual type
        size_t pos = 0;
        while ((pos = moe_kernel_src.find("TYPE", pos)) != std::string::npos) {
            moe_kernel_src.replace(pos, 4, type_str);
            pos += type_str.length();
        }

        auto moe_lib = d.get_library("moe_stream_kernel_" + type_str, [&]() {
            return moe_kernel_src;
        });

        // Encode on the EXISTING command buffer — do NOT commit here.
        // The MLX scheduler (gpu::eval / partitionedLayerCall) manages CB lifecycle.
        auto& encoder = d.get_command_encoder(stream().index);
        auto kernel   = d.get_kernel("streamed_moe_gemm", moe_lib);
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_input_array(x, 0);
        encoder.set_input_array(w, 1);
        
        // Ensure memory is allocated for output BEFORE adding to the Metal encoder
        o.set_data(allocator::malloc(o.nbytes()));
        encoder.set_output_array(o, 2);

        uint M = static_cast<uint>(x.size() / x.shape().back());
        uint K = static_cast<uint>(x.shape().back());
        uint N = static_cast<uint>(o.shape().back());

        encoder.set_bytes(M, 3);
        encoder.set_bytes(K, 4);
        encoder.set_bytes(N, 5);

        MTL::Size grid       = MTL::Size::Make(N, M, 1);
        MTL::Size threadgroup = MTL::Size::Make(8, 8, 1);
        encoder.dispatch_threads(grid, threadgroup);

        // Keep `w` alive until the encoder moves past this op.
        // add_temporary() registers `w` with the current stream so MLX's
        // completion handler guarantees `w` is freed only after GPU runs.
        d.add_temporary(w, stream().index);
    }
    
    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        throw std::runtime_error("[StreamedGatherMM] Cannot stream to CPU. Use standard memory mapping.");
    }
    
    // Auto-differentiation is not required for inference
    std::vector<array> vjp(
        const std::vector<array>& inputs,
        const std::vector<array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<array>& outputs) override {
        throw std::runtime_error("[StreamedGatherMM] backward pass (VJP) is unsupported for streamed evaluation.");
    }

    std::vector<array> jvp(
        const std::vector<array>& inputs,
        const std::vector<array>& tangents,
        const std::vector<int>& argnums) override {
        throw std::runtime_error("[StreamedGatherMM] backward pass (JVP) is unsupported for streamed evaluation.");
    }
    
    bool is_equivalent(const Primitive& other) const override {
        return false; // Dynamic state makes this generally un-cacheable across exact same passes
    }
    
    const char* name() const override {
        return "StreamedGatherMM";
    }
    
private:
    uint32_t active_expert_;
    std::shared_ptr<fast::SSDStreamer> streamer_;
    std::vector<off_t> expert_offsets_;
};

MLX_API array streamed_gather_mm(
    const array& x,
    const array& w_shape,
    uint32_t active_expert,
    std::shared_ptr<fast::SSDStreamer> streamer,
    const std::vector<off_t>& expert_offsets,
    StreamOrDevice s
) {
    // Output shape: [tokens, outputDims]
    // w_shape is [numExperts, outputDims, inputDims] — dim(1) is the projection output size
    auto out_shape = x.shape();
    out_shape.back() = w_shape.shape(1); // outputDims, NOT .back() which would be inputDims!
    
    return array(
        out_shape, x.dtype(),
        std::make_unique<StreamedGatherMM>(to_stream(s), active_expert, streamer, expert_offsets),
        {x} // ONLY x is a graph dependency, shape and indices are static!
    );
}

} // namespace mlx::core
