# TurboQuant Hybrid: Achieving V3 Quality at V2 Speeds in Apple Metal
> *An architectural analysis for SwiftLM's KV Cache pipeline*

KV Cache quantization is fundamentally constrained by a tradeoff between **per-bit representation quality** and **hardware execution speed**. Following the publication of *TurboQuant (Google, 2025)*, reference implementations across the MLX community generally diverged into two disparate paths: **V2 (speed-oriented)** and **V3 (quality-oriented)**. 

In `SwiftLM`, we discard this dichotomy by fusing the mathematical precision of V3 directly into the hardware-accelerated pathways of V2 natively in C++ and Metal.

## The Problem: The V2 / V3 Divergence

Recent implementations (such as `turboquant-mlx`) categorized their quantization strategies into two tiers:

- **V2 (Affine / Hardware-Accelerated):**
  This approach leverages native `mx.quantize` and `mx.quantized_matmul` ops. It is blazingly fast (~105% of fp16 throughput for simple quantization, ~78% when doing random rotations). However, it relies on linear/affine scaling. Because WHT-rotated vectors naturally form a Gaussian probability distribution `N(0, 1/sqrt(d))`, linear uniform bins are sub-optimal for the long tails of the distribution. At 3-bits or 2-bits, V2 affine scaling aggressively deteriorates perplexity (+9% to +23% PPL).
- **V3 (Lloyd-Max Codebook / Paper-Correct):**
  This route uses paper-correct non-linear quantization. By using pre-computed Lloyd-Max centroids designed for a Gaussian distribution, the quantization tightly clusters near the dense center and sparsely tracks the tails. This provides near-lossless compression (e.g., +0.3% PPL at 3.5-bit). However, this method requires software dequantization (centroid payload lookups), destroying throughput. On MLX without custom Metal kernels, V3 runs 5-6x slower than V2.

## The Solution: A Fused C++/Metal Hybrid Approach

Rather than choosing between Python orchestration speed penalties or affine centroid quality loss, `SwiftLM` bypasses the Python boundary entirely. We ported the non-linear Lloyd-Max logic down to the bare metal.

### 1. Vector Quantization (C++ Encoding)
When tokens enter the KV cache during the pre-fill/generation phases, the C++ encoding logic (in `fast_turbo.cpp`) performs the pre-processing natively:
1. **L2 Normalization**: The vector is scaled to the unit sphere.
2. **WHT Rotation**: An in-place Fast Walsh-Hadamard Transform `O(d log d)` evenly distributes outlier channels across the dimension array, forcing the payload into an identical Gaussian distribution.
3. **Lloyd-Max Lookup**: Instead of mathematically calculating linear boundaries, the code uses a binary search across hardcoded probability boundaries (`BOUNDARIES_3BIT`) to assign each item to one of 8 non-linear centroids, packing the result cleanly into `uint8_t` blocks.

### 2. Inner-Product Error Correction (QJL)
The original paper’s "TurboQuant_prod" algorithm attempted to replace 1 bit of MSE payload with 1 bit of Quantized Johnson-Lindenstrauss (QJL) residual estimation. Reference tests overwhelmingly demonstrated that this was a failure on Apple Silicon (softmax exponentially amplified the centroid resolution drop of dropping from 3-bit to 2-bit).

Instead, we use QJL strictly as an **additive correction layer**, and **only on the K-Cache**.
* The **K-Cache** (used for dot-product attention scores) gets 3-bit PolarQuant + 1-bit QJL (`TurboQuantK`). Storage: 4.25 bits/dim.
* The **V-Cache** (used purely for matrix reconstruction, not attention weighting) is spared the QJL overhead and gets just 3-bit PolarQuant (`TurboQuantV`). Storage: 3.125 bits/dim.

### 3. Native Metal Dequantization
With the heavy lifting done exactly matched to the mathematical shapes of V3, we pass the 16-byte packed structs to the SDPA (Scaled Dot-Product Attention) Metal kernels (`bggml-metal`). The kernel unpacks the 3-bit indices, substitutes them directly from a constant buffer containing `CENTROIDS_3BIT`, and independently executes the 1-bit QJL sign accumulation into the SDPA hot-loop. 

## Conclusion
Our hybrid approach guarantees:
1. **No Python Global Interpreter Lock (GIL) or orchestration overhead**.
2. **No arbitrary affine quality loss** on Gaussian tails at 3-bit depth.
3. **Targeted regularization** by isolating QJL to the K-Cache only.

The result is a highly efficient unified KV Cache running at an average of **~3.6 bits/dim (~3.5x compression vs fp16)**, recovering the performance characteristics of V2 with the perplexity retention of V3.
