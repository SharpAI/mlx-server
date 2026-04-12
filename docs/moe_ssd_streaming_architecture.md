# Mixture-of-Experts (MoE) SSD Streaming Architecture

The `SwiftLM` engine features an explicit out-of-core Expert Streaming architecture, designed to execute massive routing networks (like `Qwen3.5-122B-A10B`) using fractions of the memory footprint by streaming projection matrices directly from the NVMe SSD over the PCIe bus on-demand.

## 1. The GPU Command Buffer Cycle Limit

Apple Silicon's UMA architecture relies on the GPU executing command buffers formatted by the CPU. In dense models, `SwiftLM` aggressively queues execution graphs. However, MoE models inherently trigger severe branch divergence due to token-level routing probabilities.

If an MoE graph was instantiated without breaking points, an extreme `IOAccelerator` timeout would occur (the 5-second Watchdog limit). To stabilize out-of-core streaming, `SwiftLM` explicitly blocks the CPU and drains the GPU command loop back to 0 at the generation phase utilizing:

```swift
MLX.eval(expertOutput)
Stream.gpu.synchronize() // <-- GPU Release Lock
```

### The I/O Consequences
While this protects the unified bounds, it natively prevents the main loop from fetching the *next* expert. If the CPU is waiting on `gpu.synchronize()`, standard POSIX `pread` operations block, degrading throughput.

## 2. Predictive Asynchronous Prefetch Pipeline (PAPPS)

To fully untether the SSD operations from the main loop's `Metal` synchronization locks, `SwiftLM` delegates I/O into a concurrent C++ 16-worker thread pool executing behind the `Swift` boundary.

The router queue dispatches expert indexes to the `pappsPrefetch` allocator instantaneously, entirely bypassing the main loop. The workers execute asynchronous `mmap/pread()` payloads directly to memory, maintaining raw NVMe saturation while the main thread evaluates `GPU.synchronize()`.

## 3. macOS Physical Operating Bounds

### UMA File Cache Thrashing
When testing or deploying MoE streaming, the system is fundamentally locked within the macOS **Unified Memory Swap Boundary**. The `Qwen3.5-122B` safetensors files total ~65GB. 

1. **The Hard Payload**: If the `Baseline GPU alloc` from background apps (Electron, WindowServer) occupies >12GB on a 64GB machine, loading the MoE streams guarantees a Swap overlap.
2. **Page Backing**: macOS transparently maps `safetensors` reads into the "Inactive" physical memory pool. Sustained generation will expand the mapped file footprint until it consumes 100% of available RAM.
3. **The Swap Thrashing Falloff**: Once the OS memory triggers PCIe Swap writes, the internal NVMe `pread()` latency spikes, collapsing Apple Silicon NVMe sustained bandwidth arrays and pulling generation TPS downwards.

### Defining Maximum Throughput (64GB Frameworks)

The absolute theoretical limit of MoE generation speed on Apple Silicon is purely dictated by SSD `Random Read` saturation arrays:

*   **Target Payload**: `1.84 GB / token`
*   **M1/M5 Random OS API Bandwidth**: `~3.1 GB / sec -> 3.5 GB / sec`
*   **Resulting Pipeline Ceiling**: `1.69 tok/s -> 1.84 tok/s`

No parallel decoding algorithms or I/O asynchronous loops can shatter the hardware PCIe data throughput bus limit. To maximize token generation:
1. Ensure `Baseline GPU` memory usage is as low as physically possible (closing desktop frameworks).
2. For testing isolation, utilize `sudo purge` to reset OS file caching boundaries.
3. Hardcode cache memory (`maxEntries = 8192`) to prevent the backend from forcefully inflating into Swap memory pressure over long contextual inferences.
