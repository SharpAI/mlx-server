// DFlashProfiler.swift
//
// Simple profiler for DFlash performance analysis
// Measures timing for key operations and validates numerical consistency
// Saves results to JSON for comparison
//
// Usage: swift run DFlashProfiler [--model model-id] [--output path.json]

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import DFlash

// MARK: - Timing Utilities

struct TimingResult {
    let name: String
    let meanUs: Double
    let stdUs: Double
    let minUs: Double
    let maxUs: Double
    let iterations: Int
    
    func report() {
        print(String(format: "  %-40s %8.1f ± %6.1f µs  (min: %7.1f, max: %7.1f, n=%d)",
                     name, meanUs, stdUs, minUs, maxUs, iterations))
    }
}

func timeOperation(name: String, iterations: Int, fn: () -> Void) -> TimingResult {
    var times = [Double]()
    
    // Warmup
    for _ in 0..<3 { fn() }
    
    MLX.eval(MLXArray(0))  // Synchronize
    
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        fn()
        MLX.eval(MLXArray(0))  // Synchronize
        let end = DispatchTime.now().uptimeNanoseconds
        times.append(Double(end - start) / 1000.0)  // Convert to microseconds
    }
    
    let mean = times.reduce(0, +) / Double(times.count)
    let variance = times.reduce(0) { $0 + pow($1 - mean, 2) } / Double(times.count)
    let std = sqrt(variance)
    let minT = times.min() ?? 0
    let maxT = times.max() ?? 0
    
    return TimingResult(name: name, meanUs: mean, stdUs: std, minUs: minT, maxUs: maxT, iterations: iterations)
}

// MARK: - Benchmark Data Generation

func randomArray(shape: [Int], dtype: DType = .float32) -> MLXArray {
    let data = (0..<shape.reduce(1, *)).map { _ in Float.random(in: -1...1) }
    return MLXArray(data, shape: shape).asType(dtype)
}

// MARK: - Profiler Main

@main
struct DFlashProfiler {
    static func main() async throws {
        print("═══════════════════════════════════════════════════════════════")
        print("  DFlash Performance Profiler")
        print("  Device: \(Device.defaultDevice().description)")
        print("═══════════════════════════════════════════════════════════════")
        
        let results = await runBenchmarks()
        
        print("\n═══════════════════════════════════════════════════════════════")
        print("  BENCHMARK RESULTS")
        print("═══════════════════════════════════════════════════════════════")
        
        for r in results {
            r.report()
        }
        
        // Check if Metal kernels are being used
        print("\n═══════════════════════════════════════════════════════════════")
        print("  KERNEL AVAILABILITY CHECK")
        print("═══════════════════════════════════════════════════════════════")
        
        checkKernelAvailability()
        
        // Numerical consistency check
        print("\n═══════════════════════════════════════════════════════════════")
        print("  NUMERICAL CONSISTENCY CHECK")
        print("═══════════════════════════════════════════════════════════════")
        
        checkNumericalConsistency()
    }
    
    static func runBenchmarks() async -> [TimingResult] {
        var results = [TimingResult]()
        
        // Generate test data
        let B = 1
        let T = 16  // block size
        let Hk = 8
        let Hv = 16
        let Dk = 128
        let Dv = 128
        
        print("\nGenerating test data...")
        let tape = randomArray(shape: [B, T, Hv, Dv])
        let k = randomArray(shape: [B, T, Hk, Dk])
        let g3d = randomArray(shape: [B, T, Hv])      // 3D gate
        let g4d = randomArray(shape: [B, T, Hv, Dk])  // 4D gate
        let state = randomArray(shape: [B, Hv, Dv, Dk])
        
        let q = randomArray(shape: [B, T, Hk, Dk])
        let v = randomArray(shape: [B, T, Hv, Dv])
        let beta = randomArray(shape: [B, T, Hv])
        let mask = randomArray(shape: [B, T]).asType(.bool)
        
        print("\n── Metal Kernel Benchmarks (Tape Replay) ──")
        
        // Benchmark tape replay kernel with 3D gate
        let r3d = timeOperation(name: "tapeReplay (3D gate, Metal)", iterations: 20) {
            _ = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g3d, state: state)
        }
        results.append(r3d)
        
        // Benchmark tape replay kernel with 4D gate (vectorized)
        let r4d = timeOperation(name: "tapeReplay (4D gate, Metal)", iterations: 20) {
            _ = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g4d, state: state)
        }
        results.append(r4d)
        
        // Benchmark with mask
        let rMask = timeOperation(name: "tapeReplay (with mask)", iterations: 20) {
            _ = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g3d, state: state, mask: mask)
        }
        results.append(rMask)
        
        print("\n── Metal Kernel Benchmarks (GatedDelta with Tape) ──")
        
        // Benchmark GatedDelta with tape (3D gate)
        let gd3d = timeOperation(name: "gatedDelta (3D gate, Metal)", iterations: 20) {
            _ = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: g3d, beta: beta, state: state)
        }
        results.append(gd3d)
        
        // Benchmark GatedDelta with tape (4D gate)
        let gd4d = timeOperation(name: "gatedDelta (4D gate, Metal)", iterations: 20) {
            _ = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: g4d, beta: beta, state: state)
        }
        results.append(gd4d)
        
        print("\n── Fallback (Ops) Benchmarks ──")
        
        // Set env var to force fallback
        setenv("DFLASH_FORCE_OPS", "1", 1)
        
        let fb3d = timeOperation(name: "tapeReplay fallback (3D)", iterations: 5) {
            _ = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g3d, state: state)
        }
        results.append(fb3d)
        
        let fbgd = timeOperation(name: "gatedDelta fallback (3D)", iterations: 5) {
            _ = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: g3d, beta: beta, state: state)
        }
        results.append(fbgd)
        
        unsetenv("DFLASH_FORCE_OPS")
        
        // Benchmark ContextOnlyDraftKVCache operations
        print("\n── KV Cache Benchmarks ──")
        
        let cache = ContextOnlyDraftKVCache(sinkSize: 64, windowSize: 1024)
        let ctxK = randomArray(shape: [B, 512, Hk, Dk])
        let ctxV = randomArray(shape: [B, 512, Hv, Dv])
        
        let cacheResult = timeOperation(name: "KVCache append (512 tokens)", iterations: 20) {
            cache.appendContext(contextKeys: ctxK, contextValues: ctxV, numPositions: 512)
        }
        results.append(cacheResult)
        
        return results
    }
    
    static func checkKernelAvailability() {
        // Check if Metal is available
        let device = Device.defaultDevice()
        print("  Device type: \(device.deviceType)")
        
        // Check DFLASH_FORCE_OPS env var
        if ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil {
            print("  ⚠️  DFLASH_FORCE_OPS is set - using fallback ops")
        } else {
            print("  ✓ Metal kernels enabled (unless CPU)")
        }
        
        // Test small input to see if kernels work
        let tape = randomArray(shape: [1, 4, 8, 64])
        let k = randomArray(shape: [1, 4, 4, 64])
        let g = randomArray(shape: [1, 4, 8])
        let state = randomArray(shape: [1, 8, 64, 64])
        
        // This should use Metal if available
        do {
            let result = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g, state: state)
            eval(result)
            print("  ✓ Tape replay kernel executed successfully")
        } catch {
            print("  ❌ Tape replay kernel failed: \(error)")
        }
    }
    
    static func checkNumericalConsistency() {
        // Compare Metal kernel output vs fallback
        let tape = randomArray(shape: [1, 8, 16, 128])
        let k = randomArray(shape: [1, 8, 8, 128])
        let g3d = randomArray(shape: [1, 8, 16])
        let state = randomArray(shape: [1, 16, 128, 128])
        
        // Metal kernel result
        let metalResult = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g3d, state: state)
        
        // Fallback result
        setenv("DFLASH_FORCE_OPS", "1", 1)
        let fallbackResult = DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: g3d, state: state)
        unsetenv("DFLASH_FORCE_OPS")
        
        eval(metalResult)
        eval(fallbackResult)
        
        // Compute cosine similarity
        let cosSim = cosineSimilarityMetal(metalResult, fallbackResult)
        let maxDiff = maxAbsDiff(metalResult, fallbackResult)
        
        print(String(format: "  Metal vs Fallback: cos=%.6f, max_diff=%.6f", cosSim, maxDiff))
        
        if cosSim > 0.999 && maxDiff < 0.01 {
            print("  ✅ Numerical consistency: PASS")
        } else {
            print("  ❌ Numerical consistency: FAIL")
        }
    }
}

// MARK: - Comparison Utilities

func cosineSimilarityMetal(_ a: MLXArray, _ b: MLXArray) -> Float {
    let aF = a.reshaped(-1).asType(.float32)
    let bF = b.reshaped(-1).asType(.float32)
    let dot = (aF * bF).sum()
    let normA = MLX.sqrt((aF * aF).sum())
    let normB = MLX.sqrt((bF * bF).sum())
    return (dot / (normA * normB)).item(Float.self)
}

func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = MLX.abs(a.asType(.float32) - b.asType(.float32))
    return diff.max().item(Float.self)
}