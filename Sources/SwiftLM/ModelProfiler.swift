// ModelProfiler.swift — Memory-aware model partitioning framework
//
// Reads model metadata (config.json), computes memory requirements,
// compares against system resources, and produces a PartitionPlan
// that guides loading strategy and memory limit configuration.
//
// Usage:
//   let profile = try ModelProfiler.profile(modelDirectory: dir)
//   let system = ModelProfiler.systemProfile()
//   let plan = ModelProfiler.plan(model: profile, system: system, contextSize: 4096)
//   ModelProfiler.printReport(plan: plan, model: profile, system: system)

import Foundation
import MLX

// MARK: - Model Profile

/// Characteristics of a model extracted from config.json and weight files.
struct ModelProfile: Sendable {
    let modelType: String
    let numLayers: Int
    let hiddenSize: Int
    let numAttentionHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let vocabSize: Int
    let quantBits: Int
    let isMoE: Bool
    let numExperts: Int?
    let numActiveExperts: Int?
    let weightFileSizeBytes: Int
    let modelId: String

    /// Estimated total parameters in billions (rough)
    var estimatedParamsB: Double {
        // Rough formula: hidden² × 12 × layers + vocab × hidden × 2
        let perLayer = Double(hiddenSize * hiddenSize) * 12.0
        let embedding = Double(vocabSize) * Double(hiddenSize) * 2.0
        let total = perLayer * Double(numLayers) + embedding
        if isMoE, let experts = numExperts {
            // MoE: FFN part is replicated across experts
            let ffnPerLayer = Double(hiddenSize) * Double(intermediateSize) * 3.0
            let densePerLayer = perLayer - ffnPerLayer
            let moeTotal = (densePerLayer + ffnPerLayer * Double(experts)) * Double(numLayers) + embedding
            return moeTotal / 1e9
        }
        return total / 1e9
    }

    /// Weight memory in GB (from actual file sizes, most accurate)
    var weightMemoryGB: Double {
        Double(weightFileSizeBytes) / 1e9
    }

    /// KV cache memory in GB for a given context length
    func kvCacheMemoryGB(contextLength: Int) -> Double {
        // KV cache = 2 (K + V) × layers × kv_heads × head_dim × context × 2 bytes (FP16)
        let bytesPerElement = 2 // FP16
        let kvBytes = 2 * numLayers * numKVHeads * headDim * contextLength * bytesPerElement
        return Double(kvBytes) / 1e9
    }

    /// Total memory required in GB (weights + KV cache + overhead)
    func totalMemoryGB(contextLength: Int) -> Double {
        let overheadFactor = 1.2 // 20% for activations, MLX buffers, scratch space
        return weightMemoryGB * overheadFactor + kvCacheMemoryGB(contextLength: contextLength)
    }

    /// Per-layer memory in GB
    var perLayerGB: Double {
        weightMemoryGB / Double(numLayers)
    }
}

// MARK: - System Profile

/// Hardware characteristics of the current machine.
struct SystemProfile: Sendable {
    let totalRAMBytes: UInt64
    let gpuArchitecture: String
    let recommendedWorkingSetBytes: Int

    var totalRAMGB: Double { Double(totalRAMBytes) / 1e9 }
    /// RAM available for the model after reserving space for macOS (~4GB)
    var availableRAMGB: Double { max(0, totalRAMGB - 4.0) }
}

// MARK: - Partition Strategy

/// The recommended loading strategy based on model size vs available memory.
enum PartitionStrategy: String, Sendable {
    /// Model fits comfortably in memory — full GPU inference
    case fullGPU = "full_gpu"
    /// Model exceeds RAM but macOS swap can handle it with degraded performance
    case swapAssisted = "swap_assisted"
    /// Model needs layer-level CPU/GPU split (future Phase 2)
    case layerPartitioned = "layer_partitioned"
    /// Model is far too large for this machine
    case tooLarge = "too_large"

    var emoji: String {
        switch self {
        case .fullGPU: return "✅"
        case .swapAssisted: return "⚠️"
        case .layerPartitioned: return "🔀"
        case .tooLarge: return "❌"
        }
    }

    var displayName: String {
        switch self {
        case .fullGPU: return "FULL GPU"
        case .swapAssisted: return "SWAP-ASSISTED"
        case .layerPartitioned: return "LAYER PARTITIONED"
        case .tooLarge: return "TOO LARGE"
        }
    }
}

// MARK: - Partition Plan

/// The computed memory partition plan for a model on the current system.
struct PartitionPlan: Sendable {
    let strategy: PartitionStrategy
    let weightMemoryGB: Double
    let kvCacheMemoryGB: Double
    let totalRequiredGB: Double
    let systemRAMGB: Double
    let availableRAMGB: Double
    let overcommitRatio: Double
    let recommendedGPULayers: Int
    let totalLayers: Int
    let recommendedMemoryLimit: Int  // bytes
    let recommendedCacheLimit: Int   // bytes
    let estimatedTokensPerSec: Double
    let warnings: [String]

    /// Actual GPU layers after partitioning (updated by server after model load)
    var gpuLayers: Int

    var fitsInMemory: Bool { strategy == .fullGPU }

    /// JSON-compatible dictionary for the /health endpoint
    var healthInfo: [String: Any] {
        [
            "strategy": strategy.rawValue,
            "overcommit_ratio": round(overcommitRatio * 100) / 100,
            "model_weight_gb": round(weightMemoryGB * 10) / 10,
            "kv_cache_gb": round(kvCacheMemoryGB * 10) / 10,
            "total_required_gb": round(totalRequiredGB * 10) / 10,
            "system_ram_gb": round(systemRAMGB * 10) / 10,
            "gpu_layers": gpuLayers,
            "cpu_layers": totalLayers - gpuLayers,
            "total_layers": totalLayers,
            "estimated_tok_s": round(estimatedTokensPerSec * 10) / 10,
        ]
    }
}

// MARK: - Model Profiler

/// Static methods for profiling models and computing partition plans.
enum ModelProfiler {

    // MARK: Config.json parsing

    /// Minimal struct to decode the parts of config.json we need
    private struct ModelConfig: Decodable {
        let modelType: String?
        let numHiddenLayers: Int?
        let hiddenSize: Int?
        let numAttentionHeads: Int?
        let numKeyValueHeads: Int?
        let headDim: Int?
        let intermediateSize: Int?
        let vocabSize: Int?
        let numExperts: Int?
        let numExpertsPerTok: Int?
        let quantizationConfig: QuantConfig?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case numHiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case vocabSize = "vocab_size"
            case numExperts = "num_local_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case quantizationConfig = "quantization_config"
        }
    }

    private struct QuantConfig: Decodable {
        let bits: Int?
        let groupSize: Int?

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    // MARK: Profiling

    /// Profile a model from its local directory (after download).
    /// Returns nil if config.json cannot be parsed.
    static func profile(modelDirectory: URL, modelId: String) -> ModelProfile? {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            return nil
        }

        guard let config = try? JSONDecoder().decode(ModelConfig.self, from: configData) else {
            return nil
        }

        let numLayers = config.numHiddenLayers ?? 32
        let hiddenSize = config.hiddenSize ?? 4096
        let numHeads = config.numAttentionHeads ?? 32
        let numKVHeads = config.numKeyValueHeads ?? numHeads
        let headDim = config.headDim ?? (hiddenSize / numHeads)
        let intermediateSize = config.intermediateSize ?? (hiddenSize * 4)
        let vocabSize = config.vocabSize ?? 32000

        // Detect quantization
        let quantBits = config.quantizationConfig?.bits ?? detectQuantBits(modelId: modelId)

        // Detect MoE
        let isMoE = config.numExperts != nil && (config.numExperts ?? 0) > 1
        let numExperts = config.numExperts
        let numActiveExperts = config.numExpertsPerTok

        // Measure weight file sizes on disk
        let weightSize = measureWeightFiles(directory: modelDirectory)

        return ModelProfile(
            modelType: config.modelType ?? "unknown",
            numLayers: numLayers,
            hiddenSize: hiddenSize,
            numAttentionHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            quantBits: quantBits,
            isMoE: isMoE,
            numExperts: numExperts,
            numActiveExperts: numActiveExperts,
            weightFileSizeBytes: weightSize,
            modelId: modelId
        )
    }

    /// Detect quantization bits from model ID string patterns.
    private static func detectQuantBits(modelId: String) -> Int {
        let lower = modelId.lowercased()
        if lower.contains("2bit") || lower.contains("q2") || lower.contains("-2b-") { return 2 }
        if lower.contains("3bit") || lower.contains("q3") { return 3 }
        if lower.contains("4bit") || lower.contains("q4") || lower.contains("int4") { return 4 }
        if lower.contains("8bit") || lower.contains("q8") || lower.contains("int8") { return 8 }
        if lower.contains("bf16") || lower.contains("fp16") { return 16 }
        if lower.contains("fp32") || lower.contains("f32") { return 32 }
        // Default to 16-bit if unknown
        return 16
    }

    /// Sum the sizes of all weight files (.safetensors) in the directory.
    /// Handles HuggingFace Hub's symlink-to-blobs structure.
    private static func measureWeightFiles(directory: URL) -> Int {
        let fm = FileManager.default
        var totalSize = 0

        // Look for .safetensors files
        if let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsSubdirectoryDescendants]) {
            for case let fileURL as URL in enumerator {
                let name = fileURL.lastPathComponent
                if name.hasSuffix(".safetensors") || name.hasSuffix(".bin") || name.hasSuffix(".gguf") {
                    // Resolve symlinks (HF Hub stores weights as symlinks to blobs/)
                    let resolvedURL = fileURL.resolvingSymlinksInPath()
                    if let attrs = try? fm.attributesOfItem(atPath: resolvedURL.path),
                       let size = attrs[.size] as? Int {
                        totalSize += size
                    }
                }
            }
        }

        // If no weight files found, estimate from config
        if totalSize == 0 {
            // Will use estimated params for weight size calculation upstream
            return 0
        }

        return totalSize
    }

    // MARK: System Profile

    /// Get the current system's hardware profile.
    static func systemProfile() -> SystemProfile {
        let totalRAM = ProcessInfo.processInfo.physicalMemory
        let deviceInfo = GPU.deviceInfo()
        // IMPORTANT: GPU.deviceInfo().memorySize returns Apple's artificially capped
        // Metal Working Set limit (~22-23GB on a 64GB M5 Pro), NOT the full physical RAM.
        // For SSD streaming mode, we must use the actual physical RAM budget:
        // 85% of total RAM, minus a 4GB OS reservation = realistic GPU pressure limit.
        let physicalBudget = Int(Double(totalRAM) * 0.85) - (4 * 1024 * 1024 * 1024)
        let recommended = max(physicalBudget, Int(deviceInfo.memorySize))

        return SystemProfile(
            totalRAMBytes: totalRAM,
            gpuArchitecture: deviceInfo.architecture,
            recommendedWorkingSetBytes: recommended
        )
    }

    // MARK: Partition Planning

    /// Compute a partition plan for the given model on the current system.
    static func plan(model: ModelProfile, system: SystemProfile, contextSize: Int) -> PartitionPlan {
        let weightGB = model.weightMemoryGB > 0
            ? model.weightMemoryGB
            : model.estimatedParamsB * (Double(model.quantBits) / 8.0)
        let kvGB = model.kvCacheMemoryGB(contextLength: contextSize)
        let overheadFactor = 1.2
        let totalGB = weightGB * overheadFactor + kvGB
        let availableGB = system.availableRAMGB
        let overcommit = totalGB / availableGB

        var warnings: [String] = []

        // Determine strategy
        let strategy: PartitionStrategy
        if totalGB <= availableGB * 0.85 {
            strategy = .fullGPU
        } else if totalGB <= availableGB {
            strategy = .fullGPU
            warnings.append("Model uses >\(Int(totalGB / availableGB * 100))% of available RAM. Performance may degrade under memory pressure.")
        } else if totalGB <= availableGB * 2.0 {
            strategy = .swapAssisted
            warnings.append("Model exceeds RAM by \(Int((overcommit - 1) * 100))%. macOS swap will be used. Expect 2-4× slowdown.")
        } else if totalGB <= availableGB * 4.0 {
            strategy = .layerPartitioned
            warnings.append("Model is \(String(format: "%.1f", overcommit))× system RAM. Layer partitioning needed for usable performance.")
            warnings.append("GPU/CPU layer split is not yet available in MLX Swift. Falling back to swap-assisted mode.")
        } else {
            strategy = .tooLarge
            warnings.append("Model is \(String(format: "%.1f", overcommit))× system RAM. Not recommended for this machine.")
            warnings.append("Consider: distributed inference (exo), smaller quantization, or more RAM.")
        }

        // MoE-specific warnings
        if model.isMoE {
            if let total = model.numExperts, let active = model.numActiveExperts {
                warnings.append("MoE model: \(active) of \(total) experts active per token. SSD streaming possible for very large MoE models.")
            }
        }

        // Compute GPU layers (for Phase 2 readiness)
        let perLayerGB = weightGB / Double(model.numLayers)
        let kvPerLayerGB = kvGB / Double(model.numLayers)
        let perLayerTotal = (perLayerGB + kvPerLayerGB) * overheadFactor
        let maxGPULayers = perLayerTotal > 0 ? Int(availableGB / perLayerTotal) : model.numLayers
        let gpuLayers = min(model.numLayers, max(0, maxGPULayers))

        // Memory limit recommendations
        let memoryLimit: Int
        let cacheLimit: Int
        switch strategy {
        case .fullGPU:
            memoryLimit = Int(Double(system.recommendedWorkingSetBytes) * 1.5)
            cacheLimit = system.recommendedWorkingSetBytes // default
        case .swapAssisted:
            memoryLimit = Int(totalGB * 1.1 * 1e9)
            cacheLimit = 2 * 1024 * 1024 // 2MB — let OS manage caching
        case .layerPartitioned:
            memoryLimit = Int(availableGB * 0.85 * 1e9)
            cacheLimit = 2 * 1024 * 1024
        case .tooLarge:
            memoryLimit = Int(availableGB * 0.85 * 1e9)
            cacheLimit = 0
        }

        // Estimate speed (rough heuristic)
        let estimatedSpeed: Double
        switch strategy {
        case .fullGPU:
            // Rough: smaller models faster. 7B ≈ 50 tok/s, 70B ≈ 15 tok/s at Q4
            estimatedSpeed = max(5, 100 / max(1, weightGB / 2))
        case .swapAssisted:
            let baseSpeed = max(5, 100 / max(1, weightGB / 2))
            estimatedSpeed = baseSpeed / min(4, overcommit * 1.5)
        case .layerPartitioned:
            // GPU layers at full speed, CPU layers at ~1/6 speed
            let gpuFrac = Double(gpuLayers) / Double(model.numLayers)
            let cpuFrac = 1.0 - gpuFrac
            let baseSpeed = max(5, 100 / max(1, weightGB / 2))
            estimatedSpeed = baseSpeed * gpuFrac + (baseSpeed / 6) * cpuFrac
        case .tooLarge:
            estimatedSpeed = 1.0
        }

        return PartitionPlan(
            strategy: strategy,
            weightMemoryGB: weightGB,
            kvCacheMemoryGB: kvGB,
            totalRequiredGB: totalGB,
            systemRAMGB: system.totalRAMGB,
            availableRAMGB: availableGB,
            overcommitRatio: overcommit,
            recommendedGPULayers: gpuLayers,
            totalLayers: model.numLayers,
            recommendedMemoryLimit: memoryLimit,
            recommendedCacheLimit: cacheLimit,
            estimatedTokensPerSec: estimatedSpeed,
            warnings: warnings,
            gpuLayers: gpuLayers  // Initially same as recommended; updated after actual partitioning
        )
    }

    // MARK: Reporting

    /// Print a formatted memory analysis report to stdout.
    static func printReport(plan: PartitionPlan, model: ModelProfile, system: SystemProfile) {
        let separator = String(repeating: "═", count: 60)
        let thinSep = String(repeating: "─", count: 56)

        print("╔\(separator)╗")
        print("║  SwiftLM Model Memory Analysis\(String(repeating: " ", count: 25))║")
        print("╠\(separator)╣")

        // Model info
        print("║\(String(repeating: " ", count: 60))║")
        printLine("Model:", model.modelId)
        printLine("Type:", "\(model.modelType)\(model.isMoE ? " (MoE)" : " (dense)")")
        printLine("Layers:", "\(model.numLayers)")
        printLine("Params:", "~\(String(format: "%.0f", model.estimatedParamsB))B")
        printLine("Quant:", "\(model.quantBits)-bit")
        if model.isMoE, let total = model.numExperts, let active = model.numActiveExperts {
            printLine("Experts:", "\(active) active / \(total) total per layer")
        }

        // Memory requirements
        print("║\(String(repeating: " ", count: 60))║")
        print("║  \(thinSep)  ║")
        print("║  Memory Requirements\(String(repeating: " ", count: 39))║")
        print("║  \(thinSep)  ║")
        printLine("Weight files:", String(format: "%.1f GB", plan.weightMemoryGB))
        printLine("KV cache:", String(format: "%.1f GB", plan.kvCacheMemoryGB))
        printLine("Overhead (~20%):", String(format: "%.1f GB", plan.weightMemoryGB * 0.2))
        printLine("Total required:", String(format: "%.1f GB", plan.totalRequiredGB))

        // System resources
        print("║\(String(repeating: " ", count: 60))║")
        print("║  \(thinSep)  ║")
        print("║  System Resources\(String(repeating: " ", count: 42))║")
        print("║  \(thinSep)  ║")
        printLine("Total RAM:", String(format: "%.1f GB", system.totalRAMGB))
        printLine("Available:", String(format: "%.1f GB", plan.availableRAMGB) + "  (after OS reservation)")
        printLine("GPU:", system.gpuArchitecture)

        // Partition plan
        print("║\(String(repeating: " ", count: 60))║")
        print("║  \(thinSep)  ║")
        print("║  Partition Plan\(String(repeating: " ", count: 44))║")
        print("║  \(thinSep)  ║")
        printLine("Strategy:", "\(plan.strategy.emoji)  \(plan.strategy.displayName)")
        printLine("Overcommit:", String(format: "%.2f×", plan.overcommitRatio) +
                  (plan.overcommitRatio > 1 ? " (model is \(Int((plan.overcommitRatio - 1) * 100))% larger than RAM)" : " (fits)"))
        printLine("GPU layers:", "\(plan.recommendedGPULayers)/\(plan.totalLayers)")
        printLine("Est. speed:", String(format: "~%.0f tok/s", plan.estimatedTokensPerSec))

        // Warnings
        if !plan.warnings.isEmpty {
            print("║\(String(repeating: " ", count: 60))║")
            print("║  \(thinSep)  ║")
            print("║  Notes\(String(repeating: " ", count: 53))║")
            print("║  \(thinSep)  ║")
            for warning in plan.warnings {
                // Word-wrap warnings to fit in the box
                let maxLen = 54
                var remaining = warning
                while !remaining.isEmpty {
                    let line: String
                    if remaining.count <= maxLen {
                        line = remaining
                        remaining = ""
                    } else {
                        let cutIdx = remaining.index(remaining.startIndex, offsetBy: maxLen)
                        if let spaceIdx = remaining[..<cutIdx].lastIndex(of: " ") {
                            line = String(remaining[..<spaceIdx])
                            remaining = String(remaining[remaining.index(after: spaceIdx)...])
                        } else {
                            line = String(remaining.prefix(maxLen))
                            remaining = String(remaining.dropFirst(maxLen))
                        }
                    }
                    let pad = String(repeating: " ", count: max(0, 56 - line.count))
                    print("║  • \(line)\(pad)║")
                }
            }
        }

        print("║\(String(repeating: " ", count: 60))║")
        print("╚\(separator)╝")
    }

    private static func printLine(_ label: String, _ value: String) {
        let labelPad = String(repeating: " ", count: max(0, 18 - label.count))
        let totalContent = "  \(label)\(labelPad)\(value)"
        let rightPad = String(repeating: " ", count: max(0, 60 - totalContent.count))
        print("║\(totalContent)\(rightPad)║")
    }
}
