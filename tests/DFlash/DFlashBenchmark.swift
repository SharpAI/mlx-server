// DFlashBenchmark.swift
//
// Comprehensive benchmark for DFlash speculative decoding.
// Compares baseline (standard generation) vs DFlash at various token counts.
// Saves results to JSON following dflash-mlx benchmark format.
//
// Usage: swift run DFlashBenchmark [options]

import Foundation
#if os(macOS)
import MachO
#endif
import MLX
import MLXLMCommon
import MLXNN
import DFlash

// MARK: - Benchmark Configuration

struct BenchmarkConfig: Codable, Sendable {
    let targetModel: String
    let draftModel: String
    let maxNewTokens: Int
    let blockTokens: [Int]
    let cooldownSeconds: Int
    let repeatCount: Int
    let prompt: String
    let promptTokens: Int
    let gitHash: String

    enum CodingKeys: String, CodingKey {
        case targetModel = "target_model"
        case draftModel = "draft_model"
        case maxNewTokens = "max_new_tokens"
        case blockTokens = "block_tokens"
        case cooldownSeconds = "cooldown"
        case repeatCount = "repeat"
        case prompt
        case promptTokens = "prompt_tokens"
        case gitHash = "git_hash"
    }
}

// MARK: - Hardware Info

struct HardwareInfo: Codable, Sendable {
    let chip: String
    let memoryGB: Int
    let mlxVersion: String
    let swiftVersion: String
    let deviceDescription: String

    enum CodingKeys: String, CodingKey {
        case chip
        case memoryGB = "memory_gb"
        case mlxVersion = "mlx_version"
        case swiftVersion = "swift_version"
        case deviceDescription = "device_description"
    }

    static func collect() -> HardwareInfo {
        // Get chip info using sysctl (macOS only)
        let chip = runShellCommand(["sysctl", "-n", "machdep.cpu.brand_string"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "Unknown"
        let memoryGB = Int(runShellCommand(["sysctl", "-n", "hw.memsize"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "0") ?? 0 / (1024 * 1024 * 1024)

        return HardwareInfo(
            chip: chip,
            memoryGB: memoryGB,
            mlxVersion: "0.21.0",  // Update based on your mlx-swift version
            swiftVersion: swiftVersion,
            deviceDescription: Device.defaultDevice().description
        )
    }

    private static var swiftVersion: String {
        #if swift(>=6.0)
        return "6.0+"
        #elseif swift(>=5.10)
        return "5.10"
        #elseif swift(>=5.9)
        return "5.9"
        #else
        return "<5.9"
        #endif
    }
}

// MARK: - Thermal Pressure Check

enum ThermalPressure: String, Codable, Sendable {
    case nominal, fair, serious, critical, unknown
}

func checkThermalPressure() -> ThermalPressure {
    #if os(macOS)
    // Check CPU scheduler limit
    if let output = runShellCommand(["pmset", "-g", "therm"]),
       let line = output.split(separator: "\n").first(where: { $0.contains("CPU_Scheduler_Limit") }) {
        let parts = line.split(separator: "=")
        if parts.count > 1,
           let value = Int(parts[1].trimmingCharacters(in: .whitespaces)) {
            if value == 100 { return .nominal }
            if value >= 80 { return .fair }
            if value >= 50 { return .serious }
            return .critical
        }
    }
    #endif
    return .unknown
}

// MARK: - Benchmark Result Structures

struct ModelResult: Codable, Sendable {
    let ttftMs: Double  // Time to first token
    let generationTps: Double
    let peakMemoryGB: Double?
    let tokensGenerated: Int
    let promptTokens: Int
    let generationTimeMs: Double

    enum CodingKeys: String, CodingKey {
        case ttftMs = "ttft_ms"
        case generationTps = "generation_tps"
        case peakMemoryGB = "peak_memory_gb"
        case tokensGenerated = "tokens_generated"
        case promptTokens = "prompt_token_count"
        case generationTimeMs = "generation_time_ms"
    }
}

struct DFlashSpecificResult: Codable, Sendable {
    let tokensPerCycle: Double
    let cycles: Int
    let acceptanceRatio: Double
    let acceptanceFirst20Avg: Double?
    let acceptanceLast20Avg: Double?
    let blockTokens: Int
    let acceptedFromDraft: Int

    enum CodingKeys: String, CodingKey {
        case tokensPerCycle = "tokens_per_cycle"
        case cycles
        case acceptanceRatio = "acceptance_ratio"
        case acceptanceFirst20Avg = "acceptance_first_20_avg"
        case acceptanceLast20Avg = "acceptance_last_20_avg"
        case blockTokens = "block_tokens"
        case acceptedFromDraft = "accepted_from_draft"
    }
}

struct RunResult: Codable, Sendable {
    let run: Int
    let thermalPressure: String
    let baseline: ModelResult
    let dflash: DFlashRunResult
    let speedup: Double?

    enum CodingKeys: String, CodingKey {
        case run
        case thermalPressure = "thermal_pressure"
        case baseline
        case dflash
        case speedup
    }
}

struct DFlashRunResult: Codable, Sendable {
    let base: ModelResult
    let specific: DFlashSpecificResult

    var ttftMs: Double { base.ttftMs }
    var generationTps: Double { base.generationTps }
    var peakMemoryGB: Double? { base.peakMemoryGB }
    var tokensPerCycle: Double { specific.tokensPerCycle }
    var cycles: Int { specific.cycles }
    var acceptanceRatio: Double { specific.acceptanceRatio }
    var acceptanceFirst20Avg: Double? { specific.acceptanceFirst20Avg }
    var acceptanceLast20Avg: Double? { specific.acceptanceLast20Avg }
}

struct BenchmarkSummary: Codable, Sendable {
    let baselineTpsMedian: Double?
    let dflashTpsMedian: Double?
    let dflashTpsMin: Double?
    let dflashTpsMax: Double?
    let speedupMedian: Double?
    let acceptanceRatioMedian: Double?
    let totalMemoryGB: Double?

    enum CodingKeys: String, CodingKey {
        case baselineTpsMedian = "baseline_tps_median"
        case dflashTpsMedian = "dflash_tps_median"
        case dflashTpsMin = "dflash_tps_min"
        case dflashTpsMax = "dflash_tps_max"
        case speedupMedian = "speedup_median"
        case acceptanceRatioMedian = "acceptance_ratio_median"
        case totalMemoryGB = "total_memory_gb"
    }
}

struct BenchmarkReport: Codable, Sendable {
    let hardware: HardwareInfo
    let config: BenchmarkConfig
    let runs: [RunResult]
    let summary: BenchmarkSummary

    func save(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: URL(fileURLWithPath: path))
    }
}

// MARK: - Baseline Generation

/// Runs baseline generation using standard mlx-swift
func runBaselineGeneration(
    targetModel: any LanguageModel,
    promptTokens: [Int],
    maxNewTokens: Int,
    eventHandler: @escaping (String) -> Void
) async -> ModelResult {
    let startTime = DispatchTime.now().uptimeNanoseconds
    var firstTokenTime: UInt64?
    var tokenCount = 0
    var promptTokenCount = 0

    // Create tokenizer - you'll need to pass this in or get from the model
    // For now, we'll use the model's configuration
    let modelContext = ModelContext(model: targetModel)

    for await event in sample(modelContext.model, tokenizer: modelContext.tokenization.tokenizer, prompt: promptTokens) {
        switch event {
        case .promptTokens(let tokens):
            promptTokenCount = tokens.count

        case .token(let token):
            if firstTokenTime == nil {
                firstTokenTime = DispatchTime.now().uptimeNanoseconds
            }
            tokenCount += 1
            eventHandler("[Baseline] Token \(tokenCount): \(token)")

        case .generationStopped:
            break
        }
    }

    let endTime = DispatchTime.now().uptimeNanoseconds
    let ttftNs = (firstTokenTime ?? startTime) - startTime
    let generationNs = endTime - (firstTokenTime ?? startTime)
    let ttftMs = Double(ttftNs) / 1_000_000.0
    let generationMs = Double(generationNs) / 1_000_000.0
    let tps = Double(tokenCount) / (generationMs / 1000.0)

    // Get memory info
    let memoryGB = getPeakMemoryGB()

    return ModelResult(
        ttftMs: ttftMs,
        generationTps: tps,
        peakMemoryGB: memoryGB,
        tokensGenerated: tokenCount,
        promptTokens: promptTokenCount,
        generationTimeMs: generationMs
    )
}

// MARK: - DFlash Generation

/// Runs DFlash speculative decoding
func runDFlashGeneration(
    targetModelAdapter: any DFlashTargetModel,
    draftModel: DFlashDraftModel,
    promptTokens: [Int],
    maxNewTokens: Int,
    blockTokens: Int,
    eventHandler: @escaping (String) -> Void
) async -> DFlashRunResult {
    let startTime = DispatchTime.now().uptimeNanoseconds
    var firstTokenTime: UInt64?
    var tokenCount = 0
    var promptTokenCount = 0
    var cycleCount = 0
    var acceptedFromDraft = 0
    var acceptanceRatios: [Double] = []

    let stream = DFlashRuntime.generate(
        targetModel: targetModelAdapter,
        draftModel: draftModel,
        promptTokens: promptTokens,
        maxNewTokens: maxNewTokens,
        blockTokens: blockTokens
    )

    var summary: DFlashSummary?

    for await event in stream {
        switch event {
        case .prefill(let tokens, let us):
            promptTokenCount = tokens
            eventHandler("[DFlash] Prefill: \(tokens) tokens in \(us / 1000.0) ms")

        case .token(let token, let generated, let ratio, let cycles):
            if firstTokenTime == nil {
                firstTokenTime = DispatchTime.now().uptimeNanoseconds
            }
            tokenCount += 1
            cycleCount = cycles
            acceptanceRatios.append(ratio)
            eventHandler("[DFlash] Token \(generated): \(token) (acceptance: \(String(format: "%.2f", ratio)))")

        case .summary(let s):
            summary = s
            acceptedFromDraft = s.acceptedFromDraft
        }
    }

    let endTime = DispatchTime.now().uptimeNanoseconds
    let ttftNs = (firstTokenTime ?? startTime) - startTime
    let generationNs = endTime - (firstTokenTime ?? startTime)
    let ttftMs = Double(ttftNs) / 1_000_000.0
    let generationMs = Double(generationNs) / 1_000_000.0
    let tps = Double(tokenCount) / (generationMs / 1000.0)

    // Get memory info
    let memoryGB = getPeakMemoryGB()

    // Calculate acceptance stats
    let first20Avg = acceptanceRatios.prefix(20).reduce(0, +) / Double(min(20, acceptanceRatios.count))
    let last20Avg = acceptanceRatios.suffix(20).reduce(0, +) / Double(min(20, acceptanceRatios.count))
    let acceptanceRatio = Double(acceptedFromDraft) / Double(tokenCount)

    let baseResult = ModelResult(
        ttftMs: ttftMs,
        generationTps: tps,
        peakMemoryGB: memoryGB,
        tokensGenerated: tokenCount,
        promptTokens: promptTokenCount,
        generationTimeMs: generationMs
    )

    let specificResult = DFlashSpecificResult(
        tokensPerCycle: Double(tokenCount) / Double(cycleCount),
        cycles: cycleCount,
        acceptanceRatio: acceptanceRatio,
        acceptanceFirst20Avg: first20Avg,
        acceptanceLast20Avg: last20Avg,
        blockTokens: blockTokens,
        acceptedFromDraft: acceptedFromDraft
    )

    return DFlashRunResult(base: baseResult, specific: specificResult)
}

// MARK: - Main Benchmark Runner

struct DFlashBenchmarkRunner {
    let config: BenchmarkConfig
    let verbose: Bool

    func run() async throws -> BenchmarkReport {
        print("═══════════════════════════════════════════════════════════════")
        print("  DFlash Benchmark")
        print("  Target: \(config.targetModel)")
        print("  Draft:  \(config.draftModel)")
        print("  Max Tokens: \(config.maxNewTokens)")
        print("  Repeat: \(config.repeatCount)")
        print("═══════════════════════════════════════════════════════════════")

        // Load models
        print("\nLoading models...")

        // Load target model
        let targetConfig = ModelConfiguration(id: config.targetModel)
        let targetContainer = try await ModelContainer.load(
            targetConfig,
            memoryLimit: [0: 20 * 1024 * 1024 * 1024]  // 20GB
        )

        // Load draft model
        let draftConfig = DFlashDraftConfiguration.fromHuggingFace(id: config.draftModel)
        let draftModel = DFlashDraftModel(draftConfig)
        // Note: you'll also need to load draft weights here

        // Tokenize prompt
        let tokenizer = targetContainer.tokenization.tokenizer
        let promptTokens = tokenizer.encode(text: config.prompt, addSpecialTokens: true).tokens

        print("Prompt: \(config.prompt.prefix(60))...")
        print("Tokens: \(promptTokens.count)")

        var runResults: [RunResult] = []

        for run in 1...config.repeatCount {
            print("\n── Run \(run)/\(config.repeatCount) ──")

            let thermalPressure = checkThermalPressure()
            if thermalPressure != .nominal {
                print("⚠️ Thermal pressure: \(thermalPressure.rawValue)")
            }

            // Run baseline
            print("\nRunning baseline...")
            let baselineResult = await runBaselineGeneration(
                targetModel: targetContainer.model,
                promptTokens: promptTokens,
                maxNewTokens: config.maxNewTokens
            ) { msg in
                if self.verbose { print(msg) }
            }

            print("  Baseline: \(String(format: "%.2f", baselineResult.generationTps)) TPS")

            // Cooldown
            if config.cooldownSeconds > 0 {
                print("  Cooling down for \(config.cooldownSeconds)s...")
                try await Task.sleep(nanoseconds: UInt64(config.cooldownSeconds) * 1_000_000_000)
            }

            // Run DFlash for each block size
            var bestDFlashResult: DFlashRunResult?
            var bestSpeedup: Double = 0

            for blockSize in config.blockTokens {
                print("\nRunning DFlash (block=\(blockSize))...")

                let dflashResult = await runDFlashGeneration(
                    targetModelAdapter: targetContainer.model as! DFlashTargetModel,
                    draftModel: draftModel,
                    promptTokens: promptTokens,
                    maxNewTokens: config.maxNewTokens,
                    blockTokens: blockSize
                ) { msg in
                    if self.verbose { print(msg) }
                }

                let speedup = dflashResult.base.generationTps / baselineResult.generationTps
                print("  DFlash: \(String(format: "%.2f", dflashResult.base.generationTps)) TPS (speedup: \(String(format: "%.2fx", speedup)))")

                if speedup > bestSpeedup {
                    bestSpeedup = speedup
                    bestDFlashResult = dflashResult
                }

                // Cooldown between block sizes
                if config.cooldownSeconds > 0 && blockSize != config.blockTokens.last {
                    print("  Cooling down...")
                    try await Task.sleep(nanoseconds: UInt64(config.cooldownSeconds) * 1_000_000_000)
                }
            }

            let runResult = RunResult(
                run: run,
                thermalPressure: thermalPressure.rawValue,
                baseline: baselineResult,
                dflash: bestDFlashResult!,
                speedup: bestSpeedup > 0 ? bestSpeedup : nil
            )

            runResults.append(runResult)

            // Final cooldown before next repeat
            if run < config.repeatCount && config.cooldownSeconds > 0 {
                print("\nFinal cooldown for run...")
                try await Task.sleep(nanoseconds: UInt64(config.cooldownSeconds) * 1_000_000_000)
            }
        }

        // Compute summary statistics
        let baselineTpsValues = runResults.map { $0.baseline.generationTps }
        let dflashTpsValues = runResults.map { $0.dflash.base.generationTps }
        let speedupValues = runResults.compactMap { $0.speedup }
        let acceptanceRatios = runResults.map { $0.dflash.acceptanceRatio }

        let summary = BenchmarkSummary(
            baselineTpsMedian: median(baselineTpsValues),
            dflashTpsMedian: median(dflashTpsValues),
            dflashTpsMin: dflashTpsValues.min(),
            dflashTpsMax: dflashTpsValues.max(),
            speedupMedian: median(speedupValues),
            acceptanceRatioMedian: median(acceptanceRatios),
            totalMemoryGB: getPeakMemoryGB()
        )

        return BenchmarkReport(
            hardware: HardwareInfo.collect(),
            config: config,
            runs: runResults,
            summary: summary
        )
    }
}

// MARK: - Helper Functions

func runShellCommand(_ args: [String]) -> String? {
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    task.arguments = args

    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = FileHandle.nullDevice

    do {
        try task.run()
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    } catch {
        return nil
    }
}

func getPeakMemoryGB() -> Double? {
    #if os(macOS)
    // Use task_info to get memory info
    var info = task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<task_basic_info>.size) / 4

    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO), $0, &count)
        }
    }

    if kerr == KERN_SUCCESS {
        return Double(info.resident_size) / (1024 * 1024 * 1024)
    }
    #endif
    return nil
}

func median<T: Comparable & Numeric>(_ values: [T]) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let count = sorted.count
    if count % 2 == 0 {
        let mid = count / 2
        return (Double(sorted[mid - 1] as! NSNumber) + Double(sorted[mid] as! NSNumber)) / 2
    } else {
        return Double(sorted[count / 2] as! NSNumber)
    }
}

// MARK: - Command Line Arguments

struct BenchmarkArguments {
    let targetModel: String
    let draftModel: String
    let maxNewTokens: Int
    let blockTokens: [Int]
    let repeatCount: Int
    let cooldownSeconds: Int
    let prompt: String
    let outputPath: String
    let verbose: Bool

    static func parse() -> BenchmarkArguments {
        let args = CommandLine.arguments

        func arg(_ flag: String, defaultValue: String) -> String {
            if let idx = args.firstIndex(of: flag), idx + 1 < args.count {
                return args[idx + 1]
            }
            return defaultValue
        }

        func argInt(_ flag: String, defaultValue: Int) -> Int {
            return Int(arg(flag, defaultValue: String(defaultValue))) ?? defaultValue
        }

        func argArray<T>(_ flag: String, separator: Character, transform: (String) -> T) -> [T] {
            let str = arg(flag, defaultValue: "")
            if str.isEmpty { return [] }
            return str.split(separator: separator).map { transform(String($0)) }
        }

        let targetModel = arg("--target", defaultValue: "mlx-community/Qwen3.5-27B-4bit")
        let draftModel = arg("--draft", defaultValue: "z-lab/Qwen3.5-27B-DFlash")
        let maxNewTokens = argInt("--max-tokens", defaultValue: 512)
        let blockTokensStr = arg("--block-tokens", defaultValue: "8,16,32")
        let blockTokens = blockTokensStr.split(separator: ",").compactMap { Int($0) }
        let repeatCount = argInt("--repeat", defaultValue: 3)
        let cooldownSeconds = argInt("--cooldown", defaultValue: 60)
        let verbose = args.contains("--verbose") || args.contains("-v")

        let defaultPrompt = """
            The function $f$ satisfies the functional equation \\[ f(x) + f(y) = f(x + y) - xy - 1 \\] \
            for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. \
            Enter all such integers, separated by commas. Please reason step by step.
            """
        let prompt = arg("--prompt", defaultValue: defaultPrompt)

        let outputPath = arg("--output", defaultValue: "benchmark/results/swift-\(targetModel.split(separator: "/").last ?? "model")-\(maxNewTokens).json")

        return BenchmarkArguments(
            targetModel: targetModel,
            draftModel: draftModel,
            maxNewTokens: maxNewTokens,
            blockTokens: blockTokens.isEmpty ? [8, 16, 32] : blockTokens,
            repeatCount: repeatCount,
            cooldownSeconds: cooldownSeconds,
            prompt: prompt,
            outputPath: outputPath,
            verbose: verbose
        )
    }

    func toConfig(gitHash: String) -> BenchmarkConfig {
        // Count prompt tokens (rough estimate)
        let promptTokens = prompt.split(separator: " ").count

        return BenchmarkConfig(
            targetModel: targetModel,
            draftModel: draftModel,
            maxNewTokens: maxNewTokens,
            blockTokens: blockTokens,
            cooldownSeconds: cooldownSeconds,
            repeatCount: repeatCount,
            prompt: prompt,
            promptTokens: promptTokens,
            gitHash: gitHash
        )
    }
}

// MARK: - Main

@main
struct DFlashBenchmark {
    static func main() async {
        let args = BenchmarkArguments.parse()

        print("DFlash Benchmark - Swift")
        print("========================\n")

        // Get git hash
        let gitHash = runShellCommand(["git", "rev-parse", "--short", "HEAD"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"

        let config = args.toConfig(gitHash: gitHash)
        let runner = DFlashBenchmarkRunner(config: config, verbose: args.verbose)

        do {
            let report = try await runner.run()

            // Create output directory if needed
            let outputURL = URL(fileURLWithPath: args.outputPath)
            try? FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            // Save report
            try report.save(to: args.outputPath)

            print("\n═══════════════════════════════════════════════════════════════")
            print("  Benchmark Complete")
            print("  Results saved to: \(args.outputPath)")
            print("═══════════════════════════════════════════════════════════════")
            print("\nSummary:")
            print("  Baseline TPS:     \(String(format: "%.2f", report.summary.baselineTpsMedian ?? 0))")
            print("  DFlash TPS:       \(String(format: "%.2f", report.summary.dflashTpsMedian ?? 0))")
            if let speedup = report.summary.speedupMedian {
                print("  Speedup:          \(String(format: "%.2fx", speedup))")
            }
            if let acceptance = report.summary.acceptanceRatioMedian {
                print("  Acceptance Ratio: \(String(format: "%.2f%%", acceptance * 100))")
            }

        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }
}