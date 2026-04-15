// Calibrator.swift — Auto-tuning "Wisdom" system for optimal inference config
//
// FFTW-style approach: profile once per (model, hardware) pair, store optimal
// config, apply instantly on subsequent runs.
//
// On first run with a new model, the calibrator runs a short benchmark to find
// the optimal cache limit that maximizes tok/s. The result is stored in
// ~/.swiftlm/wisdom/<hash>.json and loaded directly on future runs.
//
// Usage:
//   let wisdom = try await Calibrator.calibrate(container: container, plan: plan, profile: profile)
//   // Apply: Memory.cacheLimit = wisdom.cacheLimit

import Foundation
import MLX
import MLXLMCommon
import Tokenizers

// MARK: - Wisdom Entry

/// Persisted calibration result for a specific (model, hardware) combination.
struct WisdomEntry: Codable {
    let modelId: String
    let hardwareFingerprint: String
    let cacheLimit: Int  // bytes
    let gpuLayers: Int?
    let tokPerSec: Double
    let prefillTokPerSec: Double
    let ttftMs: Double
    let memoryPeakMB: Int
    let calibratedAt: Date
    let calibrationSeconds: Double
}

// MARK: - Calibration Config

/// A single calibration trial configuration
private struct CalibrationTrial {
    let cacheLimitBytes: Int
    let label: String
}

// MARK: - Calibrator

enum Calibrator {
    
    /// Directory for wisdom files
    private static var wisdomDirectory: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".swiftlm/wisdom")
    }
    
    /// Hardware fingerprint: chip + memory + OS
    static func hardwareFingerprint() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
        let memGB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        return "\(machine)_\(memGB)GB_\(os)"
    }
    
    /// Unique key for a (model, hardware) pair
    private static func wisdomKey(modelId: String) -> String {
        let hw = hardwareFingerprint()
        let combined = "\(modelId)_\(hw)"
        // Simple hash: use the string itself, sanitized for filename
        let sanitized = combined
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "(", with: "")
            .replacingOccurrences(of: ")", with: "")
        return sanitized
    }
    
    /// Load existing wisdom for a model, if available
    static func loadWisdom(modelId: String) -> WisdomEntry? {
        let key = wisdomKey(modelId: modelId)
        let path = wisdomDirectory.appendingPathComponent("\(key).json")
        
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        
        do {
            let data = try Data(contentsOf: path)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(WisdomEntry.self, from: data)
        } catch {
            print("[SwiftLM] ⚠️  Failed to load wisdom: \(error.localizedDescription)")
            return nil
        }
    }
    
    /// Save wisdom entry to disk
    private static func saveWisdom(_ entry: WisdomEntry) throws {
        let key = wisdomKey(modelId: entry.modelId)
        let dir = wisdomDirectory
        
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        
        let path = dir.appendingPathComponent("\(key).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(entry)
        try data.write(to: path)
    }
    
    /// Run calibration: benchmark different cache limits, pick the best
    ///
    /// Runs 3-5 short inference bursts at different cache limits and
    /// measures tok/s for each. Returns the optimal configuration.
    static func calibrate(
        container: ModelContainer,
        plan: PartitionPlan,
        modelId: String,
        contextSize: Int = 4096
    ) async throws -> WisdomEntry {
        let startTime = Date()
        print("[SwiftLM] 📊 Calibrating... (this only happens once per model × hardware)")
        
        // Determine trial cache limits based on available memory
        let systemRAMBytes = Int(ProcessInfo.processInfo.physicalMemory)
        let modelWeightBytes = Int(plan.weightMemoryGB * 1e9)
        
        // Trial cache limits: from tight (just weights + 20%) to generous (50% of free RAM)
        let freeRAMBytes = systemRAMBytes - modelWeightBytes
        let trials: [CalibrationTrial] = [
            CalibrationTrial(
                cacheLimitBytes: modelWeightBytes + modelWeightBytes / 5,
                label: "tight (weights + 20%)"
            ),
            CalibrationTrial(
                cacheLimitBytes: modelWeightBytes + freeRAMBytes / 4,
                label: "moderate (weights + 25% free)"
            ),
            CalibrationTrial(
                cacheLimitBytes: modelWeightBytes + freeRAMBytes / 2,
                label: "generous (weights + 50% free)"
            ),
            CalibrationTrial(
                cacheLimitBytes: 0,  // system default (no limit)
                label: "unlimited (system default)"
            ),
        ]
        
        var bestTrial: (trial: CalibrationTrial, tokPerSec: Double, prefillTokPerSec: Double, ttft: Double)?
        
        // Calibration prompt — short enough for fast iteration, long enough to measure
        let calibrationPrompt = "Explain the concept of machine learning in three sentences."
        let maxTokens = 30  // Just enough to measure steady-state decode speed
        
        for (idx, trial) in trials.enumerated() {
            print("[SwiftLM]   Trial \(idx + 1)/\(trials.count): \(trial.label) (\(trial.cacheLimitBytes / (1024*1024))MB)")
            
            // Set cache limit for this trial
            if trial.cacheLimitBytes > 0 {
                Memory.cacheLimit = trial.cacheLimitBytes
            } else {
                // Reset to system default
                Memory.cacheLimit = 0
            }
            
            // Run inference and measure
            let result = await measureInference(
                container: container,
                prompt: calibrationPrompt,
                maxTokens: maxTokens
            )
            
            if let result = result {
                print("[SwiftLM]     → \(String(format: "%.1f", result.tokPerSec)) tok/s decode, \(String(format: "%.0f", result.ttftMs))ms TTFT")
                
                if bestTrial == nil || result.tokPerSec > bestTrial!.tokPerSec {
                    bestTrial = (trial, result.tokPerSec, result.prefillTokPerSec, result.ttftMs)
                }
            } else {
                print("[SwiftLM]     → failed, skipping")
            }
        }
        
        guard let best = bestTrial else {
            throw CalibratorError.allTrialsFailed
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        
        // Apply the winner
        if best.trial.cacheLimitBytes > 0 {
            Memory.cacheLimit = best.trial.cacheLimitBytes
        }
        
        let entry = WisdomEntry(
            modelId: modelId,
            hardwareFingerprint: hardwareFingerprint(),
            cacheLimit: best.trial.cacheLimitBytes,
            gpuLayers: plan.gpuLayers,
            tokPerSec: best.tokPerSec,
            prefillTokPerSec: best.prefillTokPerSec,
            ttftMs: best.ttft,
            memoryPeakMB: Int(Double(Memory.activeMemory) / 1e6),
            calibratedAt: Date(),
            calibrationSeconds: elapsed
        )
        
        try saveWisdom(entry)
        
        print("[SwiftLM] 📊 Calibration complete in \(String(format: "%.1f", elapsed))s")
        print("[SwiftLM]    Winner: \(best.trial.label) → \(String(format: "%.1f", best.tokPerSec)) tok/s")
        print("[SwiftLM]    Saved to ~/.swiftlm/wisdom/")
        
        return entry
    }
    
    /// Measure a single inference run
    private static func measureInference(
        container: ModelContainer,
        prompt: String,
        maxTokens: Int
    ) async -> InferenceResult? {
        do {
            // Prepare input using the same pattern as Server.swift
            let chatMessages: [Chat.Message] = [.user(prompt)]
            let userInput = UserInput(chat: chatMessages)
            let lmInput = try await container.prepare(input: userInput)
            let inputTokenCount = lmInput.text.tokens.size
            
            let result: InferenceResult = try await container.perform { context in
                var generateParams = GenerateParameters(temperature: 0.6)
                generateParams.topP = 1.0
                generateParams.topK = 50
                generateParams.minP = 0.0
                
                let ttftStart = Date()
                var firstTokenTime: Date?
                var tokenCount = 0
                
                for try await result in try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: generateParams,
                    context: context
                ) {
                    switch result {
                    case .chunk(_, tokenId: _):
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                        }
                        tokenCount += 1
                        if tokenCount >= maxTokens {
                            break
                        }
                    default:
                        break
                    }
                    if tokenCount >= maxTokens { break }
                }
                
                let ttft = firstTokenTime?.timeIntervalSince(ttftStart) ?? 0
                let decodeTime = Date().timeIntervalSince(firstTokenTime ?? ttftStart)
                let tokPerSec = decodeTime > 0 && tokenCount > 1 ? Double(tokenCount - 1) / decodeTime : 0
                let prefillTokPerSec = ttft > 0 ? Double(inputTokenCount) / ttft : 0
                
                return InferenceResult(
                    tokPerSec: tokPerSec,
                    prefillTokPerSec: prefillTokPerSec,
                    ttftMs: ttft * 1000,
                    tokenCount: tokenCount
                )
            }
            
            return result
        } catch {
            return nil
        }
    }
}

// MARK: - Supporting Types

private struct InferenceResult {
    let tokPerSec: Double
    let prefillTokPerSec: Double
    let ttftMs: Double
    let tokenCount: Int
}

enum CalibratorError: Error {
    case allTrialsFailed
}
