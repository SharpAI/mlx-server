// DFlashCosSimComparison.swift
//
// Compares intermediate values between Python and Swift DFlash implementations
// by loading Python .npy dumps and running equivalent Swift code, computing
// cosine similarity at each step.
//
// Usage: swift run DFlashCompare [--dir path/to/intermediates]

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXFast

// MARK: - NPY Loader

/// Minimal .npy loader for float32 arrays
func loadNpy(_ path: String) -> MLXArray? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        print("  ⚠️  Could not load: \(path)")
        return nil
    }
    
    // Parse numpy .npy header
    // Magic: \x93NUMPY + version + header_len + header
    guard data.count > 10,
          data[0] == 0x93,
          String(data: data[1..<6], encoding: .ascii) == "NUMPY" else {
        print("  ⚠️  Not a valid .npy file: \(path)")
        return nil
    }
    
    let majorVersion = data[6]
    let headerLen: Int
    if majorVersion == 1 {
        headerLen = Int(data[8]) | (Int(data[9]) << 8)
        let headerStart = 10
        let headerEnd = headerStart + headerLen
        
        // Parse header to get shape
        guard let headerStr = String(data: data[headerStart..<headerEnd], encoding: .ascii) else {
            return nil
        }
        
        // Extract shape from header like: {'descr': '<f4', 'shape': (1, 11, 5120), ...
        let shapeRegex = try? NSRegularExpression(pattern: "'shape': \\(([^)]*)\\)")
        let shapeMatch = shapeRegex?.firstMatch(in: headerStr, range: NSRange(headerStr.startIndex..., in: headerStr))
        
        var shape: [Int] = []
        if let match = shapeMatch,
           let range = Range(match.range(at: 1), in: headerStr) {
            let shapeStr = String(headerStr[range])
            shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            if shape.isEmpty { shape = [] }  // scalar
        }
        
        // Extract dtype
        let dtypeRegex = try? NSRegularExpression(pattern: "'descr': '([^']*)'")
        let dtypeMatch = dtypeRegex?.firstMatch(in: headerStr, range: NSRange(headerStr.startIndex..., in: headerStr))
        var isFloat32 = true
        var isInt32 = false
        if let match = dtypeMatch,
           let range = Range(match.range(at: 1), in: headerStr) {
            let dtypeStr = String(headerStr[range])
            isFloat32 = dtypeStr.contains("f4")
            isInt32 = dtypeStr.contains("i4")
        }
        
        let payloadData = data[headerEnd...]
        let totalElements = shape.reduce(1, *)
        
        if isFloat32 && totalElements * 4 <= payloadData.count {
            let floatPtr = payloadData.withUnsafeBytes { ptr in
                ptr.baseAddress!.assumingMemoryBound(to: Float.self)
            }
            let arr = MLXArray(floatPtr, shape)
            return arr
        } else if isInt32 && totalElements * 4 <= payloadData.count {
            let intPtr = payloadData.withUnsafeBytes { ptr in
                ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
            }
            return MLXArray(intPtr, shape)
        }
    } else if majorVersion == 2 {
        headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        let headerStart = 12
        let headerEnd = headerStart + headerLen
        
        guard let headerStr = String(data: data[headerStart..<headerEnd], encoding: .ascii) else {
            return nil
        }
        
        let shapeRegex = try? NSRegularExpression(pattern: "'shape': \\(([^)]*)\\)")
        let shapeMatch = shapeRegex?.firstMatch(in: headerStr, range: NSRange(headerStr.startIndex..., in: headerStr))
        
        var shape: [Int] = []
        if let match = shapeMatch,
           let range = Range(match.range(at: 1), in: headerStr) {
            let shapeStr = String(headerStr[range])
            shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }
        
        let dtypeRegex = try? NSRegularExpression(pattern: "'descr': '([^']*)'")
        let dtypeMatch = dtypeRegex?.firstMatch(in: headerStr, range: NSRange(headerStr.startIndex..., in: headerStr))
        var isFloat32 = true
        var isInt32 = false
        if let match = dtypeMatch,
           let range = Range(match.range(at: 1), in: headerStr) {
            let dtypeStr = String(headerStr[range])
            isFloat32 = dtypeStr.contains("f4")
            isInt32 = dtypeStr.contains("i4")
        }
        
        let payloadData = data[headerEnd...]
        let totalElements = shape.reduce(1, *)
        
        if isFloat32 && totalElements * 4 <= payloadData.count {
            let floatPtr = payloadData.withUnsafeBytes { ptr in
                ptr.baseAddress!.assumingMemoryBound(to: Float.self)
            }
            return MLXArray(floatPtr, shape)
        } else if isInt32 && totalElements * 4 <= payloadData.count {
            let intPtr = payloadData.withUnsafeBytes { ptr in
                ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
            }
            return MLXArray(intPtr, shape)
        }
    }
    
    print("  ⚠️  Failed to parse .npy: \(path)")
    return nil
}

// MARK: - Cosine Similarity

func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
    precondition(a.shape == b.shape, "Shape mismatch: \(a.shape) vs \(b.shape)")
    let aF = a.reshaped(-1).asType(.float32)
    let bF = b.reshaped(-1).asType(.float32)
    let dot = (aF * bF).sum()
    let normA = (aF * aF).sum()
    let normB = (bF * bF).sum()
    let denom = MLX.sqrt(normA * normB)
    let cosSim = (dot / denom).item(Float.self)
    return cosSim
}

func meanAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let aF = a.reshaped(-1).asType(.float32)
    let bF = b.reshaped(-1).asType(.float32)
    return MLX.abs(aF - bF).mean().item(Float.self)
}

// MARK: - Comparison Result

struct CompareResult {
    let name: String
    let cosSim: Float
    let mad: Float
    let shape: [Int]
    
    var pass: Bool { cosSim > 0.99 }
    
    func report() {
        let status = pass ? "✅" : "❌"
        print(String(format: "  %@ %-45s cos=%7.5f  mad=%10.6f  shape=%@", status, name, cosSim, mad, shape.map { $0.description }.joined(separator: "x")))
    }
}

// MARK: - Main Comparison

@main
struct DFlashCompare {
    static func main() async throws {
        let dir: String
        if CommandLine.arguments.count > 2 && CommandLine.arguments[1] == "--dir" {
            dir = CommandLine.arguments[2]
        } else {
            dir = URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("intermediates")
                .path
        }
        
        print("═══════════════════════════════════════════════════════════════")
        print("  DFlash Python ↔ Swift Cosine Similarity Comparison")
        print("  Intermediates dir: \(dir)")
        print("═══════════════════════════════════════════════════════════════")
        
        // Load meta
        let metaURL = URL(fileURLWithPath: dir + "/_meta.json")
        let metaData = try Data(contentsOf: metaURL)
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let promptTokens = meta["prompt_tokens"] as! [Int]
        let stagedFirst = meta["staged_first"] as! Int
        let maskTokenID = meta["mask_token_id"] as! Int
        let blockLen = meta["block_len"] as! Int
        let targetLayerIDs = meta["target_layer_ids"] as! [Int]
        let captureLayerIDs = meta["capture_layer_ids"] as! [Int]
        let draftedTokens = meta["drafted_tokens"] as! [Int]
        
        print("\nPrompt tokens: \(promptTokens)")
        print("staged_first: \(stagedFirst)")
        print("block_len: \(blockLen)")
        print("target_layer_ids: \(targetLayerIDs)")
        print("drafted_tokens (first 5): \(Array(draftedTokens.prefix(5)))")
        
        var results: [CompareResult] = []
        
        // ── Step 1: Load Python reference arrays ──
        print("\n── Loading Python reference arrays ──")
        
        func load(_ name: String) -> MLXArray? {
            return loadNpy(dir + "/" + name + ".npy")
        }
        
        guard let pyTargetHidden = load("target_hidden") else {
            print("FATAL: Could not load target_hidden")
            return
        }
        guard let pyNoiseEmbedding = load("noise_embedding") else {
            print("FATAL: Could not load noise_embedding")
            return
        }
        guard let pyProjectedHidden = load("projected_hidden") else {
            print("FATAL: Could not load projected_hidden")
            return
        }
        
        // ── Step 2: Load Swift models and run equivalent pipeline ──
        print("\n── Loading Swift models ──")
        
        // Load target model
        let targetConfig = ModelConfiguration(id: "mlx-community/Qwen3.5-27B-4bit")
        let targetContainer = try await ModelContainer.load(
            targetConfig,
            memoryLimit: [0: 20 * 1024 * 1024 * 1024]
        )
        
        // Load draft model
        let draftConfig = DFlashDraftConfiguration.fromHuggingFace(id: "z-lab/Qwen3.5-27B-DFlash")
        let draftModel = DFlashDraftModel(draftConfig)
        // TODO: load draft weights
        
        // ── Step 3: Compare step by step ──
        print("\n── Step-by-step comparison ──")
        
        // Compare target_hidden (from prefill)
        // We can't easily re-run the target model's prefill here, so compare the extracted hidden
        
        // Compare projected_hidden
        // Run Swift's projectTargetHidden on Python's target_hidden
        let swiftProjected = draftModel.projectTargetHidden(pyTargetHidden.asType(.bfloat16))
        eval(swiftProjected)
        let cosProjected = cosineSimilarity(pyProjectedHidden, swiftProjected.asType(.float32))
        let madProjected = meanAbsDiff(pyProjectedHidden, swiftProjected.asType(.float32))
        results.append(CompareResult(name: "projected_hidden", cosSim: cosProjected, mad: madProjected, shape: swiftProjected.shape.map { $0.intValue }))
        
        // Compare layer-by-layer
        for i in 0..<5 {
            // Load Python intermediates
            guard let pyAfterInputLN = load("draft_layer\(i)_after_input_ln"),
                  let pyAfterAttn = load("draft_layer\(i)_after_attn"),
                  let pyAfterMLP = load("draft_layer\(i)_after_mlp"),
                  let pyOutput = load("draft_layer\(i)_output") else {
                print("  ⚠️  Missing layer \(i) intermediates")
                continue
            }
            
            // We'll compare the Python values against each other (sanity check)
            // and also run the Swift draft model layer by layer if we can
            
            // For now, compute self-consistency and cross-layer metrics
            for (name, arr) in [
                ("draft_layer\(i)_after_input_ln", pyAfterInputLN),
                ("draft_layer\(i)_after_attn", pyAfterAttn),
                ("draft_layer\(i)_after_mlp", pyAfterMLP),
                ("draft_layer\(i)_output", pyOutput),
            ] {
                // Print stats for each Python intermediate
                let mean = arr.mean().item(Float.self)
                let maxVal = arr.max().item(Float.self)
                let minVal = arr.min().item(Float.self)
                print(String(format: "  📊 %-45s mean=%8.4f  min=%8.4f  max=%8.4f", name, mean, minVal, maxVal))
            }
        }
        
        // Compare draft_logits
        if let pyDraftLogits = load("draft_logits") {
            let pyDraftLogitsF = pyDraftLogits.asType(.float32)
            // Get top-5 tokens from Python logits at position 0
            let pos0Logits = pyDraftLogitsF[0..., 0, 0...]
            let topK = MLX.argMax(pos0Logits, axis: -1)
            print("\n  Python top token at pos 0: \(topK.item(Int32.self))")
        }
        
        // ── Summary ──
        print("\n═══════════════════════════════════════════════════════════════")
        print("  COMPARISON SUMMARY")
        print("═══════════════════════════════════════════════════════════════")
        for r in results {
            r.report()
        }
        
        let passCount = results.filter { $0.pass }.count
        let failCount = results.filter { !$0.pass }.count
        print("\n  ✅ \(passCount) passed, ❌ \(failCount) failed")
    }
}
