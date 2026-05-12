// Gemma4MTPBench — Real-model MTP speculative decoding benchmark
//
// Usage:
//   swift run -c release Gemma4MTPBench
//   swift run -c release Gemma4MTPBench --main-model /path/to/e2b-4bit
//   swift run -c release Gemma4MTPBench --main-model mlx-community/gemma-4-e2b-it-4bit \
//                                        --asst-model mlx-community/gemma-4-E2B-it-assistant-bf16
//
// Safety limits baked in: maxKVSize=512, maxTokens=50, numDraft=2

import ArgumentParser
import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

// ── Tokenizer loader that wraps swift-transformers' AutoTokenizer ─────────────

struct HFTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream)
    }
}

/// Bridge: `Tokenizers.Tokenizer` → `MLXLMCommon.Tokenizer`
struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer {
    private let t: any Tokenizers.Tokenizer
    init(_ t: any Tokenizers.Tokenizer) { self.t = t }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        t.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        t.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { t.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { t.convertIdToToken(id) }
    var bosToken: String? { t.bosToken }
    var eosToken: String? { t.eosToken }
    var unknownToken: String? { t.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try t.applyChatTemplate(
                messages: messages, tools: tools,
                additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

// ── HuggingFace cache resolver ────────────────────────────────────────────────

func resolveModelPath(_ id: String) throws -> URL {
    // 1. Local path
    if id.hasPrefix("/") || id.hasPrefix("./") || id.hasPrefix("../") {
        return URL(fileURLWithPath: id)
    }
    // 2. HuggingFace cache
    let slug = "models--" + id.replacingOccurrences(of: "/", with: "--")
    let base = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".cache/huggingface/hub/\(slug)/snapshots")
    if let snap = (try? FileManager.default.contentsOfDirectory(at: base,
        includingPropertiesForKeys: nil))?.first {
        return snap
    }
    // 3. Return the id as-is (mlx-swift-lm will resolve via HubClient)
    return URL(fileURLWithPath: id)
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

@main
struct Gemma4MTPBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "Gemma4MTPBench",
        abstract: "Benchmark Gemma4 MTP speculative decoding vs. baseline on real model weights."
    )

    @Option(name: .long, help: "Main model path or HF id")
    var mainModel: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Assistant (MTP draft) model path or HF id")
    var asstModel: String = "mlx-community/gemma-4-E2B-it-assistant-bf16"

    @Option(name: .long, help: "Prompt to generate from")
    var prompt: String = "What is the capital of France? Answer in one word."

    @Option(name: .long, help: "Max tokens to generate")
    var maxTokens: Int = 50

    @Option(name: .long, help: "KV cache size (context window)")
    var maxKVSize: Int = 512

    @Option(name: .long, help: "Number of MTP draft tokens per round")
    var numDraft: Int = 2

    @Flag(name: .long, help: "Skip baseline run (faster iteration)")
    var skipBaseline: Bool = false

    mutating func run() async throws {
        // Clamping safety limits
        maxKVSize = min(max(maxKVSize, 128), 4096)
        maxTokens = min(max(maxTokens, 1), 500)
        numDraft = min(max(numDraft, 1), 8)

        print("""
        ╔═══════════════════════════════════════════════════════════╗
        ║   Gemma 4 E2B — MTP Speculative Decoding Benchmark       ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Main:      \(mainModel)
        ║  Assistant: \(asstModel)
        ║  Prompt:    "\(prompt.prefix(50))"
        ║  maxTokens=\(maxTokens)  maxKVSize=\(maxKVSize)  numDraft=\(numDraft)
        ╚═══════════════════════════════════════════════════════════╝
        """)

        let loader = HFTokenizerLoader()
        let factory = LLMModelFactory.shared

        // ── Load main model ───────────────────────────────────────────
        print("\n[1/3] Loading main model…")
        let mainURL = try resolveModelPath(mainModel)
        print("      Path: \(mainURL.path)")
        let mainCtx = try await factory.load(from: mainURL, using: loader)
        print("      ✅ Loaded: \(type(of: mainCtx.model))")

        let params = GenerateParameters(
            maxTokens: maxTokens, maxKVSize: maxKVSize, temperature: 0.0)

        let messages = [["role": "user", "content": prompt]]
        let tokens = try mainCtx.tokenizer.applyChatTemplate(messages: messages)
        let input  = LMInput(tokens: MLXArray(tokens))

        // ── Baseline ─────────────────────────────────────────────────
        var baseTPS: Double = 0
        if !skipBaseline {
            print("\n[2/3] Baseline (no speculative decoding)…")
            var baseOut = [Int]()
            let t0 = Date()
            var it = try TokenIterator(
                input: input, model: mainCtx.model,
                cache: mainCtx.model.newCache(parameters: params),
                parameters: params)
            while let tok = it.next() {
                baseOut.append(tok)
                if let eosToken = mainCtx.tokenizer.eosTokenId, tok == eosToken { break }
            }
            let elapsed = Date().timeIntervalSince(t0)
            baseTPS = Double(baseOut.count) / elapsed
            print("      Output: \"\(mainCtx.tokenizer.decode(tokenIds: baseOut).trimmingCharacters(in: .whitespacesAndNewlines).prefix(80))\"")
            print("      Speed:  \(String(format: "%.1f", baseTPS)) tok/s  (\(baseOut.count) tokens in \(String(format: "%.2f", elapsed))s)")
        }

        // ── Load assistant model ──────────────────────────────────────
        print("\n[3/3] Loading assistant model…")
        let asstURL = try resolveModelPath(asstModel)
        print("      Path: \(asstURL.path)")
        let asstCtx = try await factory.load(from: asstURL, using: loader)
        print("      ✅ Loaded: \(type(of: asstCtx.model))")

        guard let asstModel = asstCtx.model as? Gemma4AssistantModel else {
            print("\n❌ Assistant model is not Gemma4AssistantModel — got \(type(of: asstCtx.model))")
            Foundation.exit(1)
        }
        asstModel.mainModelRef = mainCtx.model
        print("      ✅ mainModelRef injected")

        // ── MTP benchmark ─────────────────────────────────────────────
        print("\n[MTP]  Running speculative decoding (numDraft=\(numDraft))…")
        var mtpOut = [Int]()
        let mtpT0 = Date()
        var mtpIt = try MTPTokenIterator(
            input: input, model: asstModel,
            cache: mainCtx.model.newCache(parameters: params),
            parameters: params, numMTPTokens: numDraft)
        while let tok = mtpIt.next() {
            mtpOut.append(tok)
            if let eosToken = mainCtx.tokenizer.eosTokenId, tok == eosToken { break }
        }
        let mtpElapsed = Date().timeIntervalSince(mtpT0)
        let mtpTPS = Double(mtpOut.count) / mtpElapsed
        let mtpText = mainCtx.tokenizer.decode(tokenIds: mtpOut)

        print("      Output: \"\(mtpText.trimmingCharacters(in: .whitespacesAndNewlines).prefix(80))\"")
        print("      Speed:  \(String(format: "%.1f", mtpTPS)) tok/s  (\(mtpOut.count) tokens in \(String(format: "%.2f", mtpElapsed))s)")

        // ── Results ───────────────────────────────────────────────────
        print("""

        ╔═══════════════════════════════════════════════════════════╗
        ║                     RESULTS                               ║
        ╠═══════════════════════════════════════════════════════════╣
        """, terminator: "")

        if !skipBaseline {
            let speedup = mtpTPS / baseTPS
            let acceptedCount = mtpIt.acceptedDraftTokens
            let totalDrafts = mtpIt.totalDraftTokens
            let acceptRate = totalDrafts > 0 ? (Double(acceptedCount) / Double(totalDrafts)) * 100.0 : 0.0

            print("""
        ║  Baseline: \(String(format: "%.1f", baseTPS)) tok/s
        ║  MTP:      \(String(format: "%.1f", mtpTPS)) tok/s
        ║  Speedup:  \(String(format: "%.2f", speedup))x
        ║  Acceptance: \(String(format: "%.1f", acceptRate))% (\(acceptedCount)/\(totalDrafts) drafts)
        ╠═══════════════════════════════════════════════════════════╣
        """, terminator: "")

            // Correctness check

            let correctOutput = mtpText.lowercased().contains("paris")
            print("""
        ║  Output correct (contains 'paris'): \(correctOutput ? "✅" : "❌")
        ║  Speedup target (≥ 1.05x):          \(speedup >= 1.05 ? "✅" : "⚠️ ") \(String(format: "%.2f", speedup))x
        ╚═══════════════════════════════════════════════════════════╝
        """)
            if speedup < 1.0 {
                print("\n⚠️  MTP is slower than baseline — check draft model quality and numDraft setting.")
            }
        } else {
            print("""
        ║  MTP:      \(String(format: "%.1f", mtpTPS)) tok/s  (baseline skipped)
        ╚═══════════════════════════════════════════════════════════╝
        """)
        }
    }
}
