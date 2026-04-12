import Foundation

// Checking swift-transformers source for applyChatTemplate signature
let path = "/Users/simba/workspace/mlx-server/.build/checkouts/swift-transformers/Sources/Tokenizers/Tokenizer.swift"
if let text = try? String(contentsOfFile: path, encoding: .utf8) {
    if text.contains("addGenerationPrompt") {
        print("FOUND addGenerationPrompt")
    }
}
