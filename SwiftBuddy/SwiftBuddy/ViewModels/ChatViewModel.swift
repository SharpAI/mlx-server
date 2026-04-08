// ChatViewModel.swift — Bridges InferenceEngine actor to SwiftUI
import SwiftUI
import Combine
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var streamingText: String = ""
    @Published var thinkingText: String? = nil
    @Published var isGenerating: Bool = false
    @Published var config: GenerationConfig = .default
    @Published var systemPrompt: String = ""
    public var currentWing: String? = nil

    weak var engine: InferenceEngine?
    private var generationTask: Task<Void, Never>?

    // MARK: — Send

    func send(_ userText: String) async {
        guard let engine, !isGenerating else { return }

        let userMessage = ChatMessage.user(userText)
        messages.append(userMessage)

        isGenerating = true
        streamingText = ""
        thinkingText = nil
        
        // --- INVISIBLE RAG INJECTION ---
        var dynamicSystemPrompt = systemPrompt
        if let wing = currentWing, !wing.isEmpty {
            do {
                let facts = try MemoryPalaceService.shared.searchMemories(query: userText, wingName: wing)
                if !facts.isEmpty {
                    let factList = facts.map { "- [\($0.hallType)] \($0.text)" }.joined(separator: "\n")
                    let ragDirective = "\n\nCRITICAL CONTEXT FROM MEMORY PALACE:\n\(factList)\n\nYou must explicitly use these facts to answer if they are relevant to the user query."
                    dynamicSystemPrompt += ragDirective
                }
            } catch {
                print("RAG Pre-Fetch Failed: \(error.localizedDescription)")
            }
        }

        var fullMessages = messages
        if !dynamicSystemPrompt.isEmpty {
            fullMessages.insert(.system(dynamicSystemPrompt), at: 0)
        }

        generationTask = Task {
            var response = ""
            var thinking = ""
            var inThinkBlock = false

            for await token in engine.generate(messages: fullMessages, config: config) {
                guard !Task.isCancelled else { break }

                if token.isThinking {
                    thinking += token.text
                    thinkingText = thinking
                } else {
                    // Strip any residual </think> tag from visible output
                    var visible = token.text
                    if visible.contains("</think>") {
                        visible = visible.replacingOccurrences(of: "</think>", with: "")
                        inThinkBlock = false
                    }
                    if !inThinkBlock {
                        response += visible
                        streamingText = response
                    }
                }
            }

            // Commit completed message
            if !response.isEmpty {
                messages.append(.assistant(response))
            }

            streamingText = ""
            thinkingText = nil
            isGenerating = false
        }

        await generationTask?.value
    }

    // MARK: — Controls

    func stopGeneration() {
        generationTask?.cancel()
        if !streamingText.isEmpty {
            messages.append(.assistant(streamingText))
        }
        streamingText = ""
        thinkingText = nil
        isGenerating = false
    }

    func newConversation() {
        stopGeneration()
        messages = []
    }
}
