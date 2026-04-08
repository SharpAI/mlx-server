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
                // 1. WAKE-UP HOOK: Offline Persona Context Injection
                var wakeUpText = ""
                let coreFacts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "CORE IDENTITY")
                let bgFacts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "BACKGROUND STORY")
                let toneFacts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "TALK TONE")
                
                var combinedIdentity = (coreFacts + bgFacts).joined(separator: "\n")
                if !toneFacts.isEmpty {
                    combinedIdentity += "\n\nCONVERSATIONAL TONE DIRECTIVE:\n" + toneFacts.joined(separator: "\n")
                }
                
                if !combinedIdentity.isEmpty {
                    wakeUpText = "SYSTEM PERSONA DIRECTIVE:\n\(combinedIdentity)\n\n"
                    dynamicSystemPrompt = wakeUpText + dynamicSystemPrompt
                }
                
                // 2. ACTIVE RAG HOOK
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
            if let firstUserIdx = fullMessages.firstIndex(where: { $0.role == .user }) {
                let originalText = fullMessages[firstUserIdx].content
                fullMessages[firstUserIdx].content = "SYSTEM DIRECTIVE & CONTEXT:\n\(dynamicSystemPrompt)\n\nUSER PROMPT:\n\(originalText)"
            } else {
                fullMessages.insert(.user("SYSTEM DIRECTIVE & CONTEXT:\n\(dynamicSystemPrompt)"), at: 0)
            }
        }

        generationTask = Task {
            var response = ""
            var thinking = ""
            var hasRawThinkTags = false

            for await token in engine.generate(messages: fullMessages, config: config) {
                guard !Task.isCancelled else { break }

                if token.isThinking {
                    thinking += token.text
                    thinkingText = thinking
                } else {
                    response += token.text
                    
                    // Fallback cleanup if the model outputs literal <think>...</think> tags
                    // and the tokenizer isn't setting the isThinking flag correctly.
                    if response.contains("<think>") {
                        hasRawThinkTags = true
                        
                        // Try to safely extract thinking content between the tags
                        if let startRange = response.range(of: "<think>"),
                           let endRange = response.range(of: "</think>") {
                            // Extract thinking
                            let rawThinking = String(response[startRange.upperBound..<endRange.lowerBound])
                            thinkingText = rawThinking
                            
                            // Remove the entire block from the visible response
                            let before = String(response[..<startRange.lowerBound])
                            let after = String(response[endRange.upperBound...])
                            streamingText = before + after
                        } else if let startRange = response.range(of: "<think>") {
                            // We have a start tag but no end tag yet, it's currently generating the thought
                            let rawThinking = String(response[startRange.upperBound...])
                            thinkingText = rawThinking
                            
                            // Only update streaming text with what came before
                            streamingText = String(response[..<startRange.lowerBound])
                        }
                    } else if !hasRawThinkTags {
                        // Standard flow: no raw tags seen yet, just stream normally
                        streamingText = response 
                    }
                }
            }

            // Commit completed message
            if !response.isEmpty {
                // Do a final cleanup just in case
                var finalVisible = response
                if let startRange = response.range(of: "<think>"),
                   let endRange = response.range(of: "</think>") {
                    let before = String(response[..<startRange.lowerBound])
                    let after = String(response[endRange.upperBound...])
                    finalVisible = before + after
                } else if let startRange = response.range(of: "<think>") {
                     finalVisible = String(response[..<startRange.lowerBound])
                }
                
                // Trim leading newlines that often follow thought blocks
                finalVisible = finalVisible.trimmingCharacters(in: .whitespacesAndNewlines)
                
                if !finalVisible.isEmpty {
                    messages.append(.assistant(finalVisible, thinkingContent: thinkingText))
                }
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
            messages.append(.assistant(streamingText, thinkingContent: thinkingText))
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
