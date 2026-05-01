// ChatViewModel.swift — Bridges InferenceEngine actor to SwiftUI
import SwiftUI
import Combine
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var streamingText: String = ""
    @Published var thinkingText: String? = nil
    @Published var isGenerating: Bool = false
    @Published var config: GenerationConfig = .load() {
        didSet { scheduleConfigSave() }
    }
    @Published var systemPrompt: String = {
        UserDefaults.standard.string(forKey: "swiftlm.systemPrompt") ?? ""
    }() {
        didSet { scheduleSystemPromptSave() }
    }
    public var currentWing: String? = nil
    weak var engine: InferenceEngine?
    var modelContext: ModelContext?
    private var generationTask: Task<Void, Never>?
    private var activeSession: ChatSession?

    // MARK: — Debounced persistence
    // Saves are debounced at 0.5 s so rapid slider drags or keystrokes
    // don't saturate UserDefaults with synchronous writes and cause UI jank.
    private var configSaveWork: DispatchWorkItem?
    private var systemPromptSaveWork: DispatchWorkItem?

    private func scheduleConfigSave() {
        configSaveWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.config.save() }
        configSaveWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    private func scheduleSystemPromptSave() {
        systemPromptSaveWork?.cancel()
        let snapshot = systemPrompt
        let work = DispatchWorkItem {
            UserDefaults.standard.set(snapshot, forKey: "swiftlm.systemPrompt")
        }
        systemPromptSaveWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    // MARK: — Send

    func send(_ userText: String) async {
        guard let engine, !isGenerating else { return }

        let userMessage = ChatMessage.user(userText)
        messages.append(userMessage)

        if let context = modelContext, let session = activeSession {
            let turn = ChatTurn(id: userMessage.id, roleRaw: "user", content: userMessage.content, timestamp: userMessage.timestamp, session: session)
            context.insert(turn)
            try? context.save()
        }

        isGenerating = true
        streamingText = ""
        thinkingText = nil
        
        var fullMessages = messages
        
        // 1. Prepend System Persona dynamically for the MLX Engine context (Stateless & Cache-Perfect)
        let identityPayload = await buildIdentityPayload(userText: userText)
        if !identityPayload.isEmpty {
            // Remove any existing system roles to prevent duplication
            fullMessages.removeAll { $0.role == .system }
            fullMessages.insert(ChatMessage.system(identityPayload), at: 0)
        }
        
        // Squash consecutive roles to prevent Jinja alternation crashes on strict models (e.g., Gemma)
        var collapsedMessages: [ChatMessage] = []
        for msg in fullMessages {
            if let last = collapsedMessages.last, last.role == msg.role {
                collapsedMessages[collapsedMessages.count - 1].content += "\n\n" + msg.content
            } else {
                collapsedMessages.append(msg)
            }
        }
        fullMessages = collapsedMessages

        generationTask = Task {
            var latestMessages = fullMessages
            var shouldGenerateAgain = true
            var depth = 0
            
            while shouldGenerateAgain && depth < 3 {
                shouldGenerateAgain = false
                depth += 1
                
                var response = ""
                var thinking = ""
                var hasRawThinkTags = false

                for await token in engine.generate(messages: latestMessages, config: config) {
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

                // First, check if there's a tool call in the complete response
                if let toolCall = ExtractionService.extractToolCall(from: response) {
                    // Extract text BEFORE the tool call to save as assistant message
                    if let startRange = response.range(of: "<tool_call>") {
                        let textBeforeTool = String(response[..<startRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !textBeforeTool.isEmpty {
                            let msg = ChatMessage.assistant(textBeforeTool, thinkingContent: thinkingText)
                            messages.append(msg)
                            latestMessages.append(msg)
                            if let context = modelContext, let session = activeSession {
                                let turn = ChatTurn(id: msg.id, roleRaw: "assistant", content: msg.content, thinkingContent: thinkingText, timestamp: msg.timestamp, session: session)
                                context.insert(turn)
                                try? context.save()
                            }
                        }
                    }
                    
                    // Execute tool natively!
                    do {
                        let toolResult = try await MemoryPalaceTools.handleToolCall(name: toolCall.name, arguments: toolCall.parameters ?? [:])
                        let msg = ChatMessage.tool(toolResult)
                        messages.append(msg)
                        latestMessages.append(msg)
                        // Trigger generation loop again!
                        shouldGenerateAgain = true
                        continue
                    } catch {
                        let errorMsg = ChatMessage.tool("Error executing tool: \(error.localizedDescription)")
                        messages.append(errorMsg)
                        latestMessages.append(errorMsg)
                        shouldGenerateAgain = true
                        continue
                    }
                }

                // If no tool call, commit the standard completed message
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
                        let msg = ChatMessage.assistant(finalVisible, thinkingContent: thinkingText)
                        messages.append(msg)
                        if let context = modelContext, let session = activeSession {
                            let turn = ChatTurn(id: msg.id, roleRaw: "assistant", content: msg.content, thinkingContent: thinkingText, timestamp: msg.timestamp, session: session)
                            context.insert(turn)
                            try? context.save()
                        }
                    }
                }

                streamingText = ""
                thinkingText = nil
                isGenerating = false
            } // end while
        }

        await generationTask?.value
    }

    // MARK: — Controls

    func stopGeneration() {
        generationTask?.cancel()
        if !streamingText.isEmpty {
            let msg = ChatMessage.assistant(streamingText, thinkingContent: thinkingText)
            messages.append(msg)
            if let context = modelContext, let session = activeSession {
                let turn = ChatTurn(id: msg.id, roleRaw: "assistant", content: msg.content, thinkingContent: thinkingText, timestamp: msg.timestamp, session: session)
                context.insert(turn)
                try? context.save()
            }
        }
        streamingText = ""
        thinkingText = nil
        isGenerating = false
    }

    func newConversation() {
        stopGeneration()
        
        let targetWing = currentWing ?? "CORE_SYSTEM"
        
        if let context = modelContext {
            let fetchDesc = FetchDescriptor<ChatSession>()
            let allSessions = try? context.fetch(fetchDesc)
            
            // Find session matching this wing
            let session = allSessions?.first(where: { 
                if targetWing == "CORE_SYSTEM" { return $0.wingName == nil }
                return $0.wingName == targetWing 
            })
            
            if let existing = session {
                activeSession = existing
                // Restore history chronologically
                let sortedTurns = existing.turns.sorted { $0.timestamp < $1.timestamp }
                messages = sortedTurns.map { turn in
                    let role: ChatMessage.Role = turn.roleRaw == "assistant" ? .assistant : (turn.roleRaw == "system" ? .system : .user)
                    return ChatMessage(role: role, content: turn.content, thinkingContent: turn.thinkingContent, id: turn.id, timestamp: turn.timestamp)
                }
            } else {
                // Creates the fresh session!
                let wingParam = targetWing == "CORE_SYSTEM" ? nil : targetWing
                let newSession = ChatSession(wingName: wingParam)
                context.insert(newSession)
                try? context.save()
                activeSession = newSession
                messages = []
            }
        } else {
            messages = []
        }
    }
    
    func clearHistory() {
        stopGeneration()
        messages.removeAll()
        if let context = modelContext, let session = activeSession {
            // Delete all stored turns for this session
            for turn in session.turns {
                context.delete(turn)
            }
            try? context.save()
            // Reset the loaded session state so we start fresh
            session.turns.removeAll()
        }
    }
    
    // MARK: - Tool Calling & Context Injection
    public func buildIdentityPayload(userText: String = "") async -> String {
        var wakeUpText = ""
        var activeRagDirective = ""
        
        if let wing = currentWing {
            do {
                // L0: Identity (~50 tokens, Always loaded)
                let l0Identity = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "background")
                
                // L1: Critical facts - team, projects, preferences (~120 tokens, Always loaded)
                let prefs = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "preferences")
                let facts = try MemoryPalaceService.shared.fetchRoomContents(wingName: wing, roomName: "corpus")
                
                let l1Facts = prefs + facts
                
                // Truncate L1 payload to ~120 tokens (approx 500 chars) to prevent prompt bloat 
                // matching the AAAK MemPalace layers.py spec
                var l1Text = ""
                for fact in l1Facts {
                    if l1Text.count + fact.count > 600 {
                        l1Text += "\n  ... (more in L3 search)"
                        break
                    }
                    l1Text += "\n - \(fact)"
                }
                
                wakeUpText = "[SYSTEM OVERRIDE]: YOU ARE \(wing.uppercased()). YOU ARE NOT AN AI OPEN WEIGHTS MODEL. DO NOT INTRODUCE YOURSELF AS GEMINA OR LLAMA. ALWAYS RESPOND ENTIRELY IN CHARACTER.\n\n"
                
                if !l0Identity.isEmpty {
                    wakeUpText += "[L0 IDENTITY]:\n" + l0Identity.joined(separator: "\n") + "\n\n"
                }
                
                if !l1Text.isEmpty {
                    wakeUpText += "[L1 CRITICAL FACTS & PREFERENCES]:" + l1Text + "\n\n"
                }
                
                // ACTIVE RAG HOOK
                if !userText.isEmpty {
                    let facts = try MemoryPalaceService.shared.searchMemories(query: userText, wingName: wing)
                    if !facts.isEmpty {
                        let factList = facts.map { "- [\($0.hallType)] \($0.text)" }.joined(separator: "\n")
                        activeRagDirective = "\n\n[RELEVANT MEMORY CONTEXT FOR THIS TURN]:\n\(factList)\n\nYou must strictly incorporate these facts if they are relevant here."
                    }
                }
            } catch {
                print("RAG Pre-Fetch Failed: \(error.localizedDescription)")
            }
        }
        
        let toolInjection = MemoryPalaceTools.schemaManifestString
        
        return wakeUpText + systemPrompt + activeRagDirective + "\n\n" + toolInjection
    }
}
