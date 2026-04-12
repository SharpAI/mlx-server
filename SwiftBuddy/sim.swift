import Foundation

struct ChatMessage {
    var role: String
    var content: String
}

var messages: [ChatMessage] = []
var currentWing = "Lumina"

func simulateTurn(userText: String, isFirst: Bool) {
    print("\n--- TURN: \(userText) ---")
    messages.append(ChatMessage(role: "user", content: userText))
    
    var fullMessages = messages
    var dynamicSystemPrompt = "CORE IDENTITY: I am Lumina. "
    
    if isFirst {
        dynamicSystemPrompt += "RAG: fact 1 "
    } else {
        dynamicSystemPrompt += "RAG: fact 2 "
    }
    
    if let firstUserIdx = fullMessages.firstIndex(where: { $0.role == "user" }) {
        let originalText = fullMessages[firstUserIdx].content
        fullMessages[firstUserIdx].content = "SYSTEM DIRECTIVE:\n\(dynamicSystemPrompt)\n\nUSER:\n\(originalText)"
    }
    
    var squashed: [ChatMessage] = []
    for msg in fullMessages {
        if let last = squashed.last, last.role == msg.role {
            squashed[squashed.count - 1].content += "\n\n" + msg.content
        } else {
            squashed.append(msg)
        }
    }
    
    print("SENDING TO MLX (\(squashed.count) messages):")
    for (i, m) in squashed.enumerated() {
        print("[\(i)] \(m.role.uppercased()): \(m.content.prefix(50))...")
    }
    
    // Simulate MLX generating a response
    messages.append(ChatMessage(role: "assistant", content: "Response to \(userText)"))
}

simulateTurn(userText: "Hi", isFirst: true)
simulateTurn(userText: "How are you?", isFirst: false)
