// ChatMessage.swift — SwiftLM shared message model
// Used by both MLXInferenceCore and SwiftLM Chat UI

import Foundation

/// Represents a single turn in a chat conversation.
public struct ChatMessage: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let role: Role
    public var content: String
    public var thinkingContent: String?
    public let timestamp: Date

    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
    }

    public init(role: Role, content: String, thinkingContent: String? = nil, id: UUID = UUID(), timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.timestamp = timestamp
    }

    // Convenience constructors
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: .system, content: content)
    }
    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: .user, content: content)
    }
    public static func assistant(_ content: String, thinkingContent: String? = nil) -> ChatMessage {
        ChatMessage(role: .assistant, content: content, thinkingContent: thinkingContent)
    }
}
