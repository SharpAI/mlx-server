import Foundation
// Simulate the ChatViewModel state
var fullMessages: [[String: String]] = [
    ["role": "user", "content": "Hi"],
    ["role": "assistant", "content": "Hello! How can I help you today?"],
    ["role": "user", "content": "What's up?"]
]

print("Simulating squashing...")
// Simulating the squashing logic
var squashed: [[String: String]] = []
for msg in fullMessages {
    if let last = squashed.last, last["role"] == msg["role"] {
        squashed[squashed.count - 1]["content"]! += "\n\n" + msg["content"]!
    } else {
        squashed.append(msg)
    }
}
print(squashed)

// Simulating the next turn where you send another message IMMEDIATELY:
var crashedMessages = [
    ["role": "user", "content": "Hi"],
    ["role": "user", "content": "What's up?"]
]
var squashedCrashed: [[String: String]] = []
for msg in crashedMessages {
    if let last = squashedCrashed.last, last["role"] == msg["role"] {
        squashedCrashed[squashedCrashed.count - 1]["content"]! += "\n\n" + msg["content"]!
    } else {
        squashedCrashed.append(msg)
    }
}
print(squashedCrashed)
