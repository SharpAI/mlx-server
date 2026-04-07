import Foundation

/// Exposes the Memory Palace tools via OpenAI-compatible tool calling schemas.
public struct MemoryPalaceTools {
    
    public static var schemas: [[String: Any]] {
        return [
            [
                "type": "function",
                "function": [
                    "name": "mempalace_save_fact",
                    "description": "Store a new factual memory, decision, or preference in the Memory Palace. Use this to permanently record important facts.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "wing": ["type": "string", "description": "The top-level AI persona or project (e.g., 'reviewer', 'orion')"],
                            "room": ["type": "string", "description": "The specific topic or concept (e.g., 'auth-migration', 'coding-style')"],
                            "type": ["type": "string", "description": "The category of memory: 'Facts', 'Events', 'Preferences', or 'Advice'"],
                            "fact": ["type": "string", "description": "The verbatim fact to store."]
                        ],
                        "required": ["wing", "room", "type", "fact"]
                    ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_search",
                    "description": "Semantically search the Memory Palace for past facts or decisions.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "wing": ["type": "string", "description": "The top-level AI persona or project (e.g., 'reviewer')"],
                            "query": ["type": "string", "description": "The semantic query to search for."]
                        ],
                        "required": ["wing", "query"]
                    ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_list_rooms",
                    "description": "List all active topics (rooms) inside a specific wing of the Memory Palace.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "wing": ["type": "string", "description": "The top-level AI persona or project (e.g., 'reviewer')"]
                        ],
                        "required": ["wing"]
                    ]
                ]
            ]
        ]
    }
    
    @MainActor
    public static func handleToolCall(name: String, arguments: [String: Any]) async throws -> String {
        switch name {
        case "mempalace_save_fact":
            guard let wing = arguments["wing"] as? String,
                  let room = arguments["room"] as? String,
                  let fact = arguments["fact"] as? String else {
                return "Error: Missing required arguments."
            }
            let type = (arguments["type"] as? String) ?? "Facts"
            try MemoryPalaceService.shared.saveMemory(wingName: wing, roomName: room, text: fact, type: type)
            return "Successfully saved fact to wing: \(wing), room: \(room)."
            
        case "mempalace_search":
            guard let wing = arguments["wing"] as? String,
                  let query = arguments["query"] as? String else {
                return "Error: Missing required arguments."
            }
            let memories = try MemoryPalaceService.shared.searchMemories(query: query, wingName: wing)
            if memories.isEmpty { return "No relevant memories found in wing: \(wing)." }
            
            var result = "Found \(memories.count) memories:\n"
            for (idx, mem) in memories.enumerated() {
                result += "[\(idx + 1)] [\(mem.hallType) | Room: \(mem.room?.name ?? "Unknown")] \(mem.text)\n"
            }
            return result
            
        case "mempalace_list_rooms":
            guard let wing = arguments["wing"] as? String else {
                return "Error: Missing required arguments."
            }
            let rooms = try MemoryPalaceService.shared.listRooms(wingName: wing)
            if rooms.isEmpty { return "No rooms found for wing: \(wing)." }
            return "Rooms in \(wing): " + rooms.joined(separator: ", ")
            
        default:
            return "Unknown tool call: \(name)"
        }
    }
}
