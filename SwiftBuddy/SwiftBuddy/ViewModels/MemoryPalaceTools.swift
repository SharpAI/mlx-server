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
                            "type": ["type": "string", "description": "The category of memory: 'Facts', 'Events', 'Discoveries', 'Preferences', or 'Advice'"],
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
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_list_wings",
                    "description": "List all top-level wings in the Memory Palace.",
                    "parameters": [ "type": "object", "properties": [:] ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_get_taxonomy",
                    "description": "Get a full hierarchical tree of the entire Palace (Wings -> Rooms).",
                    "parameters": [ "type": "object", "properties": [:] ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_status",
                    "description": "Get statistical counts of wings, rooms, and memories across the entire Palace.",
                    "parameters": [ "type": "object", "properties": [:] ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_delete_drawer",
                    "description": "Delete a specific factual memory out of the Palace.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "wing": ["type": "string", "description": "The wing name"],
                            "room": ["type": "string", "description": "Optional room name"],
                            "fact": ["type": "string", "description": "A semantic match to the text you want removed (e.g. 'Stripe is used')."]
                        ],
                        "required": ["wing", "fact"]
                    ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_get_closet",
                    "description": "Fetch all facts inside a room to understand its full context.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "wing": ["type": "string"],
                            "room": ["type": "string"]
                        ],
                        "required": ["wing", "room"]
                    ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_kg_add",
                    "description": "Add a structured entity relationship (triple) into the Temporal Knowledge Graph.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "subject": ["type": "string"],
                            "predicate": ["type": "string"],
                            "object": ["type": "string"]
                        ],
                        "required": ["subject", "predicate", "object"]
                    ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_kg_query",
                    "description": "Query all properties and relations mapped to a specific entity.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "entity": ["type": "string"]
                        ],
                        "required": ["entity"]
                    ]
                ]
            ],
            [
                "type": "function",
                "function": [
                    "name": "mempalace_diary_save",
                    "description": "Save an internal agent reflection into the hardcoded 'diary' room of a specific wing.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "wing": ["type": "string"],
                            "reflection": ["type": "string"]
                        ],
                        "required": ["wing", "reflection"]
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
            
        case "mempalace_list_wings":
            let wings = try MemoryPalaceService.shared.listWings()
            if wings.isEmpty { return "The Palace is empty. No wings found." }
            return "Wings: " + wings.joined(separator: ", ")
            
        case "mempalace_delete_drawer":
            guard let wing = arguments["wing"] as? String,
                  let fact = arguments["fact"] as? String else { return "Error: Missing required arguments." }
            let roomName = arguments["room"] as? String
            try MemoryPalaceService.shared.deleteMemory(wingName: wing, roomName: roomName, textMatch: fact)
            return "Attempted to delete memory matching fact."
            
        case "mempalace_status":
            let stats = try MemoryPalaceService.shared.getPalaceStatus()
            return "Palace Status: \(stats.wings) Wings, \(stats.rooms) Rooms, \(stats.memories) Memories."
            
        case "mempalace_get_taxonomy":
            return try MemoryPalaceService.shared.getTaxonomy()
            
        case "mempalace_get_closet":
            guard let wing = arguments["wing"] as? String,
                  let room = arguments["room"] as? String else { return "Error: Missing arguments" }
            return try MemoryPalaceService.shared.getCloset(wingName: wing, roomName: room)
            
        case "mempalace_kg_add":
            guard let subject = arguments["subject"] as? String,
                  let predicate = arguments["predicate"] as? String,
                  let object = arguments["object"] as? String else { return "Error: Missing KG triple arguments" }
            try MemoryPalaceService.shared.addTriple(subject: subject, predicate: predicate, object: object)
            return "Knowledge Graph Triple Saved: [\(subject)] - \(predicate) -> [\(object)]"
            
        case "mempalace_kg_query":
            guard let entity = arguments["entity"] as? String else { return "Error: Missing entity argument" }
            let triples = try MemoryPalaceService.shared.queryEntity(entity)
            if triples.isEmpty { return "No knowledge properties found for entity: \(entity)" }
            let lines = triples.map { "- \($0.predicate): \($0.object)" }
            return "Entity: \(entity)\n" + lines.joined(separator: "\n")
            
        case "mempalace_diary_save":
            guard let wing = arguments["wing"] as? String,
                  let text = arguments["reflection"] as? String else { return "Error: Missing diary arguments" }
            try MemoryPalaceService.shared.saveMemory(wingName: wing, roomName: "diary", text: text, type: "hall_events")
            return "Saved reflection to diary."
            
        default:
            return "Unknown tool call: \(name)"
        }
    }
}
