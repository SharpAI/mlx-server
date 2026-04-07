import Foundation
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ExtractedMemory: Codable {
    let room: String
    let hall: String
    let fact: String
}

struct ExtractionPayload: Codable {
    let extractions: [ExtractedMemory]
}

@MainActor
public final class ExtractionService: ObservableObject {
    public static let shared = ExtractionService()
    
    @Published public var isMining: Bool = false
    @Published public var lastLog: String = ""
    
    private init() {}
    
    /// Silently prompts the local InferenceEngine to map unstructured text to JSON facts,
    /// then persists it into the targeted Wing via SwiftData.
    public func mine(textBlock: String, wing: String, engine: InferenceEngine) async {
        guard !isMining else { return }
        isMining = true
        lastLog = "Starting Extraction for Wing: \(wing)..."
        
        let systemPrompt = """
        You are a Memory Palace extraction engine.
        Analyze the following raw text. Identify highly specific facts, events, and biographical preferences.
        OUTPUT STRICTLY IN THE FOLLOWING JSON FORMAT ONLY. NEVER Output conversational text.
        
        {
          "extractions": [
            {
              "room": "Topic Category (e.g., 'Career', 'Physics', 'Personal')",
              "hall": "Category Type (must be either: 'hall_facts', 'hall_events', 'hall_preferences', 'hall_advice')",
              "fact": "The extracted fact written as a concise, timeless statement."
            }
          ]
        }
        """
        
        let messages = [
            ChatMessage(role: .system, content: systemPrompt),
            ChatMessage(role: .user, content: "Extract Memory from: \n\n" + textBlock)
        ]
        
        var jsonBuffer = ""
        lastLog = "Generating JSON Extractions..."
        
        let stream = engine.generate(messages: messages)
        for await token in stream {
            jsonBuffer += token.text
        }
        
        lastLog = "Engine finished. Parsing output..."
        
        // Clean markdown backticks if model generated them
        let cleanedJSON = cleanJSON(jsonBuffer)
        
        guard let data = cleanedJSON.data(using: .utf8) else {
            lastLog = "Error: Model returned unparsable string characters."
            isMining = false
            return
        }
        
        do {
            let payload = try JSONDecoder().decode(ExtractionPayload.self, from: data)
            for item in payload.extractions {
                try MemoryPalaceService.shared.saveMemory(
                    wingName: wing,
                    roomName: item.room,
                    text: item.fact,
                    type: item.hall
                )
            }
            lastLog = "Success! Injected \(payload.extractions.count) facts into the Palace."
        } catch {
            lastLog = "JSON Decode Error: \(error.localizedDescription)\nRaw:\n\(cleanedJSON.prefix(100))..."
            print("Failed to decode extracted memory: \(error)")
        }
        
        isMining = false
    }
    
    private func cleanJSON(_ string: String) -> String {
        // Aggressively scan for the exact bounds of the JSON dictionary object
        // by finding the first parsing bracket and the absolute last parsing bracket,
        // completely ignoring any markdown backticks or LLM conversational text.
        guard let start = string.firstIndex(of: "{"),
              let end = string.lastIndex(of: "}") else {
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(string[start...end])
    }
}
