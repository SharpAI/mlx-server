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
              "hall": "Category Type (must be either: 'hall_facts', 'hall_events', 'hall_discoveries', 'hall_preferences', 'hall_advice')",
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
        let cleanedJSON = JSONSanitizer.cleanJSON(jsonBuffer)
        
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
}

public struct JSONSanitizer {
    public static func cleanJSON(_ raw: String) -> String {
        var str = raw
        
        // Feature 9: Strip ANSI escape sequences \u001B[...m (from terminal debug spillovers)
        if let regex = try? NSRegularExpression(pattern: "\u{001B}\\[[0-9;]*[a-zA-Z]", options: []) {
            let range = NSRange(location: 0, length: str.utf16.count)
            str = regex.stringByReplacingMatches(in: str, options: [], range: range, withTemplate: "")
        }
        
        str = str.trimmingCharacters(in: .whitespacesAndNewlines)
        // Feature 6: Handle empty / whitespace payloads unconditionally
        if str.isEmpty { return "{}" }
        
        let firstBrace = str.firstIndex(of: "{")
        let firstBracket = str.firstIndex(of: "[")
        
        enum RootType { case dict, array, none }
        var type: RootType = .none
        var startIdx = str.endIndex
        
        if let fBrace = firstBrace, let fBracket = firstBracket {
            if fBrace < fBracket {
                type = .dict; startIdx = fBrace
            } else {
                type = .array; startIdx = fBracket
            }
        } else if let fBrace = firstBrace {
            type = .dict; startIdx = fBrace
        } else if let fBracket = firstBracket {
            type = .array; startIdx = fBracket
        } else {
            return "{}"
        }
        
        let substring = String(str[startIdx...])
        
        if type == .dict {
            if let lastBrace = substring.lastIndex(of: "}") {
                let candidate = String(substring[...lastBrace])
                
                // Feature 7: Fragmented JSON blobs stitching `{...} {...}` -> `{ "extractions": [...] }`
                if let regex = try? NSRegularExpression(pattern: "\\}\\s*\\{", options: []) {
                    let range = NSRange(location: 0, length: candidate.utf16.count)
                    if regex.firstMatch(in: candidate, options: [], range: range) != nil {
                        let transformed = candidate.replacingOccurrences(of: "\\}\\s*\\{", with: "},{", options: .regularExpression)
                        return "{ \"extractions\": [ \(transformed) ] }"
                    }
                }
                return candidate
            } else {
                // Feature 4: Gracefully append missing `}` closures
                return substring + "}"
            }
        } else if type == .array {
            // Feature 5: Discover array bounds [...] independently mapping to ExtractionPayload format natively
            if let lastBracket = substring.lastIndex(of: "]") {
                let candidate = String(substring[...lastBracket])
                return "{ \"extractions\": \(candidate) }"
            } else {
                return "{ \"extractions\": \(substring)] }"
            }
        }
        
        return "{}"
    }
}
