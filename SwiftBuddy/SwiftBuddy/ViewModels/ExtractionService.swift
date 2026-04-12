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

public struct ToolCall: Codable, Equatable {
    public let name: String
    public let parameters: [String: Any]?
    
    enum CodingKeys: String, CodingKey {
        case name, parameters
    }
    
    public init(name: String, parameters: [String: Any]? = nil) {
        self.name = name
        self.parameters = parameters
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        if let paramsData = try? container.decode([String: AnyCodable].self, forKey: .parameters), !paramsData.isEmpty {
            var params: [String: Any] = [:]
            for (k, v) in paramsData { params[k] = v.value }
            parameters = params
        } else {
            parameters = nil
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        if let p = parameters {
            var codableParams: [String: AnyCodable] = [:]
            for (k, v) in p { codableParams[k] = AnyCodable(v) }
            try container.encode(codableParams, forKey: .parameters)
        }
    }
    
    public static func == (lhs: ToolCall, rhs: ToolCall) -> Bool {
        lhs.name == rhs.name // Basic equality for testing
    }
}

// AnyCodable helper for loose JSON dictionary parameters
public struct AnyCodable: Codable {
    public let value: Any
    
    public init(_ value: Any) { self.value = value }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(String.self) { value = v }
        else if let v = try? container.decode(Int.self) { value = v }
        else if let v = try? container.decode(Double.self) { value = v }
        else if let v = try? container.decode(Bool.self) { value = v }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "AnyCodable unsupported type") }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let v = value as? String { try container.encode(v) }
        else if let v = value as? Int { try container.encode(v) }
        else if let v = value as? Double { try container.encode(v) }
        else if let v = value as? Bool { try container.encode(v) }
    }
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
        You are a highly intelligent Memory Palace extraction engine.
        Analyze the following raw text and distill it into highly specific, cohesive, and timeless facts or events.
        
        CRITICAL RULES:
        1. DO NOT regurgitate raw text line-by-line. You must synthesize the data.
        2. Combine fragmented sentences, dates, and titles into rich, complete paragraph-length facts.
        3. IGNORE boilerplate, headers, copyright notices, and irrelevant metadata (e.g. 'Volume 1', 'Translated by', 'Project Gutenberg').
        4. Each extracted fact MUST be a complete, descriptive sentence of at least 15 words.
        
        OUTPUT STRICTLY IN THE FOLLOWING JSON FORMAT ONLY. NEVER Output conversational text.
        
        {
          "extractions": [
            {
              "room": "Topic Category (e.g., 'Career', 'Physics', 'Personal')",
              "hall": "Category Type (must be either: 'hall_facts', 'hall_events', 'hall_discoveries', 'hall_preferences', 'hall_advice')",
              "fact": "The synthesized extract written as a comprehensive, timeless statement."
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
    
    // Feature 3: Extract tool call block from LLM stream
    public static func extractToolCall(from text: String) -> ToolCall? {
        guard let startRange = text.range(of: "<tool_call>"),
              let endRange = text.range(of: "</tool_call>") else {
            return nil
        }
        
        let jsonPayload = String(text[startRange.upperBound..<endRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard let data = jsonPayload.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(ToolCall.self, from: data)
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

public struct TextCombiner {
    
    /// Splits content into overlapping string chunks based on mempalace/miner.py RAG behavior.
    ///
    /// - Parameters:
    ///   - content: Raw text to split.
    ///   - chunkSize: Maximum characters per chunk (default 800).
    ///   - chunkOverlap: Characters to overlap when sliding to the next window (default 100).
    ///   - minChunkSize: Any extracted chunk smaller than this is dropped (default 50).
    public static func chunkText(_ content: String, chunkSize: Int = 800, chunkOverlap: Int = 100, minChunkSize: Int = 50) -> [String] {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        
        var chunks: [String] = []
        var currentIndex = text.startIndex
        
        while currentIndex < text.endIndex {
            var endIndex = text.index(currentIndex, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
            
            // Try to gently break at a clean paragraph boundary (\n\n) or line edge (\n) 
            // if we are near the chunk boundary.
            if endIndex < text.endIndex {
                let chunkRange = currentIndex..<endIndex
                let substring = text[chunkRange]
                
                if let lastDoubleNewline = substring.range(of: "\n\n", options: .backwards) {
                    let distance = text.distance(from: currentIndex, to: lastDoubleNewline.lowerBound)
                    // Only break if it's past the midpoint to avoid tiny chunks
                    if distance > chunkSize / 2 {
                        endIndex = lastDoubleNewline.lowerBound
                    } else if let lastNewline = substring.range(of: "\n", options: .backwards) {
                        let singleDistance = text.distance(from: currentIndex, to: lastNewline.lowerBound)
                        if singleDistance > chunkSize / 2 {
                            endIndex = lastNewline.lowerBound
                        }
                    }
                } else if let lastNewline = substring.range(of: "\n", options: .backwards) {
                    let singleDistance = text.distance(from: currentIndex, to: lastNewline.lowerBound)
                    if singleDistance > chunkSize / 2 {
                        endIndex = lastNewline.lowerBound
                    }
                }
            }
            
            let chunkString = String(text[currentIndex..<endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            if chunkString.count >= minChunkSize {
                chunks.append(chunkString)
            }
            
            if endIndex == text.endIndex {
                break
            }
            
            // Rewind by overlap to ensure sentences aren't cleanly sliced in half
            currentIndex = text.index(endIndex, offsetBy: -chunkOverlap, limitedBy: text.startIndex) ?? text.startIndex
            
            // Fast-forward past any immediate leading whitespace for the new chunk
            while currentIndex < text.endIndex && text[currentIndex].isWhitespace {
                currentIndex = text.index(after: currentIndex)
            }
        }
        
        return chunks
    }
}

