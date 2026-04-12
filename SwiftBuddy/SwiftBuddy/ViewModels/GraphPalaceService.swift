// GraphPalaceService.swift - Synaptic Synthesis Engine
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
public final class GraphPalaceService {
    public static let shared = GraphPalaceService()
    public var modelContext: ModelContext?

    private init() {}

    /// Performs the "2nd round of memory construction" by converting raw SQL memory chunks into topological Graph nodes.
    public func buildRelationalGraph(wingName: String, using engine: InferenceEngine? = nil, onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        print("[GraphPalace] SYNAPTIC SYNTHESIS INITIATED for \(wingName).")
        
        guard let context = modelContext else {
            print("[GraphPalace] Warning: No ModelContext attached.")
            return
        }
        
        // Fetch all MemoryEntries for this wing implicitly by querying the Wing
        let fetchDescriptor = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try? context.fetch(fetchDescriptor).first else { return }
        
        var extractionTargets: [String] = []
        var rawMemories: [String] = []
        for room in wing.rooms {
            for memory in room.memories {
                // Multimodal bridging: Pull from text (hall_facts), audio transcript (hall_audio), OCR (hall_vision)
                if memory.hallType == "hall_facts" || memory.hallType == "hall_audio" || memory.hallType == "hall_vision" {
                    rawMemories.append(memory.text)
                }
            }
        }
        
        // Massive Context Batching: Combine micro-chunks into heavy blocks
        // Target ~16,000 characters per extraction shot to maximize KV Cache efficiency
        let targetBlockSize = 16000
        var currentBlock = ""
        
        for text in rawMemories {
            if currentBlock.count + text.count > targetBlockSize {
                extractionTargets.append(currentBlock.trimmingCharacters(in: .whitespacesAndNewlines))
                currentBlock = text + "\n"
            } else {
                currentBlock += text + "\n"
            }
        }
        if !currentBlock.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            extractionTargets.append(currentBlock.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        
        print("[GraphPalace] Synthesizing edges across \(extractionTargets.count) multimodal targets.")
        
        guard let engine = engine else {
            print("[GraphPalace] Engine unavailable or not injected. Skipping LLM generation loop.")
            return
        }
        
        for (index, target) in extractionTargets.enumerated() {
            onProgress?(index + 1, extractionTargets.count, "SYNTHESIZING EDGE: \(target)")
            let prompt = buildGraphPrompt(text: target)
            let stream = engine.generate(messages: [.user(prompt)])
            var response = ""
            for await token in stream {
                response += token.text
            }
            
            if let nodeArray = parseGraphTriples(fromJSONString: response) {
                // Deduplicate within the same chunk to prevent SwiftData ephemeral collisions
                var seenIds = Set<String>()
                
                for triple in nodeArray {
                    if !seenIds.contains(triple.id) {
                        seenIds.insert(triple.id)
                        context.insert(triple)
                        print("[GraphPalace] 🧬 Edge Extracted: \(triple.subject) -> [\(triple.predicate)] -> \(triple.object)")
                    }
                }
                // Save immediately to flush temporary IDs to stable SQLite row-ids, avoiding `.unique` constraint panics on large loops
                try? context.save()
            }
        }
    }
    
    public func synthesizePersonaIndex(wingName: String, using engine: InferenceEngine, onProgress: ((Int, Int, String) -> Void)? = nil) async throws {
        print("[GraphPalace] SYNTHESIZING PERSONA INDEX for \(wingName).")
        onProgress?(1, 1, "DISTILLING GOD-NODES INTO CONDENSED PERSONA INDEX...")
        guard let context = modelContext else { return }
        
        let fetchDescriptor = FetchDescriptor<KnowledgeGraphTriple>()
        guard let allTriples = try? context.fetch(fetchDescriptor), !allTriples.isEmpty else { return }
        
        // Phase 3: Leiden-style grouping (by Subject for now)
        var groupedTriples: [String: [KnowledgeGraphTriple]] = [:]
        for triple in allTriples {
            groupedTriples[triple.subject, default: []].append(triple)
        }
        
        var indexPrompt = "Create a 100-word dense persona index mapping out the following extracted graph communities. Ignore ambiguous links. Format as a dense summary:\n\n"
        for (subject, edges) in groupedTriples {
            if edges.count > 1 { // Only count "God Nodes" (highly connected)
                indexPrompt += "Node: \(subject)\n"
                for edge in edges {
                    indexPrompt += " - \(edge.predicate) \(edge.object) (Type: \(edge.taxonomy ?? "fact"), Truth: \(edge.confidence ?? "INFERRED"))\n"
                }
                indexPrompt += "\n"
            }
        }
        
        let stream = engine.generate(messages: [.user(indexPrompt)])
        var response = ""
        for await token in stream {
            response += token.text
        }
        
        _ = try await MemoryPalaceService.shared.saveMemories(wingName: wingName, roomName: "persona_index", texts: [response], type: "hall_facts")
        print("[GraphPalace] 🧠 Persona Index Generated & Saved.")
    }
    
    public func parseGraphTriples(fromJSONString jsonString: String) -> [KnowledgeGraphTriple]? {
        var cleanText = jsonString.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleanText.hasPrefix("```json") { cleanText.removeFirst(7) }
        else if cleanText.hasPrefix("```") { cleanText.removeFirst(3) }
        if cleanText.hasSuffix("```") { cleanText.removeLast(3) }
        cleanText = cleanText.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard let data = cleanText.data(using: .utf8) else { return nil }
        guard let array = try? JSONSerialization.jsonObject(with: data, options: []) as? [[String: String]] else { return nil }
        
        var triples: [KnowledgeGraphTriple] = []
        for dict in array {
            guard let subject = dict["subject"],
                  let predicate = dict["predicate"],
                  let object = dict["object"] else {
                continue
            }
            triples.append(KnowledgeGraphTriple(
                subject: subject, 
                predicate: predicate, 
                object: object,
                taxonomy: dict["taxonomy"],
                confidence: dict["confidence"]
            ))
        }
        
        return triples.isEmpty ? nil : triples
    }
    
    public func buildGraphPrompt(text: String) -> String {
        return """
        Extract an exhaustive list of subject-predicate-object triples from the following text to form a complete knowledge graph.
        Additionally, classify each triple's 'taxonomy' as either 'fact', 'preference', 'decision', or 'relationship'.
        Additionally, assess the 'confidence' of each triple as either 'EXTRACTED' (literal) or 'INFERRED' (deduced).
        Format the output as a pure JSON array of objects, containing the keys "subject", "predicate", "object", "taxonomy", and "confidence".
        Do not output any reasoning. Return ONLY the JSON array.
        
        Text:
        \(text)
        """
    }
}
