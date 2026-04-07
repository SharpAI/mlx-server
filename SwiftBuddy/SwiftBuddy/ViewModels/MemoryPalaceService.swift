import Foundation
import SwiftData
import NaturalLanguage

@MainActor
final class MemoryPalaceService {
    static let shared = MemoryPalaceService()
    
    var modelContext: ModelContext?
    
    // Apple's Native Embedding Model
    private let embeddingModel: NLEmbedding? = {
        return NLEmbedding.sentenceEmbedding(for: .english)
    }()
    
    // MARK: - Vector Math
    
    private func cosineSimilarity(a: [Double], b: [Double]) -> Double {
        guard a.count == b.count, a.count > 0 else { return 0.0 }
        
        var dotProduct: Double = 0.0
        var normA: Double = 0.0
        var normB: Double = 0.0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        if normA == 0.0 || normB == 0.0 { return 0.0 }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
    
    private func generateEmbedding(for text: String) -> [Double]? {
        guard let model = embeddingModel else { return nil }
        return model.vector(for: text)
    }
    
    // MARK: - Palace Operations
    
    func saveMemory(wingName: String, roomName: String, text: String, type: String = "Facts") throws {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        
        // 1. Find or create Wing
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        let wing = (try? context.fetch(fetchWing).first) ?? {
            let w = PalaceWing(name: wingName)
            context.insert(w)
            return w
        }()
        
        // 2. Find or create Room in Wing
        let fetchRoom = FetchDescriptor<PalaceRoom>(predicate: #Predicate { $0.name == roomName && $0.wing?.name == wingName })
        let room = (try? context.fetch(fetchRoom).first) ?? {
            let r = PalaceRoom(name: roomName, wing: wing)
            context.insert(r)
            return r
        }()
        
        // 3. Generate Embedding & Save Memory
        let vector = generateEmbedding(for: text)
        let entry = MemoryEntry(text: text, hallType: type, embedding: vector, room: room)
        context.insert(entry)
        
        try context.save()
    }
    
    func searchMemories(query: String, wingName: String, topK: Int = 5) throws -> [MemoryEntry] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        guard let queryVector = generateEmbedding(for: query) else { return [] }
        
        // Fetch all memories directly matching the wing
        let fetchDesc = FetchDescriptor<MemoryEntry>(predicate: #Predicate { $0.room?.wing?.name == wingName })
        let allMemories = try context.fetch(fetchDesc)
        
        // Score using cosine similarity
        var scored: [(entry: MemoryEntry, score: Double)] = []
        for mem in allMemories {
            if let emb = mem.embedding {
                let score = cosineSimilarity(a: queryVector, b: emb)
                scored.append((mem, score))
            }
        }
        
        // Sort and return topK
        scored.sort { $0.score > $1.score }
        return scored.prefix(topK).map { $0.entry }
    }
    
    func listRooms(wingName: String) throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return [] }
        return wing.rooms.map { $0.name }
    }
}
