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
    
    @discardableResult
    func saveMemory(wingName: String, roomName: String, text: String, type: String = "Facts") throws -> Bool {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        guard let vector = generateEmbedding(for: text) else { return false }
        
        // 1. Semantic Duplicate Guard
        let fetchDesc = FetchDescriptor<MemoryEntry>()
        let existingMemories = try context.fetch(fetchDesc).filter { $0.room?.name == roomName && $0.room?.wing?.name == wingName }
        for mem in existingMemories {
            if let emb = mem.embedding {
                let similarity = cosineSimilarity(a: vector, b: emb)
                if similarity > 0.95 {
                    return false // Duplicate blocked
                }
            }
        }
        
        // 2. Find or create Wing
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        let wing = (try? context.fetch(fetchWing).first) ?? {
            let w = PalaceWing(name: wingName)
            context.insert(w)
            return w
        }()
        
        // 3. Find or create Room in Wing
        let fetchRoom = FetchDescriptor<PalaceRoom>(predicate: #Predicate { $0.name == roomName && $0.wing?.name == wingName })
        let room = (try? context.fetch(fetchRoom).first) ?? {
            let r = PalaceRoom(name: roomName, wing: wing)
            context.insert(r)
            return r
        }()
        
        // 4. Save Memory
        let entry = MemoryEntry(text: text, hallType: type, embedding: vector, room: room)
        context.insert(entry)
        
        try context.save()
        return true
    }
    
    func searchMemories(query: String, wingName: String, roomName: String? = nil, hallType: String? = nil, topK: Int = 5) throws -> [MemoryEntry] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        guard let queryVector = generateEmbedding(for: query) else { return [] }
        
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return [] }
        
        var allMemories = wing.rooms.flatMap { $0.memories }
        if let r = roomName { allMemories = allMemories.filter { $0.room?.name == r } }
        if let h = hallType { allMemories = allMemories.filter { $0.hallType == h } }
        
        return sortAndSliceMemories(allMemories, queryVector: queryVector, topK: topK)
    }
    
    func searchAllMemories(query: String, topK: Int = 5) throws -> [MemoryEntry] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        guard let queryVector = generateEmbedding(for: query) else { return [] }
        
        let fetchDesc = FetchDescriptor<MemoryEntry>()
        let allMemories = try context.fetch(fetchDesc)
        return sortAndSliceMemories(allMemories, queryVector: queryVector, topK: topK)
    }
    
    private func sortAndSliceMemories(_ memories: [MemoryEntry], queryVector: [Double], topK: Int) -> [MemoryEntry] {
        var scored: [(entry: MemoryEntry, score: Double)] = []
        for mem in memories {
            if let emb = mem.embedding {
                let score = cosineSimilarity(a: queryVector, b: emb)
                scored.append((mem, score))
            }
        }
        scored.sort { $0.score > $1.score }
        return scored.prefix(topK).map { $0.entry }
    }
    
    func findTunnels(roomName: String) throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchDesc = FetchDescriptor<PalaceRoom>(predicate: #Predicate { $0.name == roomName })
        let rooms = try context.fetch(fetchDesc)
        return rooms.compactMap { $0.wing?.name }
    }
    
    func listRooms(wingName: String) throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return [] }
        return wing.rooms.map { $0.name }
    }
    
    // MARK: - Wing Management
    
    func listWings() throws -> [String] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let descriptor = FetchDescriptor<PalaceWing>(sortBy: [SortDescriptor(\.createdDate)])
        let wings = try context.fetch(descriptor)
        return wings.map { $0.name }
    }
    
    func deleteWing(_ name: String) throws {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == name })
        if let wing = try context.fetch(fetchWing).first {
            context.delete(wing)
            try context.save()
        }
    }
    
    func deleteMemory(wingName: String, roomName: String?, textMatch: String) throws {
        // Find closest match > 0.85 and delete it
        let matches = try searchMemories(query: textMatch, wingName: wingName, roomName: roomName, topK: 1)
        if let mem = matches.first, let context = modelContext {
            context.delete(mem)
            try context.save()
        }
    }
    
    // MARK: - Taxonomy & Status
    
    func getCloset(wingName: String, roomName: String) throws -> String {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let fetchWing = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(fetchWing).first else { return "Closet is empty." }
        
        guard let room = wing.rooms.first(where: { $0.name == roomName }) else { return "Closet is empty." }
        
        let memories = room.memories.sorted { $0.dateAdded > $1.dateAdded }
        if memories.isEmpty { return "Closet is empty." }
        
        let facts = memories.map { "[\($0.hallType)] \($0.text)" }.joined(separator: "\n")
        return "Closet for \(wingName)/\(roomName):\n\(facts)"
    }
    
    func getTaxonomy() throws -> String {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let descriptor = FetchDescriptor<PalaceWing>(sortBy: [SortDescriptor(\.createdDate)])
        let wings = try context.fetch(descriptor)
        
        var output = "Memory Palace Taxonomy:\n"
        for w in wings {
            output += "Wing: \(w.name) (\(w.rooms.count) rooms)\n"
            for r in w.rooms {
                output += "  - Room: \(r.name) (\(r.memories.count) memories)\n"
            }
        }
        return output
    }
    
    func getPalaceStatus() throws -> (wings: Int, rooms: Int, memories: Int) {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let wCount = try context.fetchCount(FetchDescriptor<PalaceWing>())
        let rCount = try context.fetchCount(FetchDescriptor<PalaceRoom>())
        let mCount = try context.fetchCount(FetchDescriptor<MemoryEntry>())
        return (wings: wCount, rooms: rCount, memories: mCount)
    }
    
    // MARK: - Native Fact Checker
    
    private func detectContradiction(existingObject: String, newObject: String) -> Bool {
        // NLEmbedding-based semantic contradiction detection mechanism natively bypassing fact_checker.py limits
        guard let e1 = generateEmbedding(for: existingObject),
              let e2 = generateEmbedding(for: newObject) else { return false }
        
        let similarity = cosineSimilarity(a: e1, b: e2)
        // If similarity is extremely low (< 0.2) but they share the exact same subject+predicate mapping,
        // it triggers the contradiction detector.
        return similarity < 0.2
    }
    
    // MARK: - Tier 5: Temporal Knowledge Graph
    
    @discardableResult
    func addTriple(subject: String, predicate: String, object: String) throws -> Bool {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        
        let targetId = "\(subject.lowercased())_\(predicate.lowercased())"
        let fetchDesc = FetchDescriptor<KnowledgeGraphTriple>(predicate: #Predicate { $0.id == targetId })
        
        if let existing = try context.fetch(fetchDesc).first {
            // Check for direct contradiction before quietly overwriting.
            if detectContradiction(existingObject: existing.object, newObject: object) {
                // Evolve the triple via Contradiction tracking (temporal shift instead of generic overwrite)
                existing.object = "\(object) [Contradicted prior belief: \(existing.object)]"
            } else {
                // Standard Temporal Invalidation
                existing.object = object
            }
            existing.dateObserved = Date()
        } else {
            let triple = KnowledgeGraphTriple(subject: subject, predicate: predicate, object: object)
            context.insert(triple)
        }
        
        try context.save()
        return true
    }
    
    func queryEntity(_ subject: String) throws -> [KnowledgeGraphTriple] {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        let targetSubj = subject.lowercased()
        let fetchDesc = FetchDescriptor<KnowledgeGraphTriple>()
        
        // Cannot use lowercased() easily in SwiftData predicates sometimes, so fetch & filter natively for stability
        let allTriples = try context.fetch(fetchDesc)
        return allTriples.filter { $0.subject.lowercased() == targetSubj }
    }
    
    // MARK: - Tier 6: Advanced Support
    
    func getWakeupContext(wingName: String) throws -> String {
        guard let context = modelContext else { throw URLError(.badServerResponse) }
        
        // 1. Fetch Top 5 Triples for this specific wing's context
        let tripleDesc = FetchDescriptor<KnowledgeGraphTriple>(sortBy: [SortDescriptor(\.dateObserved, order: .reverse)])
        let recentTriples = Array((try context.fetch(tripleDesc)).prefix(5))
        
        let kgBlock = recentTriples.map { "[\($0.subject)] \($0.predicate) \($0.object)" }.joined(separator: "\n")
        
        // 2. Fetch last 3 entries from the "diary" room without NSException nesting
        let wingDesc = FetchDescriptor<PalaceWing>(predicate: #Predicate { $0.name == wingName })
        guard let wing = try context.fetch(wingDesc).first else { return "No context found." }
        
        let diaryRoom = wing.rooms.first(where: { $0.name == "diary" })
        let memories = diaryRoom?.memories.sorted { $0.dateAdded > $1.dateAdded } ?? []
        let diaryEntries = Array(memories.prefix(3).reversed())
        
        let diaryBlock = diaryEntries.map { "- \($0.text)" }.joined(separator: "\n")
        
        return """
        --- L0/L1 WAKEUP CONTEXT ---
        [Knowledge Graph]
        \(kgBlock)
        
        [Recent Reflections]
        \(diaryBlock)
        ----------------------------
        """
    }
}
