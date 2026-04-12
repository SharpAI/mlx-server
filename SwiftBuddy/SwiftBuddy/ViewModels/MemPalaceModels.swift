import Foundation
import SwiftData
import NaturalLanguage

@Model
final class PalaceWing {
    @Attribute(.unique) var name: String
    var createdDate: Date
    
    @Relationship(deleteRule: .cascade, inverse: \PalaceRoom.wing)
    var rooms: [PalaceRoom] = []
    
    init(name: String, createdDate: Date = Date()) {
        self.name = name
        self.createdDate = createdDate
    }
}

@Model
final class PalaceRoom {
    var name: String
    var wing: PalaceWing?
    
    @Relationship(deleteRule: .cascade, inverse: \MemoryEntry.room)
    var memories: [MemoryEntry] = []
    
    init(name: String, wing: PalaceWing? = nil) {
        self.name = name
        self.wing = wing
    }
}

@Model
final class MemoryEntry {
    var text: String
    var hallType: String
    var dateAdded: Date
    var embedding: [Double]?
    
    var room: PalaceRoom?
    
    init(text: String, hallType: String, embedding: [Double]? = nil, dateAdded: Date = Date(), room: PalaceRoom? = nil) {
        self.text = text
        self.hallType = hallType
        self.embedding = embedding
        self.dateAdded = dateAdded
        self.room = room
    }
}

@Model
public final class KnowledgeGraphTriple {
    @Attribute(.unique) public var id: String // e.g. "subject_predicate"
    public var subject: String
    public var predicate: String
    public var object: String
    public var dateObserved: Date
    public var taxonomy: String?
    public var confidence: String?
    
    init(subject: String, predicate: String, object: String, dateObserved: Date = Date(), taxonomy: String? = nil, confidence: String? = nil) {
        self.id = "\(subject.lowercased())_\(predicate.lowercased())" // Enforce temporal overwrite (one predicate per subject)
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.dateObserved = dateObserved
        self.taxonomy = taxonomy
        self.confidence = confidence
    }
}

// MARK: - Persistent Chat History 

@Model
final class ChatSession {
    @Attribute(.unique) var id: UUID
    var wingName: String? // nil applies to the 'Core System Chat'
    var createdAt: Date
    
    @Relationship(deleteRule: .cascade, inverse: \ChatTurn.session)
    var turns: [ChatTurn] = []
    
    init(id: UUID = UUID(), wingName: String? = nil, createdAt: Date = Date()) {
        self.id = id
        self.wingName = wingName
        self.createdAt = createdAt
    }
}

@Model
final class ChatTurn {
    @Attribute(.unique) var id: UUID
    var roleRaw: String      // "user", "assistant", "system"
    var content: String
    var thinkingContent: String?
    var timestamp: Date
    
    var session: ChatSession?
    
    init(id: UUID = UUID(), roleRaw: String, content: String, thinkingContent: String? = nil, timestamp: Date = Date(), session: ChatSession? = nil) {
        self.id = id
        self.roleRaw = roleRaw
        self.content = content
        self.thinkingContent = thinkingContent
        self.timestamp = timestamp
        self.session = session
    }
}
