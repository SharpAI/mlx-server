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
final class KnowledgeGraphTriple {
    @Attribute(.unique) var id: String // e.g. "subject_predicate"
    var subject: String
    var predicate: String
    var object: String
    var dateObserved: Date
    
    init(subject: String, predicate: String, object: String, dateObserved: Date = Date()) {
        self.id = "\(subject.lowercased())_\(predicate.lowercased())" // Enforce temporal overwrite (one predicate per subject)
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.dateObserved = dateObserved
    }
}
