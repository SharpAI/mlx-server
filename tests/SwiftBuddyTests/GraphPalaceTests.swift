import XCTest
@testable import SwiftBuddy
import SwiftData

#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class GraphPalaceTests: XCTestCase {
    var modelContainer: ModelContainer!
    var modelContext: ModelContext!
    var service: GraphPalaceService!
    
    override func setUp() async throws {
        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        modelContainer = try ModelContainer(for: PalaceWing.self, PalaceRoom.self, MemoryEntry.self, KnowledgeGraphTriple.self, ChatSession.self, ChatTurn.self, configurations: config)
        modelContext = modelContainer.mainContext
        
        service = GraphPalaceService.shared
        service.modelContext = modelContext
    }
    
    override func tearDown() {
        modelContainer = nil
        modelContext = nil
        service = nil
    }
    
    func testGraphPalaceSingleton() {
        XCTAssertNotNil(GraphPalaceService.shared)
    }
    
    // Testing the extraction logic. We mock the JSON payload return.
    func testExtractTriplesFromJSON() async throws {
        let rawJSON = """
        ```json
        [
            {
              "subject": "Albert Einstein",
              "predicate": "formulated",
              "object": "Theory of Relativity"
            },
            {
              "subject": "Leonardo Da Vinci",
              "predicate": "painted",
              "object": "Mona Lisa"
            }
        ]
        ```
        """
        
        let triples = service.parseGraphTriples(fromJSONString: rawJSON)
        XCTAssertNotNil(triples)
        XCTAssertEqual(triples?.count, 2)
        XCTAssertEqual(triples?.first?.subject, "Albert Einstein")
        XCTAssertEqual(triples?.first?.predicate, "formulated")
        XCTAssertEqual(triples?.first?.object, "Theory of Relativity")
    }
    
    // Harness to ensure acceptance criteria: the extraction loop correctly bypasses LLM generation
    // if the InferenceEngine is missing during execution (avoiding force-unwrap crashes).
    func testGraphPalaceSynthesisBypassWhenEngineIsNil() async throws {
        // Create mock wing with memory
        let wing = PalaceWing(name: "Test Wing")
        modelContext.insert(wing)
        let room = PalaceRoom(name: "Facts", wing: wing)
        modelContext.insert(room)
        let entry = MemoryEntry(text: "Test fact", hallType: "hall_facts", embedding: [0.1, 0.2], room: room)
        modelContext.insert(entry)
        try modelContext.save()
        
        // Assert no throw when engine is nil
        do {
            try await service.buildRelationalGraph(wingName: "Test Wing", using: nil)
            // If it reaches here without crash, bypass accepted
            XCTAssertTrue(true, "Bypass correctly handled nil inference engine.")
        } catch {
            XCTFail("Should not throw when engine is omitted, just bypass.")
        }
    }
    func testSynapticSynthesisSystemPrompt() {
        let prompt = service.buildGraphPrompt(text: "Albert Einstein liked sailing.")
        XCTAssertTrue(prompt.contains("Extract an exhaustive list"))
        XCTAssertTrue(prompt.contains("JSON array"))
        XCTAssertTrue(prompt.contains("Albert Einstein liked sailing."))
    }
}
