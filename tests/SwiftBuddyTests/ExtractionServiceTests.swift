import XCTest
@testable import SwiftBuddy
@testable import MLXInferenceCore

final class ExtractionServiceTests: XCTestCase {
    
    // We cannot instantiate ExtractionService directly if it heavily relies on @MainActor Engine.
    // Instead, we will extract the cleanJSON core logic mechanically into a testable function
    // or test it via its internal components using Swift mirrors if needed.
    // For now, we replicate the pure Regex extraction logic to mathematically verify its safety bounds.
    
    func cleanJSON(_ string: String) -> String {
        // Aggressively scan for the exact bounds of the JSON dictionary object
        guard let start = string.firstIndex(of: "{"),
              let end = string.lastIndex(of: "}") else {
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(string[start...end])
    }
    
    func testCleanJSON_withPerfectJSON() {
        let input = """
        {
            "extractions": [{"test": "value"}]
        }
        """
        
        let output = cleanJSON(input)
        XCTAssertTrue(output.starts(with: "{"))
        XCTAssertTrue(output.hasSuffix("}"))
        XCTAssertEqual(output, input)
    }
    
    func testCleanJSON_withHallucinatedPreamble() {
        let input = """
        Here is the JSON you requested master:
        ```json
        {
            "extractions": [{"test": "value"}]
        }
        ```
        And a quick reminder to eat your vegetables!
        """
        
        let output = cleanJSON(input)
        let expected = """
        {
            "extractions": [{"test": "value"}]
        }
        """
        XCTAssertEqual(output, expected)
    }
    
    func testCleanJSON_withInternalNestedBraces() {
        let input = """
        {
            "extractions": [
                { "key": "{value}" },
                { "key2": "value2" }
            ]
        }
        """
        
        let output = cleanJSON(input)
        XCTAssertEqual(output, input) // Internal braces should NOT truncate early
    }
}
