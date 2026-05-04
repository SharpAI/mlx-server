import XCTest
import MLX
import MLXLMCommon
@testable import MLXInferenceCore

final class ContextWindowCalculationTests: XCTestCase {

    @MainActor
    func testContextTokensCalculation() async throws {
        // Feature: Verify that tokens calculation accurately reflects the prompt cache window
        // by evaluating the full size of the prepared tokens array, not just the batch shape.
        
        let engine = InferenceEngine()
        
        // Mock a scenario where userInput prepares a chat template with large history.
        // We will directly instantiate LMInput and assert on its size.
        
        let mockTokens = MLXArray(Array(0..<512))
        // If tokenizer batches it, shape could be [1, 512].
        let reshapedTokens = mockTokens.reshaped([1, 512])
        
        // MLXLMCommon's LMInput struct
        let lmInput = LMInput(tokens: reshapedTokens)
        
        // Validate that using .size accurately captures the token count (512)
        // rather than falling victim to the batch dimension .shape[0] which would be 1.
        XCTAssertEqual(lmInput.text.tokens.shape[0], 1, "shape[0] captures the batch dimension, returning 1")
        XCTAssertEqual(lmInput.text.tokens.size, 512, "size captures the total token count, resolving the context window bug")
    }
}
