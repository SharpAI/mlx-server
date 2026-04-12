import XCTest
import MLXInferenceCore
@testable import MLXVLM
import MLXLMCommon
import CoreImage

final class VLMProcessorTests: XCTestCase {

    // Feature 4: Reject request with no image when model requires one
    func testVLM_RejectMissingImage() async throws {
        let registry = VLMProcessorTypeRegistry.shared
        
        // We know PaliGemmaProcessor throws if given no images
        let dummyData = "{}".data(using: .utf8)!
        let dummyTokenizer = MockTokenizer()
        
        let json = """
        {
            "image_seq_length": 256,
            "size": {"width": 224, "height": 224},
            "patch_size": 14,
            "processor_class": "PaliGemmaProcessor",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }
        """
        let config = try JSONDecoder().decode(PaliGemmaProcessorConfiguration.self, from: json.data(using: .utf8)!)
        let processor = PaliGemmaProcessor(config, tokenizer: dummyTokenizer)
        
        let input = UserInput(prompt: "Hello", images: [])
        
        do {
            _ = try await processor.prepare(input: input)
            XCTFail("Should have thrown imageRequired")
        } catch VLMError.imageRequired {
            // Success
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }
    
    // Feature 5: Text-only fallback when VLM receives no image
    func testVLM_TextOnlyFallback() async throws {
        // Qwen2VL natively supports text-only.
        let json = """
        {
            "processor_class": "Qwen2VLProcessor",
            "image_mean": [0.5],
            "image_std": [0.5],
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "min_pixels": 256,
            "max_pixels": 1024
        }
        """
        let config = try JSONDecoder().decode(Qwen2VLProcessorConfiguration.self, from: json.data(using: .utf8)!)
        let dummyTokenizer = MockTokenizer()
        let processor = Qwen2VLProcessor(config, tokenizer: dummyTokenizer)
        
        let input = UserInput(prompt: "Hello", images: [])
        let lmInput = try await processor.prepare(input: input)
        
        // Should succeed and return text only
        XCTAssertNil(lmInput.image)
        XCTAssertNotNil(lmInput.text)
    }

    // Feature 7: Image too small for ViT patch size returns graceful error
    func testVLM_ImageTooSmallError() {
        do {
            _ = try QwenVL.targetSize(height: 1, width: 1, factor: 28, minPixels: 256, maxPixels: 1024)
            // It might throw an imageProcessingFailure or processing error natively
            XCTFail("Should throw gracefully")
        } catch {
            // Accept any error as a graceful processing error.
        }
    }
}
