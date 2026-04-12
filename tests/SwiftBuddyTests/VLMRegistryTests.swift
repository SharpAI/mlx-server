import XCTest
@testable import SwiftBuddy
import MLXInferenceCore
@testable import MLXVLM
import MLXLMCommon

struct MockTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { return [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { return "" }
    func convertTokenToId(_ token: String) -> Int? { return nil }
    func convertIdToToken(_ id: Int) -> String? { return nil }
    var bosToken: String? { return nil }
    var eosToken: String? { return nil }
    var unknownToken: String? { return nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { return [] }
}

final class VLMRegistryTests: XCTestCase {
    
    // Feature 9: VLM model type registry covers all supported types
    func testVLM_TypeRegistryCompleteness() async {
        let expectedTypes: Set<String> = [
            "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_moe",
            "idefics3", "gemma3", "smolvlm", "fastvlm", "llava_qwen2", "pixtral",
            "mistral3", "lfm2_vl", "lfm2-vl", "glm_ocr"
        ]
        
        let registry = VLMTypeRegistry.shared
        let dummyData = "{}".data(using: .utf8)!
        
        for type in expectedTypes {
            do {
                _ = try await registry.createModel(configuration: dummyData, modelType: type)
                // If it succeeds with dummy data, that's fine, it means the registry works.
            } catch let ModelFactoryError.unsupportedModelType(t) {
                XCTFail("Registry is missing supported model type: \(t)")
            } catch {
                // Expected decoding error
            }
        }
    }
    
    // Feature 10: VLM processor registry covers all supported types
    func testVLM_ProcessorRegistryCompleteness() async {
        let expectedProcessors: Set<String> = [
            "PaliGemmaProcessor", "Qwen2VLProcessor", "Qwen2_5_VLProcessor", "Qwen3VLProcessor",
            "Idefics3Processor", "Gemma3Processor", "SmolVLMProcessor", "FastVLMProcessor",
            "PixtralProcessor", "Mistral3Processor", "Lfm2VlProcessor", "Glm46VProcessor"
        ]
        
        let registry = VLMProcessorTypeRegistry.shared
        let dummyData = "{}".data(using: .utf8)!
        let dummyTokenizer = MockTokenizer()
        
        for type in expectedProcessors {
            do {
                _ = try await registry.createModel(configuration: dummyData, processorType: type, tokenizer: dummyTokenizer)
                // If it succeeds with dummy data, that's fine, it means the registry works and the config was optional.
            } catch let ModelFactoryError.unsupportedModelType(t) {
                XCTFail("Registry is missing supported processor type: \(t)")
            } catch {
                // Expected decoding error or other initialization error
            }
        }
    }
    
    // Feature 11: Unsupported model_type returns clear error
    func testVLM_UnsupportedModelType() async {
        let registry = VLMTypeRegistry.shared
        do {
            let data = "{}".data(using: .utf8)!
            _ = try await registry.createModel(configuration: data, modelType: "nonexistent_model")
            XCTFail("Should have thrown error")
        } catch ModelFactoryError.unsupportedModelType(let type) {
            XCTAssertEqual(type, "nonexistent_model")
        } catch {
            XCTFail("Threw unknown error: \(error)")
        }
    }
}
