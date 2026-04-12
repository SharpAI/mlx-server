import XCTest
import MLX
import MLXInferenceCore
import MLXLLM
import Foundation

final class AudioFusionTests: XCTestCase {

    override class func setUp() {
        super.setUp()
    }
    
    // Feature 13: Gemma 4 `audio_config` is parsed from config.json
    func testAudio_Gemma4ConfigParsed() throws {
        let jsonPayload = """
        {
            "model_type": "gemma4",
            "audio_config": {
                "model_type": "gemma4_audio",
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 8
            }
        }
        """.data(using: .utf8)!
        
        let decoder = JSONDecoder()
        let config = try decoder.decode(Gemma4Configuration.self, from: jsonPayload)
        
        XCTAssertNotNil(config.audioConfig)
        XCTAssertEqual(config.audioConfig?.hiddenSize, 1024)
        XCTAssertEqual(config.audioConfig?.numHiddenLayers, 12)
        XCTAssertEqual(config.audioConfig?.numAttentionHeads, 8)
    }

    // Feature 14: Audio tokens interleaved with text tokens at correct positions
    // Feature 15: `boa_token_id` / `eoa_token_id` correctly bracket audio segments
    func testAudio_TokenInterleavingAndBoundaries() {
        let boa: Int = 255010
        let eoa: Int = 255011
        
        // Simulating the Text (101, 102) and Audio tensors array (A1, A2, A3 sequence)
        let textTokens = [101, 102]
        let numAudioEmbeddings = 3 
        
        let fusionEngine = MultimodalFusionProcessor(boaToken: boa, eoaToken: eoa)
        let fusions = fusionEngine.interleave(textTokens: textTokens, numAudioEmbeddings: numAudioEmbeddings, audioFirst: true)
        
        // Expected Media First: [BOA, dummy, dummy, dummy, EOA, 101, 102]
        XCTAssertEqual(fusions.first, boa)
        XCTAssertEqual(fusions[fusions.count - textTokens.count - 1], eoa)
        XCTAssertEqual(fusions.suffix(textTokens.count), ArraySlice(textTokens))
        XCTAssertEqual(fusions.count, 2 + numAudioEmbeddings + textTokens.count)
    }

    // Feature 16: Mixed text + audio + vision request processed without crash
    func testAudio_TrimodalRequest() throws {
        let fusionEngine = MultimodalFusionProcessor(boaToken: 255010, eoaToken: 255011)
        
        XCTAssertNoThrow(
            try fusionEngine.processTrimodal(
                text: "Describe this video clip", 
                imageBase64: "dummy_image", 
                audioBase64: "dummy_audio"
            )
        )
    }
}

// Temporary internal configurations tests
struct Gemma4ConfigurationMock: Codable {
    let modelType: String
    let audioConfig: AudioConfigMock?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
    }
}

struct AudioConfigMock: Codable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
    }
}
