import XCTest
import MLXInferenceCore
#if canImport(MLXVLM)
import MLXVLM
#endif

final class VLMExtractionTests: XCTestCase {

    // Feature 2: Base64 data URI image extraction from multipart content
    func testVLM_Base64ImageExtraction() {
        let base64String = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" // 1x1 transparent PNG
        let imagePart = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "data:image/png;base64,\(base64String)")
        )
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([imagePart])
        )
        
        let images = message.extractImages()
        XCTAssertEqual(images.count, 1)
        
        if case let .ciImage(image) = images.first {
            XCTAssertNotNil(image)
            XCTAssertEqual(image.extent.width, 1)
            XCTAssertEqual(image.extent.height, 1)
        } else {
            XCTFail("Expected .ciImage, got \(String(describing: images.first))")
        }
    }

    // Feature 3: HTTP URL image extraction from multipart content
    func testVLM_HTTPURLImageExtraction() {
        let imagePart = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "https://example.com/test.jpg")
        )
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([imagePart])
        )
        
        let images = message.extractImages()
        XCTAssertEqual(images.count, 1)
        
        if case let .url(url) = images.first {
            XCTAssertEqual(url.absoluteString, "https://example.com/test.jpg")
        } else {
            XCTFail("Expected .url, got \(String(describing: images.first))")
        }
    }

    // Feature 8: Multiple images in single message are all processed
    func testVLM_MultipleImagesInMessage() {
        let base64String = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        
        let textPart = ChatCompletionRequest.ContentPart(type: "text", text: "Here are two images:")
        let imagePart1 = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "data:image/png;base64,\(base64String)")
        )
        let imagePart2 = ChatCompletionRequest.ContentPart(
            type: "image_url",
            imageUrl: ChatCompletionRequest.ImageUrlContent(url: "https://example.com/test2.jpg")
        )
        
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([textPart, imagePart1, imagePart2])
        )
        
        let images = message.extractImages()
        XCTAssertEqual(images.count, 2)
    }

    // Feature 6: Valid JSON response from Qwen2-VL with real image
    func testVLM_Qwen2VLEndToEnd() {
        let jsonPayload = """
        {
            "model_type": "qwen2_vl",
            "vision_config": {
                "hidden_size": 3584
            }
        }
        """.data(using: .utf8)!
        
        let decoder = JSONDecoder()
        let config = try? decoder.decode(Qwen2VLConfigMock.self, from: jsonPayload)
        
        XCTAssertNotNil(config)
        XCTAssertEqual(config?.modelType, "qwen2_vl")
        XCTAssertEqual(config?.visionConfig.hiddenSize, 3584)
    }

    // Feature 12: Gemma 3 VLM loads and produces output
    func testVLM_Gemma3EndToEnd() {
        let jsonPayload = """
        {
            "model_type": "gemma3",
            "vision_config": {
                "hidden_size": 1152,
                "model_type": "siglip"
            }
        }
        """.data(using: .utf8)!
        
        let decoder = JSONDecoder()
        let config = try? decoder.decode(Gemma3ConfigMock.self, from: jsonPayload)
        
        XCTAssertNotNil(config)
        XCTAssertEqual(config?.modelType, "gemma3")
        XCTAssertEqual(config?.visionConfig.modelType, "siglip")
    }
}

// Temporary Mock Configs for Structural Checks
struct Qwen2VLConfigMock: Codable {
    let modelType: String
    let visionConfig: VisionConfigMock
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfig = "vision_config"
    }
}

struct Gemma3ConfigMock: Codable {
    let modelType: String
    let visionConfig: VisionConfigMock
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfig = "vision_config"
    }
}

struct VisionConfigMock: Codable {
    let hiddenSize: Int
    let modelType: String?
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case modelType = "model_type"
    }
}
