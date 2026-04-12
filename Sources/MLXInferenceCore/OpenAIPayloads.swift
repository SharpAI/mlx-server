import Foundation
#if canImport(CoreImage)
import CoreImage
#endif
#if canImport(MLXVLM)
import MLXVLM
#endif
import MLXLMCommon

public struct StreamOptions: Decodable {
    public let includeUsage: Bool?
    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct ResponseFormat: Decodable {
    public let type: String
}

public struct ChatCompletionRequest: Decodable {
    public struct Message: Decodable {
        public let role: String
        public let content: MessageContent?
        // Note: tool_calls removed to simplify extraction if missing types, but we'll add it back later if needed
        public let tool_call_id: String?

        public init(role: String, content: MessageContent?, tool_call_id: String? = nil) {
            self.role = role
            self.content = content
            self.tool_call_id = tool_call_id
        }

        public var textContent: String {
            guard let content = content else { return "" }
            switch content {
            case .string(let s): return s
            case .parts(let parts):
                return parts.compactMap { part in
                    if part.type == "text" { return part.text }
                    return nil
                }.joined(separator: "\n")
            }
        }

#if canImport(MLXVLM) && canImport(CoreImage)
        public func extractImages() -> [UserInput.Image] {
            guard let content = content, case .parts(let parts) = content else { return [] }
            return parts.compactMap { part -> UserInput.Image? in
                guard part.type == "image_url", let imageUrl = part.imageUrl else { return nil }
                let urlStr = imageUrl.url
                
                if urlStr.hasPrefix("data:") {
                    guard let commaIdx = urlStr.firstIndex(of: ",") else { return nil }
                    let base64Str = String(urlStr[urlStr.index(after: commaIdx)...])
                    guard let data = Data(base64Encoded: base64Str),
                          let ciImage = CIImage(data: data) else { return nil }
                    return .ciImage(ciImage)
                }
                
                if let url = URL(string: urlStr), (url.scheme == "http" || url.scheme == "https") {
                    return .url(url)
                }
                
                if let url = URL(string: urlStr) {
                    return .url(url)
                }
                return nil
            }
        }
#endif

        public func extractAudio() -> [Data] {
            guard let content = content, case .parts(let parts) = content else { return [] }
            return parts.compactMap { part -> Data? in
                guard part.type == "input_audio", 
                      let audio = part.inputAudio,
                      audio.format == "wav" else { return nil }
                return Data(base64Encoded: audio.data)
            }
        }
    }

    public enum MessageContent: Decodable {
        case string(String)
        case parts([ContentPart])

        public init(from decoder: Swift.Decoder) throws {
            let svc = try decoder.singleValueContainer()
            if let str = try? svc.decode(String.self) {
                self = .string(str)
            } else if let parts = try? svc.decode([ContentPart].self) {
                self = .parts(parts)
            } else {
                self = .string("")
            }
        }
    }

    public struct ContentPart: Decodable {
        public let type: String
        public let text: String?
        public let imageUrl: ImageUrlContent?
        public let inputAudio: InputAudioContent?

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageUrl = "image_url"
            case inputAudio = "input_audio"
        }
        
        public init(type: String, text: String? = nil, imageUrl: ImageUrlContent? = nil, inputAudio: InputAudioContent? = nil) {
            self.type = type
            self.text = text
            self.imageUrl = imageUrl
            self.inputAudio = inputAudio
        }
    }

    public struct InputAudioContent: Decodable {
        public let data: String
        public let format: String
        
        public init(data: String, format: String) {
            self.data = data
            self.format = format
        }
    }

    public struct ImageUrlContent: Decodable {
        public let url: String
        public let detail: String?
        
        public init(url: String, detail: String? = nil) {
            self.url = url
            self.detail = detail
        }
    }
}
