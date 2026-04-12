import Foundation
#if canImport(CoreImage)
import CoreImage
#endif
#if canImport(MLXVLM)
import MLXVLM
#endif

// A simplified generic equivalent to extract images for unit testing purpose.
// This matches Server.swift `ChatCompletionRequest.Message` logic but is public for testing.

public struct ImageExtractor {
    public static func extractImages(from parts: [[String: String]]) -> [CoreImage.CIImage?] {
        return parts.compactMap { part -> CoreImage.CIImage? in
            guard part["type"] == "image_url", let urlStr = part["url"] else { return nil }
            
            if urlStr.hasPrefix("data:") {
                guard let commaIdx = urlStr.firstIndex(of: ",") else { return nil }
                let base64Str = String(urlStr[urlStr.index(after: commaIdx)...])
                guard let data = Data(base64Encoded: base64Str) else { return nil }
                return CIImage(data: data)
            }
            
            // Note: In tests we might skip real HTTP loading due to blocking, 
            // but the URL string parser handles it.
            return nil
        }
    }
}
