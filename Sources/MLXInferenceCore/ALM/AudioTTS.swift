import Foundation

// Feature 17 mock schema mapping
public struct SpeechRequest: Codable {
    public let model: String
    public let input: String
    public let voice: String
    public let responseFormat: String
    
    public enum CodingKeys: String, CodingKey {
        case model, input, voice
        case responseFormat = "response_format"
    }
}

public class TTSVocoder {
    public init() {}
    
    // Feature 18: Generate raw PCM waveform data (Float array)
    public func generate(from tokens: [Int]) -> [Float] {
        // Mocking Vocoder token decoding mapping to sound bytes
        return [0.0, 0.5, -0.5, 0.0]
    }
}

public class AudioWaveformGenerator {
    
    public init() {}

    // Feature 19: Valid WAV Output with RIFF Header
    public func encodeWav(pcm: [Float], sampleRate: Int) -> Data {
        var data = Data()
        
        // standard RIFF WAVE header bytes formulation
        let chunkSize = 36 + (pcm.count * 2) // 16-bit PCM = 2 bytes per sample
        
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: Int32(chunkSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: Int32(16).littleEndian) { Array($0) }) // subchunk1 size
        data.append(contentsOf: withUnsafeBytes(of: Int16(1).littleEndian) { Array($0) }) // PCM format
        data.append(contentsOf: withUnsafeBytes(of: Int16(1).littleEndian) { Array($0) }) // 1 Channel
        data.append(contentsOf: withUnsafeBytes(of: Int32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: Int32(sampleRate * 2).littleEndian) { Array($0) }) // ByteRate
        data.append(contentsOf: withUnsafeBytes(of: Int16(2).littleEndian) { Array($0) }) // BlockAlign
        data.append(contentsOf: withUnsafeBytes(of: Int16(16).littleEndian) { Array($0) }) // bits per sample
        
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: Int32(pcm.count * 2).littleEndian) { Array($0) })
        
        for sample in pcm {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
        }
        
        return data
    }
    
    // Feature 20: Streaming audio chunks sent as Server-Sent Events
    public func encodeSSEChunk(pcm: [Float]) -> Data {
        // We encode partial data inside SSE block
        // Assuming chunk maps heavily to OpenAI JSON lines 
        let rawBase64 = encodeWav(pcm: pcm, sampleRate: 24000).base64EncodedString()
        let jsonStr = "{\"audio\":\"\(rawBase64)\"}"
        
        var chunk = Data()
        chunk.append("data: \(jsonStr)\n\n".data(using: .utf8)!)
        return chunk
    }
}
