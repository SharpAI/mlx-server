import XCTest
import MLX
import MLXInferenceCore
import Foundation

final class AudioTTSTests: XCTestCase {

    override class func setUp() {
        super.setUp()
    }
    
    // Feature 17: `/v1/audio/speech` endpoint accepts text input
    func testAudio_TTSEndpointAccepts() throws {
        let jsonPayload = """
        {
            "model": "tts-1",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "wav"
        }
        """.data(using: .utf8)!
        
        let decoder = JSONDecoder()
        let request = try decoder.decode(SpeechRequest.self, from: jsonPayload)
        
        XCTAssertEqual(request.model, "tts-1")
        XCTAssertEqual(request.input, "Hello world")
        XCTAssertEqual(request.responseFormat, "wav")
    }

    // Feature 18: TTS vocoder generates valid PCM waveform from tokens
    func testAudio_VocoderOutput() {
        // Assume text token embeddings [101, 102]
        let tokens = [101, 102]
        let vocoder = TTSVocoder()
        let pcmOutput = vocoder.generate(from: tokens)
        
        // Ensure standard audio depth generation, say 24000 PCM ticks per token
        XCTAssertGreaterThan(pcmOutput.count, 0)
    }

    // Feature 19: Generated WAV has valid header and is playable
    func testAudio_ValidWAVOutput() {
        let pcmData: [Float] = [0.0, 0.5, -0.5, 0.0]
        let audioGenerator = AudioWaveformGenerator()
        
        let wavData = audioGenerator.encodeWav(pcm: pcmData, sampleRate: 24000)
        
        XCTAssertGreaterThan(wavData.count, 44, "WAV header is 44 bytes, so file must be strictly larger")
        // Check RIFF header signature
        let signature = String(data: wavData.prefix(4), encoding: .ascii)
        XCTAssertEqual(signature, "RIFF")
    }
    
    // Feature 20: Streaming audio chunks sent as Server-Sent Events
    func testAudio_StreamingTTSOutput() {
        let pcmFrame: [Float] = [0.1, 0.2, 0.3]
        let audioGenerator = AudioWaveformGenerator()
        
        let sseChunk = audioGenerator.encodeSSEChunk(pcm: pcmFrame)
        let chunkString = String(data: sseChunk, encoding: .utf8)
        
        XCTAssertNotNil(chunkString)
        XCTAssertTrue(chunkString!.hasPrefix("data: {"))
        XCTAssertTrue(chunkString!.hasSuffix("}\n\n"))
    }
}


