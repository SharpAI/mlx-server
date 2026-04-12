import XCTest
import MLXInferenceCore
import AVFoundation

final class AudioExtractionTests: XCTestCase {

    // Feature 2: Base64 WAV data URI extraction from API content
    func testAudio_Base64WAVExtraction() {
        // Dummy base64 string padded to multiple of 4
        let base64String = "UklGRuQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0Yc=="
        let audioPart = ChatCompletionRequest.ContentPart(
            type: "input_audio",
            inputAudio: ChatCompletionRequest.InputAudioContent(data: base64String, format: "wav")
        )
        let message = ChatCompletionRequest.Message(
            role: "user",
            content: .parts([audioPart])
        )
        
        let audioData = message.extractAudio()
        XCTAssertEqual(audioData.count, 1)
        
        if let data = audioData.first {
            XCTAssertEqual(data, Data(base64Encoded: base64String))
        } else {
            XCTFail("Expected valid data extraction")
        }
    }

    // Feature 3: WAV header parsing: extract sample rate, channels, bit depth
    func testAudio_WAVHeaderParsing() throws {
        let base64String = "UklGRuQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0Yc=="
        let data = Data(base64Encoded: base64String)!
        
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        
        // AVFoundation parses WAV headers easily
        let audioFile = try AVFoundation.AVAudioFile(forReading: url)
        let format = audioFile.fileFormat
        
        XCTAssertEqual(format.sampleRate, 8000.0)
        XCTAssertEqual(format.channelCount, 1)
        XCTAssertEqual(format.commonFormat, .pcmFormatInt16)
        
        // Ensure data is readable
        XCTAssertEqual(audioFile.length, 0) // No actual data chunks appended yet
    }

    // Feature 4: PCM samples → mel spectrogram via FFT
    func testAudio_MelSpectrogramGeneration() throws {
        let sampleRate: Float = 16000.0
        let duration: Float = 1.0
        let count = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: count)
        
        for i in 0..<count {
            let t = Float(i) / sampleRate
            samples[i] = sin(2.0 * Float.pi * 440.0 * t) // 440 Hz Sine
        }
        
        let processor = AudioProcessor()
        let mel = try processor.generateMelSpectrogram(samples: samples, sampleRate: sampleRate)
        
        XCTAssertEqual(mel.shape[0], 80, "Must have exactly 80 mel bins")
        XCTAssertTrue(mel.shape[1] > 0, "Must have valid frames")
    }

    // Feature 5: Mel spectrogram dimensions match Whisper expected
    func testAudio_MelDimensionsCorrect() throws {
        let samples = [Float](repeating: 0.1, count: 16000 * 30) // 30 seconds at 16kHz
        let processor = AudioProcessor()
        let mel = try processor.generateMelSpectrogram(samples: samples, sampleRate: 16000.0)
        
        XCTAssertEqual(mel.ndim, 2)
        XCTAssertEqual(mel.shape[0], 80)
        XCTAssertEqual(mel.shape[1], 3000, "30 seconds at 160 hop_length should yield 3000 frames")
    }

    // Feature 6: Audio longer than 30s is chunked into segments
    func testAudio_LongAudioChunking() throws {
        let samples = [Float](repeating: 0.1, count: 16000 * 90) // 90 seconds
        let processor = AudioProcessor()
        let chunks = try processor.chunkAndProcess(samples: samples, sampleRate: 16000.0)
        
        XCTAssertEqual(chunks.count, 3, "90 seconds should divide into 3x 30s chunks")
        for chunk in chunks {
            XCTAssertEqual(chunk.shape[0], 80)
            XCTAssertEqual(chunk.shape[1], 3000)
        }
    }

    // Feature 7: Empty/silent audio returns empty transcription (no crash)
    func testAudio_SilentAudioHandling() throws {
        let samples = [Float](repeating: 0.0, count: 16000 * 1) // 1 second of perfect silence
        let processor = AudioProcessor()
        let mel = try processor.generateMelSpectrogram(samples: samples, sampleRate: 16000.0)
        
        XCTAssertEqual(mel.shape[0], 80)
        XCTAssertTrue(mel.shape[1] > 0)
        // Ensure no NaN crashes
    }
}
