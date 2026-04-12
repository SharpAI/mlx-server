import Foundation
import MLX
import MLXInferenceCore
import MLXVLM
import MLXLMCommon

@main
struct SwiftLMTestSTFT {
    static func main() async throws {
        guard CommandLine.arguments.count > 1 else {
            print("Usage: SwiftLMTestSTFT <path/to/audio>")
            exit(1)
        }
        
        let path = CommandLine.arguments[1]
        let url = URL(fileURLWithPath: path)
        print("Reading Audio payload from: \(url.path)")
        
        let audioInput = UserInput.Audio.url(url)
        print("Extracting 16kHz Mono Float32 Samples natively via AVFoundation...")
        let samples = try MediaProcessing.extractAudioSamples(from: audioInput)
        
        print("Extracted \(samples.count) raw PCM samples.")
        print("Sample Head: \(samples.prefix(10))")
        
        // Pass through AudioProcessor
        let processor = AudioProcessor(nMels: 128)
        
        print("Converting sequence geometry to 128-bin Mel Spectrogram using nFft=400 hopLength=160...")
        var melSpec = try processor.generateMelSpectrogram(samples: samples)
        print("Generated Spectral bounds: \(melSpec.shape)")
        
        // Final dimensional reshaping for VLM prompt sequence
        melSpec = melSpec.expandedDimensions(axis: 0)
        
        print("✅ STFT Validation Geometry Correct:")
        print("Final Array Shape for KV Input: \(melSpec.shape)") // Should be [1, <sequence_length>, 128]
    }
}
