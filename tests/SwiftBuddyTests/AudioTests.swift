import XCTest
import Foundation

final class AudioTests: XCTestCase {

    func testAudio_AudioFlagAccepted() async throws {
        let process = Process()
        
        let projectRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent() // SwiftBuddyTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // SwiftLM
            
        let debugExecutableURL = projectRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/SwiftLM")
        let releaseExecutableURL = projectRoot.appendingPathComponent(".build/arm64-apple-macosx/release/SwiftLM")
        
        let executableURL = FileManager.default.fileExists(atPath: debugExecutableURL.path) 
                            ? debugExecutableURL 
                            : releaseExecutableURL
        
        guard FileManager.default.fileExists(atPath: executableURL.path) else {
            XCTFail("Could not find SwiftLM executable at \(debugExecutableURL.path)")
            return
        }
        
        process.executableURL = executableURL
        process.arguments = ["--model", "mlx-community/Qwen2-VL-2B-Instruct-4bit", "--audio"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        try process.run()
        
        let start = Date()
        var foundLoading = false
        var accumulated = ""
        while Date().timeIntervalSince(start) < 15.0 {
            let data = pipe.fileHandleForReading.availableData
            if !data.isEmpty {
                accumulated += String(data: data, encoding: .utf8) ?? ""
                if accumulated.contains("Loading") || accumulated.contains("SwiftLM") {
                    foundLoading = true
                    process.terminate()
                    break
                }
            } else {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        process.terminate()
        
        XCTAssertTrue(foundLoading, "Output should indicate SwiftLM started successfully with --audio flag. Got: \(accumulated)")
    }
}
