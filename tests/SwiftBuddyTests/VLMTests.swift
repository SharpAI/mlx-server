import XCTest
import Foundation

final class VLMTests: XCTestCase {
    
    // Feature 1: --vision flag loads VLM instead of LLM
    func testVLM_VisionFlagLoadsVLMFactory() async throws {
        let accumulated = try await captureStartupOutput(arguments: [
            "--model", "mlx-community/Qwen2-VL-2B-Instruct-4bit", "--vision",
        ])
        let found = accumulated.contains("Loading") || accumulated.contains("VLM")
        XCTAssertTrue(found, "Output should indicate VLM is loading. Got: \(accumulated)")
    }


    private func captureStartupOutput(
        arguments: [String],
        timeout: TimeInterval = 15.0
    ) async throws -> String {
        let process = Process()
        let projectRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let debugExecutableURL = projectRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/SwiftLM")
        let releaseExecutableURL = projectRoot.appendingPathComponent(".build/arm64-apple-macosx/release/SwiftLM")
        let executableURL = FileManager.default.fileExists(atPath: debugExecutableURL.path)
            ? debugExecutableURL
            : releaseExecutableURL

        guard FileManager.default.fileExists(atPath: executableURL.path) else {
            XCTFail("Could not find SwiftLM executable at \(debugExecutableURL.path)")
            return ""
        }

        process.executableURL = executableURL
        process.arguments = arguments

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        defer {
            if process.isRunning {
                process.terminate()
            }
        }

        let start = Date()
        var accumulated = ""
        while Date().timeIntervalSince(start) < timeout {
            let data = pipe.fileHandleForReading.availableData
            if !data.isEmpty {
                accumulated += String(data: data, encoding: .utf8) ?? ""
                if accumulated.contains("Loading") || accumulated.contains("VLM") {
                    break
                }
            } else {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
        }

        return accumulated
    }
}
