import XCTest
@testable import MLXInferenceCore
import MLXLMCommon

final class ModelArchitectureProbeTests: XCTestCase {
    func testDetectsVisionModelFromLocalConfig() async throws {
        let directory = try makeTempModelDirectory(
            config: """
            {
              "model_type": "lfm2-vl",
              "vision_config": { "model_type": "siglip2_vision_model" }
            }
            """,
            preprocessor: """
            {
              "processor_class": "Lfm2VlProcessor"
            }
            """
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let info = try await ModelArchitectureProbe.inspect(
            configuration: ModelConfiguration(directory: directory)
        )

        XCTAssertEqual(info.modelType, "lfm2-vl")
        XCTAssertEqual(info.processorClass, "Lfm2VlProcessor")
        XCTAssertTrue(info.supportsVision)
    }

    func testLeavesTextModelAsNonVision() async throws {
        let directory = try makeTempModelDirectory(
            config: """
            {
              "model_type": "lfm2"
            }
            """
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let info = try await ModelArchitectureProbe.inspect(
            configuration: ModelConfiguration(directory: directory)
        )

        XCTAssertEqual(info.modelType, "lfm2")
        XCTAssertNil(info.processorClass)
        XCTAssertFalse(info.supportsVision)
    }

    private func makeTempModelDirectory(
        config: String,
        preprocessor: String? = nil
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try XCTUnwrap(config.data(using: .utf8)).write(
            to: directory.appendingPathComponent("config.json")
        )
        if let preprocessor {
            try XCTUnwrap(preprocessor.data(using: .utf8)).write(
                to: directory.appendingPathComponent("preprocessor_config.json")
            )
        }

        return directory
    }
}
