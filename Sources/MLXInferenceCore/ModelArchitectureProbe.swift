import Foundation
import MLXLMCommon

public struct ModelArchitectureInfo: Sendable, Equatable {
    public let modelType: String?
    public let processorClass: String?
    public let supportsVision: Bool

    public init(modelType: String?, processorClass: String?, supportsVision: Bool) {
        self.modelType = modelType
        self.processorClass = processorClass
        self.supportsVision = supportsVision
    }
}

public enum ModelArchitectureProbe {
    private static let knownVisionModelTypes: Set<String> = [
        "paligemma",
        "qwen2_vl",
        "qwen2-vl",
        "qwen2_5_vl",
        "qwen2.5-vl",
        "qwen3_vl",
        "qwen3-vl",
        "qwen3_5",
        "qwen3.5",
        "qwen3_5_moe",
        "idefics3",
        "gemma3",
        "gemma4",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "mistral3",
        "lfm2_vl",
        "lfm2-vl",
        "glm_ocr",
    ]

    private static let knownVisionProcessors: Set<String> = [
        "PaliGemmaProcessor",
        "Qwen2VLProcessor",
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
        "Idefics3Processor",
        "Gemma3Processor",
        "Gemma4Processor",
        "SmolVLMProcessor",
        "FastVLMProcessor",
        "PixtralProcessor",
        "Mistral3Processor",
        "Lfm2VlProcessor",
        "Glm46VProcessor",
    ]

    public static func inspect(
        configuration: ModelConfiguration,
        downloader: (any Downloader)? = nil
    ) async throws -> ModelArchitectureInfo {
        let modelDirectory = try await resolveModelDirectory(
            configuration: configuration,
            downloader: downloader
        )

        let config = try readRequiredJSON(
            at: modelDirectory.appendingPathComponent("config.json"),
        )
        let preprocessor = try readJSON(
            at: modelDirectory.appendingPathComponent("preprocessor_config.json"),
        )

        let modelType = config["model_type"] as? String
        let processorClass = preprocessor?["processor_class"] as? String
        let normalizedModelType = modelType?.lowercased().replacingOccurrences(of: ".", with: "_")

        let supportsVision =
            (normalizedModelType.map { knownVisionModelTypes.contains($0) } ?? false)
            || (processorClass.map { knownVisionProcessors.contains($0) } ?? false)
            || config["vision_config"] != nil
            || preprocessor?["image_processor_type"] != nil

        return ModelArchitectureInfo(
            modelType: modelType,
            processorClass: processorClass,
            supportsVision: supportsVision
        )
    }

    private static func resolveModelDirectory(
        configuration: ModelConfiguration,
        downloader: (any Downloader)?
    ) async throws -> URL {
        switch configuration.id {
        case .directory(let directory):
            return directory
        case .id(let id, let revision):
            guard let downloader else {
                throw ModelConfiguration.DirectoryError.unresolvedModelDirectory(id)
            }
            return try await downloader.download(
                id: id,
                revision: revision,
                matching: ["config.json", "preprocessor_config.json"],
                useLatest: false
            ) { _ in }
        }
    }

    private static func readRequiredJSON(at url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CocoaError(.fileNoSuchFile)
        }

        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let json = object as? [String: Any] else {
            throw CocoaError(.coderInvalidValue)
        }
        return json
    }

    private static func readJSON(at url: URL) throws -> [String: Any]? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }

        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data)
        return object as? [String: Any]
    }
}
