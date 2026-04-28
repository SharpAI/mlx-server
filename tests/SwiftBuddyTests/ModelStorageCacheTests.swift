import XCTest
@testable import MLXInferenceCore

final class ModelStorageCacheTests: XCTestCase {
    private let modelId = "mlx-community/Qwen3.5-122B-A10B-4bit"
    private let commit = "1234567890abcdef"

    override func setUpWithError() throws {
        try super.setUpWithError()
        try requireHarnessHFHome()
        try? FileManager.default.removeItem(at: ModelStorage.cacheRoot)
        try FileManager.default.createDirectory(at: ModelStorage.cacheRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if isHarnessHFHome() {
            try? FileManager.default.removeItem(at: ModelStorage.cacheRoot)
        }
        try super.tearDownWithError()
    }

    func testReadsNestedQwenContextLengthFromRefsMainSnapshot() throws {
        let snapshot = try makeSnapshot(
            config: """
            {
              "model_type": "qwen3_5_vl",
              "text_config": {
                "max_position_embeddings": 262144
              }
            }
            """
        )
        try writeShard(named: "model-00001-of-00014.safetensors", in: snapshot, byteCount: 2_048)

        XCTAssertEqual(ModelStorage.readMaxContextLength(for: modelId), 262_144)
    }

    func testVerifyModelIntegrityRejectsMissingIndexedShard() throws {
        let snapshot = try makeSnapshot(
            config: """
            {
              "model_type": "qwen3_5_vl",
              "text_config": { "max_position_embeddings": 262144 }
            }
            """,
            index: """
            {
              "metadata": { "total_size": 2048 },
              "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00014.safetensors"
              }
            }
            """
        )

        XCTAssertFalse(FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("model-00001-of-00014.safetensors").path))
        XCTAssertFalse(ModelStorage.verifyModelIntegrity(for: modelId))
        XCTAssertFalse(ModelStorage.isDownloaded(modelId))
    }

    func testVerifyModelIntegrityAcceptsCompleteIndexedShardFixture() throws {
        let snapshot = try makeSnapshot(
            config: """
            {
              "model_type": "qwen3_5_vl",
              "text_config": { "max_position_embeddings": 262144 }
            }
            """,
            index: """
            {
              "metadata": { "total_size": 4096 },
              "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00014.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00014.safetensors"
              }
            }
            """
        )
        try writeShard(named: "model-00001-of-00014.safetensors", in: snapshot, byteCount: 4_096)

        XCTAssertTrue(ModelStorage.verifyModelIntegrity(for: modelId))
        XCTAssertTrue(ModelStorage.isDownloaded(modelId))
    }

    func testMaterializedHubApiDirectoryIsPrimaryModelLocation() throws {
        let materialized = try makeMaterializedDirectory(
            config: """
            {
              "model_type": "qwen3_5_vl",
              "text_config": { "max_position_embeddings": 262144 }
            }
            """,
            index: """
            {
              "metadata": { "total_size": 4096 },
              "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00014.safetensors"
              }
            }
            """
        )
        try writeShard(named: "model-00001-of-00014.safetensors", in: materialized, byteCount: 4_096)

        XCTAssertEqual(ModelStorage.snapshotDirectory(for: modelId), materialized)
        XCTAssertTrue(ModelStorage.verifyModelIntegrity(for: modelId))
        XCTAssertTrue(ModelStorage.isDownloaded(modelId))
        XCTAssertEqual(ModelStorage.readMaxContextLength(for: modelId), 262_144)
    }

    func testDeleteRemovesMaterializedAndHubCacheLayouts() throws {
        let materialized = try makeMaterializedDirectory(config: "{ \"max_position_embeddings\": 32768 }")
        try writeShard(named: "model.safetensors", in: materialized, byteCount: 2_048)

        let snapshot = try makeSnapshot(config: "{ \"max_position_embeddings\": 32768 }")
        try writeShard(named: "model.safetensors", in: snapshot, byteCount: 2_048)

        XCTAssertTrue(FileManager.default.fileExists(atPath: materialized.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: snapshot.path))

        try ModelStorage.delete(modelId)

        XCTAssertFalse(FileManager.default.fileExists(atPath: materialized.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: snapshot.path))
    }

    func testVerifyModelIntegrityRejectsTinySymlinkTarget() throws {
        let snapshot = try makeSnapshot(
            config: """
            {
              "model_type": "qwen3_5_vl",
              "text_config": { "max_position_embeddings": 262144 }
            }
            """,
            index: """
            {
              "metadata": { "total_size": 4096 },
              "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00014.safetensors"
              }
            }
            """
        )

        let blob = snapshot.deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("blobs", isDirectory: true)
            .appendingPathComponent("tiny-shard")
        try FileManager.default.createDirectory(at: blob.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("tiny".utf8).write(to: blob)
        try FileManager.default.createSymbolicLink(
            at: snapshot.appendingPathComponent("model-00001-of-00014.safetensors"),
            withDestinationURL: blob
        )

        XCTAssertFalse(ModelStorage.verifyModelIntegrity(for: modelId))
        XCTAssertFalse(ModelStorage.isDownloaded(modelId))
    }

    func testIncompleteMarkerMakesModelNotDownloadedAndDiscoverable() throws {
        let snapshot = try makeSnapshot(
            config: """
            {
              "model_type": "qwen3_5_vl",
              "text_config": { "max_position_embeddings": 262144 }
            }
            """
        )
        try writeShard(named: "model.safetensors", in: snapshot, byteCount: 2_048)
        try Data("partial".utf8).write(to: snapshot.appendingPathComponent("model.safetensors.incomplete"))

        XCTAssertFalse(ModelStorage.isDownloaded(modelId))
        XCTAssertTrue(ModelStorage.hasIncompleteFiles(for: modelId))
        XCTAssertTrue(ModelStorage.scanIncompleteDownloads().contains { $0.id == modelId })
    }

    private func makeSnapshot(
        config: String,
        index: String? = nil
    ) throws -> URL {
        let modelDir = ModelStorage.cacheRoot.appendingPathComponent(ModelStorage.hubDirName(for: modelId))
        let refsDir = modelDir.appendingPathComponent("refs", isDirectory: true)
        let snapshot = modelDir
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(commit, isDirectory: true)

        try FileManager.default.createDirectory(at: refsDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try commit.write(to: refsDir.appendingPathComponent("main"), atomically: true, encoding: .utf8)
        try config.write(to: snapshot.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: snapshot.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        if let index {
            try index.write(to: snapshot.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        }
        return snapshot
    }

    private func makeMaterializedDirectory(
        config: String,
        index: String? = nil
    ) throws -> URL {
        let materialized = ModelStorage.cacheRoot
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)

        try FileManager.default.createDirectory(at: materialized, withIntermediateDirectories: true)
        try config.write(to: materialized.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: materialized.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        if let index {
            try index.write(to: materialized.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        }
        return materialized
    }

    private func writeShard(named name: String, in snapshot: URL, byteCount: Int) throws {
        let data = Data(repeating: 0x5A, count: byteCount)
        try data.write(to: snapshot.appendingPathComponent(name))
    }

    private func requireHarnessHFHome() throws {
        guard isHarnessHFHome() else {
            throw XCTSkip("ModelStorageCacheTests require HF_HOME to point at the isolated harness cache")
        }
    }

    private func isHarnessHFHome() -> Bool {
        guard let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] else { return false }
        return hfHome.contains("swiftbuddy-model-loading-harness")
    }
}
