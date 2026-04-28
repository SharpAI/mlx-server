// ModelStorage.swift — Platform-aware model storage resolution
// macOS: ~/Library/Caches/huggingface/hub/  (same as defaultHubApi)
// iOS:   ~/Library/Application Support/SwiftBuddy/Models/ (persistent, excluded from iCloud)

import Foundation

public enum ModelStorage {

    // MARK: — Platform Paths

    /// Root directory where model files are stored on this platform.
    /// This is the `downloadBase` passed to `HubApi`.
    public static var cacheRoot: URL {
        #if os(macOS)
        // macOS: Single source of truth with Python (huggingface-cli / mlx_lm)
        if let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        #else
        // iOS: Application Support — persistent, NOT purgeable, excluded from iCloud
        return applicationSupportModelsRoot
        #endif
    }

    /// iOS-specific persistent models directory.
    public static var applicationSupportModelsRoot: URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("SwiftBuddy/Models", isDirectory: true)
        ensureDirectory(base)
        return base
    }

    /// HuggingFace hub subdirectory name for a model ID.
    /// e.g. "mlx-community/Qwen2.5-7B-Instruct-4bit"
    ///   → "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
    public static func hubDirName(for modelId: String) -> String {
        "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    }

    /// Local cache directory for a model, or nil if not downloaded.
    public static func cacheDirectory(for modelId: String) -> URL? {
        materializedDirectory(for: modelId) ?? hubCacheDirectory(for: modelId)
    }

    /// Swift Hub's materialized repository directory.
    /// `HubApi(downloadBase: cacheRoot).snapshot(from:)` writes here.
    public static func materializedDirectory(for modelId: String) -> URL? {
        let dir = materializedDirectoryURL(for: modelId)
        return directoryExists(dir) ? dir : nil
    }

    private static func materializedDirectoryURL(for modelId: String) -> URL {
        cacheRoot
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)
    }

    /// Hugging Face hub cache directory used by Python tools and older SwiftBuddy paths.
    public static func hubCacheDirectory(for modelId: String) -> URL? {
        let dir = hubCacheDirectoryURL(for: modelId)
        return directoryExists(dir) ? dir : nil
    }

    private static func hubCacheDirectoryURL(for modelId: String) -> URL {
        cacheRoot.appendingPathComponent(hubDirName(for: modelId), isDirectory: true)
    }

    private static func directoryExists(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    /// True if a model's cache directory exists and contains files.
    // The snapshot directory is where safetensors files live inside the HF hub layout:
    // <cacheRoot>/models--org--name/snapshots/main/
    public static func snapshotDirectory(for modelId: String) -> URL {
        return materializedDirectory(for: modelId) ?? resolvedSnapshotDirectory(for: modelId) ?? cacheRoot
            .appendingPathComponent(hubDirName(for: modelId))
            .appendingPathComponent("snapshots/main")
    }

    /// Resolve the active snapshot directory for a model in the Hugging Face hub cache.
    /// Prefer refs/main because snapshot directories are usually commit hashes, not "main".
    public static func resolvedSnapshotDirectory(for modelId: String) -> URL? {
        guard let dir = hubCacheDirectory(for: modelId) else { return nil }

        let snapshotsDir = dir.appendingPathComponent("snapshots", isDirectory: true)
        guard FileManager.default.fileExists(atPath: snapshotsDir.path) else { return nil }

        let refsMain = dir.appendingPathComponent("refs/main")
        if let hashString = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !hashString.isEmpty {
            let snapshot = snapshotsDir.appendingPathComponent(hashString, isDirectory: true)
            if FileManager.default.fileExists(atPath: snapshot.path) {
                return snapshot
            }
        }

        let mainSnapshot = snapshotsDir.appendingPathComponent("main", isDirectory: true)
        if FileManager.default.fileExists(atPath: mainSnapshot.path) {
            return mainSnapshot
        }

        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        let directories = contents.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        return directories.count == 1 ? directories[0] : nil
    }

    public static func isDownloaded(_ modelId: String) -> Bool {
        verifyModelIntegrity(for: modelId, logFailures: false)
    }

    // MARK: — Model Config Inspection

    /// Read the model's maximum context length from its config.json.
    /// Checks `text_config.max_position_embeddings` first (VLM/MoE models),
    /// then falls back to top-level `max_position_embeddings`.
    public static func readMaxContextLength(for modelId: String) -> Int? {
        guard let config = readModelConfig(for: modelId) else { return nil }

        // VLM/MoE models nest the context length in text_config
        if let textConfig = config["text_config"] as? [String: Any],
           let maxPos = textConfig["max_position_embeddings"] as? Int {
            return maxPos
        }

        // Standard LLMs have it at top level
        if let maxPos = config["max_position_embeddings"] as? Int {
            return maxPos
        }

        // Fallback: some models use n_ctx or max_seq_len
        if let nCtx = config["n_ctx"] as? Int { return nCtx }
        if let maxSeq = config["max_seq_len"] as? Int { return maxSeq }

        return nil
    }

    /// Read the raw config.json dictionary for a downloaded model.
    /// Verifies that all required safetensors files are present in the snapshot directory.
    /// This prevents the engine from entering `.ready` state if a download was interrupted or corrupted.
    public static func verifyModelIntegrity(for modelId: String) -> Bool {
        verifyModelIntegrity(for: modelId, logFailures: true)
    }

    private static func verifyModelIntegrity(for modelId: String, logFailures: Bool) -> Bool {
        if hasIncompleteFiles(for: modelId) {
            if logFailures { print("[ModelStorage] Integrity Check Failed: Incomplete download files remain for \(modelId)") }
            return false
        }

        for directory in modelContentDirectories(for: modelId) {
            if validateModelFiles(in: directory, logFailures: logFailures) {
                return true
            }
        }

        if logFailures { print("[ModelStorage] Integrity Check Failed: No valid model files found for \(modelId)") }
        return false
    }

    private static func modelContentDirectories(for modelId: String) -> [URL] {
        var directories: [URL] = []
        if let materialized = materializedDirectory(for: modelId) {
            directories.append(materialized)
        }
        if let snapshot = resolvedSnapshotDirectory(for: modelId), !directories.contains(snapshot) {
            directories.append(snapshot)
        }
        return directories
    }

    private static func validateModelFiles(in snapshotDir: URL, logFailures: Bool) -> Bool {
        // 0. Verify core metadata files
        let requiredJsonFiles = ["config.json", "tokenizer.json"]
        for file in requiredJsonFiles {
            let path = snapshotDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: path.path) {
                // Some models might not have tokenizer.json if they use tokenizer.model, so we only strictly enforce config.json
                if file == "config.json" {
                    if logFailures { print("[ModelStorage] Integrity Check Failed: Missing \(file) in \(snapshotDir.path)") }
                    return false
                }
            } else if fileSizeResolvingSymlink(path) == 0 {
                if logFailures { print("[ModelStorage] Integrity Check Failed: \(file) is corrupted (0 bytes)") }
                return false
            }
        }

        // 1. Try to read model.safetensors.index.json
        let indexJsonPath = snapshotDir.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexJsonPath.path) {
            guard let data = try? Data(contentsOf: indexJsonPath),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = json["weight_map"] as? [String: String] else {
                return false
            }
            // Collect all unique safetensors filenames
            let requiredFiles = Set(weightMap.values)
            var totalShardBytes: Int64 = 0
            for file in requiredFiles {
                let filePath = snapshotDir.appendingPathComponent(file)
                guard let size = fileSizeResolvingSymlink(filePath) else {
                    if logFailures { print("[ModelStorage] Integrity Check Failed: Missing \(file)") }
                    return false
                }
                guard size > 1024 else {
                    if logFailures { print("[ModelStorage] Integrity Check Failed: \(file) is too small (\(size) bytes)") }
                    return false
                }
                totalShardBytes += size
            }

            if let metadata = json["metadata"] as? [String: Any],
               let expectedTensorBytes = int64Value(metadata["total_size"]),
               totalShardBytes < expectedTensorBytes {
                if logFailures {
                    print("[ModelStorage] Integrity Check Failed: shard bytes \(totalShardBytes) below index total_size \(expectedTensorBytes)")
                }
                return false
            }
            return true
        }

        // 2. If no index.json, it might be a single safetensors file model
        let singleSafetensors = snapshotDir.appendingPathComponent("model.safetensors")
        if let size = fileSizeResolvingSymlink(singleSafetensors), size > 1024 {
            return true
        }

        if logFailures { print("[ModelStorage] Integrity Check Failed: No safetensors found in \(snapshotDir.path)") }
        return false
    }

    public static func readModelConfig(for modelId: String) -> [String: Any]? {
        for directory in modelContentDirectories(for: modelId) {
            let configPath = directory.appendingPathComponent("config.json")
            guard let data = try? Data(contentsOf: configPath),
                  let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            return config
        }
        return nil
    }

    // MARK: — Disk Operations

    /// Total bytes used by all model files on disk.
    public static func totalDiskUsage() -> Int64 {
        guard FileManager.default.fileExists(atPath: cacheRoot.path) else { return 0 }
        return directorySize(at: cacheRoot)
    }

    /// Bytes used by a specific model on disk.
    public static func sizeOnDisk(for modelId: String) -> Int64 {
        associatedDirectories(for: modelId).reduce(Int64(0)) { $0 + directorySize(at: $1) }
    }

    /// Delete all cached files for a model.
    public static func delete(_ modelId: String) throws {
        var firstError: Error?
        for dir in associatedDirectories(for: modelId) {
            do {
                try FileManager.default.removeItem(at: dir)
            } catch {
                if firstError == nil { firstError = error }
            }
        }
        if let firstError { throw firstError }
    }

    // MARK: — iCloud Exclusion (iOS)

    /// Mark a URL as excluded from iCloud backup.
    /// Call this after creating any model storage directory on iOS.
    public static func excludeFromBackup(_ url: URL) {
        var mutable = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutable.setResourceValues(values)
    }

    // MARK: — Scan

    public struct ScannedModel: Sendable {
        public let modelId: String
        public let cacheDirectory: URL
        public let sizeBytes: Int64
        public let modifiedDate: Date?
    }

    /// Scan the cache root and return all recognizable downloaded models.
    /// Only returns models present in `ModelCatalog.all`.
    public static func scanDownloadedModels() -> [ScannedModel] {
        guard FileManager.default.fileExists(atPath: cacheRoot.path),
              let contents = try? FileManager.default.contentsOfDirectory(
                at: cacheRoot,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
              )
        else { return [] }

        var resultsById: [String: ScannedModel] = [:]
        for dir in contents {
            if dir.lastPathComponent.hasPrefix("models--") {
                let modelId = dir.lastPathComponent
                    .replacingOccurrences(of: "^models--", with: "", options: .regularExpression)
                    .replacingOccurrences(of: "--", with: "/")
                addScannedModelIfDownloaded(modelId: modelId, dir: dir, resultsById: &resultsById)
            } else if dir.lastPathComponent == "models" {
                guard let organizations = try? FileManager.default.contentsOfDirectory(
                    at: dir,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }

                for organization in organizations where directoryExists(organization) {
                    guard let modelDirs = try? FileManager.default.contentsOfDirectory(
                        at: organization,
                        includingPropertiesForKeys: [.contentModificationDateKey],
                        options: [.skipsHiddenFiles]
                    ) else { continue }

                    for modelDir in modelDirs where directoryExists(modelDir) {
                        let modelId = "\(organization.lastPathComponent)/\(modelDir.lastPathComponent)"
                        addScannedModelIfDownloaded(modelId: modelId, dir: modelDir, resultsById: &resultsById)
                    }
                }
            }
        }
        return resultsById.values.sorted { ($0.modifiedDate ?? .distantPast) > ($1.modifiedDate ?? .distantPast) }
    }

    private static func addScannedModelIfDownloaded(
        modelId: String,
        dir: URL,
        resultsById: inout [String: ScannedModel]
    ) {
        guard isDownloaded(modelId) else { return }

        let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        let candidate = ScannedModel(
            modelId: modelId,
            cacheDirectory: cacheDirectory(for: modelId) ?? dir,
            sizeBytes: sizeOnDisk(for: modelId),
            modifiedDate: modified
        )

        if let existing = resultsById[modelId],
           (existing.modifiedDate ?? .distantPast) >= (candidate.modifiedDate ?? .distantPast) {
            return
        }
        resultsById[modelId] = candidate
    }

    // MARK: — Incomplete Downloads

    /// A model whose download was interrupted and can be resumed.
    public struct IncompleteDownload: Identifiable, Sendable {
        public let id: String  // modelId
        public let cacheDirectory: URL
        /// Bytes downloaded so far (sum of complete + incomplete files)
        public let downloadedBytes: Int64
        /// When the partial download was last modified
        public let lastModified: Date?
    }

    /// Check whether a model directory has any `.incomplete` partial files (iOS path)
    /// or incomplete blobs (macOS HubApi path).
    public static func hasIncompleteFiles(for modelId: String) -> Bool {
        associatedDirectories(for: modelId).contains { countIncompleteFiles(in: $0) > 0 }
    }

    /// Scan the cache root for model directories that have partial downloads
    /// but are NOT fully downloaded (i.e. `isDownloaded()` returns false, or
    /// the directory contains `.incomplete` files).
    public static func scanIncompleteDownloads() -> [IncompleteDownload] {
        guard FileManager.default.fileExists(atPath: cacheRoot.path),
              let contents = try? FileManager.default.contentsOfDirectory(
                at: cacheRoot,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
              )
        else { return [] }

        var resultsById: [String: IncompleteDownload] = [:]
        for dir in contents {
            if dir.lastPathComponent.hasPrefix("models--") {
                let modelId = dir.lastPathComponent
                    .replacingOccurrences(of: "^models--", with: "", options: .regularExpression)
                    .replacingOccurrences(of: "--", with: "/")
                addIncompleteDownloadIfNeeded(modelId: modelId, dir: dir, resultsById: &resultsById)
            } else if dir.lastPathComponent == "models" {
                guard let organizations = try? FileManager.default.contentsOfDirectory(
                    at: dir,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }

                for organization in organizations where directoryExists(organization) {
                    guard let modelDirs = try? FileManager.default.contentsOfDirectory(
                        at: organization,
                        includingPropertiesForKeys: [.contentModificationDateKey],
                        options: [.skipsHiddenFiles]
                    ) else { continue }

                    for modelDir in modelDirs where directoryExists(modelDir) {
                        let modelId = "\(organization.lastPathComponent)/\(modelDir.lastPathComponent)"
                        addIncompleteDownloadIfNeeded(modelId: modelId, dir: modelDir, resultsById: &resultsById)
                    }
                }
            }
        }
        return resultsById.values.sorted { ($0.lastModified ?? .distantPast) > ($1.lastModified ?? .distantPast) }
    }

    private static func addIncompleteDownloadIfNeeded(
        modelId: String,
        dir: URL,
        resultsById: inout [String: IncompleteDownload]
    ) {
        // Skip fully completed models unless they have leftover .incomplete files.
        if isDownloaded(modelId) && !hasIncompleteFiles(for: modelId) {
            return
        }

        // Must have SOME content (not just an empty directory).
        let size = directorySize(at: dir)
        guard size > 0 else { return }

        let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        let candidate = IncompleteDownload(
            id: modelId,
            cacheDirectory: dir,
            downloadedBytes: size,
            lastModified: modified
        )

        if let existing = resultsById[modelId],
           (existing.lastModified ?? .distantPast) >= (candidate.lastModified ?? .distantPast) {
            return
        }
        resultsById[modelId] = candidate
    }

    /// Count `.incomplete` files in a directory tree.
    private static func countIncompleteFiles(in directory: URL) -> Int {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var count = 0
        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension == "incomplete" {
                count += 1
            }
        }
        return count
    }

    // MARK: — Helpers

    private static func associatedDirectories(for modelId: String) -> [URL] {
        let candidates = [
            materializedDirectoryURL(for: modelId),
            hubCacheDirectoryURL(for: modelId),
        ]

        var seen = Set<String>()
        return candidates.filter { url in
            guard directoryExists(url), !seen.contains(url.path) else { return false }
            seen.insert(url.path)
            return true
        }
    }

    private static func fileSizeResolvingSymlink(_ url: URL) -> Int64? {
        let resolved = url.resolvingSymlinksInPath()
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: resolved.path) else { return nil }
        if let size = attrs[.size] as? Int64 { return size }
        if let size = attrs[.size] as? NSNumber { return size.int64Value }
        return nil
    }

    private static func int64Value(_ value: Any?) -> Int64? {
        switch value {
        case let value as Int64: return value
        case let value as Int: return Int64(value)
        case let value as NSNumber: return value.int64Value
        case let value as String: return Int64(value)
        default: return nil
        }
    }

    private static func ensureDirectory(_ url: URL) {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        #if !os(macOS)
        excludeFromBackup(url)
        #endif
    }

    static func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
        return total
    }
}
