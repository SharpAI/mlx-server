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
        // macOS: match defaultHubApi exactly so models are shared with CLI server
        return FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("huggingface/hub")
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
        let dir = cacheRoot.appendingPathComponent(hubDirName(for: modelId))
        return FileManager.default.fileExists(atPath: dir.path) ? dir : nil
    }

    /// True if a model's cache directory exists and contains files.
    // The snapshot directory is where safetensors files live inside the HF hub layout:
    // <cacheRoot>/models--org--name/snapshots/main/
    public static func snapshotDirectory(for modelId: String) -> URL {
        return cacheRoot
            .appendingPathComponent(hubDirName(for: modelId))
            .appendingPathComponent("snapshots/main")
    }

    public static func isDownloaded(_ modelId: String) -> Bool {
        guard let dir = cacheDirectory(for: modelId) else { return false }
        // Must have a snapshots subdirectory with content
        let snapshots = dir.appendingPathComponent("snapshots")
        return FileManager.default.fileExists(atPath: snapshots.path)
    }

    // MARK: — Disk Operations

    /// Total bytes used by all model files on disk.
    public static func totalDiskUsage() -> Int64 {
        guard FileManager.default.fileExists(atPath: cacheRoot.path) else { return 0 }
        return directorySize(at: cacheRoot)
    }

    /// Bytes used by a specific model on disk.
    public static func sizeOnDisk(for modelId: String) -> Int64 {
        guard let dir = cacheDirectory(for: modelId) else { return 0 }
        return directorySize(at: dir)
    }

    /// Delete all cached files for a model.
    public static func delete(_ modelId: String) throws {
        guard let dir = cacheDirectory(for: modelId) else { return }
        try FileManager.default.removeItem(at: dir)
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

        var results: [ScannedModel] = []
        for dir in contents {
            guard dir.lastPathComponent.hasPrefix("models--") else { continue }

            // Reverse the naming convention to get the model ID
            let modelId = dir.lastPathComponent
                .replacingOccurrences(of: "^models--", with: "", options: .regularExpression)
                .replacingOccurrences(of: "--", with: "/")

            // Only include models in our curated catalog
            guard ModelCatalog.all.contains(where: { $0.id == modelId }) else { continue }
            guard isDownloaded(modelId) else { continue }  // skip partial downloads

            let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            results.append(ScannedModel(
                modelId: modelId,
                cacheDirectory: dir,
                sizeBytes: directorySize(at: dir),
                modifiedDate: modified
            ))
        }
        return results.sorted { ($0.modifiedDate ?? .distantPast) > ($1.modifiedDate ?? .distantPast) }
    }

    // MARK: — Helpers

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
