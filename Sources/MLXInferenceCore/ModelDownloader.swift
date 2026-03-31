// ModelDownloader.swift — Platform-aware model downloading
//
// macOS: LLMModelFactory.loadContainer() handles download + progress  
//         (called directly from InferenceEngine.load — no separate downloader needed)
//
// iOS:   Uses HuggingFace API to enumerate model files, then downloads
//         each file via URLSession background session to ModelStorage.cacheRoot
//         so LLMModelFactory can find them on next load without re-downloading.

import Foundation
import Hub
import MLXLMCommon

// MARK: — Download Progress

public struct DownloadFileProgress: Sendable {
    public let modelId: String
    public let fileName: String
    public let fileIndex: Int
    public let fileCount: Int
    public let fileFractionCompleted: Double
    public let totalBytesDownloaded: Int64
    public let speedBytesPerSec: Double?

    public var overallFraction: Double {
        let fileDone = Double(max(fileIndex - 1, 0)) / Double(max(fileCount, 1))
        let fileProgress = fileFractionCompleted / Double(max(fileCount, 1))
        return min(fileDone + fileProgress, 1.0)
    }

    public var speedString: String {
        guard let s = speedBytesPerSec else { return "" }
        return s >= 1_000_000
            ? String(format: "%.1f MB/s", s / 1_000_000)
            : String(format: "%.0f KB/s", s / 1_000)
    }
}

// MARK: — Downloader actor

public actor ModelDownloader {

    public static let shared = ModelDownloader()
    private init() {}

    // MARK: — iOS: URLSession background download

    #if !os(macOS)

    private lazy var backgroundSession: URLSession = {
        let config = URLSessionConfiguration.background(
            withIdentifier: "com.sharpai.swiftlmchat.modeldownload"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.allowsConstrainedNetworkAccess = true
        config.allowsExpensiveNetworkAccess = true
        return URLSession(configuration: config)
    }()

    /// HuggingFace model tree API response — only the fields we need.
    private struct HFModelInfo: Decodable {
        let siblings: [HFFile]
        struct HFFile: Decodable {
            let rfilename: String
        }
    }

    /// Fetch the file list for a model from the HuggingFace REST API.
    private func fetchFileList(modelId: String) async throws -> [String] {
        let url = URL(string: "https://huggingface.co/api/models/\(modelId)")!
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        let info = try JSONDecoder().decode(HFModelInfo.self, from: data)
        return info.siblings
            .map { $0.rfilename }
            .filter { name in
                !name.hasSuffix(".bin")     // Skip PyTorch weights
                && !name.hasSuffix(".ot")
                && !name.contains(".gguf")
            }
    }

    /// Download a single file from HuggingFace to `targetDir`.
    private func downloadFile(modelId: String, fileName: String, targetDir: URL) async throws {
        let fileURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(fileName)")!
        let destURL = targetDir.appendingPathComponent(fileName)

        // Create subdirectories if needed (e.g. for tokenizer/config subpaths)
        let parentDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        if FileManager.default.fileExists(atPath: destURL.path) { return }

        let (tmpURL, response) = try await backgroundSession.download(from: fileURL)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        try? FileManager.default.removeItem(at: destURL)
        try FileManager.default.moveItem(at: tmpURL, to: destURL)
    }

    /// Download all model files to `ModelStorage.cacheRoot` in the Hugging Face
    /// hub format expected by `LLMModelFactory.loadContainer()`.
    public func download(
        modelId: String,
        onProgress: @escaping @Sendable (DownloadFileProgress) -> Void
    ) async throws {
        let files = try await fetchFileList(modelId: modelId)

        // Target: <cacheRoot>/models--org--name/snapshots/main/
        let snapshotDir = ModelStorage.cacheRoot
            .appendingPathComponent(ModelStorage.hubDirName(for: modelId))
            .appendingPathComponent("snapshots/main")
        try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)

        // Exclude parent from iCloud backup
        ModelStorage.excludeFromBackup(
            ModelStorage.cacheRoot.appendingPathComponent(ModelStorage.hubDirName(for: modelId))
        )

        var totalDownloaded: Int64 = 0

        for (idx, fileName) in files.enumerated() {
            try Task.checkCancellation()

            let startTime = Date()
            let before = ModelStorage.directorySize(at: snapshotDir)

            onProgress(DownloadFileProgress(
                modelId: modelId,
                fileName: fileName,
                fileIndex: idx + 1,
                fileCount: files.count,
                fileFractionCompleted: 0,
                totalBytesDownloaded: totalDownloaded,
                speedBytesPerSec: nil
            ))

            try await downloadFile(modelId: modelId, fileName: fileName, targetDir: snapshotDir)

            let after = ModelStorage.directorySize(at: snapshotDir)
            let downloaded = max(0, after - before)
            totalDownloaded += downloaded
            let elapsed = max(Date().timeIntervalSince(startTime), 0.001)
            let speed = Double(downloaded) / elapsed

            onProgress(DownloadFileProgress(
                modelId: modelId,
                fileName: fileName,
                fileIndex: idx + 1,
                fileCount: files.count,
                fileFractionCompleted: 1.0,
                totalBytesDownloaded: totalDownloaded,
                speedBytesPerSec: speed
            ))
        }
    }

    #endif
}
