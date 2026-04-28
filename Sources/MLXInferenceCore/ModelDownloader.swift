// ModelDownloader.swift — Platform-aware model downloading
//
// macOS: LLMModelFactory.loadContainer() handles download + progress  
//         (called directly from InferenceEngine.load — no separate downloader needed)
//
// iOS:   Uses HuggingFace API to enumerate model files, then downloads
//         each file via URLSession to ModelStorage.cacheRoot
//         so LLMModelFactory can find them on next load without re-downloading.
//
// Both paths support:
//   • Resume after restart (partial files preserved + HTTP Range)
//   • Automatic retry with exponential backoff

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
    public let retryAttempt: Int

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

    public init(
        modelId: String,
        fileName: String,
        fileIndex: Int,
        fileCount: Int,
        fileFractionCompleted: Double,
        totalBytesDownloaded: Int64,
        speedBytesPerSec: Double?,
        retryAttempt: Int = 0
    ) {
        self.modelId = modelId
        self.fileName = fileName
        self.fileIndex = fileIndex
        self.fileCount = fileCount
        self.fileFractionCompleted = fileFractionCompleted
        self.totalBytesDownloaded = totalBytesDownloaded
        self.speedBytesPerSec = speedBytesPerSec
        self.retryAttempt = retryAttempt
    }
}

// MARK: — Retry Configuration

public struct DownloadRetryConfig: Sendable {
    /// Maximum number of retry attempts per file (0 = no retry)
    public let maxRetries: Int
    /// Initial delay before the first retry (doubles each attempt)
    public let initialDelaySeconds: Double
    /// Maximum delay cap to prevent extremely long waits
    public let maxDelaySeconds: Double

    public static let `default` = DownloadRetryConfig(
        maxRetries: 3,
        initialDelaySeconds: 2.0,
        maxDelaySeconds: 30.0
    )

    /// Calculate delay for a given attempt (exponential backoff with jitter)
    func delay(for attempt: Int) -> TimeInterval {
        let base = initialDelaySeconds * pow(2.0, Double(attempt))
        let capped = min(base, maxDelaySeconds)
        // Add ±25% jitter to prevent thundering herd
        let jitter = capped * Double.random(in: -0.25...0.25)
        return max(0.5, capped + jitter)
    }
}

// MARK: — Speed Tracker

/// Tracks download speed using an exponentially weighted moving average (EWMA)
/// over a sliding window. Produces stable, human-readable speed values instead
/// of volatile per-chunk calculations.
public final class DownloadSpeedTracker: @unchecked Sendable {
    private let lock = NSLock()

    /// Samples: (timestamp, cumulativeBytes)
    private var samples: [(time: TimeInterval, bytes: Int64)] = []
    /// How far back (seconds) to look for the rolling average
    private let windowSeconds: TimeInterval
    /// EWMA smoothing factor (0.0–1.0). Higher = more responsive, lower = smoother.
    private let alpha: Double
    /// Current EWMA speed in bytes/sec
    private var ewmaSpeed: Double = 0
    /// Absolute start time for elapsed calculation
    private let startTime: TimeInterval
    /// Total bytes at start (for resumed downloads)
    private let startBytes: Int64

    public init(windowSeconds: TimeInterval = 5.0, alpha: Double = 0.3, resumeOffset: Int64 = 0) {
        self.windowSeconds = windowSeconds
        self.alpha = alpha
        self.startTime = ProcessInfo.processInfo.systemUptime
        self.startBytes = resumeOffset
    }

    /// Record a cumulative byte count at the current time.
    public func record(totalBytes: Int64) {
        lock.lock()
        defer { lock.unlock() }

        let now = ProcessInfo.processInfo.systemUptime
        samples.append((time: now, bytes: totalBytes))

        // Prune samples older than the window
        let cutoff = now - windowSeconds
        samples.removeAll { $0.time < cutoff }

        // Calculate instantaneous speed from oldest sample in window
        if samples.count >= 2, let oldest = samples.first {
            let dt = now - oldest.time
            if dt > 0.1 {
                let instantSpeed = Double(totalBytes - oldest.bytes) / dt
                // EWMA blend
                if ewmaSpeed == 0 {
                    ewmaSpeed = instantSpeed
                } else {
                    ewmaSpeed = alpha * instantSpeed + (1 - alpha) * ewmaSpeed
                }
            }
        }
    }

    /// Current smoothed speed in bytes/sec. Returns nil if no meaningful data yet.
    public var speedBytesPerSec: Double? {
        lock.lock()
        defer { lock.unlock() }
        return ewmaSpeed > 0 ? ewmaSpeed : nil
    }

    /// Overall average speed since tracking began (bytes/sec).
    public func overallAverageSpeed(currentBytes: Int64) -> Double? {
        let elapsed = ProcessInfo.processInfo.systemUptime - startTime
        guard elapsed > 0.5 else { return nil }
        let downloaded = currentBytes - startBytes
        return downloaded > 0 ? Double(downloaded) / elapsed : nil
    }

    /// Reset for a new file while keeping the tracker alive.
    public func reset(resumeOffset: Int64 = 0) {
        lock.lock()
        defer { lock.unlock() }
        samples.removeAll()
        ewmaSpeed = 0
    }
}

// MARK: — Downloader actor

public actor ModelDownloader {

    public static let shared = ModelDownloader()
    private init() {}

    // MARK: — iOS: URLSession download with resume + retry

    #if !os(macOS)

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.allowsConstrainedNetworkAccess = true
        config.allowsExpensiveNetworkAccess = true
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600  // 1 hour for large files
        return URLSession(configuration: config)
    }()

    /// HuggingFace model tree API response — only the fields we need.
    private struct HFModelInfo: Decodable {
        let siblings: [HFFile]
        struct HFFile: Decodable {
            let rfilename: String
            let size: Int64?
        }
    }

    /// Fetch the file list for a model from the HuggingFace REST API.
    private func fetchFileList(modelId: String) async throws -> [(name: String, size: Int64?)] {
        let url = URL(string: "https://huggingface.co/api/models/\(modelId)")!
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        let info = try JSONDecoder().decode(HFModelInfo.self, from: data)
        return info.siblings
            .filter { name in
                !name.rfilename.hasSuffix(".bin")     // Skip PyTorch weights
                && !name.rfilename.hasSuffix(".ot")
                && !name.rfilename.contains(".gguf")
            }
            .map { ($0.rfilename, $0.size) }
    }

    /// Download a single file from HuggingFace to `targetDir` with resume support.
    ///
    /// Uses a `.incomplete` suffix for in-progress downloads. If a partial file
    /// exists from a previous attempt, sends an HTTP Range header to resume.
    private func downloadFile(
        modelId: String,
        fileName: String,
        expectedSize: Int64?,
        targetDir: URL,
        speedTracker: DownloadSpeedTracker,
        onProgress: @Sendable (Double, Double?) -> Void
    ) async throws {
        let fileURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(fileName)")!
        let destURL = targetDir.appendingPathComponent(fileName)
        let incompleteURL = destURL.appendingPathExtension("incomplete")

        // Create subdirectories if needed (e.g. for tokenizer/config subpaths)
        let parentDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        // Already downloaded — verify size if known, skip if good
        if FileManager.default.fileExists(atPath: destURL.path) {
            if let expected = expectedSize {
                let actual = (try? FileManager.default.attributesOfItem(atPath: destURL.path)[.size] as? Int64) ?? 0
                if actual == expected {
                    onProgress(1.0, speedTracker.speedBytesPerSec)
                    return
                }
                // Size mismatch — remove and re-download
                try? FileManager.default.removeItem(at: destURL)
            } else {
                onProgress(1.0, speedTracker.speedBytesPerSec)
                return
            }
        }

        // Check for a partial download from a previous session
        var resumeOffset: Int64 = 0
        if FileManager.default.fileExists(atPath: incompleteURL.path) {
            resumeOffset = (try? FileManager.default.attributesOfItem(atPath: incompleteURL.path)[.size] as? Int64) ?? 0
        }

        var request = URLRequest(url: fileURL)
        if resumeOffset > 0 {
            request.setValue("bytes=\(resumeOffset)-", forHTTPHeaderField: "Range")
        }

        // Stream download using bytes(for:) for progress tracking
        let (asyncBytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }

        // Handle 416 Range Not Satisfiable — partial file is stale, restart
        if http.statusCode == 416 {
            try? FileManager.default.removeItem(at: incompleteURL)
            resumeOffset = 0
            speedTracker.reset()
            // Retry without Range header
            let freshRequest = URLRequest(url: fileURL)
            let (freshBytes, freshResponse) = try await session.bytes(for: freshRequest)
            guard let freshHttp = freshResponse as? HTTPURLResponse, (200..<300).contains(freshHttp.statusCode) else {
                throw URLError(.badServerResponse)
            }
            try await streamToFile(
                asyncBytes: freshBytes,
                destURL: incompleteURL,
                resumeOffset: 0,
                totalSize: freshHttp.expectedContentLength > 0 ? freshHttp.expectedContentLength : (expectedSize ?? 0),
                speedTracker: speedTracker,
                onProgress: onProgress
            )
        } else if (200..<300).contains(http.statusCode) {
            // 200 = full content (server ignored Range), 206 = partial content (resume worked)
            let isResume = (http.statusCode == 206)
            if !isResume {
                // Server returned full content — discard partial file
                try? FileManager.default.removeItem(at: incompleteURL)
                resumeOffset = 0
                speedTracker.reset()
            }
            let totalSize: Int64
            if isResume {
                totalSize = resumeOffset + http.expectedContentLength
            } else {
                totalSize = http.expectedContentLength > 0 ? http.expectedContentLength : (expectedSize ?? 0)
            }
            try await streamToFile(
                asyncBytes: asyncBytes,
                destURL: incompleteURL,
                resumeOffset: isResume ? resumeOffset : 0,
                totalSize: totalSize,
                speedTracker: speedTracker,
                onProgress: onProgress
            )
        } else {
            throw URLError(.badServerResponse)
        }

        // Atomic move from .incomplete to final destination
        try? FileManager.default.removeItem(at: destURL)
        try FileManager.default.moveItem(at: incompleteURL, to: destURL)
        onProgress(1.0, speedTracker.speedBytesPerSec)
    }

    /// Stream async bytes to a file, appending if resuming.
    private func streamToFile(
        asyncBytes: URLSession.AsyncBytes,
        destURL: URL,
        resumeOffset: Int64,
        totalSize: Int64,
        speedTracker: DownloadSpeedTracker,
        onProgress: @Sendable (Double, Double?) -> Void
    ) async throws {
        let fileHandle: FileHandle
        if resumeOffset > 0, FileManager.default.fileExists(atPath: destURL.path) {
            fileHandle = try FileHandle(forWritingTo: destURL)
            fileHandle.seekToEndOfFile()
        } else {
            FileManager.default.createFile(atPath: destURL.path, contents: nil)
            fileHandle = try FileHandle(forWritingTo: destURL)
        }
        defer { try? fileHandle.close() }

        var bytesWritten: Int64 = resumeOffset
        var buffer = Data()
        let flushSize = 256 * 1024  // Flush every 256 KB
        var lastProgressUpdate = Date()

        for try await byte in asyncBytes {
            try Task.checkCancellation()
            buffer.append(byte)

            if buffer.count >= flushSize {
                fileHandle.write(buffer)
                bytesWritten += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)

                speedTracker.record(totalBytes: bytesWritten)

                // Throttle progress updates to ~10/sec
                let now = Date()
                if now.timeIntervalSince(lastProgressUpdate) >= 0.1 {
                    lastProgressUpdate = now
                    if totalSize > 0 {
                        onProgress(Double(bytesWritten) / Double(totalSize), speedTracker.speedBytesPerSec)
                    }
                }
            }
        }

        // Flush remaining bytes
        if !buffer.isEmpty {
            fileHandle.write(buffer)
            bytesWritten += Int64(buffer.count)
            speedTracker.record(totalBytes: bytesWritten)
        }

        if totalSize > 0 {
            onProgress(Double(bytesWritten) / Double(totalSize), speedTracker.speedBytesPerSec)
        }
    }

    /// Download all model files to `ModelStorage.cacheRoot` in the Hugging Face
    /// hub format expected by `LLMModelFactory.loadContainer()`.
    ///
    /// Supports:
    /// - **Resume after restart**: partial `.incomplete` files are preserved and resumed via HTTP Range
    /// - **Automatic retry**: transient network errors retry with exponential backoff
    public func download(
        modelId: String,
        retryConfig: DownloadRetryConfig = .default,
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
        let speedTracker = DownloadSpeedTracker()

        for (idx, file) in files.enumerated() {
            try Task.checkCancellation()

            let before = ModelStorage.directorySize(at: snapshotDir)
            var lastError: Error?

            // Reset speed tracker per-file so the EWMA starts fresh
            speedTracker.reset()

            // Retry loop with exponential backoff
            for attempt in 0...retryConfig.maxRetries {
                do {
                    if attempt > 0 {
                        let delay = retryConfig.delay(for: attempt - 1)
                        print("[ModelDownloader] Retry \(attempt)/\(retryConfig.maxRetries) for \(file.name) after \(String(format: "%.1f", delay))s")
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        try Task.checkCancellation()
                    }

                    onProgress(DownloadFileProgress(
                        modelId: modelId,
                        fileName: file.name,
                        fileIndex: idx + 1,
                        fileCount: files.count,
                        fileFractionCompleted: 0,
                        totalBytesDownloaded: totalDownloaded,
                        speedBytesPerSec: speedTracker.speedBytesPerSec,
                        retryAttempt: attempt
                    ))

                    try await downloadFile(
                        modelId: modelId,
                        fileName: file.name,
                        expectedSize: file.size,
                        targetDir: snapshotDir,
                        speedTracker: speedTracker
                    ) { fraction, speed in
                        onProgress(DownloadFileProgress(
                            modelId: modelId,
                            fileName: file.name,
                            fileIndex: idx + 1,
                            fileCount: files.count,
                            fileFractionCompleted: fraction,
                            totalBytesDownloaded: totalDownloaded,
                            speedBytesPerSec: speed,
                            retryAttempt: attempt
                        ))
                    }

                    lastError = nil
                    break  // Success — exit retry loop

                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    lastError = error
                    print("[ModelDownloader] Download failed for \(file.name): \(error.localizedDescription)")
                    // Don't retry on non-transient errors
                    if let urlError = error as? URLError {
                        switch urlError.code {
                        case .cancelled, .userCancelledAuthentication:
                            throw error
                        case .notConnectedToInternet, .networkConnectionLost,
                             .timedOut, .cannotConnectToHost, .dnsLookupFailed:
                            continue  // Transient — retry
                        default:
                            if attempt >= retryConfig.maxRetries { throw error }
                            continue
                        }
                    }
                }
            }

            if let error = lastError {
                throw error
            }

            let after = ModelStorage.directorySize(at: snapshotDir)
            let downloaded = max(0, after - before)
            totalDownloaded += downloaded

            onProgress(DownloadFileProgress(
                modelId: modelId,
                fileName: file.name,
                fileIndex: idx + 1,
                fileCount: files.count,
                fileFractionCompleted: 1.0,
                totalBytesDownloaded: totalDownloaded,
                speedBytesPerSec: speedTracker.speedBytesPerSec,
                retryAttempt: 0
            ))
        }
    }

    #endif
}
