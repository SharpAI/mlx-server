// ModelDownloadManager.swift — Orchestrates storage + download + network monitoring
// Replaces the old single-file manager with the layered ModelStorage + ModelDownloader.

import Foundation
import Network
import Combine
#if os(macOS)
import Hub
#endif

// MARK: — Downloaded Model

public struct DownloadedModel: Identifiable, Sendable {
    public let id: String
    public let cacheDirectory: URL
    public let sizeBytes: Int64
    public let modifiedDate: Date?

    public var displaySize: String {
        formatBytes(sizeBytes)
    }
}

// MARK: — Download Progress (UI-facing)

public struct ModelDownloadProgress: Sendable {
    public let modelId: String
    public let fractionCompleted: Double   // 0.0–1.0
    public let currentFile: String
    public let speedMBps: Double?

    public var speedString: String {
        guard let s = speedMBps else { return "" }
        return String(format: "%.1f MB/s", s)
    }
    public var percentString: String { "\(Int(fractionCompleted * 100))%" }
}

// MARK: — Network Status

public enum NetworkStatus: Sendable {
    case unknown
    case wifi
    case cellular
    case offline
}

// MARK: — ModelDownloadManager

@MainActor
public final class ModelDownloadManager: ObservableObject {

    // MARK: Published state
    @Published public private(set) var downloadedModels: [DownloadedModel] = []
    @Published public private(set) var incompleteDownloads: [ModelStorage.IncompleteDownload] = []
    @Published public private(set) var activeDownloads: [String: ModelDownloadProgress] = [:]
    @Published public private(set) var totalDiskUsageBytes: Int64 = 0
    @Published public private(set) var networkStatus: NetworkStatus = .unknown

    // MARK: Persistence
    private let lastModelKey = "swiftlm.lastLoadedModelId"
    public var lastLoadedModelId: String? {
        get { UserDefaults.standard.string(forKey: lastModelKey) }
        set { UserDefaults.standard.set(newValue, forKey: lastModelKey) }
    }

    // MARK: Network monitoring
    private let monitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "com.sharpai.swiftlm.netmonitor")

    // MARK: In-flight download tasks
    private var downloadTasks: [String: Task<Void, Error>] = [:]

    // MARK: iOS RAM budget
    /// On iOS, use 40% (conservative, avoids jetsam) or 55% in Performance Mode.
    /// On macOS, use 75% (generous, no jetsam).
    public static var ramBudgetFraction: Double {
        #if os(iOS)
        let performanceMode = UserDefaults.standard.bool(forKey: "swiftlm.performanceMode")
        return performanceMode ? 0.55 : 0.40
        #else
        return 0.75
        #endif
    }

    public init() {
        startNetworkMonitor()
        refresh()
    }

    deinit {
        monitor.cancel()
    }

    // MARK: — Network Monitoring

    private func startNetworkMonitor() {
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if path.status == .unsatisfied {
                    self.networkStatus = .offline
                } else if path.usesInterfaceType(.wifi) || path.usesInterfaceType(.wiredEthernet) {
                    self.networkStatus = .wifi
                } else if path.usesInterfaceType(.cellular) {
                    self.networkStatus = .cellular
                } else {
                    self.networkStatus = .unknown
                }
            }
        }
        monitor.start(queue: monitorQueue)
    }

    /// True if the current connection is cellular (may warn before large downloads).
    public var isOnCellular: Bool { networkStatus == .cellular }

    /// True if offline.
    public var isOffline: Bool { networkStatus == .offline }

    // MARK: — Storage

    /// Re-scan the cache and update published state.
    public func refresh() {
        let scanned = ModelStorage.scanDownloadedModels()
        downloadedModels = scanned.map { s in
            DownloadedModel(
                id: s.modelId,
                cacheDirectory: s.cacheDirectory,
                sizeBytes: s.sizeBytes,
                modifiedDate: s.modifiedDate
            )
        }
        totalDiskUsageBytes = downloadedModels.reduce(0) { $0 + $1.sizeBytes }

        // Scan for interrupted downloads that can be resumed
        incompleteDownloads = ModelStorage.scanIncompleteDownloads()
            .filter { incomplete in
                // Exclude models that are already actively downloading
                !activeDownloads.keys.contains(incomplete.id)
            }
    }

    public func isDownloaded(_ modelId: String) -> Bool {
        ModelStorage.isDownloaded(modelId)
    }

    /// True if a model has a partial download that can be resumed.
    public func hasIncompleteDownload(_ modelId: String) -> Bool {
        incompleteDownloads.contains { $0.id == modelId }
    }

    public func downloadedModel(for modelId: String) -> DownloadedModel? {
        downloadedModels.first(where: { $0.id == modelId })
    }

    /// Delete a model and free disk space (including any partial downloads).
    public func delete(_ modelId: String) throws {
        try ModelStorage.delete(modelId)
        refresh()
        if lastLoadedModelId == modelId { lastLoadedModelId = nil }
    }

    /// Resume an incomplete download. This calls startDownload() which will
    /// automatically resume from where it left off (partial files + HTTP Range).
    @discardableResult
    public func resumeDownload(modelId: String) -> Task<Void, Error> {
        return startDownload(modelId: modelId)
    }

    // MARK: — Download

    /// Start downloading a model.
    /// iOS: Uses ModelDownloader with per-file resume + retry.
    /// macOS: Uses HubApi.snapshot() which handles resume internally; we add retry around it.
    @discardableResult
    public func startDownload(
        modelId: String,
        retryConfig: DownloadRetryConfig = .default
    ) -> Task<Void, Error> {
        print("[ModelDownloadManager] startDownload called for \(modelId)")
        downloadTasks[modelId]?.cancel()

        let task = Task<Void, Error> {
            print("[ModelDownloadManager] Task started for \(modelId)")
            // Instantly register 0% progress so UI banners appear immediately
            // before the Hub API computes the file snapshot.
            Task { @MainActor [weak self] in
                if self?.activeDownloads[modelId] == nil {
                    print("[ModelDownloadManager] Registering 0% progress for \(modelId)")
                    self?.activeDownloads[modelId] = ModelDownloadProgress(
                        modelId: modelId,
                        fractionCompleted: 0.0,
                        currentFile: "Preparing download...",
                        speedMBps: nil
                    )
                }
            }

            do {
                defer {
                    print("[ModelDownloadManager] Defer executing, removing activeDownload for \(modelId)")
                    Task { @MainActor [weak self] in
                        self?.activeDownloads.removeValue(forKey: modelId)
                    }
                }

                #if !os(macOS)
                try await ModelDownloader.shared.download(
                    modelId: modelId,
                    retryConfig: retryConfig
                ) { [weak self] fp in
                    Task { @MainActor [weak self] in
                        self?.activeDownloads[modelId] = ModelDownloadProgress(
                            modelId: modelId,
                            fractionCompleted: fp.overallFraction,
                            currentFile: fp.fileName,
                            speedMBps: fp.speedBytesPerSec.map { $0 / 1_000_000 }
                        )
                    }
                }
                #else
                // macOS: HubApi.snapshot() already supports resume via incomplete blob
                // files and HTTP Range headers. We add retry for transient failures.
                let speedTracker = DownloadSpeedTracker()
                var lastError: Error?
                for attempt in 0...retryConfig.maxRetries {
                    do {
                        if attempt > 0 {
                            let delay = retryConfig.delay(for: attempt - 1)
                            print("[ModelDownloadManager] Retry \(attempt)/\(retryConfig.maxRetries) for \(modelId) after \(String(format: "%.1f", delay))s")
                            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                            try Task.checkCancellation()
                            speedTracker.reset()
                        }

                        let hub = HubApi(downloadBase: ModelStorage.cacheRoot)
                        print("[ModelDownloadManager] Calling hub.snapshot for \(modelId)")
                        _ = try await hub.snapshot(
                            from: modelId,
                            matching: ["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"],
                            progressHandler: { @Sendable [weak self] progress in
                                // Feed cumulative bytes into the EWMA tracker
                                speedTracker.record(totalBytes: progress.completedUnitCount)
                                let smoothedSpeed = speedTracker.speedBytesPerSec

                                Task { @MainActor [weak self] in
                                    let pct = progress.fractionCompleted
                                    self?.activeDownloads[modelId] = ModelDownloadProgress(
                                        modelId: modelId,
                                        fractionCompleted: pct,
                                        currentFile: attempt > 0 ? "(retry \(attempt))" : "",
                                        speedMBps: smoothedSpeed.map { $0 / 1_000_000 }
                                    )
                                }
                            }
                        )
                        print("[ModelDownloadManager] hub.snapshot FINISHED SUCCESSFULLY for \(modelId)")
                        lastError = nil
                        break  // Success
                    } catch is CancellationError {
                        print("[ModelDownloadManager] Task was CANCELLED for \(modelId)")
                        throw CancellationError()
                    } catch {
                        lastError = error
                        print("[ModelDownloadManager] Download failed for \(modelId): \(error.localizedDescription)")
                        // Only retry transient network errors
                        if let urlError = error as? URLError {
                            switch urlError.code {
                            case .cancelled, .userCancelledAuthentication:
                                throw error
                            case .notConnectedToInternet, .networkConnectionLost,
                                 .timedOut, .cannotConnectToHost, .dnsLookupFailed:
                                continue
                            default:
                                if attempt >= retryConfig.maxRetries { throw error }
                                continue
                            }
                        }
                        // Non-URLError (e.g. auth failure) — don't retry
                        throw error
                    }
                }
                if let error = lastError { throw error }
                #endif

                Task { @MainActor [weak self] in
                    self?.activeDownloads.removeValue(forKey: modelId)
                    self?.lastLoadedModelId = modelId
                    self?.refresh()
                }
            } catch {
                print("\n[ModelDownloadManager] HuggingFace Download Failed for \(modelId): \(error.localizedDescription)\n")
                throw error
            }
        }

        downloadTasks[modelId] = task
        return task
    }

    /// Cancel an in-progress download.
    public func cancelDownload(modelId: String) {
        downloadTasks[modelId]?.cancel()
        downloadTasks.removeValue(forKey: modelId)
        activeDownloads.removeValue(forKey: modelId)
    }

    /// Update progress from an external source (e.g. InferenceEngine's loadContainer callback).
    public func updateProgress(_ progress: ModelDownloadProgress) {
        activeDownloads[progress.modelId] = progress
    }

    /// Clear progress for a model (called on completion or cancellation).
    public func clearProgress(modelId: String) {
        activeDownloads.removeValue(forKey: modelId)
    }

    // MARK: — iOS RAM filtering

    /// Returns models appropriate for this device, applying the platform RAM budget.
    public func modelsForDevice() -> [ModelEntry] {
        let device = DeviceProfile.current
        let usableRAM = device.physicalRAMGB * Self.ramBudgetFraction
        return ModelCatalog.all.filter { $0.ramRequiredGB <= usableRAM }
    }

    // MARK: — Cellular Download Threshold (bytes)

    /// Models larger than this threshold trigger a cellular warning on iOS.
    public static let cellularWarnThresholdBytes: Int64 = 200 * 1_024 * 1_024  // 200 MB

    public func shouldWarnForCellular(modelId: String) -> Bool {
        guard isOnCellular else { return false }
        guard let entry = ModelCatalog.all.first(where: { $0.id == modelId }) else { return false }
        let estimatedBytes = Int64(entry.ramRequiredGB * 1_073_741_824)  // rough estimate
        return estimatedBytes > Self.cellularWarnThresholdBytes
    }
}

// MARK: — Helpers

private func formatBytes(_ bytes: Int64) -> String {
    let gb = Double(bytes) / 1_073_741_824
    let mb = Double(bytes) / 1_048_576
    if gb >= 1.0 { return String(format: "%.1f GB", gb) }
    return String(format: "%.0f MB", mb)
}
