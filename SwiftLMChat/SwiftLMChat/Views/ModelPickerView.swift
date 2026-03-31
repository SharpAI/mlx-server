// ModelPickerView.swift — Model selection with download status and management
import SwiftUI

struct ModelPickerView: View {
    @EnvironmentObject private var engine: InferenceEngine
    let onSelect: (String) -> Void

    @State private var device = DeviceProfile.current
    @State private var showManagement = false
    @State private var pendingCellularModelId: String? = nil  // awaiting cellular confirm

    private var downloadManager: ModelDownloadManager { engine.downloadManager }

    // iOS uses tighter RAM budget (40%) — macOS uses 75%
    private var recommendedModels: [ModelEntry] {
        downloadManager.modelsForDevice()
    }
    private var otherModels: [ModelEntry] {
        ModelCatalog.all.filter { model in
            !recommendedModels.contains(where: { $0.id == model.id })
        }
    }

    var body: some View {
        NavigationStack {
            List {
                // Device info + storage header
                deviceHeader

                // Downloaded models section (if any)
                if !downloadManager.downloadedModels.isEmpty {
                    Section {
                        ForEach(downloadManager.downloadedModels) { downloaded in
                            if let entry = ModelCatalog.all.first(where: { $0.id == downloaded.id }) {
                                ModelRow(
                                    model: entry,
                                    downloadStatus: .downloaded(sizeString: downloaded.displaySize),
                                    fitStatus: ModelCatalog.fitStatus(for: entry, on: device),
                                    downloadProgress: downloadManager.activeDownloads[entry.id],
                                    onTap: { handleModelTap(entry.id) },
                                    onDelete: { try? downloadManager.delete(entry.id) }
                                )
                            }
                        }
                    } header: {
                        HStack {
                            Text("Downloaded")
                            Spacer()
                            Button("Manage") { showManagement = true }
                                .font(.caption)
                        }
                    }
                }

                // Recommended models not yet downloaded
                let notDownloaded = recommendedModels.filter { !downloadManager.isDownloaded($0.id) }
                if !notDownloaded.isEmpty {
                    Section("Recommended — tap to download & load") {
                        ForEach(notDownloaded) { model in
                            ModelRow(
                                model: model,
                                downloadStatus: downloadStatus(for: model.id),
                                fitStatus: ModelCatalog.fitStatus(for: model, on: device),
                                downloadProgress: downloadManager.activeDownloads[model.id],
                                onTap: { handleModelTap(model.id) },
                                onDelete: nil
                            )
                        }
                    }
                }

                // Larger models
                let largeNotDownloaded = otherModels.filter { !downloadManager.isDownloaded($0.id) }
                if !largeNotDownloaded.isEmpty {
                    Section("Larger models (flash streaming)") {
                        ForEach(largeNotDownloaded) { model in
                            ModelRow(
                                model: model,
                                downloadStatus: downloadStatus(for: model.id),
                                fitStatus: ModelCatalog.fitStatus(for: model, on: device),
                                downloadProgress: downloadManager.activeDownloads[model.id],
                                onTap: { handleModelTap(model.id) },
                                onDelete: nil
                            )
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Models")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onSelect("") }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button { showManagement = true } label: {
                        Image(systemName: "externaldrive")
                    }
                }
            }
            .sheet(isPresented: $showManagement) {
                ModelManagementView()
                    .environmentObject(engine)
            }
            // Cellular download warning
            .confirmationDialog(
                "Download on Cellular?",
                isPresented: .init(
                    get: { pendingCellularModelId != nil },
                    set: { if !$0 { pendingCellularModelId = nil } }
                ),
                titleVisibility: .visible
            ) {
                if let modelId = pendingCellularModelId {
                    let name = ModelCatalog.all.first(where: { $0.id == modelId })?.displayName ?? modelId
                    let size = ModelCatalog.all.first(where: { $0.id == modelId }).map {
                        String(format: "~%.1f GB", $0.ramRequiredGB)
                    } ?? ""
                    Button("Download \(size) on Cellular", role: .destructive) {
                        let id = modelId
                        pendingCellularModelId = nil
                        onSelect(id)
                    }
                    Button("Cancel", role: .cancel) { pendingCellularModelId = nil }
                }
            } message: {
                Text("This model is large and may use significant cellular data.")
            }
            .onAppear { downloadManager.refresh() }
        }
        #if os(macOS)
        .frame(minWidth: 500, minHeight: 600)
        #endif
    }

    private var deviceHeader: some View {
        Section {
            // Network status banner
            if downloadManager.isOffline {
                Label("No internet — downloaded models only.", systemImage: "wifi.slash")
                    .font(.caption).foregroundStyle(.orange)
            } else if downloadManager.isOnCellular {
                Label("Cellular connection — large downloads may incur charges.", systemImage: "antenna.radiowaves.left.and.right")
                    .font(.caption).foregroundStyle(.orange)
            }

            // Thermal warning
            if engine.thermalLevel.isThrottled {
                Label(engine.thermalLevel.displayString, systemImage: "thermometer.high")
                    .font(.caption).foregroundStyle(.red)
            }

            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 2) {
                    Label(String(format: "%.0f GB RAM", device.physicalRAMGB), systemImage: "memorychip")
                        .font(.subheadline.weight(.medium))
                    Text("Apple Silicon · Metal GPU")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if downloadManager.totalDiskUsageBytes > 0 {
                    VStack(alignment: .trailing, spacing: 2) {
                        Label(formatBytes(downloadManager.totalDiskUsageBytes), systemImage: "externaldrive.fill")
                            .font(.subheadline.weight(.medium))
                        Text("models on disk")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func handleModelTap(_ modelId: String) {
        // Don't download if offline and not already cached
        if downloadManager.isOffline && !downloadManager.isDownloaded(modelId) { return }
        // Warn before large cellular downloads
        if downloadManager.shouldWarnForCellular(modelId) && !downloadManager.isDownloaded(modelId) {
            pendingCellularModelId = modelId
        } else {
            onSelect(modelId)
        }
    }

    private func downloadStatus(for modelId: String) -> ModelDownloadStatus {
        if downloadManager.isDownloaded(modelId) {
            let size = downloadManager.downloadedModel(for: modelId)?.displaySize ?? ""
            return .downloaded(sizeString: size)
        }
        if downloadManager.activeDownloads[modelId] != nil {
            return .downloading
        }
        return .notDownloaded
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1.0 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }
}

// MARK: — Download Status Enum

enum ModelDownloadStatus {
    case notDownloaded
    case downloading
    case downloaded(sizeString: String)
}

// MARK: — Model Row

struct ModelRow: View {
    let model: ModelEntry
    let downloadStatus: ModelDownloadStatus
    let fitStatus: ModelCatalog.FitStatus
    let downloadProgress: ModelDownloadProgress?
    let onTap: () -> Void
    let onDelete: (() -> Void)?

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                modelIcon

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(model.displayName)
                            .font(.headline)
                            .foregroundStyle(.primary)
                        if let badge = model.badge {
                            Text(badge)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.12))
                                .clipShape(Capsule())
                        }
                    }

                    // Subtitle: size/quantization or download progress
                    if let progress = downloadProgress {
                        HStack(spacing: 4) {
                            ProgressView(value: progress.fractionCompleted)
                                .frame(width: 80)
                            Text("\(progress.percentString) · \(progress.speedString)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        HStack(spacing: 4) {
                            Text("\(model.parameterSize) · \(model.quantization)")
                            Text("·")
                            switch downloadStatus {
                            case .downloaded(let size):
                                Text(size).foregroundStyle(.secondary)
                            case .notDownloaded:
                                Text("\(model.ramRequiredGB, specifier: "%.1f")GB RAM")
                                    .foregroundStyle(.secondary)
                            case .downloading:
                                Text("Downloading…").foregroundStyle(.blue)
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                trailingBadge
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if let onDelete {
                Button(role: .destructive, action: onDelete) {
                    Label("Delete", systemImage: "trash")
                }
            }
        }
    }

    private var modelIcon: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(iconGradient)
                .frame(width: 44, height: 44)
            Image(systemName: iconName)
                .font(.title3)
                .foregroundStyle(.white)
        }
    }

    @ViewBuilder
    private var trailingBadge: some View {
        switch downloadStatus {
        case .downloaded:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.title3)
        case .downloading:
            ProgressView()
                .controlSize(.small)
        case .notDownloaded:
            HStack(spacing: 2) {
                switch fitStatus {
                case .fits:
                    Image(systemName: "icloud.and.arrow.down")
                        .foregroundStyle(.blue)
                case .tight:
                    Image(systemName: "icloud.and.arrow.down")
                        .foregroundStyle(.orange)
                case .requiresFlash:
                    Label("Flash", systemImage: "bolt.fill")
                        .font(.caption).foregroundStyle(.purple)
                case .tooLarge:
                    Image(systemName: "exclamationmark.circle")
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var iconGradient: LinearGradient {
        switch fitStatus {
        case .fits:    return LinearGradient(colors: [.blue, .cyan],     startPoint: .topLeading, endPoint: .bottomTrailing)
        case .tight:   return LinearGradient(colors: [.orange, .yellow], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .requiresFlash: return LinearGradient(colors: [.purple, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .tooLarge: return LinearGradient(colors: [.gray, .gray],   startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }

    private var iconName: String {
        if model.isMoE { return "square.grid.3x3.fill" }
        switch model.parameterSize {
        case let s where s.contains("0.5"): return "hare.fill"
        case let s where s.contains("3"):   return "bolt.fill"
        case let s where s.contains("7"):   return "brain"
        default: return "sparkles"
        }
    }
}
