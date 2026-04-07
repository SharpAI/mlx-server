// ModelPickerView.swift — Model selection with HuggingFace live search
import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

// MARK: — Main Picker View

struct ModelPickerView: View {
    @EnvironmentObject private var engine: InferenceEngine
    let onSelect: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showHFSearch = false
    @State private var device = DeviceProfile.current
    @State private var showManagement = false
    @State private var pendingCellularModelId: String? = nil

    private var downloadManager: ModelDownloadManager { engine.downloadManager }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                CatalogTab(
                    downloadManager: downloadManager,
                    device: device,
                    onTap: handleModelTap,
                    showManagement: $showManagement,
                    onSearchHFTap: { showHFSearch = true }
                )
            }
            .navigationTitle("Models")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showManagement = true
                    } label: {
                        Label("Manage", systemImage: "externaldrive.badge.minus")
                    }
                }
            }
            .sheet(isPresented: $showManagement) {
                ModelManagementView()
                    .environmentObject(engine)
            }
            .alert(
                "Use Cellular Data?",
                isPresented: Binding(
                    get: { pendingCellularModelId != nil },
                    set: { if !$0 { pendingCellularModelId = nil } }
                )
            ) {
                Button("Download") {
                    if let id = pendingCellularModelId { onSelect(id) }
                    pendingCellularModelId = nil
                }
                Button("Cancel", role: .cancel) { pendingCellularModelId = nil }
            } message: {
                Text("This model is large. Downloading over cellular may incur data charges.")
            }
        }
        .sheet(isPresented: $showHFSearch) {
            NavigationStack {
                ZStack {
                    SwiftBuddyTheme.background.ignoresSafeArea()
                    HFSearchTab(onSelect: { id in
                        showHFSearch = false
                        onSelect(id)
                    })
                }
                .navigationTitle("Search HuggingFace")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Cancel") { showHFSearch = false }
                            .foregroundStyle(SwiftBuddyTheme.accent)
                    }
                }
            }
            .environmentObject(engine)
        }
    }

    private func handleModelTap(_ modelId: String) {
        if downloadManager.isOffline && !downloadManager.isDownloaded(modelId) { return }
        if downloadManager.shouldWarnForCellular(modelId: modelId) && !downloadManager.isDownloaded(modelId) {
            pendingCellularModelId = modelId
        } else {
            onSelect(modelId)
        }
    }
}

// MARK: — Catalog Tab (curated list)

private struct CatalogTab: View {
    let downloadManager: ModelDownloadManager
    let device: DeviceProfile
    let onTap: (String) -> Void
    @Binding var showManagement: Bool
    let onSearchHFTap: () -> Void

    private var recommendedModels: [ModelEntry] { downloadManager.modelsForDevice() }
    private var otherModels: [ModelEntry] {
        ModelCatalog.all.filter { m in !recommendedModels.contains(where: { $0.id == m.id }) }
    }

    var body: some View {
        List {
            deviceHeader

            if !downloadManager.downloadedModels.isEmpty {
                Section {
                    ForEach(downloadManager.downloadedModels) { downloaded in
                        if let entry = ModelCatalog.all.first(where: { $0.id == downloaded.id }) {
                            ModelRow(
                                model: entry,
                                downloadStatus: .downloaded(sizeString: downloaded.displaySize),
                                fitStatus: ModelCatalog.fitStatus(for: entry, on: device),
                                downloadProgress: downloadManager.activeDownloads[entry.id],
                                onTap: { onTap(entry.id) },
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

            if !recommendedModels.isEmpty {
                Section("Recommended for your device") {
                    ForEach(recommendedModels) { model in
                        ModelRow(
                            model: model,
                            downloadStatus: downloadManager.isDownloaded(model.id) ? .downloaded(sizeString: "") : .available,
                            fitStatus: ModelCatalog.fitStatus(for: model, on: device),
                            downloadProgress: downloadManager.activeDownloads[model.id],
                            onTap: { onTap(model.id) },
                            onDelete: nil
                        )
                    }
                }
            }

            if !otherModels.isEmpty {
                Section("All Models") {
                    ForEach(otherModels) { model in
                        ModelRow(
                            model: model,
                            downloadStatus: downloadManager.isDownloaded(model.id) ? .downloaded(sizeString: "") : .available,
                            fitStatus: ModelCatalog.fitStatus(for: model, on: device),
                            downloadProgress: downloadManager.activeDownloads[model.id],
                            onTap: { onTap(model.id) },
                            onDelete: nil
                        )
                    }
                }
            }
            
            Section {
                Button(action: onSearchHFTap) {
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(.blue)
                        Text("Search HuggingFace MLX models")
                        Spacer()
                        Image(systemName: "chevron.right")
                    }
                    .padding(14)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
                }
                .buttonStyle(.plain)
            }
        }
        .listStyle(.inset)
    }

    private var deviceHeader: some View {
        Section {
            HStack(spacing: 12) {
                Image(systemName: "memorychip")
                    .font(.title2)
                    .foregroundStyle(.blue)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Apple Silicon")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                    Text(String(format: "%.0f GB RAM", device.physicalRAMGB))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if downloadManager.isOffline {
                    Label("Offline", systemImage: "wifi.slash")
                        .font(.caption.bold())
                        .foregroundStyle(.orange)
                }
            }
            .padding(.vertical, 4)
        }
    }
}

// MARK: — HuggingFace Search Tab

struct HFSearchTab: View {
    let onSelect: (String) -> Void

    @StateObject private var service = HFModelSearchService.shared
    @State private var query = ""
    @State private var sort = HFSortOption.trending

    var body: some View {
        VStack(spacing: 0) {
            // ── Search bar + sort ──────────────────────────────────────────
            VStack(spacing: 8) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search MLX models…", text: $query)
                        .textFieldStyle(.plain)
                        .autocorrectionDisabled()
                    if !query.isEmpty {
                        Button { query = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(8)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))

                Toggle("Strict MLX Formatting Only", isOn: $service.strictMLX)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .toggleStyle(.switch)
                    .padding(.horizontal, 4)
                    .onChange(of: service.strictMLX) { _, _ in
                        service.search(query: query, sort: sort)
                    }

                // ─────────────────────────────────────────────────────────────
                // Sort tags (Trending, Likes, etc)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(HFSortOption.allCases, id: \.self) { option in
                            Button {
                                sort = option
                                service.search(query: query, sort: sort)
                            } label: {
                                Text(option.label)
                                    .font(.caption.weight(.medium))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5)
                                    .background(
                                        sort == option ? Color.accentColor : Color.secondary.opacity(0.15),
                                        in: Capsule()
                                    )
                                    .foregroundStyle(sort == option ? .white : .primary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)

            Divider()

            // ── Results ────────────────────────────────────────────────────
            if service.isSearching && service.results.isEmpty {
                Spacer()
                ProgressView("Searching HuggingFace…")
                    .foregroundStyle(.secondary)
                Spacer()
            } else if let err = service.errorMessage {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                        .foregroundStyle(.orange)
                    Text(err)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding()
                Spacer()
            } else if service.results.isEmpty && !query.isEmpty {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No MLX models found for \"\(query)\"")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            } else {
                List {
                    ForEach(service.results) { model in
                        HFModelRow(model: model, onSelect: onSelect)
                    }
                    if service.hasMore {
                        HStack {
                            Spacer()
                            Button("Load More") { service.loadMore() }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                            Spacer()
                        }
                        .listRowSeparator(.hidden)
                    }
                }
                .listStyle(.inset)
                .overlay(alignment: .bottom) {
                    if service.isSearching {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Loading…").font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(6)
                        .background(.regularMaterial, in: Capsule())
                        .padding(.bottom, 8)
                    }
                }
            }
        }
        .onChange(of: query) { _, newValue in
            service.search(query: newValue, sort: sort)
        }
        .onAppear {
            if service.results.isEmpty {
                service.search(query: "", sort: sort)
            }
        }
    }
}

// MARK: — HF Model Row

private struct HFModelRow: View {
    let model: HFModelResult
    let onSelect: (String) -> Void

    var body: some View {
        Button {
            onSelect(model.id)
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    // Model name — strip "mlx-community/" prefix for cleanliness
                    Text(model.repoName)
                        .font(.system(.subheadline, design: .default, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    Text(model.id)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        if model.isMlxCommunity {
                            badge("mlx-community", color: .blue)
                        }
                        if model.isMoE {
                            badge("MoE", color: .purple)
                        }
                        if let size = model.paramSizeHint {
                            badge(size, color: .orange)
                        }
                    }
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 3) {
                    if !model.downloadsDisplay.isEmpty {
                        Text(model.downloadsDisplay)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    if !model.likesDisplay.isEmpty {
                        Text(model.likesDisplay)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.pink)
                    }
                    Image(systemName: "arrow.down.circle")
                        .font(.title3)
                        .foregroundStyle(.blue)
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func badge(_ label: String, color: Color) -> some View {
        Text(label)
            .font(.system(size: 9, weight: .bold))
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

// MARK: — ModelRow (reused by catalog tab — unchanged logic, cleaner layout)

enum DownloadStatus {
    case downloaded(sizeString: String)
    case available
    case downloading(progress: Double)
}

struct ModelRow: View {
    let model: ModelEntry
    let downloadStatus: DownloadStatus
    let fitStatus: ModelCatalog.FitStatus
    let downloadProgress: ModelDownloadProgress?
    let onTap: () -> Void
    let onDelete: (() -> Void)?

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                // ── Left: name + metadata ─────────────────────────────────
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(model.displayName)
                            .font(.system(.subheadline, design: .default, weight: .semibold))
                            .foregroundStyle(.primary)
                        if let badge = model.badge {
                            Text(badge)
                                .font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(.blue.opacity(0.12), in: Capsule())
                                .foregroundStyle(.blue)
                        }
                    }

                    HStack(spacing: 6) {
                        Text(model.parameterSize)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("•")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        Text(model.quantization)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if model.isMoE {
                            Text("MoE")
                                .font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(.purple.opacity(0.12), in: Capsule())
                                .foregroundStyle(.purple)
                        }
                    }

                    // Download progress bar
                    if let progress = downloadProgress {
                        ProgressView(value: progress.fractionCompleted)
                            .tint(.blue)
                        if let speed = progress.speedMBps {
                            Text(String(format: "%.1f MB/s", speed))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()

                // ── Right: status indicator ───────────────────────────────
                VStack(alignment: .trailing, spacing: 3) {
                    statusBadge
                    Text(String(format: "%.0f GB", model.ramRequiredGB))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)
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

    @ViewBuilder
    private var statusBadge: some View {
        switch downloadStatus {
        case .downloaded:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.title3)
        case .available:
            switch fitStatus {
            case .fits:
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.blue)
                    .font(.title3)
            case .tight:
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.orange)
                    .font(.title3)
            case .requiresFlash:
                Image(systemName: "externaldrive.badge.wifi")
                    .foregroundStyle(.indigo)
                    .font(.title3)
            case .tooLarge:
                Image(systemName: "xmark.circle")
                    .foregroundStyle(.red)
                    .font(.title3)
            }
        case .downloading(let p):
            ZStack {
                Circle()
                    .stroke(Color.secondary.opacity(0.2), lineWidth: 2)
                Circle()
                    .trim(from: 0, to: p)
                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .rotationEffect(.degrees(-90))
            }
            .frame(width: 22, height: 22)
        }
    }
}
