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
    @State private var deletionError: String? = nil

    private var downloadManager: ModelDownloadManager { engine.downloadManager }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                CatalogTab(
                    downloadManager: downloadManager,
                    device: device,
                    onTap: handleModelTap,
                    showManagement: $showManagement,
                    onSearchHFTap: { showHFSearch = true },
                    onDeleteModel: deleteModel
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
            .safeAreaInset(edge: .bottom) {
                if let (modelId, progress) = downloadManager.activeDownloads.first {
                    FloatingDownloadBanner(modelId: modelId, progress: progress)
                        .padding(.vertical, 8)
                }
            }
            .alert("Deletion Error", isPresented: Binding(
                get: { deletionError != nil },
                set: { if !$0 { deletionError = nil } }
            ), actions: {
                Button("OK") { deletionError = nil }
            }, message: {
                Text(deletionError ?? "")
            })
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
            .safeAreaInset(edge: .bottom) {
                if let (modelId, progress) = engine.downloadManager.activeDownloads.first {
                    FloatingDownloadBanner(modelId: modelId, progress: progress)
                        .padding(.vertical, 8)
                }
            }
            .frame(minWidth: 600, minHeight: 600)
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

    private func deleteModel(_ modelId: String) {
        do {
            if case .ready(let id) = engine.state, id == modelId {
                engine.unload()
                RunLoop.main.run(until: Date().addingTimeInterval(0.2))
            }
            try downloadManager.delete(modelId)
        } catch {
            deletionError = error.localizedDescription
        }
    }
}

enum CatalogCategory: String, CaseIterable, Identifiable {
    case staffPicks = "Staff Picks"
    case downloaded = "Downloaded"
    case all = "All Models"
    var id: String { rawValue }
}

private struct CatalogTab: View {
    let downloadManager: ModelDownloadManager
    let device: DeviceProfile
    let onTap: (String) -> Void
    @Binding var showManagement: Bool
    let onSearchHFTap: () -> Void
    let onDeleteModel: (String) -> Void

    @State private var selectedCategory: CatalogCategory? = .staffPicks

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedCategory) {
                ForEach(CatalogCategory.allCases) { category in
                    Text(category.rawValue).tag(category)
                }
            }
            .navigationTitle("Categories")
            #if os(macOS)
            .navigationSplitViewColumnWidth(min: 150, ideal: 200, max: 250)
            #endif
        } detail: {
            ScrollView {
                deviceHeader
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 280), spacing: 16)], spacing: 16) {
                    Group {
                        switch selectedCategory {
                        case .downloaded:
                            if downloadManager.downloadedModels.isEmpty {
                                Text("No models downloaded locally.").foregroundStyle(.secondary)
                            } else {
                                ForEach(downloadManager.downloadedModels) { downloaded in
                                    let entry = ModelCatalog.all.first(where: { $0.id == downloaded.id }) ?? ModelEntry(id: downloaded.id, displayName: String(downloaded.id.split(separator: "/").last ?? ""), parameterSize: "Hub Model", quantization: "Native", ramRequiredGB: 0, ramRecommendedGB: 0)
                                    ModelCard(model: entry, downloadStatus: .downloaded(sizeString: downloaded.displaySize), fitStatus: ModelCatalog.fitStatus(for: entry, on: device), downloadProgress: downloadManager.activeDownloads[entry.id], onTap: { onTap(entry.id) }, onDelete: { onDeleteModel(entry.id) })
                                }
                            }
                        case .staffPicks:
                            ForEach(ModelCatalog.staffPicks) { model in
                                ModelCard(model: model, downloadStatus: downloadManager.isDownloaded(model.id) ? .downloaded(sizeString: "") : .available, fitStatus: ModelCatalog.fitStatus(for: model, on: device), downloadProgress: downloadManager.activeDownloads[model.id], onTap: { onTap(model.id) }, onDelete: nil)
                            }
                        case .all:
                            ForEach(ModelCatalog.all) { model in
                                ModelCard(model: model, downloadStatus: downloadManager.isDownloaded(model.id) ? .downloaded(sizeString: "") : .available, fitStatus: ModelCatalog.fitStatus(for: model, on: device), downloadProgress: downloadManager.activeDownloads[model.id], onTap: { onTap(model.id) }, onDelete: nil)
                            }
                        case .none:
                            Text("Select a category").foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle(selectedCategory?.rawValue ?? "")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: onSearchHFTap) {
                        Label("Search HF", systemImage: "magnifyingglass")
                    }
                }
            }
        }
    }

    private var deviceHeader: some View {
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
        .padding(.vertical, 8)
        .padding(.horizontal, 16)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 16)
        .padding(.top, 16)
    }
}

// MARK: — HuggingFace Search Tab

struct HFSearchTab: View {
    let onSelect: (String) -> Void

    @ObservedObject private var service = HFModelSearchService.shared
    @State private var query = ""
    @State private var sort = HFSortOption.trending
    @State private var sizeFilter = HFSizeFilter.all

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
                        service.search(query: query, sort: sort, sizeFilter: sizeFilter)
                    }

                // ─────────────────────────────────────────────────────────────
                // Sort tags (Trending, Likes, etc)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(HFSortOption.allCases, id: \.self) { option in
                            Button {
                                sort = option
                                service.search(query: query, sort: sort, sizeFilter: sizeFilter)
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

                // Size filter segmented bar
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(HFSizeFilter.allCases, id: \.self) { filter in
                            Button {
                                sizeFilter = filter
                                service.search(query: query, sort: sort, sizeFilter: sizeFilter)
                            } label: {
                                Text(filter.label)
                                    .font(.caption.weight(.medium))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5)
                                    .background(
                                        sizeFilter == filter ? Color.accentColor.opacity(0.2) : Color.clear,
                                        in: RoundedRectangle(cornerRadius: 6)
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 6)
                                            .stroke(sizeFilter == filter ? Color.accentColor.opacity(0.5) : Color.clear, lineWidth: 1)
                                    )
                                    .foregroundStyle(sizeFilter == filter ? Color.accentColor : .secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(3)
                    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
            
            Divider()

            // ── Results ────────────────────────────────────────────────────
            if service.isSearching && service.results.isEmpty {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Searching HuggingFace…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let err = service.errorMessage {
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
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if service.results.isEmpty && !query.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No MLX models found for \"\(query)\"")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 16) {
                        ForEach(service.results) { model in
                            HFModelRow(model: model, onSelect: onSelect)
                            Divider()
                        }
                        if service.hasMore {
                            Button("Load More") { service.loadMore() }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                                .padding(.top, 4)
                        }
                    }
                    .padding()
                }
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
            service.search(query: newValue, sort: sort, sizeFilter: sizeFilter)
        }
        .onAppear {
            if service.results.isEmpty {
                service.search(query: "", sort: sort, sizeFilter: sizeFilter)
            }
        }
    }
}

// MARK: — HF Model Row

private struct HFModelRow: View {
    let model: HFModelResult
    let onSelect: (String) -> Void
    
    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager
    @State private var pendingLoad = false
    
    private var isDownloaded: Bool { downloadManager.isDownloaded(model.id) }
    private var activeProgress: ModelDownloadProgress? { downloadManager.activeDownloads[model.id] }

    var body: some View {
        Button {
            if isDownloaded {
                onSelect(model.id)
            } else if activeProgress == nil && !pendingLoad {
                pendingLoad = true
                Task {
                    _ = await downloadManager.startDownload(modelId: model.id).result
                    // Fallback reset if the download abruptly errors out offline without completing
                    if !isDownloaded { pendingLoad = false }
                }
            }
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
                        badge(model.formatDisplay, color: model.formatDisplay == "GGUF" ? .indigo : .mint)
                        if model.isMoE {
                            badge("MoE", color: .purple)
                        }
                        if let size = model.paramSizeHint {
                            badge(size, color: .orange)
                        }
                        if let storage = model.storageDisplay {
                            badge(storage, color: .gray)
                        }
                    }
                    
                    if let progress = activeProgress {
                        ProgressView(value: progress.fractionCompleted)
                            .tint(.blue)
                            .padding(.vertical, 2)
                        
                        if let speed = progress.speedMBps {
                            Text(String(format: "%.1f MB/s", speed))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
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
                    
                    if isDownloaded {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title3)
                            .foregroundStyle(.green)
                            .padding(.top, 2)
                    } else if activeProgress != nil || pendingLoad {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.top, 2)
                    } else {
                        Image(systemName: "arrow.down.circle")
                            .font(.title3)
                            .foregroundStyle(.blue)
                            .padding(.top, 2)
                    }
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onChange(of: isDownloaded) { _, newValue in
            if newValue && pendingLoad {
                pendingLoad = false
                onSelect(model.id)
            }
        }
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

// MARK: — ModelCard (reused by catalog tab)

enum DownloadStatus {
    case downloaded(sizeString: String)
    case available
    case downloading(progress: Double)
}

struct ModelCard: View {
    let model: ModelEntry
    let downloadStatus: DownloadStatus
    let fitStatus: ModelCatalog.FitStatus
    let downloadProgress: ModelDownloadProgress?
    let onTap: () -> Void
    let onDelete: (() -> Void)?

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top) {
                    Text(model.displayName)
                        .font(.headline)
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                    Spacer()
                    if let badge = model.badge {
                        Text(badge)
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(.blue.opacity(0.12), in: Capsule())
                            .foregroundStyle(.blue)
                    }
                }

                HStack(spacing: 6) {
                    Text(model.parameterSize)
                    Text("•")
                    Text(model.quantization)
                    if model.isMoE {
                        Text("MoE")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.purple.opacity(0.12), in: Capsule())
                            .foregroundStyle(.purple)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if let progress = downloadProgress {
                    ProgressView(value: progress.fractionCompleted)
                        .tint(.blue)
                    if let speed = progress.speedMBps {
                        Text(String(format: "%.1f MB/s", speed))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer(minLength: 4)

                HStack {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 8, height: 8)
                        Text(statusText)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    actionIcon
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .contextMenu {
            if case .downloaded = downloadStatus, let deleteAction = onDelete {
                Button(role: .destructive, action: deleteAction) {
                    Label("Delete Model", systemImage: "trash")
                }
            }
        }
    }

    private var statusText: String {
        switch fitStatus {
        case .fits: return "Fits comfortably"
        case .tight: return "Tight (slow)"
        case .requiresFlash: return "Flash Streaming"
        case .tooLarge: return "Too large"
        }
    }

    private var statusColor: Color {
        switch fitStatus {
        case .fits: return .green
        case .tight: return .yellow
        case .requiresFlash: return .orange
        case .tooLarge: return .red
        }
    }

    @ViewBuilder
    private var actionIcon: some View {
        switch downloadStatus {
        case .downloaded(let size):
            HStack(spacing: 4) {
                if !size.isEmpty {
                    Text(size).font(.caption2).foregroundColor(.secondary)
                }
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            }
        case .downloading(_):
            EmptyView()
        case .available:
            Image(systemName: "arrow.down.circle")
                .foregroundColor(.blue)
        }
    }
}

// MARK: — Floating Download Banner

struct FloatingDownloadBanner: View {
    let modelId: String
    let progress: ModelDownloadProgress

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .stroke(SwiftBuddyTheme.accent.opacity(0.2), lineWidth: 3)
                Circle()
                    .trim(from: 0, to: progress.fractionCompleted)
                    .stroke(
                        SwiftBuddyTheme.avatarGradient,
                        style: StrokeStyle(lineWidth: 3, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .animation(.linear(duration: 0.3), value: progress.fractionCompleted)
            }
            .frame(width: 30, height: 30)

            VStack(alignment: .leading, spacing: 2) {
                Text("Downloading \(modelId.split(separator: "/").last ?? "")")
                    .font(.system(.subheadline, design: .default, weight: .bold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .lineLimit(1)

                HStack {
                    Text("\(Int(progress.fractionCompleted * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)

                    if let speed = progress.speedMBps {
                        Text("• \(String(format: "%.1f MB/s", speed))")
                            .font(.caption)
                            .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    }
                }
            }
            Spacer()
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(SwiftBuddyTheme.accent.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.15), radius: 10, y: 5)
        .padding(.horizontal)
    }
}
