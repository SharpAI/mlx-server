// ModelManagementView.swift — Disk usage overview and model deletion
import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ModelManagementView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @Environment(\.dismiss) private var dismiss
    @State private var showDeleteAll = false
    @State private var showHFSearch = false
    @State private var deletionError: String? = nil

    @EnvironmentObject private var dm: ModelDownloadManager

    var body: some View {
        NavigationStack {
            Group {
                if dm.downloadedModels.isEmpty {
                    emptyState
                } else {
                    modelList
                }
            }
            .navigationTitle("Downloaded Models")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
                if !dm.downloadedModels.isEmpty {
                    ToolbarItem(placement: .destructiveAction) {
                        Button("Delete All", role: .destructive) {
                            showDeleteAll = true
                        }
                        .foregroundStyle(.red)
                    }
                }
            }
            .confirmationDialog(
                "Delete all \(dm.downloadedModels.count) downloaded models?",
                isPresented: $showDeleteAll,
                titleVisibility: .visible
            ) {
                Button("Delete All Models", role: .destructive) {
                    deleteAllModels()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will free \(formatBytes(dm.totalDiskUsageBytes)) of storage and cannot be undone.")
            }
            .alert("Deletion Error", isPresented: Binding(
                get: { deletionError != nil },
                set: { if !$0 { deletionError = nil } }
            ), actions: {
                Button("OK") { deletionError = nil }
            }, message: {
                Text(deletionError ?? "")
            })
            .safeAreaInset(edge: .bottom) {
                if let (modelId, progress) = dm.activeDownloads.first {
                    FloatingDownloadBanner(modelId: modelId, progress: progress)
                        .padding(.vertical, 8)
                        .background(
                            Rectangle()
                                .fill(SwiftBuddyTheme.background)
                                .shadow(color: .black.opacity(0.1), radius: 5, y: -2)
                        )
                }
            }
            .sheet(isPresented: $showHFSearch) {
                NavigationStack {
                    ZStack {
                        #if os(macOS)
                        Color(nsColor: .windowBackgroundColor).ignoresSafeArea()
                        #else
                        Color(hue: 0.67, saturation: 0.20, brightness: 0.07).ignoresSafeArea()
                        #endif
                        
                        // We must duplicate this manually wrapped view component from ModelPickerView
                        HFSearchTab(onSelect: { id in
                            showHFSearch = false
                            dismiss()
                            Task { await engine.load(modelId: id) }
                        })
                    }
                    .navigationTitle("Search HuggingFace Hub")
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Close") { showHFSearch = false }
                        }
                    }
                }
                .safeAreaInset(edge: .bottom) {
                    if let (modelId, progress) = dm.activeDownloads.first {
                        FloatingDownloadBanner(modelId: modelId, progress: progress)
                            .padding(.vertical, 8)
                            // Match the background style of the floating banner in chat
                            .background(
                                Rectangle()
                                    .fill(SwiftBuddyTheme.background)
                                    .shadow(color: .black.opacity(0.1), radius: 5, y: -2)
                            )
                    }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 500)
        #endif
    }

    // MARK: — Model List

    private var modelList: some View {
        List {
            // Central HF Search Entry
            Section {
                Button { showHFSearch = true } label: {
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(.blue)
                        Text("Search HuggingFace MLX Models")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
            }
            
            // Storage summary card
            Section {
                storageCard
            }

            // Individual models
            Section("Models") {
                ForEach(dm.downloadedModels) { downloaded in
                    downloadedModelRow(downloaded)
                }
            }

            // Cache info
            Section {
                LabeledContent("Cache Location") {
                   Text(ModelStorage.cacheRoot.path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.trailing)
                }
            } footer: {
                Text("Models are stored in the HuggingFace hub cache. Deleting here removes them from all apps that use this cache.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .listStyle(.inset)
    }

    // MARK: — Storage Card

    private var storageCard: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Total Downloaded")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(formatBytes(dm.totalDiskUsageBytes))
                        .font(.title2.weight(.bold))
                }
                Spacer()
                Image(systemName: "externaldrive.fill")
                    .font(.largeTitle)
                    .foregroundStyle(
                        LinearGradient(colors: [.blue, .purple], startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
            }

            // Usage bar
            if dm.downloadedModels.count > 1 {
                GeometryReader { geo in
                    HStack(spacing: 2) {
                        ForEach(dm.downloadedModels) { model in
                            let fraction = Double(model.sizeBytes) / Double(dm.totalDiskUsageBytes)
                            colorForModel(model.id)
                                .frame(width: max(2, geo.size.width * fraction))
                                .frame(height: 8)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }
                .frame(height: 8)

                // Legend
                FlowLayout(spacing: 6) {
                    ForEach(dm.downloadedModels) { model in
                        HStack(spacing: 4) {
                            colorForModel(model.id)
                                .frame(width: 8, height: 8)
                                .clipShape(Circle())
                            Text(model.id.components(separatedBy: "/").last ?? model.id)
                                .font(.caption2)
                                .lineLimit(1)
                        }
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: — Model Row

    private func downloadedModelRow(_ downloaded: DownloadedModel) -> some View {
        let entry = ModelCatalog.all.first(where: { $0.id == downloaded.id })
        let isLoaded: Bool = {
            if case .ready(let id) = engine.state { return id == downloaded.id }
            return false
        }()

        return Button {
            dismiss()
            Task { await engine.load(modelId: downloaded.id) }
        } label: {
            HStack(spacing: 12) {
                // Icon
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(colorForModel(downloaded.id))
                        .frame(width: 36, height: 36)
                    Image(systemName: entry?.isMoE == true ? "square.grid.3x3.fill" : "brain")
                        .font(.callout)
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(entry?.displayName ?? downloaded.id.components(separatedBy: "/").last ?? downloaded.id)
                            .font(.headline)
                            .foregroundStyle(.primary)
                        if isLoaded {
                            Text("IN USE")
                                .font(.caption2.weight(.bold))
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .background(Color.green.opacity(0.15))
                                .foregroundStyle(.green)
                                .clipShape(Capsule())
                        }
                    }
                    HStack(spacing: 4) {
                        Text(downloaded.displaySize)
                            .font(.caption).foregroundStyle(.secondary)
                        if let date = downloaded.modifiedDate {
                            Text("·")
                                .foregroundStyle(.secondary)
                            Text(date, style: .relative)
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()

                // Size indicator
                Text(downloaded.displaySize)
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            Button(role: .destructive) {
                deleteModel(downloaded.id)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
        .contextMenu {
            Button(role: .destructive) {
                deleteModel(downloaded.id)
            } label: {
                Label("Delete from cache", systemImage: "trash")
            }
            Button {
                #if os(macOS)
                NSWorkspace.shared.open(downloaded.cacheDirectory)
                #endif
            } label: {
                Label("Show in Finder", systemImage: "folder")
            }
        }
    }

    // MARK: — Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "externaldrive.badge.questionmark")
                .font(.system(size: 56))
                .foregroundStyle(.tertiary)
            Text("No models downloaded")
                .font(.title3.weight(.semibold))
            Text("Models are downloaded automatically when you select them from the model list.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            
            Button { showHFSearch = true } label: {
                HStack {
                    Image(systemName: "magnifyingglass")
                    Text("Search HuggingFace MLX Models")
                }
            }
            .buttonStyle(.borderedProminent)
            
            Button("Cancel") { dismiss() }
                .foregroundStyle(.secondary)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: — Helpers

    private func deleteModel(_ modelId: String) {
        do {
            // Unload the currently loaded model BEFORE attempting filesystem deletion
            // This releases MLX mmap file locks, preventing macOS from throwing Access/IO errors.
            if case .ready(let id) = engine.state, id == modelId {
                engine.unload()
                // Yield the main thread to ensure deallocation completes
                RunLoop.main.run(until: Date().addingTimeInterval(0.2))
            }
            try dm.delete(modelId)
        } catch {
            deletionError = error.localizedDescription
        }
    }

    private func deleteAllModels() {
        // Unload first to free mmap file locks
        if case .ready = engine.state {
            engine.unload()
            RunLoop.main.run(until: Date().addingTimeInterval(0.2))
        }
        
        let ids = dm.downloadedModels.map { $0.id }
        for id in ids {
            try? dm.delete(id)
        }
    }

    private func colorForModel(_ modelId: String) -> Color {
        let colors: [Color] = [.blue, .purple, .green, .orange, .pink, .teal, .indigo, .mint]
        let hash = abs(modelId.hashValue)
        return colors[hash % colors.count]
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1.0 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }
}

// MARK: — Simple Flow Layout

struct FlowLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = layout(subviews: subviews, in: proposal.width ?? 0)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = layout(subviews: subviews, in: bounds.width)
        for (idx, frame) in result.frames.enumerated() {
            guard idx < subviews.count else { break }
            subviews[idx].place(at: CGPoint(x: bounds.minX + frame.minX, y: bounds.minY + frame.minY),
                                proposal: ProposedViewSize(frame.size))
        }
    }

    private func layout(subviews: Subviews, in width: CGFloat) -> (size: CGSize, frames: [CGRect]) {
        var frames: [CGRect] = []
        var x: CGFloat = 0, y: CGFloat = 0, maxH: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > width && x > 0 {
                x = 0; y += maxH + spacing; maxH = 0
            }
            frames.append(CGRect(origin: CGPoint(x: x, y: y), size: size))
            x += size.width + spacing
            maxH = max(maxH, size.height)
        }
        return (CGSize(width: width, height: y + maxH), frames)
    }
}
