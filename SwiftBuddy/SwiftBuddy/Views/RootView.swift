// RootView.swift — Adaptive root layout: tab bar on iOS, sidebar on macOS
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct RootView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var appearance: AppearanceStore
    @StateObject private var viewModel = ChatViewModel()
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]

    // iOS: tab selection
    @State private var selectedTab: Tab = .chat

    // macOS sheets
    @State private var showModelPicker = false
    @State private var showSettings = false
    @State private var showMap = false
    @State private var showTextIngestion = false

    enum Tab { case chat, models, palace, miner, settings }

    var body: some View {
        Group {
            #if os(macOS)
            macOSLayout
                .sheet(isPresented: $showModelPicker) {
                    ModelPickerView(onSelect: { modelId in
                        showModelPicker = false
                        Task { await engine.load(modelId: modelId) }
                    })
                    .environmentObject(engine)
                }
                .sheet(isPresented: $showSettings) {
                    SettingsView(viewModel: viewModel)
                        .environmentObject(appearance)
                }
                .sheet(isPresented: $showMap) {
                    PalaceVisualizerView()
                        .frame(width: 800, height: 600)
                }
                .sheet(isPresented: $showTextIngestion) {
                    TextIngestionView()
                        .environmentObject(engine)
                }
                .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
                    showModelPicker = true
                }
                .onAppear {
                    viewModel.engine = engine
                }
                .onChange(of: engine.state) { _, state in
                }
            #else
            iOSTabView
                .onAppear { viewModel.engine = engine }
            #endif
        }
    }

    // MARK: — iOS Tab View

    #if os(iOS)
    private var iOSTabView: some View {
        TabView(selection: $selectedTab) {
            // ── Chat Tab ──────────────────────────────────────────────────
            NavigationStack {
                List {
                    Section("Conversations") {
                        NavigationLink {
                            ChatView(viewModel: viewModel)
                                .environmentObject(engine)
                                .onAppear { 
                                    viewModel.currentWing = nil
                                    viewModel.newConversation()
                                }
                        } label: {
                            Label("Core System Chat", systemImage: "sparkles")
                        }
                    }
                    
                    Section("Friends (Personas)") {
                        ForEach(wings) { wing in
                            NavigationLink {
                                ChatView(viewModel: viewModel)
                                    .environmentObject(engine)
                                    .onAppear { 
                                        viewModel.currentWing = wing.name 
                                        viewModel.newConversation()
                                    }
                            } label: {
                                Label(wing.name, systemImage: "person.crop.circle")
                            }
                        }
                    }
                }
                .navigationTitle("Connections")
            }
            .tabItem {
                Label("Chat", systemImage: selectedTab == .chat
                      ? "bubble.left.and.bubble.right.fill"
                      : "bubble.left.and.bubble.right")
            }
            .tag(Tab.chat)

            // ── Models Tab ────────────────────────────────────────────────
            NavigationStack {
                ModelsView(viewModel: viewModel)
                    .environmentObject(engine)
            }
            .tabItem {
                Label("Models", systemImage: selectedTab == .models ? "cpu.fill" : "cpu")
            }
            .tag(Tab.models)
            .badge(engine.downloadManager.activeDownloads.isEmpty
                   ? 0
                   : engine.downloadManager.activeDownloads.count)

            // ── Palace Tab ──────────────────────────────────────────────
            NavigationStack {
                PalaceVisualizerView()
            }
            .tabItem {
                Label("Palace", systemImage: selectedTab == .palace ? "brain.head.profile" : "brain")
            }
            .tag(Tab.palace)
            
            // ── Miner Tab ──────────────────────────────────────────────
            NavigationStack {
                TextIngestionView()
                    .environmentObject(engine)
                    .navigationTitle("Memory Miner")
            }
            .tabItem {
                Label("Miner", systemImage: selectedTab == .miner ? "hammer.fill" : "hammer")
            }
            .tag(Tab.miner)

            // ── Settings Tab ──────────────────────────────────────────────
            NavigationStack {
                SettingsView(viewModel: viewModel, isTab: true)
                    .environmentObject(appearance)
            }
            .tabItem {
                Label("Settings", systemImage: selectedTab == .settings ? "gearshape.fill" : "gearshape")
            }
            .tag(Tab.settings)
        }
        .tint(SwiftBuddyTheme.accent)
        // Navigate to Models tab when a model load is requested from chat
        .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
            selectedTab = .models
        }
    }
    #endif

    // MARK: — macOS Split View

    #if os(macOS)
    private var macOSLayout: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 0) {
                // ── Branded sidebar header ────────────────────────────────
                sidebarHeader
                Divider()
                    .background(SwiftBuddyTheme.divider)

                // ── Engine status ─────────────────────────────────────────
                engineStatusSection
                Divider()
                    .background(SwiftBuddyTheme.divider)

                // ── Actions list ──────────────────────────────────────────
                List {
                    Section("Conversations") {
                        Button {
                            viewModel.currentWing = nil
                            viewModel.newConversation()
                        } label: {
                            Label("Core Chat", systemImage: "sparkles")
                                .foregroundStyle(SwiftBuddyTheme.accent)
                        }
                        .buttonStyle(.plain)
                        
                        Button {
                            showMap = true
                        } label: {
                            Label("Memory Map", systemImage: "map.fill")
                                .foregroundStyle(.orange)
                        }
                        .buttonStyle(.plain)
                    }
                    
                    Section("Tools") {
                        Button {
                            showTextIngestion = true
                        } label: {
                            Label("Text Ingestion", systemImage: "hammer.fill")
                                .foregroundStyle(SwiftBuddyTheme.cyan)
                        }
                        .buttonStyle(.plain)
                    }
                    
                    Section("Friends (Personas)") {
                        ForEach(wings) { wing in
                            Button {
                                viewModel.currentWing = wing.name
                                viewModel.newConversation()
                            } label: {
                                Label(wing.name, systemImage: "person.crop.circle")
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
                .background(SwiftBuddyTheme.background)
            }
            .frame(minWidth: 220)
            .background(SwiftBuddyTheme.background)
        } content: {
            ChatView(
                viewModel: viewModel,
                showSettings: $showSettings,
                showModelPicker: $showModelPicker
            )
            .frame(minWidth: 400)
            .background(SwiftBuddyTheme.background)
            .navigationTitle("Chat")
        } detail: {
            InspectorView(
                showModelPicker: $showModelPicker
            )
            .frame(minWidth: 250)
            .background(SwiftBuddyTheme.background)
        }
    }

    // Branded header — bolt icon + SwiftBuddy wordmark + version chip
    private var sidebarHeader: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(SwiftBuddyTheme.heroGradient)
                    .frame(width: 32, height: 32)
                Image(systemName: "bolt.fill")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)
            }
            .shadow(color: SwiftBuddyTheme.accent.opacity(0.40), radius: 6)

            VStack(alignment: .leading, spacing: 1) {
                Text("SwiftBuddy")
                    .font(.system(.subheadline, weight: .bold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Chat")
                    .font(.caption2)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }

            Spacer()

            Text("v1.0")
                .font(.system(size: 9, weight: .bold))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(SwiftBuddyTheme.accent.opacity(0.18), in: Capsule())
                .foregroundStyle(SwiftBuddyTheme.accent)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    // Engine status row in sidebar
    private var engineStatusSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Engine")
                .font(.caption.weight(.semibold))
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                .textCase(.uppercase)
                .padding(.horizontal, 14)
                .padding(.top, 10)

            engineStateView
                .padding(.horizontal, 14)
                .padding(.bottom, 10)
        }
    }

    @ViewBuilder
    private var engineStateView: some View {
        switch engine.state {
        case .idle:
            Button("Load Model") { showModelPicker = true }
                .buttonStyle(.borderedProminent)
                .tint(SwiftBuddyTheme.accent)
                .controlSize(.small)

        case .loading:
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini).tint(SwiftBuddyTheme.accent)
                Text("Loading…")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }

        case .downloading(let progress, let speed):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: progress).tint(SwiftBuddyTheme.accent)
                Text("\(Int(progress * 100))% · \(speed)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }

        case .ready(let modelId):
            HStack(spacing: 6) {
                Circle()
                    .fill(SwiftBuddyTheme.success)
                    .frame(width: 7, height: 7)
                Text(modelId.components(separatedBy: "/").last ?? modelId)
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .lineLimit(1)
            }

        case .generating:
            HStack(spacing: 6) {
                GeneratingDots()
                Text("Generating…")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }

        case .error(let msg):
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.error)
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.error)
                    .lineLimit(2)
            }
        }
    }
    #endif
}
