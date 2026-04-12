// RootView.swift — Adaptive root layout: tab bar on iOS, sidebar on macOS
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct RootView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var appearance: AppearanceStore
    @Environment(\.modelContext) private var modelContext
    @StateObject private var viewModel = ChatViewModel()
    @StateObject private var registry = RegistryService.shared
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]

    // iOS: tab selection
    @State private var selectedTab: Tab = .chat

    // macOS sheets
    @State private var showModelPicker = false
    @State private var showSettings = false
    @State private var showPersonaDiscovery = false
    @State private var showMap = false
    @State private var showMindPalace = false
    @State private var showTextIngestion = false
    @State private var showModelManagement = false
    enum Tab { case chat, models, palace, mindPalace, miner, settings }

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
                .sheet(isPresented: $showMindPalace) {
                    MindPalaceView()
                        .frame(minWidth: 800, minHeight: 600)
                }
                .sheet(isPresented: $showTextIngestion) {
                    TextIngestionView()
                        .environmentObject(engine)
                }
                .sheet(isPresented: $showModelManagement) {
                    ModelManagementView()
                        .environmentObject(engine)
                }
                .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
                    showModelPicker = true
                }
                .onReceive(NotificationCenter.default.publisher(for: .showTextIngestion)) { _ in
                    showTextIngestion = true
                }
                .onReceive(NotificationCenter.default.publisher(for: .showModelManagement)) { _ in
                    showModelManagement = true
                }
                .onReceive(NotificationCenter.default.publisher(for: .showPersonaDiscovery)) { _ in
                    showPersonaDiscovery = true
                }
                .onAppear {
                    viewModel.engine = engine
                    viewModel.modelContext = modelContext
                }
                .onChange(of: engine.state) { _, state in
                }
                .overlay {
                    if registry.isSyncing {
                        PersonaExtractionOverlay(registry: registry)
                    }
                }
            #else
            iOSTabView
                .onAppear { 
                    viewModel.engine = engine 
                    viewModel.modelContext = modelContext
                }
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
                            .swipeActions {
                                Button(role: .destructive) {
                                    modelContext.delete(wing)
                                    try? modelContext.save()
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                                
                                Button {
                                    Task {
                                        registry.lastSyncLog = "RE-SYNTHESIZING \(wing.name)..."
                                        registry.isSyncing = true
                                        try? await GraphPalaceService.shared.buildRelationalGraph(wingName: wing.name, using: engine) { current, total, text in
                                            Task { @MainActor in
                                                registry.extractionProcessed = current
                                                registry.extractionTotal = total
                                                registry.currentChunkText = text
                                            }
                                        }
                                        try? await GraphPalaceService.shared.synthesizePersonaIndex(wingName: wing.name, using: engine) { current, total, text in
                                            Task { @MainActor in
                                                registry.extractionProcessed = current
                                                registry.extractionTotal = total
                                                registry.currentChunkText = text
                                            }
                                        }
                                        registry.lastSyncLog = "Finished processing \(wing.name)!"
                                        registry.isSyncing = false
                                    }
                                } label: {
                                    Label("Synthesize Graph", systemImage: "network")
                                }
                                .tint(.purple)
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
                Label("Memory Map", systemImage: selectedTab == .palace ? "brain.head.profile" : "brain")
            }
            .tag(Tab.palace)
            
            // ── Mind Palace Graph ───────────────────────────────────────
            NavigationStack {
                MindPalaceView()
            }
            .tabItem {
                Label("Mind Palace", systemImage: "network")
            }
            .tag(Tab.mindPalace)
            
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
                        
                        Button {
                            showMindPalace = true
                        } label: {
                            Label("Mind Palace", systemImage: "network")
                                .foregroundStyle(.purple)
                        }
                        .buttonStyle(.plain)
                    }
                    
                    Section {
                        ForEach(wings) { wing in
                            Button {
                                viewModel.currentWing = wing.name
                                viewModel.newConversation()
                            } label: {
                                Label(wing.name, systemImage: "person.crop.circle")
                            }
                            .buttonStyle(.plain)
                            .contextMenu {
                                Button {
                                    Task {
                                        registry.lastSyncLog = "RE-SYNTHESIZING \(wing.name)..."
                                        registry.isSyncing = true
                                        try? await GraphPalaceService.shared.buildRelationalGraph(wingName: wing.name, using: engine) { current, total, text in
                                            Task { @MainActor in
                                                registry.extractionProcessed = current
                                                registry.extractionTotal = total
                                                registry.currentChunkText = text
                                            }
                                        }
                                        try? await GraphPalaceService.shared.synthesizePersonaIndex(wingName: wing.name, using: engine) { current, total, text in
                                            Task { @MainActor in
                                                registry.extractionProcessed = current
                                                registry.extractionTotal = total
                                                registry.currentChunkText = text
                                            }
                                        }
                                        registry.lastSyncLog = "Finished processing \(wing.name)!"
                                        registry.isSyncing = false
                                    }
                                } label: {
                                    Label("Re-Synthesize Graph", systemImage: "network")
                                }
                                
                                Button(role: .destructive) {
                                    modelContext.delete(wing)
                                    try? modelContext.save()
                                } label: {
                                    Label("Delete Persona", systemImage: "trash")
                                }
                            }
                        }
                    } header: {
                        HStack {
                            Text("Friends (Personas)")
                            Spacer()
                            Button {
                                showPersonaDiscovery = true
                            } label: {
                                Image(systemName: "plus")
                                    .foregroundStyle(SwiftBuddyTheme.accent)
                                    .padding(.top, 4)
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
        } detail: {
            ChatView(
                viewModel: viewModel,
                showSettings: $showSettings,
                showModelPicker: $showModelPicker
            )
            .frame(minWidth: 400)
            .background(SwiftBuddyTheme.background)
            .navigationTitle("Chat")
            .sheet(isPresented: $showPersonaDiscovery) {
                PersonaDiscoveryView(registry: registry)
            }
        }
    }

    // Branded header — gear icon (settings trigger) + SwiftBuddy wordmark
    private var sidebarHeader: some View {
        HStack(spacing: 10) {
            Button {
                showSettings = true
            } label: {
                ZStack {
                    Circle()
                        .fill(SwiftBuddyTheme.heroGradient)
                        .frame(width: 32, height: 32)
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(.white)
                }
                .shadow(color: SwiftBuddyTheme.accent.opacity(0.40), radius: 6)
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 1) {
                Text("SwiftBuddy")
                    .font(.system(.subheadline, weight: .bold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Configuration")
                    .font(.caption2)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }

            Spacer()
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
                
                Spacer()
                
                Button {
                    showModelManagement = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }
                .buttonStyle(.plain)
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

struct PersonaExtractionOverlay: View {
    @ObservedObject var registry: RegistryService
    @StateObject private var monitor = SystemMonitorService.shared
    @State private var isBlinking = false
    
    var body: some View {
        ZStack {
            // Dark transparent backing
            Color.black.opacity(0.85)
                .edgesIgnoringSafeArea(.all)
                
            VStack(alignment: .leading, spacing: 20) {
                // Header
                HStack {
                    Image(systemName: "cpu")
                        .font(.system(size: 24))
                        .foregroundColor(.green)
                        .symbolEffect(.pulse)
                    
                    Text("CONSCIOUSNESS SYNTHESIS")
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                    
                    Spacer()
                    
                    Text(isBlinking ? "_" : "")
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                        .onAppear {
                            withAnimation(Animation.easeInOut(duration: 0.5).repeatForever()) {
                                isBlinking.toggle()
                            }
                        }
                }
                
                Divider().background(Color.green.opacity(0.5))
                
                // Hardware Telemetry
                HStack(spacing: 20) {
                    Text("CPU: \(String(format: "%.0f%%", monitor.cpuLoad * 100))")
                    Text("SYS MEM: \(formatBytes(monitor.memoryUsedBytes))")
                    Text("GPU MAP: \(formatBytes(monitor.vramUsedBytes))")
                }
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.green.opacity(0.8))
                
                // Active Extraction Telemetry
                VStack(alignment: .leading, spacing: 10) {
                    Text("> \(registry.lastSyncLog.uppercased())")
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                    
                    if registry.extractionTotal > 0 {
                        HStack {
                            Text("TARGET SECTOR: [\(registry.extractionPhase.uppercased())]")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.green.opacity(0.8))
                            Spacer()
                            Text("\(registry.extractionProcessed)/\(registry.extractionTotal) VECTORS")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.green.opacity(0.8))
                        }
                        
                        // Cyberpunk Progress Bar 
                        GeometryReader { proxy in
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.green.opacity(0.2))
                                    .frame(height: 12)
                                    .border(Color.green, width: 1)
                                
                                Rectangle()
                                    .fill(Color.green)
                                    .frame(width: proxy.size.width * CGFloat(registry.extractionProcessed) / CGFloat(max(1, registry.extractionTotal)), height: 12)
                                    .animation(.spring(), value: registry.extractionProcessed)
                            }
                        }
                        .frame(height: 12)
                        
                        // Scroll Matrix Text Preview
                        ScrollViewReader { scrollProxy in
                            ScrollView {
                                Text(registry.currentChunkText)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.green.opacity(0.6))
                                    .multilineTextAlignment(.leading)
                                    .lineSpacing(4)
                                    .id("bottom")
                            }
                            .frame(height: 120)
                            .padding()
                            .background(Color.black)
                            .border(Color.green.opacity(0.5), width: 1)
                            .onChange(of: registry.currentChunkText) { _ in
                                scrollProxy.scrollTo("bottom")
                            }
                        }
                    } else {
                        // Downloading Phase Waiter
                        HStack {
                            Text("ESTABLISHING MANIFOLD UPLINK...")
                                .font(.system(size: 14, design: .monospaced))
                                .foregroundColor(.green.opacity(0.6))
                            ProgressView()
                                .controlSize(.small)
                                .tint(.green)
                        }
                        .padding(.top, 20)
                    }
                }
            }
            .padding(30)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.black.opacity(0.9))
                    .border(Color.green.opacity(0.6), width: 2)
            )
            .frame(maxWidth: 600, maxHeight: 400)
            .shadow(color: .green.opacity(0.4), radius: 20, x: 0, y: 0)
        }
        .zIndex(100)
    }
    
    private func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
    }
}
