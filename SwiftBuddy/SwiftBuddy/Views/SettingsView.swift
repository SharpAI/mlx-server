// SettingsView.swift — Full SwiftLM parameter dashboard + console log (iOS tab or macOS sheet)
import Observation
import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct SettingsView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var appearance: AppearanceStore
    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var server: ServerManager
    @Environment(\.dismiss) private var dismiss

    /// When true, the view is embedded as a tab (no Done button on iOS)
    var isTab: Bool = false

    @State private var selectedTab: SettingsTab = .generation
    @State private var draftServerConfiguration = ServerStartupConfiguration.load()
    @State private var showRestartNotification = false
    @State private var endpointCopied = false
    @State private var showAppliedBadge = false
    @State private var toastHideWork: DispatchWorkItem? = nil
    @State private var cliCopied = false
    @State private var serverSaveMessage = "Server settings saved"
    @State private var restartNotificationRequiresAction = false

    // iOS-specific: performance mode toggle (read from UserDefaults)
    @AppStorage("swiftlm.performanceMode") private var performanceMode: Bool = false

    private var ramGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
    }

    private var currentModelIsMoE: Bool {
        guard case .ready(let modelId) = engine.state else { return false }
        return ModelCatalog.all.first(where: { $0.id == modelId })?.isMoE ?? false
    }

    private var effectiveStreamExpertsSetting: Bool {
        viewModel.config.effectiveStreamExperts(defaultingTo: currentModelIsMoE)
    }

    private var ssdStreamingBinding: Binding<Bool> {
        Binding(
            get: { effectiveStreamExpertsSetting },
            set: { viewModel.config.streamExperts = $0 }
        )
    }

    enum SettingsTab: String, CaseIterable {
        case generation = "Generation"
        case engine = "Engine"
        case appearance = "Appearance"
        case console = "Console"
        case about = "About"

        var icon: String {
            switch self {
            case .generation:  return "slider.horizontal.3"
            case .engine:      return "cpu"
            case .appearance:  return "paintpalette"
            case .console:     return "terminal"
            case .about:       return "info.circle"
            }
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                SwiftBuddyTheme.background.ignoresSafeArea()

                VStack(spacing: 0) {
                    // ── Tab Picker ──────────────────────────────────────────
                    tabPicker
                        .padding(.horizontal, 16)
                        .padding(.top, 12)
                        .padding(.bottom, 8)

                    // ── Tab Content ─────────────────────────────────────────
                    switch selectedTab {
                    case .generation:
                        generationTab
                    case .engine:
                        engineTab
                    case .appearance:
                        appearanceTab
                    case .console:
                        consoleTab
                    case .about:
                        aboutTab
                    }
                }
            }
            .navigationTitle("SwiftLM Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(isTab ? .large : .inline)
            .toolbarBackground(SwiftBuddyTheme.background.opacity(0.90), for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            #endif
            .toolbar {
                if !isTab {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                            .foregroundStyle(SwiftBuddyTheme.accent)
                    }
                }
            }
            .overlay(alignment: .top) {
                if showRestartNotification {
                    restartNotificationBanner
                        .padding(.horizontal, 16)
                        .padding(.top, 12)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .onAppear {
                draftServerConfiguration = server.startupConfiguration
            }
            #if os(macOS)
            .frame(minWidth: 520, minHeight: 580)
            #endif
        }
    }

    // MARK: — Tab Picker

    private var tabPicker: some View {
        HStack(spacing: 0) {
            ForEach(SettingsTab.allCases, id: \.self) { tab in
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) { selectedTab = tab }
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 16, weight: .medium))
                        Text(tab.rawValue)
                            .font(.caption2.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(
                        selectedTab == tab
                            ? SwiftBuddyTheme.accent.opacity(0.15)
                            : Color.clear
                    )
                    .foregroundStyle(
                        selectedTab == tab
                            ? SwiftBuddyTheme.accent
                            : SwiftBuddyTheme.textTertiary
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(SwiftBuddyTheme.surface.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 1)
        )
    }

    // MARK: — Generation Tab

    private var generationTab: some View {
        ScrollView {
            VStack(spacing: 16) {
                parameterCard("Sampling") {
                    sliderRow(
                        label: "Temperature", icon: "thermometer.medium",
                        value: Binding(
                            get: { Double(viewModel.config.temperature) },
                            set: { viewModel.config.temperature = Float($0) }
                        ),
                        range: 0...2, step: 0.05, format: "%.2f",
                        tint: SwiftBuddyTheme.warning,
                        hint: "Higher = more creative, lower = more focused"
                    )
                    sliderRow(
                        label: "Top P", icon: "chart.bar.xaxis",
                        value: Binding(
                            get: { Double(viewModel.config.topP) },
                            set: { viewModel.config.topP = Float($0) }
                        ),
                        range: 0...1, step: 0.05, format: "%.2f",
                        tint: SwiftBuddyTheme.accentSecondary,
                        hint: "Nucleus sampling: cumulative probability threshold"
                    )
                    sliderRow(
                        label: "Top K", icon: "line.3.horizontal.decrease.circle",
                        value: Binding(
                            get: { Double(viewModel.config.topK) },
                            set: { viewModel.config.topK = Int($0) }
                        ),
                        range: 1...200, step: 1, format: "%.0f",
                        tint: SwiftBuddyTheme.accent,
                        hint: "Limit sampling to top K candidates"
                    )
                    sliderRow(
                        label: "Min P", icon: "arrow.down.to.line",
                        value: Binding(
                            get: { Double(viewModel.config.minP) },
                            set: { viewModel.config.minP = Float($0) }
                        ),
                        range: 0...0.5, step: 0.01, format: "%.2f",
                        tint: SwiftBuddyTheme.success,
                        hint: "Minimum probability filter (0 = disabled)"
                    )
                }

                parameterCard("Output") {
                    sliderRow(
                        label: "Max Response Tokens", icon: "text.word.spacing",
                        value: Binding(
                            get: { Double(viewModel.config.maxTokens) },
                            set: { viewModel.config.maxTokens = Int($0) }
                        ),
                        range: 128...16384.0, step: 128, format: "%.0f",
                        tint: SwiftBuddyTheme.accent,
                        hint: engine.maxContextWindow > 0
                            ? "Max output per reply. Model context window: \(engine.maxContextWindow / 1000)K tokens"
                            : "Max tokens generated per response (context window shown once model loads)"
                    )
                    sliderRow(
                        label: "Repetition Penalty", icon: "repeat.circle",
                        value: Binding(
                            get: { Double(viewModel.config.repetitionPenalty) },
                            set: { viewModel.config.repetitionPenalty = Float($0) }
                        ),
                        range: 1.0...2.0, step: 0.01, format: "%.2f",
                        tint: SwiftBuddyTheme.success,
                        hint: "Higher = less repeating, 1.0 = disabled"
                    )

                    // Seed — optional reproducibility
                    HStack {
                        Label("Seed", systemImage: "number")
                            .foregroundStyle(SwiftBuddyTheme.textPrimary)
                            .font(.callout)
                        Spacer()
                        if let seed = viewModel.config.seed {
                            Text("\(seed)")
                                .foregroundStyle(SwiftBuddyTheme.textSecondary)
                                .font(.callout.monospacedDigit())
                            Stepper("", value: Binding(
                                get: { Int(min(seed, UInt64(Int.max))) },
                                set: { viewModel.config.seed = UInt64($0) }
                            ), in: 0...Int.max)
                            .labelsHidden()
                            Button {
                                viewModel.config.seed = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
                            }
                            .buttonStyle(.plain)
                        } else {
                            Text("Random")
                                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                                .font(.callout)
                            Button {
                                viewModel.config.seed = UInt64.random(in: 0...UInt64(Int.max))
                            } label: {
                                Image(systemName: "lock.fill")
                                    .foregroundStyle(SwiftBuddyTheme.accent)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.vertical, 2)
                    if viewModel.config.seed != nil {
                        Text("Fixed seed — same input will produce identical output")
                            .font(.caption2)
                            .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    }
                }

                parameterCard("Reasoning") {
                    toggleRow(
                        label: "Thinking Mode", icon: "brain.head.profile",
                        isOn: $viewModel.config.enableThinking,
                        tint: SwiftBuddyTheme.accentSecondary,
                        hint: "Step-by-step reasoning for Qwen3.5, DeepSeek-R1"
                    )
                }

                parameterCard("Performance") {
                    toggleRow(
                        label: "SSD Streaming", icon: "internaldrive",
                        isOn: ssdStreamingBinding,
                        tint: SwiftBuddyTheme.warning,
                        hint: "Stream MoE expert weights from NVMe (requires model reload)"
                    )
                    toggleRow(
                        label: "TurboQuant KV", icon: "bolt.badge.clock",
                        isOn: $viewModel.config.turboKV,
                        tint: SwiftBuddyTheme.success,
                        hint: "3-bit KV compression for massive context windows"
                    )
                }

                parameterCard("System Prompt") {
                    TextEditor(text: $viewModel.systemPrompt)
                        .frame(minHeight: 80)
                        .font(.callout.monospaced())
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                        .scrollContentBackground(.hidden)
                        .padding(8)
                        .background(SwiftBuddyTheme.background.opacity(0.5))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    Text("Injected as the system message before every conversation.")
                        .font(.caption2)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }

                // Reset button
                Button(role: .destructive) {
                    viewModel.config = .default   // didSet triggers config.save()
                    viewModel.systemPrompt = ""    // didSet clears UserDefaults key
                } label: {
                    HStack {
                        Image(systemName: "arrow.counterclockwise")
                        Text("Reset to Defaults")
                    }
                    .font(.callout.weight(.medium))
                    .foregroundStyle(SwiftBuddyTheme.error)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(SwiftBuddyTheme.error.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)

                Spacer(minLength: 20)
            }
            .padding(.top, 8)
        }
        // Generation params are hot-applied per request — no restart needed.
        // Flash a brief badge so the user knows the change was captured.
        .onChange(of: viewModel.config.temperature)       { flashApplied() }
        .onChange(of: viewModel.config.topP)              { flashApplied() }
        .onChange(of: viewModel.config.topK)              { flashApplied() }
        .onChange(of: viewModel.config.minP)              { flashApplied() }
        .onChange(of: viewModel.config.maxTokens)         { flashApplied() }
        .onChange(of: viewModel.config.repetitionPenalty) { flashApplied() }
        .onChange(of: viewModel.config.enableThinking)    { flashApplied() }
        .onChange(of: viewModel.config.kvBits)            { flashApplied() }
        .onChange(of: viewModel.config.prefillSize)       { flashApplied() }
        .onChange(of: viewModel.config.seed)              { flashApplied() }
        .overlay(alignment: .top) {
            if showAppliedBadge {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(SwiftBuddyTheme.success)
                        .font(.caption)
                    Text("Applied — takes effect on next message")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .background(SwiftBuddyTheme.success.opacity(0.12))
                .clipShape(Capsule())
                .overlay(Capsule().strokeBorder(SwiftBuddyTheme.success.opacity(0.3), lineWidth: 1))
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
                .animation(.easeInOut(duration: 0.2), value: showAppliedBadge)
            }
        }
    }

    // MARK: — Engine Tab

    private var engineTab: some View {
        ScrollView {
            VStack(spacing: 16) {
                parameterCard("Local API Server") {
                    // ── Endpoint URL card (tap to copy) ─────────────────────
                    let endpointURL = "http://\(server.host):\(server.port)"
                    Button {
                        copyEndpoint(endpointURL)
                    } label: {
                        HStack(spacing: 12) {
                            // Status dot
                            Circle()
                                .fill(server.isOnline ? SwiftBuddyTheme.success : SwiftBuddyTheme.textTertiary)
                                .frame(width: 8, height: 8)
                                .shadow(color: server.isOnline ? SwiftBuddyTheme.success.opacity(0.6) : .clear,
                                        radius: 4)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(server.isOnline ? "Online" : "Offline")
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(server.isOnline ? SwiftBuddyTheme.success : SwiftBuddyTheme.textTertiary)
                                Text(endpointURL)
                                    .font(.system(.callout, design: .monospaced))
                                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                            }

                            Spacer()

                            // Copy / confirm icon
                            Image(systemName: endpointCopied ? "checkmark" : "doc.on.doc")
                                .font(.caption)
                                .foregroundStyle(endpointCopied ? SwiftBuddyTheme.success : SwiftBuddyTheme.textTertiary)
                                .animation(.easeInOut(duration: 0.2), value: endpointCopied)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity)
                        .background(SwiftBuddyTheme.background.opacity(0.6))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .strokeBorder(
                                    server.isOnline
                                        ? SwiftBuddyTheme.success.opacity(0.3)
                                        : Color.white.opacity(0.07),
                                    lineWidth: 1
                                )
                        )
                    }
                    .buttonStyle(.plain)

                    // Quick-use hint for external tools
                    if server.isOnline {
                        Text("Compatible with OpenAI SDK, LM Studio, Continue, Cursor")
                            .font(.caption2)
                            .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    }

                    toggleRow(
                        label: "Start Server on Launch", icon: "power.circle",
                        isOn: $draftServerConfiguration.autoStart,
                        tint: SwiftBuddyTheme.success
                    )

                    textFieldRow(
                        label: "Host", icon: "network",
                        text: $draftServerConfiguration.host,
                        placeholder: "127.0.0.1"
                    )

                    stepperRow(
                        label: "Port", icon: "number.circle",
                        value: $draftServerConfiguration.port,
                        range: 1...65_535
                    )

                    stepperRow(
                        label: "Parallel Slots", icon: "square.stack.3d.up",
                        value: $draftServerConfiguration.parallelSlots,
                        range: 1...16
                    )

                    textFieldRow(
                        label: "CORS Origin", icon: "globe",
                        text: $draftServerConfiguration.corsOrigin,
                        placeholder: "Optional"
                    )

                    secureFieldRow(
                        label: "API Key", icon: "key",
                        text: $draftServerConfiguration.apiKey,
                        placeholder: "Optional"
                    )

                    HStack(spacing: 10) {
                        Button {
                            saveServerConfiguration()
                        } label: {
                            Label("Save Server Settings", systemImage: "square.and.arrow.down")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(SwiftBuddyTheme.accent)
                        .disabled(draftServerConfiguration.normalized == server.startupConfiguration)

                        if server.restartRequired {
                            Button {
                                restartServer()
                            } label: {
                                Label("Restart", systemImage: "arrow.clockwise")
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }

                parameterCard("KV Cache") {
                    HStack {
                        Label("KV Bits", systemImage: "number.circle")
                            .foregroundStyle(SwiftBuddyTheme.textPrimary)
                            .font(.callout)
                        Spacer()
                        Picker("", selection: Binding(
                            get: { viewModel.config.kvBits ?? 0 },
                            set: { viewModel.config.kvBits = $0 == 0 ? nil : $0 }
                        )) {
                            Text("Off").tag(0)
                            Text("4-bit").tag(4)
                            Text("8-bit").tag(8)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 200)
                    }
                    .padding(.vertical, 2)

                    sliderRow(
                        label: "KV Group Size", icon: "square.3.layers.3d",
                        value: Binding(
                            get: { Double(viewModel.config.kvGroupSize) },
                            set: { viewModel.config.kvGroupSize = Int($0) }
                        ),
                        range: 32...128, step: 32, format: "%.0f",
                        tint: SwiftBuddyTheme.warning,
                        hint: "Applies to quantized KV cache during generation"
                    )
                }

                parameterCard("Prefill") {
                    sliderRow(
                        label: "Prefill Chunk Size", icon: "rectangle.split.3x3",
                        value: Binding(
                            get: { Double(viewModel.config.prefillSize) },
                            set: { viewModel.config.prefillSize = Int($0) }
                        ),
                        range: 64...2048, step: 64, format: "%.0f",
                        tint: SwiftBuddyTheme.accentSecondary,
                        hint: "Lower values prevent GPU timeout on large models"
                    )
                }

                parameterCard("Advanced Engine") {
                    // ── TurboKV (per-request, no reload needed) ──────────────────────────
                    toggleRow(
                        label: "TurboKV Compression", icon: "memorychip",
                        isOn: $viewModel.config.turboKV,
                        tint: SwiftBuddyTheme.warning,
                        hint: "3-bit PolarQuant+QJL compression for KV history >8K tokens. Halves long-context RAM — applied per request"
                    )

                    Divider().background(SwiftBuddyTheme.divider)

                    // ── SSD Expert Streaming (load-time — shows reload prompt) ────
                    VStack(alignment: .leading, spacing: 6) {
                        toggleRow(
                            label: "SSD Expert Streaming", icon: "externaldrive.fill",
                            isOn: ssdStreamingBinding,
                            tint: SwiftBuddyTheme.accentSecondary,
                            hint: "mmap expert weights from NVMe — only active expert pages stay in RAM. Auto-enabled for MoE catalog models."
                        )
                        if effectiveStreamExpertsSetting != currentModelIsMoE {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack(spacing: 6) {
                                    Image(systemName: "arrow.clockwise.circle.fill")
                                        .foregroundStyle(SwiftBuddyTheme.warning)
                                        .font(.caption)
                                    Text("Reload model to apply this change")
                                        .font(.caption2.weight(.medium))
                                        .foregroundStyle(SwiftBuddyTheme.warning)
                                    Spacer()
                                    Button("Reload") {
                                        let currentId: String? = {
                                            if case .ready(let id) = engine.state { return id }
                                            return nil
                                        }()
                                        if let id = currentId {
                                            Task {
                                                engine.unload()
                                                await engine.load(modelId: id)
                                            }
                                        }
                                    }
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(SwiftBuddyTheme.accent)
                                    .buttonStyle(.plain)
                                }

                                switch engine.state {
                                case .loading(let progress, let stage):
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Text(stage)
                                                .font(.caption2.weight(.medium))
                                                .foregroundStyle(SwiftBuddyTheme.textSecondary)
                                            Spacer()
                                            Text("\(Int(progress * 100))%")
                                                .font(.caption2.monospacedDigit())
                                                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                                        }
                                        ProgressView(value: progress)
                                            .tint(SwiftBuddyTheme.accent)
                                    }
                                case .downloading(let progress, let speed):
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Text("Downloading model files")
                                                .font(.caption2.weight(.medium))
                                                .foregroundStyle(SwiftBuddyTheme.textSecondary)
                                            Spacer()
                                            Text("\(Int(progress * 100))% · \(speed)")
                                                .font(.caption2.monospacedDigit())
                                                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                                        }
                                        ProgressView(value: progress)
                                            .tint(SwiftBuddyTheme.accent)
                                    }
                                default:
                                    EmptyView()
                                }
                            }
                            .padding(.horizontal, 4)
                            .padding(.vertical, 6)
                            .background(SwiftBuddyTheme.warning.opacity(0.08))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                #if os(iOS)
                parameterCard("iOS Performance") {
                    toggleRow(
                        label: "Performance Mode", icon: "bolt.fill",
                        isOn: $performanceMode,
                        tint: SwiftBuddyTheme.accent,
                        hint: "Use 55% RAM budget (vs. 40%) — enables larger models on \(String(format: "%.0f GB", ramGB)) device"
                    )
                    toggleRow(
                        label: "Auto-Unload in Background", icon: "iphone.slash",
                        isOn: Binding(
                            get: { UserDefaults.standard.bool(forKey: "swiftlm.autoOffload") == false
                                    ? true
                                    : UserDefaults.standard.bool(forKey: "swiftlm.autoOffload") },
                            set: { UserDefaults.standard.set($0, forKey: "swiftlm.autoOffload") }
                        ),
                        tint: SwiftBuddyTheme.success,
                        hint: "Frees GPU memory when the app backgrounds"
                    )
                }
                #endif

                // ── CLI Equivalent ──────────────────────────────────────────
                parameterCard("CLI Equivalent") {
                    Text("Run standalone server with these settings:")
                        .font(.caption2)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)

                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(cliCommand)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(SwiftBuddyTheme.textSecondary)
                            .textSelection(.enabled)
                            .padding(.vertical, 6)
                    }

                    Button {
                        copyCLI()
                    } label: {
                        Label(
                            cliCopied ? "Copied!" : "Copy Command",
                            systemImage: cliCopied ? "checkmark" : "doc.on.doc"
                        )
                        .font(.caption.weight(.medium))
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(cliCopied ? SwiftBuddyTheme.success : SwiftBuddyTheme.accent)
                    .animation(.easeInOut(duration: 0.2), value: cliCopied)
                }
                Spacer(minLength: 20)
            }
            .padding(.top, 8)
        }
    }

    // MARK: - Appearance Tab

    // Use local state for the picker to avoid triggering a @Published write
    // directly from within a view update cycle, which causes the crash:
    // "Publishing changes from within view updates is not allowed"
    @State private var localColorScheme: String = "dark"

    private var appearanceTab: some View {
        ScrollView {
            VStack(spacing: 16) {
                parameterCard("Theme") {
                    HStack {
                        Label("Color Scheme", systemImage: "paintpalette")
                            .foregroundStyle(SwiftBuddyTheme.textPrimary)
                            .font(.callout)
                        Spacer()
                    }
                    Picker("", selection: $localColorScheme) {
                        Text("Dark").tag("dark")
                        Text("Light").tag("light")
                        Text("System").tag("system")
                    }
                    .pickerStyle(.segmented)
                    .tint(SwiftBuddyTheme.accent)
                    .onChange(of: localColorScheme) { newValue in
                        // Defer the @Published write to avoid the view update crash
                        Task { @MainActor in
                            appearance.preference = newValue
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
        .onAppear {
            localColorScheme = appearance.preference
        }
    }

    // MARK: — Console Tab

    private var consoleTab: some View {
        VStack(spacing: 0) {
            // Console header
            HStack {
                Image(systemName: "terminal")
                    .foregroundStyle(SwiftBuddyTheme.accent)
                Text("SwiftLM Console")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
                Button {
                    ConsoleLog.shared.clear()
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider().background(SwiftBuddyTheme.divider)

            // Console output
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(ConsoleLog.shared.entries) { entry in
                            consoleEntryRow(entry)
                                .id(entry.id)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                .onChange(of: ConsoleLog.shared.entries.count) {
                    if let last = ConsoleLog.shared.entries.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }
            .background(Color.black.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal, 12)
            .padding(.bottom, 12)
        }
    }

    private func consoleEntryRow(_ entry: ConsoleLog.Entry) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(entry.timestamp)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
            Image(systemName: entry.level.icon)
                .font(.system(size: 9))
                .foregroundStyle(entry.level.color)
                .frame(width: 12)
            Text(entry.message)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(entry.level.textColor)
                .textSelection(.enabled)
        }
        .padding(.vertical, 1)
    }

    // MARK: — About Tab

    private var aboutTab: some View {
        ScrollView {
            VStack(spacing: 16) {
                parameterCard("System Info") {
                    aboutRow("SwiftBuddy Chat", value: "1.0")
                    aboutRow("Engine", value: "MLX Swift")
                    aboutRow("Backend", value: "Metal GPU")
                    aboutRow("Platform", value: {
                        #if os(iOS)
                        return "iOS / iPadOS"
                        #else
                        return "macOS"
                        #endif
                    }())
                    aboutRow("RAM", value: String(format: "%.0f GB", ramGB))
                }

                parameterCard("Active Model") {
                    switch engine.state {
                    case .ready(let modelId):
                        aboutRow("Model", value: modelId.components(separatedBy: "/").last ?? modelId)
                    case .idle:
                        aboutRow("Model", value: "None loaded")
                    default:
                        aboutRow("Model", value: engine.state.shortLabel)
                    }
                    if engine.activeContextTokens > 0 {
                        aboutRow("Context Tokens", value: "\(engine.activeContextTokens)")
                    }
                }

                parameterCard("Quick Actions") {
                    Button {
                        NotificationCenter.default.post(name: .showModelManagement, object: nil)
                        dismiss()
                    } label: {
                        quickActionRow("Model Configuration", icon: "cpu.fill")
                    }
                    .buttonStyle(.plain)

                    Button {
                        NotificationCenter.default.post(name: .showTextIngestion, object: nil)
                        dismiss()
                    } label: {
                        quickActionRow("Text Ingestion Miner", icon: "hammer.fill")
                    }
                    .buttonStyle(.plain)

                    Button {
                        NotificationCenter.default.post(name: .showModelManagement, object: nil)
                        dismiss()
                    } label: {
                        quickActionRow("Manage Downloaded Models", icon: "externaldrive.badge.minus")
                    }
                    .buttonStyle(.plain)
                }

                Spacer(minLength: 20)
            }
            .padding(.top, 8)
        }
    }

    // MARK: — Reusable Components

    private var restartNotificationBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.clockwise.circle.fill")
                .foregroundStyle(SwiftBuddyTheme.warning)
            VStack(alignment: .leading, spacing: 2) {
                Text(serverSaveMessage)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text(restartNotificationRequiresAction
                     ? "Restart the local API server to apply startup changes."
                     : "Changes will apply the next time the server starts.")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }
            Spacer()
            if restartNotificationRequiresAction {
                Button("Restart") {
                    restartServer()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .tint(SwiftBuddyTheme.accent)
            }
            Button {
                withAnimation { showRestartNotification = false }
            } label: {
                Image(systemName: "xmark")
                    .font(.caption.weight(.bold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(SwiftBuddyTheme.textTertiary)
        }
        .padding(12)
        .background(SwiftBuddyTheme.surface.opacity(0.96))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(SwiftBuddyTheme.warning.opacity(0.45), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.18), radius: 14, y: 6)
    }

    private func flashApplied() {
        withAnimation { showAppliedBadge = true }
        // Cancel any pending hide before scheduling a new one to prevent
        // stacked closures from causing flicker when sliders are dragged rapidly.
        toastHideWork?.cancel()
        let work = DispatchWorkItem {
            withAnimation { showAppliedBadge = false }
        }
        toastHideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 2, execute: work)
    }

    /// Build the equivalent `swift run SwiftLM` command from current settings.
    private var cliCommand: String {
        buildCLICommand(
            config: viewModel.config,
            host: server.host,
            port: server.port,
            parallel: server.startupConfiguration.parallelSlots,
            apiKeySet: !server.startupConfiguration.apiKey.isEmpty,
            modelId: {
                if case .ready(let id) = engine.state { return id }
                return nil
            }()
        )
    }

    private func copyCLI() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(cliCommand, forType: .string)
        #else
        UIPasteboard.general.string = cliCommand
        #endif
        withAnimation { cliCopied = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation { cliCopied = false }
        }
    }

    private func copyEndpoint(_ url: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(url, forType: .string)
        #else
        UIPasteboard.general.string = url
        #endif
        withAnimation { endpointCopied = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation { endpointCopied = false }
        }
    }

    private func saveServerConfiguration() {
        let changed = server.saveStartupConfiguration(draftServerConfiguration)
        draftServerConfiguration = server.startupConfiguration
        if server.restartRequired {
            serverSaveMessage = changed ? "Server settings saved" : "Server restart required"
            restartNotificationRequiresAction = true
            withAnimation(.easeInOut(duration: 0.2)) {
                showRestartNotification = true
            }
        } else {
            serverSaveMessage = "Server settings saved"
            restartNotificationRequiresAction = false
            withAnimation(.easeInOut(duration: 0.2)) {
                showRestartNotification = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showRestartNotification = false
                }
            }
        }
    }

    private func restartServer() {
        server.restart(engine: engine)
        serverSaveMessage = "Server restarted"
        withAnimation(.easeInOut(duration: 0.2)) {
            showRestartNotification = false
        }
    }

    @ViewBuilder
    private func parameterCard<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption.weight(.bold))
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                .tracking(0.8)

            VStack(alignment: .leading, spacing: 12) {
                content()
            }
            .padding(14)
            .background(SwiftBuddyTheme.surface.opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(Color.white.opacity(0.06), lineWidth: 1)
            )
        }
        .padding(.horizontal, 16)
    }

    private func sliderRow(
        label: String, icon: String,
        value: Binding<Double>,
        range: ClosedRange<Double>, step: Double, format: String,
        tint: Color, hint: String? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label(label, systemImage: icon)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .font(.callout)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .monospacedDigit()
                    .font(.callout.weight(.medium))
            }
            Slider(value: value, in: range, step: step)
                .tint(tint)
            if let hint = hint {
                Text(hint)
                    .font(.caption2)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }
        }
    }

    private func toggleRow(
        label: String, icon: String,
        isOn: Binding<Bool>,
        tint: Color, hint: String? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle(isOn: isOn) {
                VStack(alignment: .leading, spacing: 2) {
                    Label(label, systemImage: icon)
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                        .font(.callout)
                    if let hint = hint {
                        Text(hint)
                            .font(.caption2)
                            .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    }
                }
            }
            .tint(tint)
        }
    }

    private func textFieldRow(
        label: String, icon: String,
        text: Binding<String>, placeholder: String
    ) -> some View {
        HStack(spacing: 12) {
            Label(label, systemImage: icon)
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .font(.callout)
            Spacer()
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 240)
        }
    }

    private func secureFieldRow(
        label: String, icon: String,
        text: Binding<String>, placeholder: String
    ) -> some View {
        HStack(spacing: 12) {
            Label(label, systemImage: icon)
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .font(.callout)
            Spacer()
            SecureField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 240)
        }
    }

    private func stepperRow(
        label: String, icon: String,
        value: Binding<Int>, range: ClosedRange<Int>
    ) -> some View {
        Stepper(value: value, in: range) {
            HStack {
                Label(label, systemImage: icon)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .font(.callout)
                Spacer()
                Text("\(value.wrappedValue)")
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .font(.callout.monospacedDigit())
            }
        }
    }

    private func aboutRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .font(.callout)
            Spacer()
            Text(value)
                .foregroundStyle(SwiftBuddyTheme.textSecondary)
                .font(.callout.weight(.medium))
        }
    }

    private func quickActionRow(_ label: String, icon: String) -> some View {
        HStack {
            Label(label, systemImage: icon)
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .font(.callout)
            Spacer()
            Image(systemName: "chevron.right")
                .font(.caption2)
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: — Console Log Service

@Observable
final class ConsoleLog {
    static let shared = ConsoleLog()

    struct Entry: Identifiable {
        let id = UUID()
        let date: Date
        let level: Level
        let message: String

        var timestamp: String {
            Self.formatter.string(from: date)
        }

        private static let formatter: DateFormatter = {
            let f = DateFormatter()
            f.dateFormat = "HH:mm:ss.SSS"
            return f
        }()

        enum Level {
            case info, warning, error, debug

            var icon: String {
                switch self {
                case .info:    return "info.circle.fill"
                case .warning: return "exclamationmark.triangle.fill"
                case .error:   return "xmark.circle.fill"
                case .debug:   return "ant.fill"
                }
            }

            var color: Color {
                switch self {
                case .info:    return .blue
                case .warning: return .orange
                case .error:   return .red
                case .debug:   return .gray
                }
            }

            var textColor: Color {
                switch self {
                case .info:    return .white.opacity(0.9)
                case .warning: return .orange.opacity(0.9)
                case .error:   return .red.opacity(0.9)
                case .debug:   return .white.opacity(0.5)
                }
            }
        }
    }

    private(set) var entries: [Entry] = []
    private let maxEntries = 500

    func log(_ message: String, level: Entry.Level = .info) {
        let entry = Entry(date: Date(), level: level, message: message)
        DispatchQueue.main.async {
            self.entries.append(entry)
            if self.entries.count > self.maxEntries {
                self.entries.removeFirst(self.entries.count - self.maxEntries)
            }
        }
    }

    func info(_ message: String)    { log(message, level: .info) }
    func warning(_ message: String) { log(message, level: .warning) }
    func error(_ message: String)   { log(message, level: .error) }
    func debug(_ message: String)   { log(message, level: .debug) }

    func clear() {
        entries.removeAll()
    }
}
