// SettingsView.swift — Inference + appearance settings (iOS tab or macOS sheet)
import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct SettingsView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var appearance: AppearanceStore
    @Environment(\.dismiss) private var dismiss

    /// When true, the view is embedded as a tab (no Done button on iOS)
    var isTab: Bool = false

    // iOS-specific: performance mode toggle (read from UserDefaults)
    @AppStorage("swiftlm.performanceMode") private var performanceMode: Bool = false

    private var ramGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
    }

    var body: some View {
        ZStack {
            SwiftBuddyTheme.background.ignoresSafeArea()

            Form {
                // ── Generation ────────────────────────────────────────────────
                Section {
                    temperatureRow
                    maxTokensRow
                    topPRow
                    repetitionPenaltyRow
                } header: {
                    sectionLabel("Generation", icon: "slider.horizontal.3")
                }

                // ── Advanced ──────────────────────────────────────────────────
                Section {
                    thinkingToggle
                } header: {
                    sectionLabel("Advanced", icon: "gearshape.2")
                }

                // ── Appearance ────────────────────────────────────────────────
                Section {
                    appearancePicker
                } header: {
                    sectionLabel("Appearance", icon: "paintpalette")
                }

                // ── System Prompt ─────────────────────────────────────────────
                Section {
                    TextEditor(text: $viewModel.systemPrompt)
                        .frame(minHeight: 88)
                        .font(.callout)
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                        .scrollContentBackground(.hidden)
                        .background(Color.clear)
                } header: {
                    sectionLabel("System Prompt", icon: "text.bubble")
                } footer: {
                    Text("Injected as the system message before every conversation.")
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }

                // ── Performance (iOS-only) ─────────────────────────────────────
                #if os(iOS)
                Section {
                    performanceModeRow
                    autoOffloadRow
                } header: {
                    sectionLabel("Performance", icon: "cpu")
                } footer: {
                    Text("Performance Mode loosens the RAM budget from 40% to 55%, allowing larger models on your \(String(format: "%.0f GB", ramGB)) device.")
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }
                #endif

                // ── Reset ─────────────────────────────────────────────────────
                Section {
                    Button(role: .destructive) {
                        viewModel.config = .default
                        viewModel.systemPrompt = ""
                    } label: {
                        HStack {
                            Spacer()
                            Label("Reset to Defaults", systemImage: "arrow.counterclockwise")
                                .foregroundStyle(SwiftBuddyTheme.error)
                            Spacer()
                        }
                    }
                }

                // ── About ─────────────────────────────────────────────────────
                Section {
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
                } header: {
                    sectionLabel("About", icon: "info.circle")
                }
            }
            .scrollContentBackground(.hidden)
        }
        .navigationTitle("Settings")
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
        #if os(macOS)
        .frame(width: 440, height: 640)
        #endif
    }

    // MARK: — Row Helpers

    private var temperatureRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Temperature", systemImage: "thermometer.medium")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
                Text(String(format: "%.2f", viewModel.config.temperature))
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .monospacedDigit()
                    .font(.callout)
            }
            Slider(value: Binding(
                get: { Double(viewModel.config.temperature) },
                set: { viewModel.config.temperature = Float($0) }
            ), in: 0...2, step: 0.05)
            .tint(SwiftBuddyTheme.warning)
            Text("Higher = more creative, lower = more focused")
                .font(.caption2)
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
        }
        .padding(.vertical, 2)
    }

    private var maxTokensRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Max Tokens", systemImage: "text.word.spacing")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
                Text("\(viewModel.config.maxTokens)")
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .monospacedDigit()
                    .font(.callout)
            }
            Slider(value: Binding(
                get: { Double(viewModel.config.maxTokens) },
                set: { viewModel.config.maxTokens = Int($0) }
            ), in: 128...8192, step: 128)
            .tint(SwiftBuddyTheme.accent)
        }
        .padding(.vertical, 2)
    }

    private var topPRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Top P", systemImage: "chart.bar.xaxis")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
                Text(String(format: "%.2f", viewModel.config.topP))
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .monospacedDigit()
                    .font(.callout)
            }
            Slider(value: Binding(
                get: { Double(viewModel.config.topP) },
                set: { viewModel.config.topP = Float($0) }
            ), in: 0...1, step: 0.05)
            .tint(SwiftBuddyTheme.accentSecondary)
        }
        .padding(.vertical, 2)
    }

    private var repetitionPenaltyRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label("Repetition Penalty", systemImage: "repeat.circle")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
                Text(String(format: "%.2f", viewModel.config.repetitionPenalty))
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .monospacedDigit()
                    .font(.callout)
            }
            Slider(value: Binding(
                get: { Double(viewModel.config.repetitionPenalty) },
                set: { viewModel.config.repetitionPenalty = Float($0) }
            ), in: 1.0...2.0, step: 0.01)
            .tint(SwiftBuddyTheme.success)
            Text("Higher = less repeating, 1.0 = disabled (can cause echoing)")
                .font(.caption2)
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
        }
        .padding(.vertical, 2)
    }

    private var thinkingToggle: some View {
        Toggle(isOn: $viewModel.config.enableThinking) {
            VStack(alignment: .leading, spacing: 2) {
                Label("Thinking Mode", systemImage: "brain.head.profile")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Step-by-step reasoning for Qwen3, DeepSeek-R1, and compatible models")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }
        }
        .tint(SwiftBuddyTheme.accentSecondary)
    }

    private var appearancePicker: some View {
        Picker("Color Scheme", selection: $appearance.preference) {
            HStack {
                Image(systemName: "moon.fill")
                Text("Dark")
            }.tag("dark")

            HStack {
                Image(systemName: "sun.max.fill")
                Text("Light")
            }.tag("light")

            HStack {
                Image(systemName: "circle.lefthalf.filled")
                Text("System")
            }.tag("system")
        }
        .pickerStyle(.segmented)
        .tint(SwiftBuddyTheme.accent)
    }

    #if os(iOS)
    private var performanceModeRow: some View {
        Toggle(isOn: $performanceMode) {
            VStack(alignment: .leading, spacing: 2) {
                Label("Performance Mode", systemImage: "bolt.fill")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Use 55% RAM budget (vs. 40%) — enables more models on 6 GB devices")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }
        }
        .tint(SwiftBuddyTheme.accent)
    }

    @State private var tempEngine: InferenceEngine? = nil

    private var autoOffloadRow: some View {
        // We can't easily get the engine here without prop drilling,
        // so we persist via UserDefaults and InferenceEngine reads it on next launch.
        Toggle(isOn: Binding(
            get: { UserDefaults.standard.bool(forKey: "swiftlm.autoOffload") == false
                    ? true  // default true
                    : UserDefaults.standard.bool(forKey: "swiftlm.autoOffload") },
            set: { UserDefaults.standard.set($0, forKey: "swiftlm.autoOffload") }
        )) {
            VStack(alignment: .leading, spacing: 2) {
                Label("Auto-Unload in Background", systemImage: "iphone.slash")
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Frees GPU memory when the app backgrounds (recommended on iPhone)")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }
        }
        .tint(SwiftBuddyTheme.success)
    }
    #endif

    private func aboutRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
            Spacer()
            Text(value)
                .foregroundStyle(SwiftBuddyTheme.textSecondary)
        }
    }

    private func sectionLabel(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .foregroundStyle(SwiftBuddyTheme.textTertiary)
            .font(.footnote.weight(.semibold))
            .textCase(.uppercase)
    }
}
