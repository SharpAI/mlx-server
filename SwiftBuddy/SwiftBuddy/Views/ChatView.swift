// ChatView.swift — Premium chat interface (iOS + macOS)
import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var engine: InferenceEngine

    // macOS-only sheet control (iOS: these are tabs)
    var showSettings: Binding<Bool>? = nil
    var showModelPicker: Binding<Bool>? = nil

    @State private var inputText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        ZStack {
            // ── Deep canvas background ───────────────────────────────────────
            SwiftBuddyTheme.background.ignoresSafeArea()

            VStack(spacing: 0) {
                // ── Message list ─────────────────────────────────────────────
                messageList

                // ── Engine state banner ──────────────────────────────────────
                engineBanner

                // ── Input bar ────────────────────────────────────────────────
                inputBar
            }
        }
        .navigationTitle("SwiftBuddy Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { iOSToolbar }
        .toolbarBackground(SwiftBuddyTheme.background.opacity(0.90), for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        #else
        .toolbar { macOSToolbar }
        #endif
    }

    // MARK: — Message List

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if viewModel.messages.isEmpty && viewModel.streamingText.isEmpty {
                    emptyStateView
                        .frame(maxWidth: .infinity)
                        .padding(.top, 60)
                } else {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                                .environmentObject(engine)
                        }
                        if !viewModel.streamingText.isEmpty || viewModel.thinkingText != nil {
                            StreamingBubble(
                                text: viewModel.streamingText,
                                thinkingText: viewModel.thinkingText
                            )
                            .id("streaming")
                            .environmentObject(engine)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 14)
                    .padding(.top, 12)
                    .padding(.bottom, 8)
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .onTapGesture { inputFocused = false }
            .onChange(of: viewModel.streamingText) { _, _ in
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("bottom")
                }
            }
        }
    }

    // MARK: — Empty State

    @ViewBuilder
    private var emptyStateView: some View {
        switch engine.state {

        case .downloading(let progress, let speed):
            VStack(spacing: 20) {
                downloadRing(progress: progress)
                VStack(spacing: 6) {
                    Text("Downloading model…")
                        .font(.headline)
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    Text(speed.isEmpty ? "Preparing…" : speed)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                }
                Text("You'll be able to chat once the download completes.")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }

        case .loading:
            VStack(spacing: 16) {
                ZStack {
                    Circle()
                        .stroke(SwiftBuddyTheme.accent.opacity(0.15), lineWidth: 3)
                        .frame(width: 64, height: 64)
                    ProgressView()
                        .controlSize(.large)
                        .tint(SwiftBuddyTheme.accent)
                }
                Text("Loading model into Metal GPU…")
                    .font(.subheadline)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }

        case .idle:
            idleEmptyState

        case .error(let msg):
            VStack(spacing: 14) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(SwiftBuddyTheme.error)
                Text("Load failed")
                    .font(.headline)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }

        case .ready, .generating:
            VStack(spacing: 14) {
                // Brand mark
                brandMark
                Text("Start a conversation")
                    .font(.headline)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Type a message below to begin.")
                    .font(.subheadline)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }
        }
    }

    // Brand mark — animated bolt in gradient ring
    private var brandMark: some View {
        ZStack {
            Circle()
                .fill(SwiftBuddyTheme.heroGradient)
                .frame(width: 80, height: 80)
                .shadow(color: SwiftBuddyTheme.accent.opacity(0.35), radius: 18)

            Image(systemName: "bolt.fill")
                .font(.system(size: 34, weight: .semibold))
                .foregroundStyle(
                    LinearGradient(colors: [.white, SwiftBuddyTheme.cyan],
                                   startPoint: .top, endPoint: .bottom)
                )
        }
    }

    // Idle empty state — brand mark + tagline
    private var idleEmptyState: some View {
        VStack(spacing: 20) {
            brandMark

            VStack(spacing: 6) {
                Text("SwiftBuddy Chat")
                    .font(.title2.weight(.bold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)

                Text("Run any model. Locally. Instantly.")
                    .font(.subheadline)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [SwiftBuddyTheme.accent, SwiftBuddyTheme.cyan],
                            startPoint: .leading, endPoint: .trailing
                        )
                    )
            }

            Text("Go to the **Models** tab to download\na model and start chatting.")
                .font(.caption)
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }

    // Download ring
    private func downloadRing(progress: Double) -> some View {
        ZStack {
            Circle()
                .stroke(SwiftBuddyTheme.accent.opacity(0.15), lineWidth: 6)
            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    SwiftBuddyTheme.avatarGradient,
                    style: StrokeStyle(lineWidth: 6, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 0.3), value: progress)
            Text("\(Int(progress * 100))%")
                .font(.caption.monospacedDigit().weight(.semibold))
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
        }
        .frame(width: 72, height: 72)
    }

    // MARK: — Engine Banner (slim status strip above input)

    @ViewBuilder
    private var engineBanner: some View {
        switch engine.state {
        case .idle:
            bannerRow(icon: "cpu", text: "No model loaded", color: SwiftBuddyTheme.textTertiary)
        case .loading:
            HStack(spacing: 8) {
                ProgressView().controlSize(.mini).tint(SwiftBuddyTheme.accent)
                Text("Loading model…")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(SwiftBuddyTheme.surface.opacity(0.90))
        case .downloading(let p, let speed):
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Downloading…")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    Spacer()
                    Text("\(Int(p * 100))% · \(speed)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                }
                ProgressView(value: p).tint(SwiftBuddyTheme.accent)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(SwiftBuddyTheme.surface.opacity(0.90))
        case .error(let msg):
            bannerRow(icon: "exclamationmark.triangle.fill", text: msg, color: SwiftBuddyTheme.error)
        case .ready, .generating:
            EmptyView()
        }
    }

    private func bannerRow(icon: String, text: String, color: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon).foregroundStyle(color)
            Text(text)
                .font(.caption)
                .foregroundStyle(color)
                .lineLimit(2)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(SwiftBuddyTheme.surface.opacity(0.90))
    }

    // MARK: — Input Bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 10) {
            // Text field with frosted glass pill
            HStack(alignment: .bottom) {
                TextField("Message", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(.body))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .lineLimit(1...8)
                    .focused($inputFocused)
                    .onSubmit {
                        #if os(macOS)
                        sendMessage()
                        #endif
                    }
                    .disabled(!engine.state.canSend)
                    .accentColor(SwiftBuddyTheme.accent)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial)
            .background(SwiftBuddyTheme.surface.opacity(0.70))
            .clipShape(RoundedRectangle(cornerRadius: SwiftBuddyTheme.radiusXL))
            .overlay(
                RoundedRectangle(cornerRadius: SwiftBuddyTheme.radiusXL)
                    .strokeBorder(
                        inputFocused
                            ? SwiftBuddyTheme.accent.opacity(0.55)
                            : Color.white.opacity(0.08),
                        lineWidth: inputFocused ? 1.5 : 1
                    )
                    .animation(SwiftBuddyTheme.quickSpring, value: inputFocused)
            )
            .glowRing(active: inputFocused)

            // Send / Stop button
            if viewModel.isGenerating {
                Button(action: viewModel.stopGeneration) {
                    ZStack {
                        Circle()
                            .fill(SwiftBuddyTheme.error.opacity(0.18))
                            .frame(width: 40, height: 40)
                        Image(systemName: "stop.fill")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(SwiftBuddyTheme.error)
                    }
                }
                .buttonStyle(.plain)
            } else {
                Button(action: sendMessage) {
                    ZStack {
                        Circle()
                            .fill(canSend ? AnyShapeStyle(SwiftBuddyTheme.userBubbleGradient) : AnyShapeStyle(Color.white.opacity(0.08)))
                            .frame(width: 40, height: 40)
                        Image(systemName: "arrow.up")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundStyle(canSend ? .white : SwiftBuddyTheme.textTertiary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: .command)
                .animation(SwiftBuddyTheme.quickSpring, value: canSend)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(SwiftBuddyTheme.background.opacity(0.95))
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespaces).isEmpty && engine.state.canSend
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !viewModel.isGenerating else { return }
        inputText = ""
        Task { await viewModel.send(text) }
    }

    // MARK: — Toolbars

    #if os(iOS)
    @ToolbarContentBuilder
    private var iOSToolbar: some ToolbarContent {
        // Animated status pill (center)
        ToolbarItem(placement: .principal) {
            modelStatusPill
        }
        // Keyboard dismiss
        ToolbarItem(placement: .topBarLeading) {
            if inputFocused {
                Button { inputFocused = false } label: {
                    Image(systemName: "keyboard.chevron.compact.down")
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                }
                .transition(.opacity)
            }
        }
        // New conversation
        ToolbarItem(placement: .topBarTrailing) {
            Button { viewModel.newConversation() } label: {
                Image(systemName: "square.and.pencil")
                    .foregroundStyle(SwiftBuddyTheme.accent)
            }
        }
    }

    private var modelStatusPill: some View {
        HStack(spacing: 5) {
            if case .generating = engine.state {
                GeneratingDots()
            } else {
                Circle()
                    .fill(engine.state.statusColor)
                    .frame(width: 7, height: 7)
            }
            Text(engine.state.shortLabel)
                .font(.caption.weight(.medium))
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial)
        .background(SwiftBuddyTheme.surface.opacity(0.70))
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Color.white.opacity(0.09), lineWidth: 1))
    }
    #endif

    #if os(macOS)
    @ToolbarContentBuilder
    private var macOSToolbar: some ToolbarContent {
        ToolbarItem {
            Button { viewModel.newConversation() } label: {
                Label("New Chat", systemImage: "square.and.pencil")
            }
        }
        ToolbarItem {
            Button { showSettings?.wrappedValue = true } label: {
                Label("Settings", systemImage: "slider.horizontal.3")
            }
        }
    }
    #endif
}

// MARK: — ModelState Extensions

extension ModelState {
    var canSend: Bool {
        if case .ready = self { return true }
        return false
    }

    var statusColor: Color {
        switch self {
        case .idle:                       return SwiftBuddyTheme.textTertiary
        case .loading, .downloading:      return SwiftBuddyTheme.warning
        case .ready:                      return SwiftBuddyTheme.success
        case .generating:                 return SwiftBuddyTheme.accent
        case .error:                      return SwiftBuddyTheme.error
        }
    }

    var shortLabel: String {
        switch self {
        case .idle:                        return "No model"
        case .loading:                     return "Loading…"
        case .downloading(let p, _):       return "\(Int(p * 100))% downloading"
        case .ready(let modelId):          return modelId.components(separatedBy: "/").last ?? modelId
        case .generating:                  return "Generating"
        case .error:                       return "Error"
        }
    }
}
