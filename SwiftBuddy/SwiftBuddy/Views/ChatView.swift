// ChatView.swift — Premium chat interface (iOS + macOS)
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var engine: InferenceEngine
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]

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
        .navigationTitle(viewModel.currentWing != nil ? "Chatting with \(viewModel.currentWing!)" : "SwiftBuddy Chat")
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
                            MessageBubble(
                                message: message,
                                isRPGMode: viewModel.currentWing != nil,
                                personaName: viewModel.currentWing
                            )
                            .id(message.id)
                            .environmentObject(engine)
                        }
                        if !viewModel.streamingText.isEmpty || viewModel.thinkingText != nil {
                            StreamingBubble(
                                text: viewModel.streamingText,
                                thinkingText: viewModel.thinkingText,
                                isRPGMode: viewModel.currentWing != nil,
                                personaName: viewModel.currentWing
                            )
                            .id("generating")
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
            DownloadAnimationView(progress: progress, speed: speed)

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
                Text(viewModel.currentWing != nil ? "System Linked to \(viewModel.currentWing!)" : "SwiftBuddy Chat")
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
                TextField(viewModel.currentWing != nil ? "Message \(viewModel.currentWing!)..." : "Message", text: $inputText, axis: .vertical)
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
        // Persona map selector
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                Button("No Persona") { viewModel.currentWing = nil }
                Divider()
                ForEach(wings) { wing in
                    Button(wing.name) { viewModel.currentWing = wing.name }
                }
            } label: {
                Image(systemName: viewModel.currentWing == nil ? "brain" : "brain.head.profile")
                    .foregroundStyle(viewModel.currentWing == nil ? SwiftBuddyTheme.textSecondary : .orange)
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
            Menu {
                Button("No Persona") { viewModel.currentWing = nil }
                Divider()
                ForEach(wings) { wing in
                    Button(wing.name) { viewModel.currentWing = wing.name }
                }
            } label: {
                Image(systemName: viewModel.currentWing == nil ? "brain" : "brain.head.profile")
                    .foregroundStyle(viewModel.currentWing == nil ? SwiftBuddyTheme.textSecondary : .orange)
            }
        }
        
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
import SwiftUI

struct DownloadAnimationView: View {
    let progress: Double
    let speed: String
    
    @State private var isAnimating = false
    @State private var textFlicker = false
    
    var body: some View {
        VStack(spacing: 30) {
            ZStack {
                // Background Ambient Glow
                Circle()
                    .fill(SwiftBuddyTheme.accent.opacity(0.1))
                    .frame(width: 140, height: 140)
                    .blur(radius: isAnimating ? 20 : 10)
                
                // Outer Runic Circle (Dashed, Rotating Clockwise)
                Circle()
                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [4, 8, 2, 8]))
                    .foregroundStyle(SwiftBuddyTheme.accent.opacity(0.4))
                    .frame(width: 130, height: 130)
                    .rotationEffect(.degrees(isAnimating ? 360 : 0))
                    .animation(
                        .linear(duration: 20).repeatForever(autoreverses: false),
                        value: isAnimating
                    )
                
                // Middle Ritual Circle (Thick Dashed, Rotating Counter-Clockwise)
                Circle()
                    .stroke(style: StrokeStyle(lineWidth: 2, dash: [10, 5, 2, 5]))
                    .foregroundStyle(SwiftBuddyTheme.accent.opacity(0.6))
                    .frame(width: 100, height: 100)
                    .rotationEffect(.degrees(isAnimating ? -360 : 0))
                    .animation(
                        .linear(duration: 15).repeatForever(autoreverses: false),
                        value: isAnimating
                    )
                
                // Dynamic Completion Progress Arc (Liquid Arc filling up)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        SwiftBuddyTheme.avatarGradient,
                        style: StrokeStyle(lineWidth: 4, lineCap: .round)
                    )
                    .frame(width: 115, height: 115)
                    .rotationEffect(.degrees(-90))
                    .animation(.spring(response: 0.5, dampingFraction: 0.8), value: progress)
                    .shadow(color: SwiftBuddyTheme.accent, radius: progress > 0 ? 5 : 0)
                
                // Core "Persona Soul" Crystal
                Image(systemName: "diamond.inset.filled")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 36, height: 36)
                    .foregroundStyle(SwiftBuddyTheme.cyan)
                    .symbolEffect(.pulse, options: .repeating)
                    .shadow(color: SwiftBuddyTheme.cyan, radius: isAnimating ? 15 : 5)
                    .scaleEffect(isAnimating ? 1.1 : 0.9)
                    .animation(
                        .easeInOut(duration: 1.5).repeatForever(autoreverses: true),
                        value: isAnimating
                    )
            }
            .frame(width: 150, height: 150)
            .padding(.top, 20)
            
            // Decrypting Text Area
            VStack(spacing: 8) {
                Text("SUMMONING PERSONA")
                    .font(.system(.subheadline, design: .monospaced, weight: .bold))
                    .tracking(4)
                    .foregroundStyle(SwiftBuddyTheme.cyan)
                    .shadow(color: SwiftBuddyTheme.cyan.opacity(0.5), radius: 2)
                    .opacity(textFlicker ? 0.8 : 1.0)
                    .animation(.randomFlicker, value: textFlicker)
                
                HStack(alignment: .lastTextBaseline, spacing: 4) {
                    Text("\(Int(progress * 100))")
                        .font(.system(size: 32, design: .monospaced))
                        .fontWeight(.heavy)
                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    Text("%")
                        .font(.system(size: 20, design: .monospaced))
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                }
                
                Text(speed.isEmpty ? "Casting initial logic runes..." : speed)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(SwiftBuddyTheme.accent)
                    .opacity(0.8)
            }
        }
        .onAppear {
            isAnimating = true
            textFlicker = true
        }
    }
}

// Helper for "cryptographic" flickering effect on the Summon banner
extension Animation {
    static var randomFlicker: Animation {
        .easeInOut(duration: 0.1).repeatForever(autoreverses: true).delay(Double.random(in: 0...0.5))
    }
}
