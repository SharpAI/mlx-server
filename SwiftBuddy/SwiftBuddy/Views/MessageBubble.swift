// MessageBubble.swift — Premium chat message bubbles (iOS + macOS)
import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Static Message Bubble
// ─────────────────────────────────────────────────────────────────────────────

struct MessageBubble: View {
    let message: ChatMessage
    var isRPGMode: Bool = false
    var personaName: String? = nil
    
    @State private var showTimestamp = false
    @State private var thinkingExpanded = false
    @EnvironmentObject private var engine: InferenceEngine

    var isUser: Bool { message.role == .user }

    var body: some View {
        if isRPGMode {
            rpgLayout
        } else {
            standardLayout
        }
    }
    
    private var standardLayout: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isUser { Spacer(minLength: 52) }

            if !isUser {
                AvatarView(
                    isGenerating: false,
                    size: 30
                )
            }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                if isUser {
                    userBubble
                } else {
                    assistantBubble
                }

                if showTimestamp {
                    Text(message.timestamp, style: .time)
                        .font(.caption2)
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .onTapGesture {
                withAnimation(SwiftBuddyTheme.quickSpring) {
                    showTimestamp.toggle()
                }
            }

            if !isUser { Spacer(minLength: 52) }
        }
    }
    
    // MARK: - RPG Layout
    private var rpgLayout: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Nameplate Header
            HStack {
                Text(isUser ? "YOU" : (personaName?.uppercased() ?? "SYSTEM"))
                    .font(.caption.weight(.heavy))
                    .foregroundStyle(isUser ? SwiftBuddyTheme.cyan : .orange)
                    .tracking(1.5)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            
            // Explicit System Cleanups (don't show the injected context matrix to the user!)
            let cleanText = isUser ? message.content.replacingOccurrences(of: "SYSTEM DIRECTIVE & CONTEXT:(.*?)USER PROMPT:\\n", with: "", options: .regularExpression) : message.content
            
            // Body Text
            VStack(alignment: .leading, spacing: 6) {
                if let thinking = message.thinkingContent, !thinking.isEmpty {
                    ThinkingPanel(text: thinking, isExpanded: $thinkingExpanded)
                        .padding(.horizontal, 14)
                }
                
                Text(cleanText)
                    .font(.system(.body, design: .serif))
                    .lineSpacing(4)
                    .textSelection(.enabled)
                    .foregroundStyle(.white.opacity(0.95))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
                    .padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            ZStack {
                Color.black.opacity(0.55)
                LinearGradient(colors: [isUser ? SwiftBuddyTheme.cyan.opacity(0.05) : .orange.opacity(0.05), .clear], startPoint: .leading, endPoint: .trailing)
            }
        )
        .overlay(
            Rectangle()
                .stroke(isUser ? SwiftBuddyTheme.cyan.opacity(0.4) : .orange.opacity(0.4), lineWidth: 1)
        )
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
    }

    // MARK: — User Bubble

    private var userBubble: some View {
        Text(message.content)
            .font(.system(.body, design: .default))
            .textSelection(.enabled)
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(SwiftBuddyTheme.userBubbleGradient)
            .clipShape(UserBubbleShape())
            .shadow(
                color: SwiftBuddyTheme.accent.opacity(0.30),
                radius: 6, x: 0, y: 3
            )
    }

    // MARK: — Assistant Bubble

    private var assistantBubble: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let thinking = message.thinkingContent, !thinking.isEmpty {
                ThinkingPanel(text: thinking, isExpanded: $thinkingExpanded)
            }
            
            if !message.content.isEmpty {
                Text(message.content)
                    .font(.system(.body, design: .default))
                    .textSelection(.enabled)
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.ultraThinMaterial)
                    .background(SwiftBuddyTheme.surface.opacity(0.80))
                    .clipShape(AssistantBubbleShape())
                    .overlay(
                        AssistantBubbleShape()
                            .stroke(Color.white.opacity(0.08), lineWidth: 1)
                    )
                    .shadow(
                        color: SwiftBuddyTheme.shadowBubble.color,
                        radius: SwiftBuddyTheme.shadowBubble.radius,
                        x: SwiftBuddyTheme.shadowBubble.x,
                        y: SwiftBuddyTheme.shadowBubble.y
                    )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Live Streaming Bubble
// ─────────────────────────────────────────────────────────────────────────────

struct StreamingBubble: View {
    let text: String
    let thinkingText: String?
    var isRPGMode: Bool = false
    var personaName: String? = nil

    @EnvironmentObject private var engine: InferenceEngine
    @State private var thinkingExpanded = true

    var body: some View {
        if isRPGMode {
            rpgStreamingLayout
        } else {
            standardStreamingLayout
        }
    }
    
    private var standardStreamingLayout: some View {
        HStack(alignment: .bottom, spacing: 8) {
            AvatarView(isGenerating: true, size: 30)

            VStack(alignment: .leading, spacing: 6) {
                // ── Thinking section ─────────────────────────────────────────
                if let thinking = thinkingText, !thinking.isEmpty {
                    ThinkingPanel(text: thinking, isExpanded: $thinkingExpanded)
                }

                // ── Response text ────────────────────────────────────────────
                if !text.isEmpty {
                    streamingText
                } else if thinkingText == nil || thinkingText?.isEmpty == true {
                    // Show typing indicator only when there's no content at all
                    typingDots
                }
            }

            Spacer(minLength: 52)
        }
    }
    
    private var rpgStreamingLayout: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Nameplate Header
            HStack {
                Text(personaName?.uppercased() ?? "SYSTEM")
                    .font(.caption.weight(.heavy))
                    .foregroundStyle(.orange)
                    .tracking(1.5)
                Spacer()
                GeneratingDots()
                    .scaleEffect(0.8)
                    .opacity(0.7)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            
            // Body Text
            VStack(alignment: .leading, spacing: 6) {
                if let thinking = thinkingText, !thinking.isEmpty {
                    ThinkingPanel(text: thinking, isExpanded: $thinkingExpanded)
                        .padding(.horizontal, 14)
                }
                
                if !text.isEmpty {
                    HStack(alignment: .bottom, spacing: 0) {
                        Text(text)
                            .font(.system(.body, design: .serif))
                            .lineSpacing(4)
                            .textSelection(.enabled)
                            .foregroundStyle(.white.opacity(0.95))
                        BlinkingCursor()
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
                    .padding(.top, 2)
                } else if thinkingText == nil || thinkingText?.isEmpty == true {
                    typingDots
                        .padding(.horizontal, 16)
                        .padding(.bottom, 16)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            ZStack {
                Color.black.opacity(0.55)
                LinearGradient(colors: [.orange.opacity(0.05), .clear], startPoint: .leading, endPoint: .trailing)
            }
        )
        .overlay(
            Rectangle()
                .stroke(.orange.opacity(0.4), lineWidth: 1)
        )
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
    }

    private var streamingText: some View {
        // Inline blinking cursor via attributed string approach
        HStack(alignment: .bottom, spacing: 0) {
            Text(text)
                .font(.system(.body, design: .default))
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .textSelection(.enabled)
            BlinkingCursor()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
        .background(SwiftBuddyTheme.surface.opacity(0.80))
        .clipShape(AssistantBubbleShape())
        .overlay(
            AssistantBubbleShape()
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
        .shadow(
            color: SwiftBuddyTheme.shadowBubble.color,
            radius: SwiftBuddyTheme.shadowBubble.radius,
            x: SwiftBuddyTheme.shadowBubble.x,
            y: SwiftBuddyTheme.shadowBubble.y
        )
    }

    private var typingDots: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                BouncingDot(index: i)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .background(SwiftBuddyTheme.surface.opacity(0.80))
        .clipShape(AssistantBubbleShape())
        .overlay(
            AssistantBubbleShape()
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Thinking Panel
// ─────────────────────────────────────────────────────────────────────────────

private struct ThinkingPanel: View {
    let text: String
    @Binding var isExpanded: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header toggle
            Button {
                withAnimation(SwiftBuddyTheme.spring) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "brain.filled.head.profile")
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.accentSecondary)
                    Text("Thinking…")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(SwiftBuddyTheme.accentSecondary)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(SwiftBuddyTheme.textTertiary)
                        .rotationEffect(.degrees(isExpanded ? 0 : -90))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
            }
            .buttonStyle(.plain)

            // Expandable content
            if isExpanded {
                ScrollView {
                    Text(text)
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
                .frame(maxHeight: 160)
            }
        }
        .background(SwiftBuddyTheme.thinkingGradient)
        .clipShape(RoundedRectangle(cornerRadius: SwiftBuddyTheme.radiusMedium))
        .overlay(
            RoundedRectangle(cornerRadius: SwiftBuddyTheme.radiusMedium)
                .strokeBorder(SwiftBuddyTheme.accentSecondary.opacity(0.20), lineWidth: 1)
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Blinking Cursor
// ─────────────────────────────────────────────────────────────────────────────

private struct BlinkingCursor: View {
    @State private var visible = true

    var body: some View {
        RoundedRectangle(cornerRadius: 1.5)
            .frame(width: 2.5, height: 17)
            .foregroundStyle(SwiftBuddyTheme.accent)
            .opacity(visible ? 1 : 0)
            .animation(
                .easeInOut(duration: 0.52).repeatForever(autoreverses: true),
                value: visible
            )
            .onAppear { visible = false }
            .padding(.leading, 1)
            .padding(.bottom, 1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Bouncing Dots (typing indicator)
// ─────────────────────────────────────────────────────────────────────────────

private struct BouncingDot: View {
    let index: Int
    @State private var bouncing = false

    var body: some View {
        Circle()
            .frame(width: 7, height: 7)
            .foregroundStyle(SwiftBuddyTheme.textSecondary)
            .offset(y: bouncing ? -5 : 0)
            .animation(
                .easeInOut(duration: 0.45)
                    .repeatForever(autoreverses: true)
                    .delay(Double(index) * 0.14),
                value: bouncing
            )
            .onAppear { bouncing = true }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Bubble Shapes
// ─────────────────────────────────────────────────────────────────────────────

/// User bubble: rounded top-left + bottom, small top-right corner.
struct UserBubbleShape: Shape {
    let r: CGFloat = 18
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.minX + r, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX - 4, y: rect.minY))
        p.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + 4),
                       control: CGPoint(x: rect.maxX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - r))
        p.addQuadCurve(to: CGPoint(x: rect.maxX - r, y: rect.maxY),
                       control: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX + r, y: rect.maxY))
        p.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - r),
                       control: CGPoint(x: rect.minX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.minY + r))
        p.addQuadCurve(to: CGPoint(x: rect.minX + r, y: rect.minY),
                       control: CGPoint(x: rect.minX, y: rect.minY))
        p.closeSubpath()
        return p
    }
}

/// Assistant bubble: small top-left (tail side), large radii elsewhere.
struct AssistantBubbleShape: Shape {
    let r: CGFloat = 18
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.minX + 4, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX - r, y: rect.minY))
        p.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + r),
                       control: CGPoint(x: rect.maxX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - r))
        p.addQuadCurve(to: CGPoint(x: rect.maxX - r, y: rect.maxY),
                       control: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX + r, y: rect.maxY))
        p.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - r),
                       control: CGPoint(x: rect.minX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.minY + 4))
        p.addQuadCurve(to: CGPoint(x: rect.minX + 4, y: rect.minY),
                       control: CGPoint(x: rect.minX, y: rect.minY))
        p.closeSubpath()
        return p
    }
}

// Keep BubbleShape for any legacy reference
typealias BubbleShape = UserBubbleShape
