// Theme.swift — SwiftBuddy Chat design system
// Single source of truth for colors, gradients, radii, and animations.
import SwiftUI

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Color Tokens
// ─────────────────────────────────────────────────────────────────────────────

public enum SwiftBuddyTheme {

    // ── Background layers ─────────────────────────────────────────────────────
    #if os(macOS)
    public static let background = Color(nsColor: .windowBackgroundColor)
    public static let surface = Color(nsColor: .controlBackgroundColor)
    public static let surfaceElevated = Color(nsColor: .gridColor)
    #else
    /// Deep navy-black canvas — the app's primary background.
    public static let background = Color(hue: 0.67, saturation: 0.20, brightness: 0.07)
    /// Slightly elevated surface for cards and panels.
    public static let surface = Color(hue: 0.67, saturation: 0.18, brightness: 0.12)
    /// Second elevation — dialogs, popovers.
    public static let surfaceElevated = Color(hue: 0.67, saturation: 0.15, brightness: 0.17)
    #endif
    /// Subtle divider / separator.
    public static let divider = Color.white.opacity(0.08)

    // ── Brand accents ─────────────────────────────────────────────────────────
    /// Primary accent — vivid indigo.
    public static let accent = Color(hue: 0.70, saturation: 0.90, brightness: 0.95)
    /// Secondary accent — electric violet.
    public static let accentSecondary = Color(hue: 0.76, saturation: 0.85, brightness: 0.95)
    /// Cyan highlight — used in avatars and MoE badges.
    public static let cyan = Color(hue: 0.54, saturation: 0.80, brightness: 0.95)

    // ── Semantic ──────────────────────────────────────────────────────────────
    public static let success = Color(hue: 0.40, saturation: 0.70, brightness: 0.80)
    public static let warning = Color(hue: 0.10, saturation: 0.85, brightness: 0.95)
    public static let error   = Color(hue: 0.02, saturation: 0.80, brightness: 0.90)

    // ── Text ──────────────────────────────────────────────────────────────────
    #if os(macOS)
    public static let textPrimary   = Color(nsColor: .labelColor)
    public static let textSecondary = Color(nsColor: .secondaryLabelColor)
    public static let textTertiary  = Color(nsColor: .tertiaryLabelColor)
    #else
    public static let textPrimary   = Color.white
    public static let textSecondary = Color.white.opacity(0.60)
    public static let textTertiary  = Color.white.opacity(0.35)
    #endif

    // ─────────────────────────────────────────────────────────────────────────
    // MARK: — Gradients
    // ─────────────────────────────────────────────────────────────────────────

    /// User message bubble fill.
    public static let userBubbleGradient = LinearGradient(
        colors: [
            Color(hue: 0.70, saturation: 0.80, brightness: 0.90),
            Color(hue: 0.76, saturation: 0.82, brightness: 0.88)
        ],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    /// AI avatar ring gradient.
    public static let avatarGradient = LinearGradient(
        colors: [
            Color(hue: 0.70, saturation: 0.85, brightness: 0.95),
            Color(hue: 0.54, saturation: 0.80, brightness: 0.95)
        ],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    /// Hero card gradient (active model card).
    public static let heroGradient = LinearGradient(
        colors: [
            Color(hue: 0.70, saturation: 0.75, brightness: 0.30),
            Color(hue: 0.76, saturation: 0.80, brightness: 0.22)
        ],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    /// Thinking panel tint.
    public static let thinkingGradient = LinearGradient(
        colors: [
            Color(hue: 0.76, saturation: 0.40, brightness: 0.18),
            Color(hue: 0.72, saturation: 0.35, brightness: 0.16)
        ],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    // ─────────────────────────────────────────────────────────────────────────
    // MARK: — Corner Radii
    // ─────────────────────────────────────────────────────────────────────────

    public static let radiusSmall:  CGFloat = 8
    public static let radiusMedium: CGFloat = 14
    public static let radiusLarge:  CGFloat = 20
    public static let radiusXL:     CGFloat = 28

    // ─────────────────────────────────────────────────────────────────────────
    // MARK: — Animation
    // ─────────────────────────────────────────────────────────────────────────

    public static let spring     = Animation.spring(response: 0.4, dampingFraction: 0.75)
    public static let quickSpring = Animation.spring(response: 0.25, dampingFraction: 0.80)

    // ─────────────────────────────────────────────────────────────────────────
    // MARK: — Shadows
    // ─────────────────────────────────────────────────────────────────────────

    public struct ShadowStyle {
        let color: Color
        let radius: CGFloat
        let x: CGFloat
        let y: CGFloat
    }

    public static let shadowCard = ShadowStyle(
        color: Color.black.opacity(0.35), radius: 12, x: 0, y: 6
    )
    public static let shadowBubble = ShadowStyle(
        color: Color.black.opacity(0.20), radius: 4, x: 0, y: 2
    )
    public static let shadowGlow = ShadowStyle(
        color: Color(hue: 0.70, saturation: 0.80, brightness: 0.90).opacity(0.45),
        radius: 16, x: 0, y: 0
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — View Modifiers
// ─────────────────────────────────────────────────────────────────────────────

extension View {
    /// Apply a glowing indigo ring — used on focused input and hero cards.
    func glowRing(color: Color = SwiftBuddyTheme.accent, radius: CGFloat = 8, active: Bool = true) -> some View {
        self.shadow(color: active ? color.opacity(0.55) : .clear, radius: radius)
    }

    /// Glassmorphic card surface.
    func glassCard(cornerRadius: CGFloat = SwiftBuddyTheme.radiusMedium) -> some View {
        self
            .background(.ultraThinMaterial)
            .background(SwiftBuddyTheme.surface.opacity(0.65))
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .strokeBorder(Color.white.opacity(0.09), lineWidth: 1)
            )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Reusable badge helpers
// ─────────────────────────────────────────────────────────────────────────────

struct ThemedBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.system(size: 9, weight: .bold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Animated generating indicator (three dots)
// ─────────────────────────────────────────────────────────────────────────────

struct GeneratingDots: View {
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .frame(width: 5, height: 5)
                    .foregroundStyle(SwiftBuddyTheme.accent)
                    .scaleEffect(phase == i ? 1.5 : 0.8)
                    .opacity(phase == i ? 1.0 : 0.45)
                    .animation(
                        .easeInOut(duration: 0.45).repeatForever().delay(Double(i) * 0.18),
                        value: phase
                    )
            }
        }
        .onAppear { withAnimation { phase = 1 } }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: — Pulsing avatar ring
// ─────────────────────────────────────────────────────────────────────────────

struct AvatarView: View {
    var isGenerating: Bool = false
    var size: CGFloat = 30

    @State private var pulse: Bool = false

    var body: some View {
        ZStack {
            // Outer glow ring when generating
            if isGenerating {
                Circle()
                    .stroke(SwiftBuddyTheme.accent.opacity(pulse ? 0.55 : 0.15), lineWidth: 2)
                    .frame(width: size + 8, height: size + 8)
                    .scaleEffect(pulse ? 1.12 : 1.0)
                    .animation(
                        .easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                        value: pulse
                    )
            }

            // Avatar circle
            Circle()
                .fill(SwiftBuddyTheme.avatarGradient)
                .frame(width: size, height: size)
                .overlay(
                    Image(systemName: "bolt.fill")
                        .font(.system(size: size * 0.40, weight: .semibold))
                        .foregroundStyle(.white)
                )
        }
        .onAppear { if isGenerating { pulse = true } }
        .onChange(of: isGenerating) { _, gen in
            pulse = gen
        }
    }
}
