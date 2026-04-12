// ModelCatalog.swift — Device-aware model recommendations for SwiftLM Chat
import Foundation

/// A curated model entry with memory requirements and metadata.
public struct ModelEntry: Identifiable, Sendable {
    public let id: String          // HuggingFace model ID
    public let displayName: String
    public let parameterSize: String  // e.g. "3B", "7B"
    public let quantization: String   // e.g. "4-bit"
    public let ramRequiredGB: Double  // Conservative RAM required
    public let ramRecommendedGB: Double // Ideal RAM for good performance
    public let isMoE: Bool
    public let supportsVision: Bool
    public var badge: String?        // e.g. "⚡ Fast", "🧠 Smart"

    public init(
        id: String,
        displayName: String,
        parameterSize: String,
        quantization: String,
        ramRequiredGB: Double,
        ramRecommendedGB: Double,
        isMoE: Bool = false,
        supportsVision: Bool = false,
        badge: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.parameterSize = parameterSize
        self.quantization = quantization
        self.ramRequiredGB = ramRequiredGB
        self.ramRecommendedGB = ramRecommendedGB
        self.isMoE = isMoE
        self.supportsVision = supportsVision
        self.badge = badge
    }
}

/// Device memory profile used for model recommendation.
public struct DeviceProfile: Sendable {
    public let physicalRAMGB: Double
    public let isAppleSilicon: Bool

    public static var current: DeviceProfile {
        let ram = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        return DeviceProfile(physicalRAMGB: ram, isAppleSilicon: true)
    }
}

/// Curated catalog of MLX-compatible models with device-aware recommendations.
public enum ModelCatalog {

    /// All available models, ordered from smallest to largest.
    public static let all: [ModelEntry] = [
        ModelEntry(
            id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            displayName: "Qwen 2.5 0.5B",
            parameterSize: "0.5B",
            quantization: "4-bit",
            ramRequiredGB: 0.5,
            ramRecommendedGB: 1.0,
            badge: "⚡ Tiny"
        ),
        ModelEntry(
            id: "mlx-community/Phi-3.5-mini-instruct-4bit",
            displayName: "Phi-3.5 Mini",
            parameterSize: "3.8B",
            quantization: "4-bit",
            ramRequiredGB: 2.1,
            ramRecommendedGB: 3.0,
            badge: "⚡ Fast"
        ),
        ModelEntry(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            displayName: "Llama 3.2 3B",
            parameterSize: "3B",
            quantization: "4-bit",
            ramRequiredGB: 1.8,
            ramRecommendedGB: 2.5,
            badge: "🦙 Popular"
        ),
        ModelEntry(
            id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            displayName: "Qwen 2.5 7B",
            parameterSize: "7B",
            quantization: "4-bit",
            ramRequiredGB: 4.2,
            ramRecommendedGB: 6.0,
            badge: "🧠 Smart"
        ),
        ModelEntry(
            id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            displayName: "Mistral 7B",
            parameterSize: "7B",
            quantization: "4-bit",
            ramRequiredGB: 4.1,
            ramRecommendedGB: 6.0
        ),
        ModelEntry(
            id: "mlx-community/Qwen2.5-14B-Instruct-4bit",
            displayName: "Qwen 2.5 14B",
            parameterSize: "14B",
            quantization: "4-bit",
            ramRequiredGB: 8.5,
            ramRecommendedGB: 12.0,
            badge: "🧠 Powerful"
        ),
        ModelEntry(
            id: "mlx-community/Qwen2.5-32B-Instruct-4bit",
            displayName: "Qwen 2.5 32B",
            parameterSize: "32B",
            quantization: "4-bit",
            ramRequiredGB: 19.0,
            ramRecommendedGB: 24.0,
            badge: "🔬 Expert"
        ),
        // ── Qwen3 dense series ───────────────────────────────────────────────
        // Naming: mlx-community/Qwen3-{size}-4bit (no -Instruct suffix)
        ModelEntry(
            id: "mlx-community/Qwen3-0.6B-4bit",
            displayName: "Qwen 3 0.6B",
            parameterSize: "0.6B",
            quantization: "4-bit",
            ramRequiredGB: 0.5,
            ramRecommendedGB: 1.0,
            badge: "⚡ Tiny"
        ),
        ModelEntry(
            id: "mlx-community/Qwen3-1.7B-4bit",
            displayName: "Qwen 3 1.7B",
            parameterSize: "1.7B",
            quantization: "4-bit",
            ramRequiredGB: 1.1,
            ramRecommendedGB: 2.0,
            badge: "⚡ Fast"
        ),
        ModelEntry(
            id: "mlx-community/Qwen3-4B-4bit",
            displayName: "Qwen 3 4B",
            parameterSize: "4B",
            quantization: "4-bit",
            ramRequiredGB: 2.4,
            ramRecommendedGB: 4.0,
            badge: "🧠 Smart"
        ),
        ModelEntry(
            id: "mlx-community/Qwen3-8B-4bit",
            displayName: "Qwen 3 8B",
            parameterSize: "8B",
            quantization: "4-bit",
            ramRequiredGB: 4.9,
            ramRecommendedGB: 6.0,
            badge: "🧠 Powerful"
        ),
        ModelEntry(
            id: "mlx-community/Qwen3-14B-4bit",
            displayName: "Qwen 3 14B",
            parameterSize: "14B",
            quantization: "4-bit",
            ramRequiredGB: 8.5,
            ramRecommendedGB: 12.0,
            badge: "🔬 Expert"
        ),
        ModelEntry(
            id: "mlx-community/Qwen3-32B-4bit",
            displayName: "Qwen 3 32B",
            parameterSize: "32B",
            quantization: "4-bit",
            ramRequiredGB: 19.0,
            ramRecommendedGB: 24.0,
            badge: "💎 Flagship"
        ),
        // ── MoE models: ramRequiredGB = peak-resident (active experts only via mmap streaming)
        // File sizes are much larger but only active expert pages are in RAM at inference time.
        // These run via ExpertStreamingConfig on iPad Pro M4 (16GB+) and macOS.
        ModelEntry(
            id: "mlx-community/Qwen3-30B-A3B-4bit",
            displayName: "Qwen 3 30B MoE",
            parameterSize: "30B (active 3B)",
            quantization: "4-bit",
            ramRequiredGB: 4.5,
            ramRecommendedGB: 8.0,
            isMoE: true,
            badge: "⚡ MoE Fast"
        ),
        // Confirmed by user — tested on macOS with SSD streaming
        ModelEntry(
            id: "mlx-community/Qwen3.5-35B-A3B-4bit",
            displayName: "Qwen 3.5 35B MoE",
            parameterSize: "35B (active 3B)",
            quantization: "4-bit",
            ramRequiredGB: 5.5,
            ramRecommendedGB: 10.0,
            isMoE: true,
            badge: "⚡ MoE Turbo"
        ),
    ]

    /// Returns models that will fit on the given device profile.
    /// - Parameter device: The device to filter for
    /// - Parameter safetyMargin: Fraction of RAM to keep free for OS (default 25%)
    public static func recommended(
        for device: DeviceProfile = .current,
        safetyMargin: Double = 0.25
    ) -> [ModelEntry] {
        let usableRAM = device.physicalRAMGB * (1.0 - safetyMargin)
        return all.filter { $0.ramRequiredGB <= usableRAM }
    }

    /// Returns the single best default model for the device.
    public static func defaultModel(for device: DeviceProfile = .current) -> ModelEntry {
        let candidates = recommended(for: device)
        // Pick the largest model that fits comfortably
        return candidates.last ?? all.first!
    }

    /// Memory fit status for a model on a given device.
    public enum FitStatus {
        case fits          // Comfortably fits in RAM
        case tight         // Fits but will be slow (>80% RAM)
        case requiresFlash // Requires flash streaming (MoE > RAM)
        case tooLarge      // Exceeds device capability
    }

    public static func fitStatus(
        for model: ModelEntry,
        on device: DeviceProfile = .current
    ) -> FitStatus {
        let ram = device.physicalRAMGB
        // On 6 GB devices: 75% = 4.5 GB comfortable, 90% = 5.4 GB tight.
        // MoE models with expert streaming can run at up to 3x RAM via NVMe paging.
        if model.ramRequiredGB <= ram * 0.75 { return .fits }
        if model.ramRequiredGB <= ram * 0.90 { return .tight }
        if model.isMoE && model.ramRequiredGB <= ram * 3.0 { return .requiresFlash }
        return .tooLarge
    }
}

