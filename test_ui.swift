import SwiftUI

public struct HFModelResult: Identifiable {
    public let id: String
    public let repoName: String
    public let isMlxCommunity: Bool
    public let isMoE: Bool
    public let paramSizeHint: String?
    public let storageDisplay: String?
    public let downloadsDisplay: String
    public let likesDisplay: String
    public let formatDisplay: String
}

private struct HFModelRow: View {
    let model: HFModelResult
    let onSelect: (String) -> Void

    var body: some View {
        Button {
            onSelect(model.id)
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.repoName)
                        .font(.system(.subheadline, design: .default, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    Text(model.id)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        if model.isMlxCommunity {
                            badge("mlx-community", color: .blue)
                        }
                        badge(model.formatDisplay, color: model.formatDisplay == "GGUF" ? .indigo : .mint)
                        if model.isMoE {
                            badge("MoE", color: .purple)
                        }
                        if let size = model.paramSizeHint {
                            badge(size, color: .orange)
                        }
                        if let storage = model.storageDisplay {
                            badge(storage, color: .gray)
                        }
                    }
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 3) {
                    if !model.downloadsDisplay.isEmpty {
                        Text(model.downloadsDisplay)
                    }
                    if !model.likesDisplay.isEmpty {
                        Text(model.likesDisplay)
                    }
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func badge(_ label: String, color: Color) -> some View {
        Text(label)
            .font(.system(size: 9, weight: .bold))
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

let m = HFModelResult(id: "mlx-community/test", repoName: "test", isMlxCommunity: true, isMoE: false, paramSizeHint: "7B", storageDisplay: "1.2 GB", downloadsDisplay: "1K", likesDisplay: "50", formatDisplay: "MLX")
print("Compiles fine!")
