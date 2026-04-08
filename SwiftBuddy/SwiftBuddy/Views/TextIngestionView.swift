import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct TextIngestionView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var engine: InferenceEngine
    @StateObject private var extractionService = ExtractionService.shared
    
    @State private var textToMine: String = ""
    @State private var targetWing: String = "Einstein"
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Label("Memory Miner", systemImage: "hammer.fill")
                        .font(.title2.bold())
                        .padding(.bottom, 10)
                    
                    Text("Extract deep offline memory vectors into your custom Palace architectures instantly.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Target Wing Persona (e.g., Einstein, Root_Logic):")
                            .font(.caption.bold())
                        TextField("Target Wing", text: $targetWing)
                            .textFieldStyle(.roundedBorder)
                        
                        Text("Unfiltered Raw Text / Context / Source:")
                            .font(.caption.bold())
                            .padding(.top, 10)
                        TextField("Paste raw text or context...", text: $textToMine, axis: .vertical)
                            .lineLimit(8...16)
                            .textFieldStyle(.roundedBorder)
                        
                        Button(action: {
                            Task {
                                await extractionService.mine(textBlock: textToMine, wing: targetWing, engine: engine)
                                textToMine = ""
                            }
                        }) {
                            Text(extractionService.isMining ? "Mining..." : "Extract to Palace")
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(extractionService.isMining || textToMine.isEmpty)
                        .padding(.top, 10)
                        
                        if !extractionService.lastLog.isEmpty {
                            Text(extractionService.lastLog)
                                .font(.caption2.monospaced())
                                .foregroundColor(.secondary)
                                .padding(.top, 5)
                        }
                    }
                    .padding(24)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
                    )
                }
                .padding(40)
            }
            
            #if os(macOS)
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 20))
                    .foregroundStyle(.secondary.opacity(0.8))
            }
            .buttonStyle(.plain)
            .padding(20)
            #endif
        }
        .frame(minWidth: 500, minHeight: 500)
        .background(SwiftBuddyTheme.background.ignoresSafeArea())
    }
}
