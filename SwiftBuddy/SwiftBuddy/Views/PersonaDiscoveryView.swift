import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct PersonaDiscoveryView: View {
    @ObservedObject var registry: RegistryService
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var engine: InferenceEngine
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]
    
    var body: some View {
        NavigationStack {
            ZStack {
                SwiftBuddyTheme.background.ignoresSafeArea()
                
                List {
                    Section {
                        Button {
                            Task { await registry.fetchAvailablePersonas() }
                        } label: {
                            HStack {
                                Label("Refresh Cloud Directory", systemImage: "arrow.triangle.2.circlepath")
                                Spacer()
                                if registry.isSyncing {
                                    ProgressView().controlSize(.small)
                                }
                            }
                        }
                        .disabled(registry.isSyncing)
                    } footer: {
                        Text("Connects to the cloud registry to find new AI personas and memory templates.")
                    }
                    
                    if !registry.availablePersonas.isEmpty {
                        Section("Available Personas") {
                            ForEach(registry.availablePersonas, id: \.self) { personaName in
                                let friendlyName = personaName.replacingOccurrences(of: "_", with: " ")
                                let isDownloaded = wings.contains(where: { $0.name == friendlyName })
                                
                                HStack {
                                    Label(friendlyName, systemImage: "person.crop.circle")
                                        .foregroundStyle(SwiftBuddyTheme.textPrimary)
                                    Spacer()
                                    
                                    if isDownloaded {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(.green)
                                    } else {
                                        Button {
                                            dismiss()
                                            Task { await registry.downloadPersona(name: personaName, using: engine) }
                                        } label: {
                                            Image(systemName: "icloud.and.arrow.down")
                                                .foregroundStyle(SwiftBuddyTheme.accent)
                                        }
                                        .buttonStyle(.plain)
                                        .disabled(registry.isSyncing)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    } else if !registry.isSyncing {
                        VStack(spacing: 12) {
                            Image(systemName: "cloud.bolt")
                                .font(.system(size: 32))
                                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                            Text("No personas found.")
                                .foregroundStyle(SwiftBuddyTheme.textSecondary)
                        }
                        .frame(maxWidth: .infinity, minHeight: 100)
                        .listRowBackground(Color.clear)
                    }
                }
                .scrollContentBackground(.hidden)
                .background(SwiftBuddyTheme.background)
            }
            .navigationTitle("Discover Personas")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        #if os(macOS)
        .frame(width: 400, height: 500)
        #endif
    }
}
