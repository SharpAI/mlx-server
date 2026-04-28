import SwiftUI
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif
import SwiftData

struct InspectorView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var server: ServerManager
    @Binding var showModelManagement: Bool
    
    @Query(sort: \PalaceWing.name) var wings: [PalaceWing]
    @StateObject private var registryService = RegistryService.shared
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                
                // MARK: - Telemetry Dashboard
                ResourceDashboardView()
                    .padding(.bottom, 10)
                
                // MARK: - API Server Status
                Section {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Local API Server", systemImage: "network")
                            .font(.headline)
                        
                        HStack {
                            Circle()
                                .fill(server.isOnline ? Color.green : Color.orange)
                                .frame(width: 8, height: 8)
                            Text(server.isOnline ? "Online" : "Offline")
                                .font(.subheadline)
                                .bold()
                                .foregroundStyle(server.isOnline ? .green : .orange)
                            Spacer()
                            Text("\(server.host):\(server.port)")
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        
                        Text(server.isOnline
                             ? "Ready for /v1/chat/completions requests."
                             : "Server will use saved SwiftLM startup settings when started.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.top, 4)
                        
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)
                } header: {
                    Text("NETWORKING").font(.caption).foregroundColor(.secondary)
                }
                
                // MARK: - Active Model Info
                Section {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Current Model", systemImage: "cpu")
                            .font(.headline)
                        
                        if case .ready(let modelId) = engine.state {
                            Text(modelId.components(separatedBy: "/").last ?? modelId)
                                .font(.body.monospaced())
                                .lineLimit(1)
                                .truncationMode(.middle)
                            
                            HStack {
                                Button("Unload") {
                                    engine.unload()
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        } else {
                            Text("No model loaded.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            
                            Button("Load Model") {
                                showModelManagement = true
                            }
                            .buttonStyle(.borderedProminent)
                            //.tint(SwiftBuddyTheme.accent)
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)
                } header: {
                    Text("INFERENCE").font(.caption).foregroundColor(.secondary)
                }
                
                // MARK: - Tools
                Section {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Web Browsing Tool", systemImage: "globe")
                            .font(.headline)
                        
                        HStack {
                            Circle()
                                .fill(Color.orange) // Switch to green when enabled
                                .frame(width: 8, height: 8)
                            Text("Standby")
                                .font(.subheadline)
                                .bold()
                                .foregroundStyle(.orange)
                        }
                        
                        Text("Uses DuckDuckGo & SwiftSoup")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.top, 4)
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)
                } header: {
                    Text("TOOLS").font(.caption).foregroundColor(.secondary)
                }
                

                
                // MARK: - Cloud Persona Registry
                Section {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label("Cloud Registry", systemImage: "sparkles.rectangle.stack")
                                .font(.headline)
                            Spacer()
                            Button(action: {
                                Task { await registryService.fetchAvailablePersonas() }
                            }) {
                                Image(systemName: "arrow.triangle.2.circlepath")
                            }
                            .buttonStyle(.plain)
                            .disabled(registryService.isSyncing)
                        }
                        
                        if registryService.availablePersonas.isEmpty {
                            Text("Click refresh to discover personas.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        } else {
                            ForEach(registryService.availablePersonas, id: \.self) { personaName in
                                HStack {
                                    Text(personaName.replacingOccurrences(of: "_", with: " "))
                                        .font(.subheadline)
                                    Spacer()
                                    Button("Install") {
                                        Task { await registryService.downloadPersona(name: personaName, using: engine) }
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .controlSize(.mini)
                                    .disabled(registryService.isSyncing || wings.contains(where: { $0.name == personaName.replacingOccurrences(of: "_", with: " ") }))
                                }
                            }
                        }
                        
                        if !registryService.lastSyncLog.isEmpty {
                            Text(registryService.lastSyncLog)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .padding(.top, 4)
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)
                    .fixedSize(horizontal: false, vertical: true)
                } header: {
                    Text("DISCOVER PERSONAS").font(.caption).foregroundColor(.secondary)
                }
                
                Spacer()
            }
            .padding()
        }
        .navigationTitle("System Status")
    }
}
