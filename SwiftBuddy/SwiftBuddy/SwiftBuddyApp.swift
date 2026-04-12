// SwiftBuddyApp.swift — App entry point (iOS + macOS)
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

// MARK: — Appearance Store (persists dark/light/system preference)

final class AppearanceStore: ObservableObject {
    private static let key = "swiftlm.colorScheme"   // "dark" | "light" | "system"

    @Published var preference: String {
        didSet { UserDefaults.standard.set(preference, forKey: Self.key) }
    }

    init() {
        preference = UserDefaults.standard.string(forKey: Self.key) ?? "dark"
    }

    var colorScheme: ColorScheme? {
        switch preference {
        case "dark":  return .dark
        case "light": return .light
        default:      return nil
        }
    }
}

// MARK: — App

@main
struct SwiftBuddyApp: App {
    @StateObject private var engine = InferenceEngine()
    @StateObject private var appearance = AppearanceStore()
    @StateObject private var server = ServerManager()

    var body: some Scene {
        WindowGroup {
            MainContentView(engine: engine, appearance: appearance, server: server)
                .modelContainer(for: [PalaceWing.self, PalaceRoom.self, MemoryEntry.self, KnowledgeGraphTriple.self, ChatSession.self, ChatTurn.self])
        }
        #if os(macOS)
        
        Window("Telemetry Dashboard", id: "telemetry-dashboard") {
            ResourceDashboardView()
                .padding()
                .frame(minWidth: 350, minHeight: 400)
                .background(SwiftBuddyTheme.background)
        }
        .windowResizability(.contentSize)
        .windowStyle(.hiddenTitleBar)
        
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Model") {
                Button("Choose Model…") {
                    NotificationCenter.default.post(name: .showModelPicker, object: nil)
                }.keyboardShortcut("m", modifiers: [.command, .shift])
                Button("Unload Model") {
                    engine.unload()
                }
            }
            CommandMenu("Tools") {
                Button("Telemetry Dashboard") {
                    NotificationCenter.default.post(name: .showTelemetryDashboard, object: nil)
                }.keyboardShortcut("t", modifiers: [.command, .shift])
            }
        }
        #endif
    }
}

extension Notification.Name {
    static let showModelPicker = Notification.Name("showModelPicker")
    static let showTextIngestion = Notification.Name("showTextIngestion")
    static let showPersonaDiscovery = Notification.Name("showPersonaDiscovery")
    static let showModelManagement = Notification.Name("showModelManagement")
    static let showTelemetryDashboard = Notification.Name("showTelemetryDashboard")
}

// Intermediary view to safely access SwiftData environment
struct MainContentView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.openWindow) private var openWindow
    
    @ObservedObject var engine: InferenceEngine
    @ObservedObject var appearance: AppearanceStore
    @ObservedObject var server: ServerManager
    
    var body: some View {
        RootView()
            .environmentObject(engine)
            .environmentObject(appearance)
            .environmentObject(server)
            .preferredColorScheme(appearance.colorScheme)
            .accentColor(SwiftBuddyTheme.accent)
            .tint(SwiftBuddyTheme.accent)
            .onAppear {
                MemoryPalaceService.shared.modelContext = modelContext
                GraphPalaceService.shared.modelContext = modelContext
                server.start(engine: engine)
                
                // Pre-load the JSON personas so the UI Wings instantly populate!
                PersonaLoader.loadDevDefaults()
                
                // Automatically resume the last selected model via UserDefaults
                if let lastModel = engine.downloadManager.lastLoadedModelId {
                    Task {
                        // Prevent loading if we're already loading or ready
                        if case .idle = engine.state {
                            await engine.load(modelId: lastModel)
                        }
                    }
                }
            }
            #if os(macOS)
            .onReceive(NotificationCenter.default.publisher(for: .showTelemetryDashboard)) { _ in
                openWindow(id: "telemetry-dashboard")
            }
            #endif
    }
}

