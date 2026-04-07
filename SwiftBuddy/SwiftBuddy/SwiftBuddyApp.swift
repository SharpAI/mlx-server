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
                .modelContainer(for: [PalaceWing.self, PalaceRoom.self, MemoryEntry.self])
        }
        #if os(macOS)
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
        }
        #endif
    }
}

extension Notification.Name {
    static let showModelPicker = Notification.Name("showModelPicker")
}

// Intermediary view to safely access SwiftData environment
struct MainContentView: View {
    @Environment(\.modelContext) private var modelContext
    
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
                server.start(engine: engine)
                
                // Pre-load the JSON personas so the UI Wings instantly populate!
                PersonaLoader.loadDevDefaults()
            }
    }
}

