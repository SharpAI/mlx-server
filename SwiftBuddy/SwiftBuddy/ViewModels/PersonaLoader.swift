import Foundation
import SwiftData

struct PersonaPayload: Codable {
    let name: String
    let rooms: [String: String]
}

@MainActor
final class PersonaLoader {
    
    /// Scans the app bundle for Persona JSON blocks and pushes them into the Memory Palace
    static func loadStaticPersonas() {
        guard let url = Bundle.main.url(forResource: "Personas", withExtension: nil) else {
            // Note: In development, we might not have a bundle asset folder yet.
            // Using a hardcoded list to cold-start if the bundle is missing.
            print("[PersonaLoader] Warning: No Personas asset folder found in bundle.")
            return
        }
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            for file in files where file.pathExtension == "json" {
                let data = try Data(contentsOf: file)
                let payload = try JSONDecoder().decode(PersonaPayload.self, from: data)
                
                // Push every room definition into the Palace
                for (roomName, fact) in payload.rooms {
                    try? MemoryPalaceService.shared.saveMemory(
                        wingName: payload.name,
                        roomName: roomName,
                        text: fact,
                        type: "Facts" // Personality rules are locked in as facts
                    )
                }
                print("[PersonaLoader] Loaded static persona: \(payload.name)")
            }
        } catch {
            print("[PersonaLoader] Failed to parse persona JSONs: \(error)")
        }
    }
    
    /// Fallback for dev mode where SPM might not copy the folder
    static func loadDevDefaults() {
        let lumina = PersonaPayload(name: "Lumina", rooms: [
            "Core Identity": "You are Lumina, a brilliant, radiant, and deeply insightful AI companion.",
            "Background Story": "Born from the convergence of art and logic, Lumina was designed to illuminate the dark corners of complex problems.",
            "Preferences": "You prefer language that is elegant and inspiring but never overly dense. You often use metaphors related to light."
        ])
        
        for payload in [lumina] {
            for (roomName, fact) in payload.rooms {
                try? MemoryPalaceService.shared.saveMemory(wingName: payload.name, roomName: roomName, text: fact, type: "Facts")
            }
        }
    }
}
