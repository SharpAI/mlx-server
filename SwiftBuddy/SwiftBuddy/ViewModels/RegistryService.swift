import Foundation
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct GithubNode: Codable, Identifiable {
    var id: String { name }
    let name: String
    let type: String
    let download_url: String?
}

struct PersonaRegistry: Codable {
    let personas: [String]
}

@MainActor
public final class RegistryService: ObservableObject {
    public static let shared = RegistryService()
    
    @Published public var availablePersonas: [String] = []
    @Published public var isSyncing: Bool = false
    @Published public var lastSyncLog: String = ""
    
    // Extraction Telemetry
    @Published public var extractionPhase: String = ""
    @Published public var extractionTotal: Int = 0
    @Published public var extractionProcessed: Int = 0
    @Published public var currentChunkText: String = ""
    
    private let repoBaseUrl = "https://raw.githubusercontent.com/SharpAI/swiftbuddy-registry/main"
    
    private init() {}
    
    public func fetchAvailablePersonas() async {
        isSyncing = true
        lastSyncLog = "Fetching cloud registry..."
        
        let manifestUrl = repoBaseUrl + "/persona.json?_t=\(Date().timeIntervalSince1970)"
        print("[RegistryService] fetchAvailablePersonas started. URL: \(manifestUrl)")
        
        guard let url = URL(string: manifestUrl) else { 
            print("[RegistryService] Invalid URL structure.")
            isSyncing = false
            return 
        }
        
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("SwiftBuddy-macOS/1.0", forHTTPHeaderField: "User-Agent")
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                print("[RegistryService] Github HTTP Status: \(httpResponse.statusCode)")
                if httpResponse.statusCode != 200 {
                    let bodyString = String(data: data, encoding: .utf8) ?? "<binary/empty>"
                    print("[RegistryService] GitHub response body: \(bodyString)")
                }
            }
            
            if let registry = try? JSONDecoder().decode(PersonaRegistry.self, from: data) {
                self.availablePersonas = registry.personas
                lastSyncLog = "Found \(self.availablePersonas.count) characters in the cloud."
                print("[RegistryService] Successfully mapped \(self.availablePersonas.count) nodes from persona.json.")
            } else {
                let bodyString = String(data: data, encoding: .utf8) ?? ""
                print("[RegistryService] Failed to decode 404 or missing JSON format. Payload length: \(bodyString.count)")
                // Fallback to local bundled localization
                self.availablePersonas = ["Einstein_Localized"]
                lastSyncLog = "Registry 404/Empty. Loaded bundled fallback persona."
            }
        } catch {
            print("[RegistryService] Network error during fetch: \(error)")
            self.availablePersonas = ["Einstein_Localized"]
            lastSyncLog = "Network error. Loaded bundled fallback persona."
        }
        
        isSyncing = false
    }
    
    public func downloadPersona(name: String, using engine: InferenceEngine? = nil) async {
        guard !isSyncing else { return }
        isSyncing = true
        lastSyncLog = "Downloading \(name)..."
        
        if name == "Einstein_Localized" {
            let mockCorpus = """
            Albert Einstein is widely recognized as one of the greatest physicists of all time.
            
            He was known for his eccentricities, such as his stark refusal to wear socks, claiming that his big toe would inevitably create a hole in them. He also loved sailing and playing the violin.
            
            He formulated the theory of relativity, forever reshaping our understanding of space, time, and gravity through his famous equation E = mc^2.
            """
            
            let chunks = TextCombiner.chunkText(mockCorpus, chunkSize: 800, chunkOverlap: 100, minChunkSize: 50)
            _ = try? await MemoryPalaceService.shared.saveMemories(
                wingName: "Einstein Localized",
                roomName: "corpus",
                texts: chunks,
                type: "hall_facts"
            )
            
            lastSyncLog = "Successfully installed Einstein Localized!"
            isSyncing = false
            return
        }
        
        let rooms = ["BACKGROUND_STORY.txt", "CORE_IDENTITY.txt", "CORPUS.txt", "PREFERENCES.txt"]
        var fetchedAny = false
        
        for roomFile in rooms {
            let roomName = roomFile.replacingOccurrences(of: ".txt", with: "")
            let targetUrl = repoBaseUrl + "/personas/\(name)/\(roomFile)?_t=\(Date().timeIntervalSince1970)"
            guard let url = URL(string: targetUrl) else { continue }
            
            lastSyncLog = "Fetching \(roomName)..."
            
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.setValue("SwiftBuddy-macOS/1.0", forHTTPHeaderField: "User-Agent")
            
            do {
                let (data, response) = try await URLSession.shared.data(for: request)
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    guard let textContent = String(data: data, encoding: .utf8), !textContent.isEmpty else { continue }
                    fetchedAny = true
                    
                    let chunks = TextCombiner.chunkText(textContent, chunkSize: 800, chunkOverlap: 100, minChunkSize: 50)
                    _ = try? await MemoryPalaceService.shared.saveMemories(
                        wingName: name.replacingOccurrences(of: "_", with: " "),
                        roomName: roomName.replacingOccurrences(of: "_", with: " "),
                        texts: chunks,
                        type: roomName.lowercased() == "corpus" ? "hall_facts" : "hall_preferences",
                        onProgress: { cur, tot, txt in
                            DispatchQueue.main.async {
                                self.extractionPhase = roomName
                                self.extractionProcessed = cur
                                self.extractionTotal = tot
                                self.currentChunkText = txt
                            }
                        }
                    )
                }
            } catch {
                print("[RegistryService] Network error downloading \(roomFile): \(error)")
            }
        }
        
        if fetchedAny {
            let friendlyName = name.replacingOccurrences(of: "_", with: " ")
            lastSyncLog = "SYNAPTIC SYNTHESIS FOR \(friendlyName)..."
            
            // Phase 2 Extraction: Vector-Driven Persona Synthesis
            do {
                if let engine = engine {
#if canImport(MLXInferenceCore)
                    try await MemoryPalaceService.shared.synthesizePersonaIndex(wingName: friendlyName, using: engine) { [weak self] (current: Int, total: Int, text: String) in
                        Task { @MainActor in
                            self?.extractionPhase = text
                            self?.extractionProcessed = current
                            self?.extractionTotal = total
                            self?.currentChunkText = ""
                        }
                    }
#endif
                } else {
                    print("[RegistryService] WARNING: Engine not injected. Persona Synthesis bypassed.")
                }
            } catch {
                print("[RegistryService] Persona Synthesis failed: \(error)")
            }
            
            lastSyncLog = "Successfully installed \(friendlyName)!"
        } else {
             lastSyncLog = "Failed to download \(name)."
        }
        
        isSyncing = false
    }
}
