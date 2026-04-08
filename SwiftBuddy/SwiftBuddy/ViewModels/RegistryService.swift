import Foundation
import SwiftData

struct GithubNode: Codable, Identifiable {
    var id: String { name }
    let name: String
    let type: String
    let download_url: String?
}

@MainActor
public final class RegistryService: ObservableObject {
    public static let shared = RegistryService()
    
    @Published public var availablePersonas: [String] = []
    @Published public var isSyncing: Bool = false
    @Published public var lastSyncLog: String = ""
    
    private let repoUrl = "https://api.github.com/repos/SharpAI/swiftbuddy-registry/contents/personas"
    
    private init() {}
    
    public func fetchAvailablePersonas() async {
        isSyncing = true
        lastSyncLog = "Fetching cloud registry..."
        print("[RegistryService] fetchAvailablePersonas started. URL: \(repoUrl)")
        
        guard let url = URL(string: repoUrl) else { 
            print("[RegistryService] Invalid URL structure.")
            isSyncing = false
            return 
        }
        
        var request = URLRequest(url: url)
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
            
            if let nodes = try? JSONDecoder().decode([GithubNode].self, from: data) {
                self.availablePersonas = nodes.filter { $0.type == "dir" }.map { $0.name }
                lastSyncLog = "Found \(self.availablePersonas.count) characters in the cloud."
                print("[RegistryService] Successfully mapped \(self.availablePersonas.count) nodes.")
            } else {
                let bodyString = String(data: data, encoding: .utf8) ?? ""
                print("[RegistryService] Failed to decode 404 or missing array format. Payload length: \(bodyString.count)")
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
    
    public func downloadPersona(name: String) async {
        guard !isSyncing else { return }
        isSyncing = true
        lastSyncLog = "Downloading \(name)..."
        
        if name == "Einstein_Localized" {
            let mockCorpus = """
            Albert Einstein is widely recognized as one of the greatest physicists of all time.
            
            He was known for his eccentricities, such as his stark refusal to wear socks, claiming that his big toe would inevitably create a hole in them. He also loved sailing and playing the violin.
            
            He formulated the theory of relativity, forever reshaping our understanding of space, time, and gravity through his famous equation E = mc^2.
            """
            
            let chunks = mockCorpus.components(separatedBy: "\n\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                
            for chunk in chunks {
                try? MemoryPalaceService.shared.saveMemory(
                    wingName: "Einstein Localized",
                    roomName: "corpus",
                    text: chunk,
                    type: "hall_facts"
                )
            }
            
            lastSyncLog = "Successfully installed Einstein Localized!"
            isSyncing = false
            return
        }
        
        let personaUrl = repoUrl + "/\(name)"
        guard let url = URL(string: personaUrl) else { return }
        
        var request = URLRequest(url: url)
        request.setValue("SwiftBuddy-macOS/1.0", forHTTPHeaderField: "User-Agent")
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse {
                print("[RegistryService] Download \(name) HTTP Status: \(httpResponse.statusCode)")
                if httpResponse.statusCode != 200 {
                    let bodyString = String(data: data, encoding: .utf8) ?? "<binary/empty>"
                    print("[RegistryService] GitHub response body: \(bodyString)")
                }
            }
            
            if let files = try? JSONDecoder().decode([GithubNode].self, from: data) {
                for file in files where file.type == "file" && file.name.hasSuffix(".txt") {
                    let roomName = file.name.replacingOccurrences(of: ".txt", with: "")
                    guard let dlURLString = file.download_url, let dlURL = URL(string: dlURLString) else { continue }
                    
                    lastSyncLog = "Fetching \(roomName)..."
                    
                    var dlRequest = URLRequest(url: dlURL)
                    dlRequest.setValue("SwiftBuddy-macOS/1.0", forHTTPHeaderField: "User-Agent")
                    let (fileData, _) = try await URLSession.shared.data(for: dlRequest)
                    guard let textContent = String(data: fileData, encoding: .utf8) else { continue }
                    
                    let chunks = textContent.components(separatedBy: "\n\n")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                    
                    for chunk in chunks {
                        try? MemoryPalaceService.shared.saveMemory(
                            wingName: name.replacingOccurrences(of: "_", with: " "),
                            roomName: roomName.replacingOccurrences(of: "_", with: " "),
                            text: chunk,
                            type: roomName.lowercased() == "corpus" ? "hall_facts" : "hall_preferences"
                        )
                    }
                }
                lastSyncLog = "Successfully installed \(name.replacingOccurrences(of: "_", with: " "))!"
            } else {
                lastSyncLog = "Failed to parse persona files."
            }
        } catch {
            print("[RegistryService] Network error downloading \(name): \(error)")
            lastSyncLog = "Failed to download \(name): \(error.localizedDescription)"
        }
        
        isSyncing = false
    }
}
