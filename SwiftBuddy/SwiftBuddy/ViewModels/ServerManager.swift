import Foundation
import Hummingbird
import NIOCore
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

@MainActor
final class ServerManager: ObservableObject {
    @Published var isOnline = false
    @Published var port: Int = 8080
    
    // In a real implementation this would hold the Hummingbird App and tie into `engine`
    private var task: Task<Void, Never>?
    
    func start(engine: InferenceEngine) {
        guard !isOnline else { return }
        
        task = Task {
            do {
                let router = Router()
                
                router.get("/health") { _, _ -> Response in
                    let buffer = ByteBuffer(string: #"{"status": "ok", "message": "SwiftBuddy Local Server"}"#)
                    return Response(status: .ok, body: .init(byteBuffer: buffer))
                }
                
                // Simple V1 models mock
                router.get("/v1/models") { _, _ -> Response in
                    let buffer = ByteBuffer(string: #"{"object": "list", "data": [{"id": "local", "object": "model"}]}"#)
                    return Response(status: .ok, body: .init(byteBuffer: buffer))
                }
                
                let app = Application(
                    router: router,
                    configuration: .init(address: .hostname("127.0.0.1", port: 8080))
                )
                
                self.isOnline = true
                self.port = 8080
                
                try await app.runService()
            } catch {
                print("Server failed: \(error)")
                self.isOnline = false
            }
        }
    }
    
    func stop() {
        task?.cancel()
        task = nil
        isOnline = false
    }
}
