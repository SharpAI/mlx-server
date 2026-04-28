import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct ServerStartupConfiguration: Codable, Equatable, Sendable {
    var autoStart: Bool = true
    var host: String = "127.0.0.1"
    var port: Int = 5413
    var parallelSlots: Int = 1
    var corsOrigin: String = ""
    var apiKey: String = ""

    private static let storageKey = "swiftlm.server.startupConfiguration"

    var normalized: ServerStartupConfiguration {
        var copy = self
        copy.host = copy.host.trimmingCharacters(in: .whitespacesAndNewlines)
        if copy.host.isEmpty { copy.host = "127.0.0.1" }
        copy.port = min(max(copy.port, 1), 65_535)
        copy.parallelSlots = max(copy.parallelSlots, 1)
        copy.corsOrigin = copy.corsOrigin.trimmingCharacters(in: .whitespacesAndNewlines)
        copy.apiKey = copy.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return copy
    }

    static func load() -> ServerStartupConfiguration {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode(ServerStartupConfiguration.self, from: data) else {
            return ServerStartupConfiguration()
        }
        return decoded.normalized
    }

    func save() {
        guard let data = try? JSONEncoder().encode(normalized) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

private var swiftBuddyJSONHeaders: HTTPFields {
    HTTPFields([HTTPField(name: .contentType, value: "application/json")])
}

private func swiftBuddyJSONString(_ value: String) -> String {
    guard let data = try? JSONEncoder().encode(value),
          let string = String(data: data, encoding: .utf8) else {
        return "\"\""
    }
    return string
}

private struct SwiftBuddyCORSMiddleware<Context: RequestContext>: RouterMiddleware {
    let allowedOrigin: String

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        if request.method == .options {
            return Response(status: .noContent, headers: corsHeaders(for: request))
        }

        var response = try await next(request, context)
        for field in corsHeaders(for: request) {
            response.headers.append(field)
        }
        return response
    }

    private func corsHeaders(for request: Request) -> HTTPFields {
        var fields: [HTTPField] = []
        if allowedOrigin == "*" {
            fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Origin")!, value: "*"))
        } else {
            let requestOrigin = request.headers[values: HTTPField.Name("Origin")!].first ?? ""
            if requestOrigin == allowedOrigin {
                fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Origin")!, value: allowedOrigin))
                fields.append(HTTPField(name: HTTPField.Name("Vary")!, value: "Origin"))
            }
        }
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Methods")!, value: "GET, POST, OPTIONS"))
        fields.append(HTTPField(name: HTTPField.Name("Access-Control-Allow-Headers")!, value: "Content-Type, Authorization, X-SwiftLM-Prefill-Progress"))
        return HTTPFields(fields)
    }
}

private struct SwiftBuddyAPIKeyMiddleware<Context: RequestContext>: RouterMiddleware {
    let apiKey: String

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        let path = request.uri.path
        if path == "/health" || path == "/metrics" {
            return try await next(request, context)
        }

        let authHeader = request.headers[values: .authorization].first ?? ""
        if authHeader == "Bearer \(apiKey)" || authHeader == apiKey {
            return try await next(request, context)
        }

        return Response(
            status: .unauthorized,
            headers: swiftBuddyJSONHeaders,
            body: .init(byteBuffer: ByteBuffer(string: #"{"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}"#))
        )
    }
}

@MainActor
final class ServerManager: ObservableObject {
    @Published var isOnline = false
    @Published var host: String = "127.0.0.1"
    @Published var port: Int = 5413
    @Published private(set) var startupConfiguration: ServerStartupConfiguration
    @Published private(set) var runningConfiguration: ServerStartupConfiguration?
    @Published private(set) var restartRequired = false
    
    // In a real implementation this would hold the Hummingbird App and tie into `engine`
    private var task: Task<Void, Never>?

    init() {
        let configuration = ServerStartupConfiguration.load()
        self.startupConfiguration = configuration
        self.host = configuration.host
        self.port = configuration.port
    }

    func start(engine: InferenceEngine) {
        guard !isOnline else { return }
        let configuration = startupConfiguration.normalized

        task = Task {
            do {
                let router = Router()

                if !configuration.corsOrigin.isEmpty {
                    router.add(middleware: SwiftBuddyCORSMiddleware(allowedOrigin: configuration.corsOrigin))
                }

                if !configuration.apiKey.isEmpty {
                    router.add(middleware: SwiftBuddyAPIKeyMiddleware(apiKey: configuration.apiKey))
                }

                router.get("/health") { _, _ -> Response in
                    let body = """
                    {"status":"ok","message":"SwiftBuddy Local Server","host":\(swiftBuddyJSONString(configuration.host)),"port":\(configuration.port),"parallel":\(configuration.parallelSlots),"cors":\(swiftBuddyJSONString(configuration.corsOrigin.isEmpty ? "disabled" : configuration.corsOrigin)),"auth":"\(configuration.apiKey.isEmpty ? "disabled" : "enabled")"}
                    """
                    let buffer = ByteBuffer(string: body)
                    return Response(status: .ok, headers: swiftBuddyJSONHeaders, body: .init(byteBuffer: buffer))
                }

                // Simple V1 models mock
                router.get("/v1/models") { _, _ -> Response in
                    let buffer = ByteBuffer(string: #"{"object": "list", "data": [{"id": "local", "object": "model"}]}"#)
                    return Response(status: .ok, headers: swiftBuddyJSONHeaders, body: .init(byteBuffer: buffer))
                }
                
                let app = Application(
                    router: router,
                    configuration: .init(address: .hostname(configuration.host, port: configuration.port))
                )

                self.isOnline = true
                self.host = configuration.host
                self.port = configuration.port
                self.runningConfiguration = configuration
                self.restartRequired = false
                ConsoleLog.shared.info("Server online at http://\(configuration.host):\(configuration.port)")

                try await app.runService()
            } catch {
                print("Server failed: \(error)")
                ConsoleLog.shared.error("Server failed: \(error.localizedDescription)")
                self.isOnline = false
            }
        }
    }

    @discardableResult
    func saveStartupConfiguration(_ configuration: ServerStartupConfiguration) -> Bool {
        let normalized = configuration.normalized
        let changed = normalized != startupConfiguration
        startupConfiguration = normalized
        host = normalized.host
        port = normalized.port
        normalized.save()
        restartRequired = isOnline && runningConfiguration != nil && runningConfiguration != normalized
        if changed {
            ConsoleLog.shared.info("Server startup configuration saved")
        }
        return changed
    }

    func restart(engine: InferenceEngine) {
        stop()
        start(engine: engine)
    }

    func stop() {
        task?.cancel()
        task = nil
        isOnline = false
        runningConfiguration = nil
        restartRequired = false
    }
}
