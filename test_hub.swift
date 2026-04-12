import Foundation
import Hub

let cache = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("huggingface_test_swift_hub")
let hub = HubApi(downloadBase: cache)

Task {
    do {
        let url = try await hub.snapshot(from: "mlx-community/Qwen2.5-3B-4bit", matching: ["*.safetensors", "*.json", "*.txt"])
        print("Success! url: \(url)")
    } catch {
        print("Error: \(error)")
    }
}
RunLoop.main.run(until: Date(timeIntervalSinceNow: 5))
