import Foundation

struct HFModelDetails: Decodable {
    let usedStorage: Int64?
    let tags: [String]?
}

Task {
    let url = URL(string: "https://huggingface.co/api/models/mlx-community/Qwen2.5-3B-4bit")!
    let (data, _) = try await URLSession.shared.data(from: url)
    let details = try JSONDecoder().decode(HFModelDetails.self, from: data)
    print("Storage:", details.usedStorage ?? -1)
    print("Tags:", details.tags ?? [])
}
RunLoop.main.run(until: Date(timeIntervalSinceNow: 2))
