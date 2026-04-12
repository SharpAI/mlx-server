import Foundation

public struct HFModelResult: Identifiable, Sendable, Decodable {
    public let id: String
    public let likes: Int?
    public let downloads: Int?
    public let pipeline_tag: String?
    public let tags: [String]?
    public var usedStorage: Int64? = nil
}

Task {
    do {
        let url = URL(string: "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=20&offset=0&full=false&library=mlx")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let page = try JSONDecoder().decode([HFModelResult].self, from: data)
        print("Decoded \(page.count) models successfully.")
    } catch {
        print("Decode Error: \(error)")
    }
}
RunLoop.main.run(until: Date(timeIntervalSinceNow: 2))
