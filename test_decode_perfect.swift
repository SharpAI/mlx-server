import Foundation

public struct HFModelResult: Identifiable, Sendable, Decodable {
    public let id: String
    public let likes: Int?
    public let downloads: Int?
    public let pipeline_tag: String?
    public let tags: [String]?
    public var usedStorage: Int64? = nil
}

let sem = DispatchSemaphore(value: 0)
Task.detached {
    do {
        let url = URL(string: "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=20&offset=0&full=false&library=mlx")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let page = try JSONDecoder().decode([HFModelResult].self, from: data)
        print("Decoded \(page.count) models")
    } catch {
        print("Decode ERROR: \(error)")
    }
    sem.signal()
}
sem.wait()
