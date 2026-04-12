import Foundation

struct HFModelResult: Decodable {
    let id: String
    let tags: [String]?
    var usedStorage: Int64? = nil
}

let sem = DispatchSemaphore(value: 0)
Task {
    do {
        let url = URL(string: "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=5&offset=0&full=false&library=mlx")!
        let (data, _) = try await URLSession.shared.data(from: url)
        var page = try JSONDecoder().decode([HFModelResult].self, from: data)
        print("Decoded \(page.count) models. Fetching storage sizes...")
        
        try await withThrowingTaskGroup(of: (Int, Int64?).self) { group in
            for i in 0..<page.count {
                let modelId = page[i].id
                group.addTask {
                    let detailUrl = URL(string: "https://huggingface.co/api/models/\(modelId)")!
                    let (detailData, _) = try await URLSession.shared.data(from: detailUrl)
                    struct HFFullDetails: Decodable {
                        let usedStorage: Int64?
                    }
                    let details = try? JSONDecoder().decode(HFFullDetails.self, from: detailData)
                    return (i, details?.usedStorage)
                }
            }
            for try await (index, size) in group {
                if let size = size {
                    page[index].usedStorage = size
                }
            }
        }
        print("Success! First model size: \(page[0].usedStorage ?? -1)")
    } catch {
        print("Failure: \(error)")
    }
    sem.signal()
}
sem.wait()
