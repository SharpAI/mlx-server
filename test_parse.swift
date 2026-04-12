import Foundation

struct HFModelResult: Decodable {
    let id: String
    var paramSizeHint: String? {
        let repoName = String(id.split(separator: "/").last ?? "")
        let patterns = [#"(\d+)[xX](\d+)[Bb]"#, #"(\d+\.?\d*)[Bb]"#]
        for pattern in patterns {
            if let match = repoName.range(of: pattern, options: .regularExpression) {
                return String(repoName[match])
            }
        }
        return nil
    }
}

let sem = DispatchSemaphore(value: 0)
Task.detached {
    for offset in [0, 20, 40] {
        let url = URL(string: "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=20&offset=\(offset)&full=false&library=mlx")!
        let (d, _) = try await URLSession.shared.data(from: url)
        let page = try JSONDecoder().decode([HFModelResult].self, from: d)
        print("Page \((offset/20)+1):")
        for m in page { print("   \(m.id) -> \(m.paramSizeHint ?? "nil")") }
    }
    sem.signal()
}
sem.wait()
