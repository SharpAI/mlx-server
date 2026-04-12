import Foundation

let sem = DispatchSemaphore(value: 0)
Task.detached {
    let url1 = URL(string: "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=20&offset=0&full=false&library=mlx")!
    let url2 = URL(string: "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=20&offset=20&full=false&library=mlx")!
    let (d1, _) = try! await URLSession.shared.data(from: url1)
    let (d2, _) = try! await URLSession.shared.data(from: url2)
    print("Page 1 length:", d1.count)
    print("Page 2 length:", d2.count)
    print("Page 1 == Page 2:", d1 == d2)
    sem.signal()
}
sem.wait()
