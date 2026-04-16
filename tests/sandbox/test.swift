import Foundation

let data = """
{
  "model": "mlx-community/Qwen3.5-122B-A10B-4bit"
}
""".data(using: .utf8)!

struct Req: Decodable {
    let repetitionPenalty: Double?
}
let req = try JSONDecoder().decode(Req.self, from: data)
print("Decoded:", req.repetitionPenalty ?? "nil")
