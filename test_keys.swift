import Foundation
import MLX

let weights = try! MLX.load(url: URL(fileURLWithPath: "/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/18f3418f2da5426ec6e4967b4c96bdd2d0002ee4/model-00001-of-00004.safetensors"))
for key in weights.keys {
    if key.contains("per_layer") {
        print(key)
    }
}
