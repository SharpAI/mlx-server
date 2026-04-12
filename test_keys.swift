import Foundation
import MLX
import MLXNN

MLX.GPU.set(cacheLimit: 10 * 1024 * 1024)
let norm = RMSNorm(dimensions: 128)
print("RMSNorm parameters:", norm.parameters().keys)
