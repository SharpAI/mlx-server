import Foundation
import MLX
import MLXNN

class DummyModule: Module {
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray
    override init() {
        self._layerScalar.wrappedValue = MLXArray(ones: [1])
        super.init()
    }
}
let m = DummyModule()
print(m.parameters())
