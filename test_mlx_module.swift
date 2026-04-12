import MLX
import MLXNN
import Foundation

class MyModule: Module {
    var bare_array: MLXArray
    @ModuleInfo(key: "wrapped_array") var wrapped_array: Linear

    init() {
        self.bare_array = MLXArray.zeros([10])
        self._wrapped_array.wrappedValue = Linear(10, 10)
    }
}

let m = MyModule()
print(m.parameters().keys)
