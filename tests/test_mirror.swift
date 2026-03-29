
@propertyWrapper
struct ModuleInfo<Value> {
    var wrappedValue: Value
    var key: String
}
class SwitchGLU {}
class MLP { 
  @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU = SwitchGLU()
}
let m = MLP()
let mirror = Mirror(reflecting: m)
for child in mirror.children { print(child.label!) }

