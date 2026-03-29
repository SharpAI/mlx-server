import MLXNN; class MLP: Module { @ModuleInfo(key: "proj") var proj = Linear(10, 10) }; let p = MLP(); print(p.leafModules().flattened().count)
