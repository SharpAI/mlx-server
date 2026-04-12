import Foundation

// simulate python logic
let total_layers = 42
let layer_types = [
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global"
]
let num_kv_shared_layers = 18

let first_kv_shared_layer_idx = total_layers - num_kv_shared_layers
prev_layers = layer_types[0..<first_kv_shared_layer_idx]

for layer_idx in range(42):
    is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx
    if is_kv_shared_layer:
        # find last non-shared of same type
        ltype = layer_types[layer_idx]
        rev_prev = list(reversed(prev_layers))
        idx_in_rev = rev_prev.index(ltype)
        kv_shared_layer_index = len(prev_layers) - 1 - idx_in_rev
        store_full = False
        print(f"Layer {layer_idx} ({ltype}): shared = True, uses cache from {kv_shared_layer_index}")
    else:
        ltype = layer_types[layer_idx]
        rev_prev = list(reversed(prev_layers))
        idx_in_rev = rev_prev.index(ltype)
        kv_shared_layer_index = None
        store_full = (layer_idx == len(prev_layers) - 1 - idx_in_rev)
        print(f"Layer {layer_idx} ({ltype}): shared = False, store = {store_full}")
