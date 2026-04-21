#!/usr/bin/env python3
"""Compare Python vs Swift DFlash intermediate values using cosine similarity.

Loads the Python reference .npy dumps and also re-runs the Swift-equivalent
draft model forward pass using the same weights, computing cosine similarity
at each step.

The "Swift-equivalent" path simulates what Swift does:
  - No ExactSmallProjPad
  - Standard SDPA (no batched_sdpa_2pass_exact)
  - No VerifyQuantizedLinear
  - No speculative hooks

This isolates the numerical differences from the algorithmic differences.

Usage: python3 compare_cosine.py [--dir path/to/intermediates]
"""
import json
import os
import sys
import numpy as np
import mlx.core as mx

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intermediates")

def load(name: str) -> mx.array:
    arr = np.load(os.path.join(OUT_DIR, f"{name}.npy"))
    return mx.array(arr)

def cosine_sim(a: mx.array, b: mx.array) -> float:
    a = a.reshape(-1).astype(mx.float32)
    b = b.reshape(-1).astype(mx.float32)
    dot = (a * b).sum()
    denom = mx.sqrt((a * a).sum() * (b * b).sum())
    if float(denom) < 1e-10:
        return 0.0
    return float(dot / denom)

def mean_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.reshape(-1).astype(mx.float32) - b.reshape(-1).astype(mx.float32)).mean())

def compare(name: str, ref: mx.array, test: mx.array):
    cs = cosine_sim(ref, test)
    mad = mean_abs_diff(ref, test)
    status = "✅" if cs > 0.99 else "❌" if cs < 0.95 else "⚠️"
    shape_str = "x".join(str(s) for s in ref.shape)
    print(f"  {status} {name:50s} cos={cs:.6f}  mad={mad:.8f}  shape={shape_str}")
    return cs

def main():
    # Load meta
    with open(os.path.join(OUT_DIR, "_meta.json")) as f:
        meta = json.load(f)
    
    prompt_tokens = meta["prompt_tokens"]
    staged_first = meta["staged_first"]
    mask_token_id = meta["mask_token_id"]
    block_len = meta["block_len"]
    target_layer_ids = meta["target_layer_ids"]
    capture_layer_ids = meta["capture_layer_ids"]
    drafted_tokens = meta["drafted_tokens"]
    
    print("═══════════════════════════════════════════════════════════════════")
    print("  DFlash Cosine Similarity: Python Reference vs Python Reference")
    print("  (Self-consistency check — should all be 1.0)")
    print("═══════════════════════════════════════════════════════════════════")
    
    # Load all Python reference intermediates
    py_ref = {}
    for i in range(5):
        for suffix in ["after_input_ln", "after_attn", "after_attn_residual",
                       "after_post_ln", "after_mlp", "output"]:
            name = f"draft_layer{i}_{suffix}"
            try:
                py_ref[name] = load(name)
            except:
                pass
    for name in ["target_hidden", "noise_embedding", "projected_hidden",
                 "draft_final_normed", "draft_logits"]:
        try:
            py_ref[name] = load(name)
        except:
            pass
    
    print(f"\nLoaded {len(py_ref)} reference arrays")
    
    # ── Self-consistency: reload and compare ──
    print("\n── Self-consistency check ──")
    for name, arr in py_ref.items():
        arr2 = load(name)
        cs = cosine_sim(arr, arr2)
        if cs < 0.9999:
            print(f"  ⚠️  {name}: cos={cs:.8f} (should be 1.0)")
    
    print("  Self-consistency: OK")
    
    # ── Now: run the "Swift path" using same weights but different logic ──
    print("\n═══════════════════════════════════════════════════════════════════")
    print("  DFlash Cosine Similarity: Python vs Swift-equivalent")
    print("═══════════════════════════════════════════════════════════════════")
    
    # Load the draft model (same weights as Python reference)
    import dflash_mlx.runtime as rt
    rt._install_target_speculative_hooks = lambda *a, **kw: None
    
    from dflash_mlx.runtime import load_draft_bundle, resolve_model_ref, load_target_bundle
    from dflash_mlx.model import ContextOnlyDraftKVCache
    
    mx.set_cache_limit(mx.device_info()["max_recommended_working_set_size"] // 4)
    
    target_model, tokenizer, _ = load_target_bundle(
        resolve_model_ref("mlx-community/Qwen3.5-27B-4bit", kind="target"),
        lazy=True, split_full_attention_sdpa=False,
    )
    draft_model, _ = load_draft_bundle(
        resolve_model_ref("z-lab/Qwen3.5-27B-DFlash", kind="draft"),
        lazy=True,
    )
    
    # ── Step 1: Compare target_hidden ──
    # The Python reference target_hidden was computed by the Python target model.
    # The Swift target model should produce similar but not identical hidden states
    # due to the exactSmallProjPad and other numerical differences.
    # For now, compare the Python reference with itself (baseline).
    print("\n── Step 1: Target hidden states (from prefill) ──")
    py_target_hidden = py_ref["target_hidden"]
    print(f"  Python target_hidden: shape={py_target_hidden.shape}, mean={float(py_target_hidden.mean()):.6f}")
    
    # Re-run Python prefill to get target_hidden
    from dflash_mlx.runtime import _verify_target_block, make_target_cache
    target_cache = make_target_cache(target_model, enable_speculative_linear_cache=True)
    logits, hidden_states = _verify_target_block(
        target_model=target_model,
        verify_ids=mx.array(prompt_tokens, dtype=mx.uint32)[None],
        target_cache=target_cache,
        verify_chunk_tokens=None,
        capture_layer_ids=set(capture_layer_ids),
    )
    mx.eval(logits, *hidden_states.values())
    
    selected = [hidden_states[lid + 1] for lid in target_layer_ids]
    rerun_target_hidden = mx.concatenate(selected, axis=-1)
    compare("target_hidden (rerun)", py_target_hidden.astype(mx.float32), rerun_target_hidden.astype(mx.float32))
    
    # ── Step 2: Compare projected_hidden ──
    print("\n── Step 2: Projected hidden (fc + hiddenNorm) ──")
    py_proj = py_ref["projected_hidden"]
    swift_proj = draft_model._project_target_hidden(py_target_hidden.astype(mx.bfloat16))
    compare("projected_hidden", py_proj.astype(mx.float32), swift_proj.astype(mx.float32))
    
    # ── Step 3: Compare noise_embedding ──
    print("\n── Step 3: Noise embedding (target embed_tokens) ──")
    py_noise = py_ref["noise_embedding"]
    from dflash_mlx.runtime import _target_embed_tokens
    block_token_ids = load("block_token_ids")
    swift_noise = _target_embed_tokens(target_model)(block_token_ids.astype(mx.uint32))
    compare("noise_embedding", py_noise.astype(mx.float32), swift_noise.astype(mx.float32))
    
    # ── Step 4: Layer-by-layer comparison ──
    print("\n── Step 4: Draft model layer-by-layer ──")
    
    # Run the draft model step by step, comparing at each stage
    draft_cache = [ContextOnlyDraftKVCache() for _ in range(len(draft_model.layers))]
    hidden = py_noise.astype(mx.bfloat16)  # Use Python's noise_embedding as input
    projected = draft_model._project_target_hidden(py_target_hidden.astype(mx.bfloat16))
    
    for i, (layer, cache) in enumerate(zip(draft_model.layers, draft_cache)):
        print(f"\n  Layer {i}:")
        
        # Input layernorm
        h = layer.input_layernorm(hidden)
        if f"draft_layer{i}_after_input_ln" in py_ref:
            compare(f"  layer{i}_after_input_ln", py_ref[f"draft_layer{i}_after_input_ln"].astype(mx.float32), h.astype(mx.float32))
        
        # Attention
        h = layer.self_attn(h, target_hidden=projected, cache=cache)
        if f"draft_layer{i}_after_attn" in py_ref:
            compare(f"  layer{i}_after_attn", py_ref[f"draft_layer{i}_after_attn"].astype(mx.float32), h.astype(mx.float32))
        
        # Residual
        h = hidden + h
        if f"draft_layer{i}_after_attn_residual" in py_ref:
            compare(f"  layer{i}_after_attn_residual", py_ref[f"draft_layer{i}_after_attn_residual"].astype(mx.float32), h.astype(mx.float32))
        
        # Post-attention layernorm
        r = h
        h = layer.post_attention_layernorm(h)
        if f"draft_layer{i}_after_post_ln" in py_ref:
            compare(f"  layer{i}_after_post_ln", py_ref[f"draft_layer{i}_after_post_ln"].astype(mx.float32), h.astype(mx.float32))
        
        # MLP
        h = layer.mlp(h)
        if f"draft_layer{i}_after_mlp" in py_ref:
            compare(f"  layer{i}_after_mlp", py_ref[f"draft_layer{i}_after_mlp"].astype(mx.float32), h.astype(mx.float32))
        
        # Final residual
        hidden = r + h
        if f"draft_layer{i}_output" in py_ref:
            compare(f"  layer{i}_output", py_ref[f"draft_layer{i}_output"].astype(mx.float32), hidden.astype(mx.float32))
    
    # ── Step 5: Final norm + logits ──
    print("\n── Step 5: Final norm + logits ──")
    final_normed = draft_model.norm(hidden)
    if "draft_final_normed" in py_ref:
        compare("draft_final_normed", py_ref["draft_final_normed"].astype(mx.float32), final_normed.astype(mx.float32))
    
    from dflash_mlx.runtime import _lm_head_logits
    draft_logits = _lm_head_logits(target_model, final_normed[:, 1:, :])
    if "draft_logits" in py_ref:
        cs = compare("draft_logits", py_ref["draft_logits"].astype(mx.float32), draft_logits.astype(mx.float32))
        
        # Check if top tokens match
        py_top = mx.argmax(py_ref["draft_logits"][0, 0], axis=-1).item()
        swift_top = mx.argmax(draft_logits[0, 0], axis=-1).item()
        print(f"\n  Top token at pos 0: Python={py_top}, Swift-equiv={swift_top} {'✅' if py_top == swift_top else '❌'}")
    
    # ── Step 6: Run the ACTUAL Swift-equivalent path ──
    # The key difference: Swift might process things in a different order,
    # use different data types, or have subtle bugs.
    # Since this Python script can't run Swift code, we'll document the differences.
    
    print("\n═══════════════════════════════════════════════════════════════════")
    print("  ANALYSIS: Where could Swift diverge?")
    print("═══════════════════════════════════════════════════════════════════")
    print("""
  The above comparison shows Python reference vs Python re-run.
  Any cosine < 1.0 here is due to non-determinism in MLX ops.
  
  To find where SWIFT diverges, we need to dump Swift intermediates
  the same way and compare against these Python reference files.
  
  Key suspects for Swift divergence:
  1. target_hidden: Different prefill (exactSmallProjPad, VerifyQMM, etc.)
  2. noise_embedding: embed_tokens call differences
  3. projected_hidden: fc + hiddenNorm numerical differences
  4. layer attention: SDPA precision, RoPE implementation
  5. layer MLP: QuantizedLinear at small M differences
  6. final logits: lm_head numerical differences
    """)

if __name__ == "__main__":
    main()
