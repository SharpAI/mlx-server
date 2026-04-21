#!/usr/bin/env python3
"""Compare Python vs Swift DFlash intermediate values using cosine similarity.

Loads Python reference .npy dumps from intermediates/ and Swift dumps
from swift_dumps/ (or custom dir), computing cosine similarity at each step.

Usage: python3 compare_swift_python.py [--swift-dir /tmp/dflash_swift_dumps]
"""
import json
import os
import sys
import argparse
import numpy as np
import mlx.core as mx

def cosine_sim(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two arrays."""
    if a.shape != b.shape:
        print(f"    ⚠️  Shape mismatch: {a.shape} vs {b.shape}")
        # Try to broadcast or slice
        min_dims = [min(a.shape[i], b.shape[i]) for i in range(len(a.shape))]
        slices_a = tuple(slice(0, m) for m in min_dims)
        slices_b = tuple(slice(0, m) for m in min_dims)
        a = a[slices_a]
        b = b[slices_b]
    a = a.reshape(-1).astype(mx.float32)
    b = b.reshape(-1).astype(mx.float32)
    dot = (a * b).sum()
    denom = mx.sqrt((a * a).sum() * (b * b).sum())
    if float(denom) < 1e-10:
        return 0.0
    return float(dot / denom)

def mean_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.reshape(-1).astype(mx.float32) - b.reshape(-1).astype(mx.float32)).mean())

def load_npy(path: str) -> mx.array:
    arr = np.load(path)
    return mx.array(arr)

def compare(name: str, ref: mx.array, test: mx.array) -> float:
    cs = cosine_sim(ref, test)
    mad = mean_abs_diff(ref, test)
    if cs > 0.99:
        status = "✅"
    elif cs > 0.95:
        status = "⚠️"
    else:
        status = "❌"
    shape_str = "x".join(str(s) for s in ref.shape)
    print(f"  {status} {name:50s} cos={cs:.6f}  mad={mad:.8f}  shape={shape_str}")
    return cs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "intermediates"))
    parser.add_argument("--swift-dir", default="/tmp/dflash_swift_dumps")
    args = parser.parse_args()
    
    py_dir = args.py_dir
    swift_dir = args.swift_dir
    
    print("═══════════════════════════════════════════════════════════════════")
    print("  DFlash Python ↔ Swift Cosine Similarity Comparison")
    print(f"  Python dir:  {py_dir}")
    print(f"  Swift dir:   {swift_dir}")
    print("═══════════════════════════════════════════════════════════════════")
    
    # Load meta
    with open(os.path.join(py_dir, "_meta.json")) as f:
        meta = json.load(f)
    
    prompt_tokens = meta["prompt_tokens"]
    staged_first = meta["staged_first"]
    block_len = meta["block_len"]
    target_layer_ids = meta["target_layer_ids"]
    drafted_tokens = meta["drafted_tokens"]
    
    print(f"\n  Python: prompt={len(prompt_tokens)} tokens, staged_first={staged_first}")
    print(f"  Python: target_layer_ids={target_layer_ids}")
    print(f"  Python: drafted_tokens[:5]={drafted_tokens[:5]}")
    
    results = []
    
    # ── 1. Target hidden states ──
    print("\n── 1. Target hidden states (from prefill) ──")
    try:
        py_target = load_npy(os.path.join(py_dir, "target_hidden.npy"))
        sw_target = load_npy(os.path.join(swift_dir, "swift_target_hidden.npy"))
        cs = compare("target_hidden", py_target, sw_target)
        results.append(("target_hidden", cs))
    except Exception as e:
        print(f"  ⚠️  Could not compare target_hidden: {e}")
    
    # ── 2. Noise embedding ──
    print("\n── 2. Noise embedding (target embed_tokens) ──")
    try:
        py_noise = load_npy(os.path.join(py_dir, "noise_embedding.npy"))
        sw_noise = load_npy(os.path.join(swift_dir, "swift_noise_embedding.npy"))
        cs = compare("noise_embedding", py_noise, sw_noise)
        results.append(("noise_embedding", cs))
    except Exception as e:
        print(f"  ⚠️  Could not compare noise_embedding: {e}")
    
    # ── 3. Projected hidden ──
    print("\n── 3. Projected hidden (fc + hiddenNorm) ──")
    try:
        py_proj = load_npy(os.path.join(py_dir, "projected_hidden.npy"))
        sw_proj = load_npy(os.path.join(swift_dir, "swift_projected_hidden.npy"))
        cs = compare("projected_hidden", py_proj, sw_proj)
        results.append(("projected_hidden", cs))
    except Exception as e:
        print(f"  ⚠️  Could not compare projected_hidden: {e}")
    
    # ── 4. Draft model layer outputs ──
    print("\n── 4. Draft model layer outputs ──")
    for i in range(5):
        try:
            py_layer = load_npy(os.path.join(py_dir, f"draft_layer{i}_output.npy"))
            sw_layer = load_npy(os.path.join(swift_dir, f"swift_draft_layer{i}_output.npy"))
            cs = compare(f"draft_layer{i}_output", py_layer, sw_layer)
            results.append((f"draft_layer{i}_output", cs))
        except Exception as e:
            print(f"  ⚠️  Could not compare layer{i}_output: {e}")
    
    # ── 5. Draft final normed ──
    print("\n── 5. Draft final normed ──")
    try:
        py_final = load_npy(os.path.join(py_dir, "draft_final_normed.npy"))
        sw_final = load_npy(os.path.join(swift_dir, "swift_draft_final_normed.npy"))
        cs = compare("draft_final_normed", py_final, sw_final)
        results.append(("draft_final_normed", cs))
    except Exception as e:
        print(f"  ⚠️  Could not compare draft_final_normed: {e}")
    
    # ── 6. Draft logits ──
    print("\n── 6. Draft logits ──")
    try:
        py_logits = load_npy(os.path.join(py_dir, "draft_logits.npy"))
        sw_logits = load_npy(os.path.join(swift_dir, "swift_draft_logits.npy"))
        cs = compare("draft_logits", py_logits, sw_logits)
        results.append(("draft_logits", cs))
        
        # Check top tokens
        print("\n  Top tokens comparison:")
        for pos in range(min(3, py_logits.shape[1])):
            py_top = int(mx.argmax(mx.array(py_logits[0, pos]), axis=-1))
            sw_top = int(mx.argmax(mx.array(sw_logits[0, pos]), axis=-1))
            match = "✅" if py_top == sw_top else "❌"
            print(f"    pos {pos}: Python={py_top}, Swift={sw_top} {match}")
    except Exception as e:
        print(f"  ⚠️  Could not compare draft_logits: {e}")
    
    # ── 7. Prefill logits (last position) ──
    print("\n── 7. Prefill logits ──")
    try:
        py_prefill = load_npy(os.path.join(py_dir, "prefill_logits.npy"))
        sw_prefill = load_npy(os.path.join(swift_dir, "swift_prefill_logits.npy"))
        # Compare only last position
        py_last = py_prefill[:, -1, :]
        sw_last = sw_prefill[:, -1, :]
        cs = compare("prefill_logits (last pos)", py_last, sw_last)
        results.append(("prefill_logits_last", cs))
        
        # Check staged_first
        py_top = int(mx.argmax(mx.array(py_last[0]), axis=-1))
        sw_top = int(mx.argmax(mx.array(sw_last[0]), axis=-1))
        print(f"  staged_first: Python={py_top}, Swift={sw_top} {'✅' if py_top == sw_top else '❌'}")
    except Exception as e:
        print(f"  ⚠️  Could not compare prefill_logits: {e}")
    
    # ── Summary ──
    print("\n═══════════════════════════════════════════════════════════════════")
    print("  SUMMARY")
    print("═══════════════════════════════════════════════════════════════════")
    
    if not results:
        print("  No comparisons made!")
        return
    
    # Sort by cosine similarity (worst first)
    results.sort(key=lambda x: x[1])
    
    print("\n  Divergence ranking (worst → best):")
    for name, cs in results:
        bar = "█" * int(cs * 40)
        status = "✅" if cs > 0.99 else "⚠️" if cs > 0.95 else "❌"
        print(f"  {status} {name:45s} cos={cs:.6f}  {bar}")
    
    worst_name, worst_cs = results[0]
    if worst_cs < 0.95:
        print(f"\n  🔍 BIGGEST DIVERGENCE: {worst_name} (cos={worst_cs:.6f})")
        print(f"  This is the first place to investigate!")
    elif worst_cs < 0.99:
        print(f"\n  ⚠️  Small divergence at: {worst_name} (cos={worst_cs:.6f})")
    else:
        print(f"\n  ✅ All comparisons >0.99 cosine similarity!")

if __name__ == "__main__":
    main()
