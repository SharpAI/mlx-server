#!/usr/bin/env python3
"""Dump Python DFlash intermediate values for cross-language comparison.

Outputs .npy files and a _meta.json with token IDs and scalar values.
Run: python3 dump_python_intermediates.py
"""
import json
import os
import sys
import numpy as np
import mlx.core as mx

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/intermediates"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Patch hooks out so we compare bare numerical paths ──
import dflash_mlx.runtime as rt
rt._install_target_speculative_hooks = lambda *a, **kw: None

from dflash_mlx.runtime import (
    load_target_bundle, load_draft_bundle, resolve_model_ref,
    _target_embed_tokens, _lm_head_logits, greedy_tokens_with_mask,
    _verify_target_block, make_target_cache,
)
from dflash_mlx.model import ContextOnlyDraftKVCache

mx.set_cache_limit(mx.device_info()["max_recommended_working_set_size"] // 4)

PROMPT = "Hello"
BLOCK_LEN = 16
USE_CHAT_TEMPLATE = True

def save(name: str, arr: mx.array):
    # Convert MLX array to numpy via float32 to avoid bfloat16 issues
    # For integer arrays, cast to int32 first
    if mx.issubdtype(arr.dtype, mx.integer):
        np_arr = np.array(arr.astype(mx.int32), copy=True)
    else:
        np_arr = np.array(arr.astype(mx.float32), copy=True)
    np.save(f"{OUT_DIR}/{name}.npy", np_arr)
    print(f"  saved {name}: shape={arr.shape} dtype={arr.dtype}")

def main():
    print("Loading models …")
    target_model, tokenizer, _ = load_target_bundle(
        resolve_model_ref("mlx-community/Qwen3.5-27B-4bit", kind="target"),
        lazy=True, split_full_attention_sdpa=False,
    )
    draft_model, _ = load_draft_bundle(
        resolve_model_ref("z-lab/Qwen3.5-27B-DFlash", kind="draft"),
        lazy=True,
    )

    # ── 1. Prompt tokens ──
    from dflash_mlx.runtime import _prepare_prompt_tokens
    prompt_tokens = _prepare_prompt_tokens(tokenizer, PROMPT, use_chat_template=USE_CHAT_TEMPLATE)
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    # ── 2. Target prefill ──
    target_cache = make_target_cache(target_model, enable_speculative_linear_cache=True)
    target_layer_ids = list(draft_model.target_layer_ids)
    capture_layer_ids = {int(lid) + 1 for lid in target_layer_ids}

    logits, hidden_states = _verify_target_block(
        target_model=target_model,
        verify_ids=mx.array(prompt_tokens, dtype=mx.uint32)[None],
        target_cache=target_cache,
        verify_chunk_tokens=None,
        capture_layer_ids=capture_layer_ids,
    )
    mx.eval(logits, *hidden_states.values())

    save("prefill_logits", logits)
    for lid in capture_layer_ids:
        save(f"hidden_layer_{lid}", hidden_states[lid])

    # ── 3. Extract context feature ──
    selected = [hidden_states[layer_id + 1] for layer_id in target_layer_ids]
    target_hidden = mx.concatenate(selected, axis=-1)
    save("target_hidden", target_hidden)

    # ── 4. staged_first ──
    staged_first = greedy_tokens_with_mask(logits[:, -1, :], None)
    staged_first_id = int(staged_first.item())
    print(f"staged_first = {staged_first_id} = {repr(tokenizer.decode([staged_first_id]))}")

    # ── 5. Draft model inputs ──
    mask_token_id = int(draft_model.mask_token_id)
    block_token_ids = mx.concatenate(
        [staged_first[:1], mx.full((BLOCK_LEN - 1,), mask_token_id, dtype=mx.uint32)]
    )
    noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
    save("noise_embedding", noise_embedding)
    save("block_token_ids", block_token_ids[None])

    # ── 6. Draft model: project target hidden ──
    projected_hidden = draft_model._project_target_hidden(target_hidden)
    save("projected_hidden", projected_hidden)

    # ── 7. Draft model: layer-by-layer ──
    draft_cache = [ContextOnlyDraftKVCache() for _ in range(len(draft_model.layers))]
    hidden_states_draft = noise_embedding

    for i, (layer, layer_cache) in enumerate(zip(draft_model.layers, draft_cache)):
        # input layernorm
        h = layer.input_layernorm(hidden_states_draft)
        save(f"draft_layer{i}_after_input_ln", h)

        # attention
        h = layer.self_attn(h, target_hidden=projected_hidden, cache=layer_cache)
        save(f"draft_layer{i}_after_attn", h)

        # residual + attention
        h = hidden_states_draft + h
        save(f"draft_layer{i}_after_attn_residual", h)

        # post-attention layernorm
        r = h
        h = layer.post_attention_layernorm(h)
        save(f"draft_layer{i}_after_post_ln", h)

        # MLP
        h = layer.mlp(h)
        save(f"draft_layer{i}_after_mlp", h)

        # final residual
        hidden_states_draft = r + h
        save(f"draft_layer{i}_output", hidden_states_draft)

    # ── 8. Final norm + logits ──
    draft_final = draft_model.norm(hidden_states_draft)
    save("draft_final_normed", draft_final)

    draft_logits = _lm_head_logits(target_model, draft_final[:, 1:, :])
    save("draft_logits", draft_logits)

    drafted = greedy_tokens_with_mask(draft_logits, None)
    drafted_list = drafted.tolist()
    if isinstance(drafted_list[0], list):
        drafted_list = drafted_list[0]
    print(f"drafted tokens: {drafted_list[:5]}")
    print(f"drafted text: {repr(tokenizer.decode(drafted_list[:5]))}")

    # ── 9. Verify logits (target forward on draft tokens) ──
    verify_ids = mx.concatenate([staged_first[:1], drafted[0, :BLOCK_LEN - 1]], axis=0)[None]
    save("verify_ids", verify_ids)

    # ── Meta ──
    meta = {
        "prompt_tokens": prompt_tokens,
        "staged_first": staged_first_id,
        "mask_token_id": mask_token_id,
        "block_len": BLOCK_LEN,
        "target_layer_ids": target_layer_ids,
        "capture_layer_ids": list(capture_layer_ids),
        "drafted_tokens": drafted_list,
    }
    with open(f"{OUT_DIR}/_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved to {OUT_DIR}/_meta.json")

if __name__ == "__main__":
    main()
