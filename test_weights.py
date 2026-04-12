from mlx.core import load
import math
import glob

f = glob.glob("/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/*/*.safetensors")[0]
st = load(f)

for k, v in st.items():
    if "embed_tokens_per_layer" in k or "embed_tokens" in k:
        print(f"{k}: shape {v.shape}")
        if "per_layer" in k:
            import mlx.core as mx
            embeds = v
            # Token 258880 is image
            if 258880 < embeds.shape[0]:
                img_token = embeds[258880]
                max_val = mx.max(mx.abs(img_token)).item()
                print(f"  Token 258880 (Image) max abs value: {max_val}")
                img_token2 = embeds[258881]
                max_val2 = mx.max(mx.abs(img_token2)).item()
                print(f"  Token 258881 (Audio) max abs value: {max_val2}")
