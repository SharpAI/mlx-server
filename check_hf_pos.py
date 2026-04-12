from mlx.core import load
import mlx.core as mx
import glob

f = glob.glob("/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/*/*.safetensors")[0]
st = load(f)

table = st["vision_tower.patch_embedder.position_embedding_table"]
print(f"Table shape: {table.shape}")

# Check magnitude of index 0 vs index 1
mag0 = mx.max(mx.abs(table[0])).item()
mag1 = mx.max(mx.abs(table[1])).item()

print(f"Max abs val for index 0: {mag0}")
print(f"Max abs val for index 1: {mag1}")
