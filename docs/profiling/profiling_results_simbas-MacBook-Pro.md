### `mlx-community/Qwen3.6-35B-A3B-4bit` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 4.01s | 32.10 tok/s | N/A | 18.9 GB | 33.6 GB |
| Dense/Vanilla | 40000 | 26.41s | 23.99 tok/s | N/A | 49.4 GB | 64.2 GB |
| Dense/Vanilla | 100000 | 151.76s | 18.64 tok/s | N/A | 49.3 GB | 63.9 GB |
| SSD Stream | 512 | 1.81s | 15.01 tok/s | N/A | 4.5 GB | 18.8 GB |
| SSD Stream | 40000 | 28.89s | 5.13 tok/s | N/A | 37.4 GB | 51.7 GB |
| SSD Stream | 100000 | 100.72s | 4.08 tok/s | N/A | 49.4 GB | 63.9 GB |
| TurboQuant | 512 | 0.44s | 33.14 tok/s | N/A | 18.9 GB | 33.3 GB |
| TurboQuant | 40000 | 20.90s | 2.54 tok/s | N/A | 22.7 GB | 37.0 GB |
| TurboQuant | 100000 | 60.30s | 4.73 tok/s | N/A | 27.7 GB | 42.0 GB |
| SSD + TurboQuant | 512 | 1.64s | 14.51 tok/s | N/A | 4.5 GB | 19.3 GB |
| SSD + TurboQuant | 40000 | 27.56s | 5.39 tok/s | N/A | 8.5 GB | 23.2 GB |
| SSD + TurboQuant | 100000 | 75.59s | 3.86 tok/s | N/A | 13.6 GB | 28.3 GB |
| SSD + 16-Worker Prefetch | 512 | 0.94s | 16.70 tok/s | N/A | 4.5 GB | 19.4 GB |
| SSD + 16-Worker Prefetch | 40000 | 28.88s | 5.17 tok/s | N/A | 37.4 GB | 51.9 GB |
| SSD + 16-Worker Prefetch | 100000 | 101.96s | 3.79 tok/s | N/A | 49.4 GB | 63.9 GB |
### `Thump604/DeepSeek-V4-Flash-MLX-Q3-mixed-gs128-affine` — Context & Memory Profile

Context depths tested: 512,40000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (OS) | GPU_Alloc (virtual) | GPU_InUse peak (physical) |
|---|---|---|---|---|---|---|---|
| SSD Stream | 512 | 6.80s | 4.65 tok/s | N/A | 17.0 GB | 28.4 GB | 16.7 GB |
| SSD Stream | 40000 | 565.02s | 0.32 tok/s | N/A | 48.3 GB | 60.5 GB | 12.5 GB |
| SSD + TurboQuant | 512 | 6.35s | 4.78 tok/s | N/A | 16.9 GB | 29.5 GB | 16.8 GB |
| SSD + TurboQuant | 40000 | 363.76s | 4.16 tok/s | N/A | 28.3 GB | 40.6 GB | 16.8 GB |
| SSD + 16-Worker Prefetch | 512 | 5.84s | 4.43 tok/s | N/A | 16.9 GB | 29.3 GB | 16.6 GB |
| SSD + 16-Worker Prefetch | 40000 | 565.50s | 0.32 tok/s | N/A | 48.3 GB | 60.9 GB | 13.6 GB |

> **Active RAM (OS)**: Memory wired into physical RAM by macOS (from server log).
> **GPU_Alloc (virtual)**: Total GPU address-space allocation including SSD-backed pages — the TRUE memory demand, can exceed physical RAM.
> **GPU_InUse peak (physical)**: Peak physical RAM occupied by the GPU during the entire request (prefill + generation), sampled every 0.5 s. This is the real active footprint — for SSD-streaming configs it reflects the high-water mark while layers are being read, not a post-generation snapshot.
