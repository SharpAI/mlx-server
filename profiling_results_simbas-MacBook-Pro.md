### `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 0.71s | 448.43 tok/s | N/A | 1.1 GB | 17.4 GB |
| Dense/Vanilla | 40000 | 15.85s | 55.73 tok/s | N/A | 49.4 GB | 65.9 GB |
| Dense/Vanilla | 100000 | 45.36s | 21.33 tok/s | N/A | 49.4 GB | 65.6 GB |
| SSD Stream | 512 | 0.15s | 117.66 tok/s | N/A | 0.7 GB | 17.2 GB |
| SSD Stream | 40000 | 13.35s | 42.22 tok/s | N/A | 49.4 GB | 65.4 GB |
| SSD Stream | 100000 | 61.34s | 13.97 tok/s | N/A | 49.3 GB | 65.5 GB |
| TurboQuant | 512 | 0.06s | 439.35 tok/s | N/A | 1.1 GB | 17.2 GB |
| TurboQuant | 40000 | 28.85s | 1.89 tok/s | N/A | 19.3 GB | 35.2 GB |
| TurboQuant | 100000 | 103.60s | 1.18 tok/s | N/A | 40.2 GB | 56.5 GB |
| SSD + TurboQuant | 512 | 0.14s | 117.85 tok/s | N/A | 0.7 GB | 17.3 GB |
| SSD + TurboQuant | 40000 | 30.50s | 1.74 tok/s | N/A | 5.9 GB | 22.1 GB |
| SSD + TurboQuant | 100000 | 111.63s | 1.11 tok/s | N/A | 14.1 GB | 30.1 GB |
| SSD + 16-Worker Prefetch | 512 | 0.14s | 117.63 tok/s | N/A | 0.7 GB | 16.9 GB |
| SSD + 16-Worker Prefetch | 40000 | 17.43s | 35.13 tok/s | N/A | 49.4 GB | 65.7 GB |
| SSD + 16-Worker Prefetch | 100000 | 55.61s | 18.80 tok/s | N/A | 49.4 GB | 64.9 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
