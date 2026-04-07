### `mlx-community/gemma-4-26b-a4b-it-4bit` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 0.44s | 33.01 tok/s | N/A | 15.8 GB | 23.4 GB |
| Dense/Vanilla | 40000 | 28.61s | 20.17 tok/s | N/A | 49.4 GB | 57.0 GB |
| Dense/Vanilla | 100000 | 85.08s | 15.74 tok/s | N/A | 48.5 GB | 56.7 GB |
| SSD Stream | 512 | 1.46s | 10.75 tok/s | N/A | 14.1 GB | 22.2 GB |
| SSD Stream | 40000 | 40.49s | 10.35 tok/s | N/A | 16.3 GB | 24.2 GB |
| SSD Stream | 100000 | 100.85s | 8.98 tok/s | N/A | 19.7 GB | 27.6 GB |
| TurboQuant | 512 | 0.44s | 28.97 tok/s | N/A | 15.8 GB | 23.7 GB |
| TurboQuant | 40000 | 28.25s | 3.91 tok/s | N/A | 31.7 GB | 39.4 GB |
| TurboQuant | 100000 | 97.28s | 3.93 tok/s | N/A | 49.4 GB | 57.3 GB |
| SSD + TurboQuant | 512 | 1.49s | 11.35 tok/s | N/A | 14.1 GB | 22.0 GB |
| SSD + TurboQuant | 40000 | 44.68s | 2.49 tok/s | N/A | 14.9 GB | 22.5 GB |
| SSD + TurboQuant | 100000 | 143.53s | 1.65 tok/s | N/A | 15.2 GB | 22.3 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
