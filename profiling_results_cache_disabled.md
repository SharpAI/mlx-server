### `Qwen3.5-122B-A10B-4bit` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| SSD Stream | 512 | 6.66s | 1.58 tok/s | N/A | 11.2 GB | 36.2 GB |
| SSD Stream | 40000 | 164.54s | 1.30 tok/s | N/A | 49.0 GB | 74.3 GB |
| SSD Stream | 100000 | 475.08s | 0.60 tok/s | N/A | 49.4 GB | 73.1 GB |
| SSD + TurboQuant | 512 | 8.99s | 1.54 tok/s | N/A | 11.2 GB | 34.8 GB |
| SSD + TurboQuant | 40000 | 130.95s | 1.09 tok/s | N/A | 18.2 GB | 42.0 GB |
| SSD + TurboQuant | 100000 | 334.97s | 0.69 tok/s | N/A | 27.9 GB | 52.1 GB |
| SSD + 16-Worker Prefetch | 512 | 8.61s | 1.55 tok/s | N/A | 11.2 GB | 35.6 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
