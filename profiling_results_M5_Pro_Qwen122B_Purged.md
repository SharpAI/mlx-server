### `Qwen3.5-122B-A10B-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| SSD Stream | 512 | 10.32s | 1.54 tok/s | N/A | 43.2 GB | 69.5 GB |
| SSD + TurboQuant | 512 | 10.74s | 1.55 tok/s | N/A | 43.2 GB | 69.5 GB |
| SSD + 16-Worker Prefetch | 512 | 8.62s | 1.71 tok/s | N/A | 43.4 GB | 69.4 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
