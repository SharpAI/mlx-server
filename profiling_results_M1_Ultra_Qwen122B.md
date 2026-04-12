### `Qwen3.5-122B-A10B-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| SSD Stream | 512 | 11.54s | 1.51 tok/s | N/A | 43.2 GB | 68.3 GB |
| SSD + TurboQuant | 512 | 11.21s | 1.52 tok/s | N/A | 43.2 GB | 68.3 GB |
| SSD + 16-Worker Prefetch | 512 | 8.70s | 1.67 tok/s | N/A | 43.4 GB | 68.5 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
