### `Qwen3.5-122B-A10B-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| SSD Stream | 512 | 11.85s | 1.54 tok/s | N/A | 23.2 GB | 48.1 GB |
| SSD + TurboQuant | 512 | 11.87s | 1.53 tok/s | N/A | 23.2 GB | 48.1 GB |
| SSD + 16-Worker Prefetch | 512 | 9.81s | 1.82 tok/s | N/A | 23.4 GB | 48.1 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
