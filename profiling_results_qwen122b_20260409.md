### `Qwen3.5-122B-A10B-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| SSD Stream | 512 | 299.66s | 0.01 tok/s | N/A | 64.3 GB | 88.2 GB |
| SSD + TurboQuant | 512 | 9.46s | 3.00 tok/s | N/A | 11.2 GB | 34.9 GB |
| SSD + 16-Worker Prefetch | 512 | 5.95s | 3.80 tok/s | N/A | 11.2 GB | 34.9 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
