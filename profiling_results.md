### `gemma-4-26b-a4b-it-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 0.67s | 30.90 tok/s | N/A | 16.0 GB | 41.5 GB |
| SSD Stream | 512 | 4.08s | 4.41 tok/s | N/A | 12.0 GB | 37.2 GB |
| TurboQuant | 512 | 0.46s | 30.96 tok/s | N/A | 16.1 GB | 41.3 GB |
| SSD + TurboQuant | 512 | 4.01s | 4.45 tok/s | N/A | 12.0 GB | 37.4 GB |
| SSD + 16-Worker Prefetch | 512 | 3.17s | 4.48 tok/s | N/A | 12.1 GB | 37.2 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
