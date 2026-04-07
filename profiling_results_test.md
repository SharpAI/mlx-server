### `gemma-4-26b-a4b-it-4bit` — Context & Memory Profile

Context depths tested: 512

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 3.61s | 28.16 tok/s | N/A | 15.8 GB | 24.2 GB |
| SSD Stream | 512 | 2.10s | 7.27 tok/s | N/A | 14.1 GB | 22.4 GB |
| TurboQuant | 512 | 0.65s | 21.40 tok/s | N/A | 15.8 GB | 24.2 GB |
| SSD + TurboQuant | 512 | 1.19s | 7.79 tok/s | N/A | 14.1 GB | 22.5 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
