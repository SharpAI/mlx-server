### `gemma-4-26b-a4b-it-4bit` — Context & Memory Profile

Context depths tested: 16

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 16 | 0.24s | 26.32 tok/s | N/A | 14.6 GB | 29.6 GB |
| SSD Stream | 16 | 2.55s | 2.90 tok/s | N/A | 3.5 GB | 18.6 GB |
| TurboQuant | 16 | 0.22s | 26.13 tok/s | N/A | 14.6 GB | 29.6 GB |
| SSD + TurboQuant | 16 | 2.71s | 2.86 tok/s | N/A | 3.5 GB | 18.6 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
