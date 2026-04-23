# DFlash Swift Benchmarking Tools

This directory contains comprehensive benchmarking tools for DFlash speculative decoding.

## Files

### 1. DFlashBenchmark.swift (NEW)
Full end-to-end benchmark comparing baseline vs DFlash performance.

**Features:**
- Compares standard generation vs DFlash speculative decoding
- Multiple block sizes tested per run
- Thermal pressure monitoring
- Automatic cooldown between runs
- Saves detailed JSON results

**Usage:**
```bash
swift run DFlashBenchmark \
  --target mlx-community/Qwen3.5-27B-4bit \
  --draft z-lab/Qwen3.5-27B-DFlash \
  --max-tokens 1024 \
  --block-tokens 8,16,32 \
  --repeat 3 \
  --cooldown 60 \
  --output benchmark/results/my-benchmark.json
```

**Options:**
- `--target`: Target model ID (default: mlx-community/Qwen3.5-27B-4bit)
- `--draft`: Draft model ID (default: z-lab/Qwen3.5-27B-DFlash)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--block-tokens`: Comma-separated block sizes to test (default: 8,16,32)
- `--repeat`: Number of repeat runs (default: 3)
- `--cooldown`: Cooldown seconds between runs (default: 60)
- `--prompt`: Custom prompt text
- `--output`: Output JSON path
- `--verbose` / `-v`: Enable verbose output

**Output Format:**
```json
{
  "hardware": {
    "chip": "Apple M5 Max",
    "memory_gb": 64,
    "mlx_version": "0.21.0",
    "swift_version": "6.0+",
    "device_description": "..."
  },
  "config": {
    "target_model": "mlx-community/Qwen3.5-27B-4bit",
    "draft_model": "z-lab/Qwen3.5-27B-DFlash",
    "max_new_tokens": 1024,
    "block_tokens": [8, 16, 32],
    "repeat": 3,
    "cooldown": 60,
    "prompt": "...",
    "prompt_tokens": 102,
    "git_hash": "abc1234"
  },
  "runs": [
    {
      "run": 1,
      "thermal_pressure": "nominal",
      "baseline": {
        "ttft_ms": 1210.6,
        "generation_tps": 33.3,
        "peak_memory_gb": 15.4,
        "tokens_generated": 1024,
        "prompt_token_count": 102,
        "generation_time_ms": 30750.0
      },
      "dflash": {
        "ttft_ms": 357.3,
        "generation_tps": 78.8,
        "peak_memory_gb": 19.2,
        "tokens_per_cycle": 10.04,
        "cycles": 102,
        "acceptance_ratio": 0.90,
        "acceptance_first_20_avg": 6.6,
        "acceptance_last_20_avg": 7.45,
        "block_tokens": 16,
        "accepted_from_draft": 922
      },
      "speedup": 2.37
    }
  ],
  "summary": {
    "baseline_tps_median": 33.55,
    "dflash_tps_median": 79.02,
    "dflash_tps_min": 78.78,
    "dflash_tps_max": 80.08,
    "speedup_median": 2.37,
    "acceptance_ratio_median": 0.90,
    "total_memory_gb": 19.21
  }
}
```

### 2. DFlashProfiler.swift
Low-level kernel profiler for Metal vs fallback performance.

**Usage:**
```bash
swift run DFlashProfiler
```

**Features:**
- Benchmarks Metal kernel performance
- Compares vs Python reference
- Validates numerical consistency

### 3. DFlashCosSimComparison.swift
Compares intermediate values between Python and Swift implementations.

**Usage:**
```bash
swift run DFlashCompare --dir tests/DFlashComparison/intermediates
```

## Python Comparison

The benchmark format is compatible with `dflash-mlx/benchmark/` results:
- Same JSON structure
- Same metrics (TPS, TTFT, acceptance ratio)
- Same hardware info collection

You can compare Swift vs Python results by loading both JSON files and comparing the `summary` sections.

## Results Directory

Create a `results/` directory here or specify custom output paths:
```bash
mkdir -p tests/DFlashComparison/results
swift run DFlashBenchmark --output tests/DFlashComparison/results/benchmark.json
```

## Performance Tuning Tips

1. **Thermal Throttling**: The benchmark monitors thermal pressure. If you see values other than "nominal", increase `--cooldown` or wait for the chip to cool.

2. **Block Size Selection**: 
   - 8 tokens: Better for shorter prompts
   - 16 tokens: Good balance (default in DFlash paper)
   - 32 tokens: May help for very long contexts

3. **Memory**: DFlash uses more memory due to running both target and draft models. Monitor `peak_memory_gb` in results.

4. **Repeat Count**: Use `--repeat 5` or more for statistically significant results on variable workloads.
