# ⚡️ SwiftLM

A blazingly fast, native Swift inference server that serves [MLX](https://github.com/ml-explore/mlx) models with a strict **OpenAI-compatible API**. 

No Python runtime, no Global Interpreter Lock (GIL), no unnecessary memory copies. Just bare-metal Apple Silicon performance compiled to a single binary.

## 🚀 Features

- 🍎 **100% Native Apple Silicon**: Powered natively by Metal and Swift. 
- 🔌 **OpenAI-compatible**: Drop-in replacement for OpenAI SDKs (`/v1/chat/completions`, streaming, etc).
- 🧠 **Smart Model Routing**: Loads HuggingFace format models directly, with native Safetensors parsing.
- ⚡️ **TurboQuantization Integrated**: Custom low-level MLX Metal primitives that apply extremely fast quantization for KV caching out-of-the-box.
- 💾 **SSD Expert Streaming**: *Experimental* zero-copy streaming that swaps Mixture of Experts (MoE) layers directly from the NVMe SSD to the GPU command buffer without trashing macOS Unified Memory (prevents Watchdog OS kernel panics on 122B+ models).
- 🎛️ **Granular Memory Control**: Integrated Layer Partitioning (`--gpu-layers`) and Wisdom Auto-Calibration for squeezing massive models into RAM.

---

## ⚡️ TurboQuantization: KV Cache Compression

`SwiftLM` implements **TurboQuant** (AISTATS/ICLR 2026) for on-the-fly KV cache compression, enabling long-context inference with drastically reduced memory. At 3 bits/coordinate, the KV cache is compressed ~5.8× vs FP16 with near-zero accuracy loss.

The algorithm runs in two stages per KV vector:

**Stage 1 — PolarQuant (2 bits):**
1. Extract L2 norm: `‖x‖`
2. Normalize: `x̂ = x / ‖x‖`
3. Rotate: `y = R @ x̂`  (random orthogonal R via Fast Walsh-Hadamard Transform — O(d log d))
4. Quantize each coordinate to nearest Lloyd-Max centroid (optimal for post-rotation Gaussian distribution)
- → Store: `(2-bit indices[d], float16 norm)`

**Stage 2 — QJL residual (1 bit):**
1. Dequantize Stage 1 → `x̂_mse`
2. Compute residual: `r = x - x̂_mse`
3. Project: `z = S @ r`  (S ~ N(0,1) random matrix)
4. Sign-bit encode: `signs = sign(z) ∈ {+1, -1}`
- → Store: `(1-bit signs[d], float16 residual_norm)`

**Total: 3 bits/coord + 32-bit norm ≈ 5.8× compression vs FP16**

> *K cache uses full TurboQuant (Stage 1 + Stage 2) to preserve attention dot-product accuracy. V cache uses Stage 1 only (PolarQuant MSE) since MSE-optimal reconstruction doesn't need the QJL residual stage.*

Reference implementation: [`turboquant_plus`](https://github.com/TheTom/turboquant_plus) (Python) | Paper: [TurboQuant, AISTATS 2026](https://aistats.org)

---

## 💻 Tested Hardware & Benchmarks

To reliably run massive 122B parameter MoE models over SSD streaming, `SwiftLM` was designed and benchmarked natively on the following hardware:

- **Machine**: MacBook Pro, Apple M5 Pro
- **Memory**: 64 GB Unified Memory
- **Model**: Qwen3.5-122B-A10B-4bit
- **SSD**: Internal Apple NVMe (Zero-Copy Streaming)

> **⚠️ Quantization Disclaimer**: While heavier quantization shrinks the required memory footprint, **4-bit quantization** remains the strict production standard for MoE models. Our metrics indicated that aggressive 2-bit quantization heavily destabilizes JSON grammars—routinely producing broken keys like `\name\` instead of `"name"`—which systematically breaks OpenAI-compatible tool calling.

---

## 🛠️ Quick Start

### Fastest: Download Pre-built Binary
The absolute fastest way to get started is to [download the latest pre-compiled macOS binary](https://github.com/SharpAI/SwiftLM/releases) directly from the Releases page. Just extract it and run!

### Build from Source

```bash
swift build -c release
```

### Run (Downloads model natively on first launch)

```bash
.build/release/SwiftLM \
  --model Qwen3.5-122B-A10B-4bit \
  --stream-experts true \
  --port 5413
```

*(Note: Add `--stream-experts=true` if you are attempting to run oversized MoE models like Qwen3.5 122B to bypass macOS virtual memory swapping!)*

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health + loaded model capabilities |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (LLM and VLM support, multi-turn, system prompts) |

## 💻 Usage Examples

### Chat Completion (Streaming)
Drop-in compatible with standard OpenAI HTTP consumers:
```bash
curl http://localhost:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-122B-A10B-4bit",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are Aegis-AI, a local home security agent. Output strictly in JSON format."},
      {"role": "user", "content": "Clip 1: Delivery person drops package at 14:02. Clip 2: Delivery person walks away down driveway at 14:03. Do these clips represent the same security event? Output a JSON object with a `duplicate` boolean and a `reason` string."}
    ]
  }'
```
---


## ⚙️ CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model ID or local path |
| `--port` | `5413` | Port to listen on |
| `--host` | `127.0.0.1` | Host to bind |
| `--max-tokens` | `2048` | Max tokens limit per generation |
| `--gpu-layers` | `model_default`| Restrict the amount of layers allocated to GPU hardware |
| `--stream-experts` | `false` | Enable experimental SSD streaming for MoE model expert matrices |

## 📦 Requirements

- macOS 14.0+
- Apple Silicon (M1/M2/M3/M4/M5)
- Xcode Command Line Tools
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

## 📄 Dependencies & License

Built entirely on the hard work of the Apple MLX community.
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) — Event-driven Swift HTTP server

### 🙏 TurboQuant Credits

The TurboQuant KV cache compression implemented in `SwiftLM` is directly based on the following open-source work and research:

- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache)** — The primary reference for the C and Metal GPU implementation. The `turbo-wht.h` Fast Walsh-Hadamard kernel, WHT sign arrays (seed=42), Lloyd-Max centroid tables, and the `ggml-turbo-quant.c` quantize/dequantize logic were ported directly from this repository into our MLX C++ and Metal backend.

- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Python reference implementation used to validate the algorithm math, codebook construction (Lloyd's algorithm for N(0, 1/d)), and KV cache integration design.

- **TurboQuant Paper** — *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*, Zandieh et al., AISTATS/ICLR 2026. The two-stage PolarQuant + QJL algorithm described in Section 3 and Appendix A is the mathematical foundation of this implementation.

- **[amirzandieh/QJL](https://github.com/amirzandieh/QJL)** — Original Quantized Johnson-Lindenstrauss (QJL) 1-bit residual correction implementation by the paper authors.

**MIT License**
