# ⚡️ mlx-server

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

## 🆚 Why `mlx-server`? (vs. llama.cpp & python mlx-lm)

| Feature | `mlx-server` (Swift) | `llama.cpp` (Metal) | `python mlx-lm` |
| :--- | :--- | :--- | :--- |
| **Backend Math** | Official Apple MLX (Metal) | Custom Metal Shaders | Official Apple MLX |
| **Target Hardware** | Consumer Apple Silicon | Universal (CPU/Mac) | Consumer Apple Silicon |
| **Concurrency / GIL** | 🟢 **Zero GIL** (Swift async) | 🟢 **Zero GIL** (C++) | 🔴 **GIL Bottlenecked** (Python) |
| **Model Format** | Native HF (Safetensors) | GGUF (Requires Conversion) | Native HF (Safetensors) |
| **MoE Memory Footprint**| 🟢 **Direct SSD Streaming** | 🟡 CPU `mmap` Swapping | 🔴 OS Swap (High pressure) |
| **KV Cache** | 🟢 **TurboQuantization** | 🟢 Aggressive Quantization | 🟡 Standard Python Hooks |
| **Dependencies** | None (Single Native Binary) | None (Single Native Binary) | Python Runtime, `pip` |

**The TL;DR:**
- Use **`llama.cpp`** if you prefer GGUF formats and are running cross-platform on Windows/Linux.
- Use **`python mlx-lm`** if you are explicitly prototyping ML code or data science scripts in Python.
- Use **`mlx-server`** if you want the absolute maximum MLX inference performance on macOS for serving an API (e.g. for multi-agent workflows, long-running REST APIs, or local deployment) without the Python GIL blocking simultaneous request streaming.

---

## 💻 Tested Hardware & Benchmarks

To reliably run massive 122B parameter MoE models over SSD streaming, `mlx-server` was designed and benchmarked natively on the following hardware:

- **Machine**: MacBook Pro, Apple M5 Pro
- **Memory**: 64 GB Unified Memory
- **Model**: Qwen3.5-122B-A10B-4bit
- **SSD**: Internal Apple NVMe (Zero-Copy Streaming)

> **⚠️ Quantization Disclaimer**: While heavier quantization shrinks the required memory footprint, **4-bit quantization** remains the strict production standard for MoE models. Our metrics indicated that aggressive 2-bit quantization heavily destabilizes JSON grammars—routinely producing broken keys like `\name\` instead of `"name"`—which systematically breaks OpenAI-compatible tool calling.

---

## 🛠️ Quick Start

### Fastest: Download Pre-built Binary
The absolute fastest way to get started is to [download the latest pre-compiled macOS binary](https://github.com/SharpAI/mlx-server/releases) directly from the Releases page. Just extract it and run!

### Build from Source

```bash
swift build -c release
```

### Run (Downloads model natively on first launch)

```bash
.build/release/mlx-server \
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

## 🛡️ Aegis-AI & System Integration

`mlx-server` is designed to be a completely transparent, drop-in substitution for `llama-server` or cloud VLM gateways within local intelligence platforms like **Aegis-AI**, offering dramatically faster zero-latency inference on macOS instances.

When configuring local inference workflows (e.g., within `~/.aegis-ai/llm-config.json`), apply the following integration details:

### 1. Gateway Emulation
`mlx-server` exposes a fully standard OpenAI-compatible API layer:
- **`GET /health`**: Exposes advanced JSON containing GPU metrics and VRAM allocations.
- **`GET /v1/models`**: Lists actively loaded topologies.
- **`POST /v1/chat/completions`**: Supports both `stream: true` and `stream: false`. Natively handles tool-calls, system prompts, and temperature variables.

### 2. Process Configuration (The Sidecar)
When your agent engine spins up `mlx-server` using standard subprocess management, you must be explicitly aware of the memory requirements for *Mixture of Expert* (MoE) models.

> [!CAUTION]
> **Critical Memory Routing (`--stream-experts`)**  
> When managing MoE models (e.g., `Qwen3.5-122B-A10B` where active parameters are significantly smaller than the total payload), you **must artificially append** the `--stream-experts true` flag to the process arguments. If omitted, macOS will inevitably suffer a `Data Abort` memory-mapping fault when mapping > 100GB of tensors onto the unified GPU hardware.


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

**MIT License**
