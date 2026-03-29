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

| Feature | `mlx-server` (Swift) | `llama.cpp` (Metal) | `python mlx-lm` | `vLLM` (Flash-MoE) |
| :--- | :--- | :--- | :--- | :--- |
| **Backend Math** | Official Apple MLX (Metal) | Custom Metal Shaders | Official Apple MLX | NVIDIA CUDA / Triton |
| **Target Hardware** | Consumer Apple Silicon | Universal (CPU/Mac) | Consumer Apple Silicon | Datacenter NVIDIA GPUs |
| **Concurrency / GIL** | 🟢 **Zero GIL** (Swift async) | 🟢 **Zero GIL** (C++) | 🔴 **GIL Bottlenecked** (Python) | 🟢 **Zero GIL** (C++/Python) |
| **Model Format** | Native HF (Safetensors) | GGUF (Requires Conversion) | Native HF (Safetensors) | Native HF (Safetensors) |
| **MoE Memory Footprint**| 🟢 **Direct SSD Streaming** | 🟡 CPU `mmap` Swapping | 🔴 OS Swap (High pressure) | 🟢 **Flash-MoE** (High VRAM required) |
| **KV Cache** | 🟢 **TurboQuantization** | 🟢 Aggressive Quantization | 🟡 Standard Python Hooks | 🟢 PagedAttention |
| **Dependencies** | None (Single Native Binary) | None (Single Native Binary) | Python Runtime, `pip` | Heavy CUDA Python Environment |

**The TL;DR:**
- Use **`llama.cpp`** if you prefer GGUF formats and are running cross-platform on Windows/Linux.
- Use **`python mlx-lm`** if you are explicitly prototyping ML code or data science scripts in Python.
- Use **`mlx-server`** if you want the absolute maximum MLX inference performance on macOS for serving an API (e.g. for multi-agent workflows, long-running REST APIs, or local deployment) without the Python GIL blocking simultaneous request streaming.

---

## 💻 Tested Hardware & Benchmarks

To reliably run massive 122B parameter MoE models over SSD streaming, `mlx-server` was designed and benchmarked on the following hardware:

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB Unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, 17.5 GB/s sequential read (measured)
- **OS**: macOS 26.2 (Darwin 25.2.0)

> **⚠️ Quantization Disclaimer**: While heavier quantization shrinks the required memory footprint, **4-bit quantization** remains the strict production standard for MoE models. Our metrics indicated that aggressive 2-bit quantization heavily destabilizes JSON grammars—routinely producing broken keys like `\name\` instead of `"name"`—which systematically breaks OpenAI-compatible tool calling.

---

## 🛠️ Quick Start

### Build

```bash
swift build -c release
```

### Run (Downloads model natively on first launch)

```bash
.build/release/mlx-server \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
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
    "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "stream": true,
    "messages": [{"role": "user", "content": "Explain the speed of light."}]
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
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

## 📄 Dependencies & License

Built entirely on the hard work of the Apple MLX community.
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) — Event-driven Swift HTTP server 

**MIT License**
