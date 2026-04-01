# ⚡️ SwiftLM

A blazingly fast, native Swift inference server that serves [MLX](https://github.com/ml-explore/mlx) models with a strict **OpenAI-compatible API**. 

No Python runtime, no Global Interpreter Lock (GIL), no unnecessary memory copies. Just bare-metal Apple Silicon performance compiled to a single binary.

<p align="center">
  <img src="docs/demo.gif" width="320" alt="SwiftLM Chat iOS demo" />
</p>

## 🚀 Features

- 🍎 **100% Native Apple Silicon**: Powered natively by Metal and Swift. 
- 🔌 **OpenAI-compatible**: Drop-in replacement for OpenAI SDKs (`/v1/chat/completions`, streaming, etc).
- 🧠 **Smart Model Routing**: Loads HuggingFace format models directly, with native Safetensors parsing.
- ⚡️ **TurboQuantization Integrated**: Custom low-level MLX Metal primitives that apply extremely fast quantization for KV caching out-of-the-box.
- 💾 **SSD Expert Streaming**: *Experimental* zero-copy streaming that swaps Mixture of Experts (MoE) layers directly from the NVMe SSD to the GPU command buffer without trashing macOS Unified Memory (prevents Watchdog OS kernel panics on 122B+ models).
- 🎛️ **Granular Memory Control**: Integrated Layer Partitioning (`--gpu-layers`) and Wisdom Auto-Calibration for squeezing massive models into RAM.

---

## ⚡️ TurboQuantization: KV Cache Compression

`SwiftLM` implements a **hybrid V2+V3 TurboQuant architecture** for on-the-fly KV cache compression. At roughly ~3.6 bits per coordinate overall, the KV cache is compressed ~3.5× vs FP16 with near-zero accuracy loss.

### By combining V2 Speed with V3 Quality:
Recent reproductions of the TurboQuant algorithm (e.g., `turboquant-mlx`) revealed two distinct paths:
1. **V2 (Hardware-Accelerated)**: Fast, but uses linear affine quantization which degrades quality at 3-bit.
2. **V3 (Paper-Correct)**: Excellent quality using non-linear Lloyd-Max codebooks, but painfully slow due to software dequantization.

**We built the "Holy Grail" hybrid:** We ported the V3 non-linear Lloyd-Max codebooks directly into the native C++ encoding path, and process the dequantization natively in fused Metal (`bggml-metal`) shaders. This achieves **V3 quality at V2 speeds**, completely detached from Python overhead.

### The Algorithm:

**K-Cache (3-bit PolarQuant + 1-bit QJL) = 4.25 bits/dim**
1. Extract L2 norm and normalize: `x̂ = x / ‖x‖`
2. Apply Fast Walsh-Hadamard Transform (WHT) rotation to distribute outliers evenly.
3. Quantize each coordinate using **3-bit non-linear Lloyd-Max centroids**.
4. Compute the residual error between the original vector and the quantized approximation.
5. Project the residual via a random Johnson-Lindenstrauss (QJL) matrix and store the 1-bit signs.
*(Why QJL? QJL acts as an additional regularizer that prevents centroid resolution loss from degrading the attention dot-product.)*

**V-Cache (3-bit PolarQuant) = 3.125 bits/dim**
Because the V-cache matrix is not used for inner-product attention scoring, the QJL error correction provides no benefit. We cleanly disable QJL for the V-cache, extracting an additional 25% memory savings without sacrificing quality.

Reference implementations: [`turboquant-mlx`](https://github.com/sharpner/turboquant-mlx) | [`turboquant_plus`](https://github.com/TheTom/turboquant_plus) | Paper: [TurboQuant, Google 2504.19874](https://arxiv.org/abs/2504.19874)

---

## 💻 Tested Hardware & Benchmarks

To reliably run massive 122B parameter MoE models over SSD streaming, `SwiftLM` was designed and benchmarked natively on the following hardware:

- **Machine**: MacBook Pro, Apple M5 Pro
- **Memory**: 64 GB Unified Memory
- **Model**: Qwen3.5-122B-A10B-4bit
- **SSD**: Internal Apple NVMe (Zero-Copy Streaming)

> **⚠️ Quantization Disclaimer**: While heavier quantization shrinks the required memory footprint, **4-bit quantization** remains the strict production standard for MoE models. Our metrics indicated that aggressive 2-bit quantization heavily destabilizes JSON grammars—routinely producing broken keys like `\name\` instead of `"name"`—which systematically breaks OpenAI-compatible tool calling.

---

---

## 📱 SwiftLM Chat — iOS App

A native iPhone & iPad companion app that downloads MLX models directly from HuggingFace and runs inference on-device via MLX Swift.

### Features
- **Tab UI**: Chat · Models · Settings
- **Live download progress** with speed indicator and circular progress ring
- **Model catalog**: Qwen3, Phi-3.5, Mistral, Llama — with on-device RAM fit indicators
- **HuggingFace search** — find any `mlx-community` model by name
- **Context-aware empty states** — downloading ring, loading spinner, idle prompt
- **iOS lifecycle hardened** — model unload only fires on true background (not notification banners); 30-second grace period on app-switch

### Build & Run (iOS)

```bash
cd SwiftLMChat
python3 generate_xcodeproj.py       # Generates SwiftLMChat.xcodeproj
open SwiftLMChat.xcodeproj
```

Then in Xcode:
1. Select the **SwiftLMChat** target → **Signing & Capabilities**
2. Set your **Team** (your Apple Developer account)
3. Select your iPhone as the run destination
4. ⌘R to build and run

> **Note for contributors**: The `.xcodeproj` is git-ignored (it contains your personal Team ID). Run `generate_xcodeproj.py` after cloning to regenerate it locally. Your Team ID is never committed.

---

## 🛠️ Quick Start (macOS Server)

### Fastest: Download Pre-built Binary

Download the latest release tarball from the [Releases page](https://github.com/SharpAI/SwiftLM/releases).
The archive is **self-contained** — `default.metallib` is bundled alongside the binary.

```bash
tar -xzf SwiftLM-<version>-macos-arm64.tar.gz

# Run from the extracted directory — default.metallib must be co-located with the binary
./SwiftLM --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 5413
```

> **⚠️ Metal GPU Error?** If you see `Failed to load the default metallib`, it means `default.metallib` is missing from the directory you are running `SwiftLM` from. Make sure you run the binary **from the extracted folder** and do not move the binary without also moving `default.metallib` alongside it.

### Build from Source

```bash
# Must clone recursively — default.metallib ships inside the mlx-swift submodule
git clone --recursive https://github.com/SharpAI/SwiftLM
cd SwiftLM
swift build -c release
```

`default.metallib` is a pre-built artifact inside the `mlx-swift` submodule, version-matched to the Swift binary. Copy it next to the binary before running:

```bash
cp LocalPackages/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/default.metallib \
   .build/release/

.build/release/SwiftLM \
  --model mlx-community/Qwen3.5-122B-A10B-4bit \
  --stream-experts \
  --port 5413
```

> **⚠️ Do NOT use Python's `mlx-metal` package as a source for `mlx.metallib`.**  
> While `uv run --with mlx-metal python -c "...shutil.copy(metallib, ...)"` will get the server to start, the pip `mlx-metal` package is a **different version** of MLX than what this binary was compiled against. The version mismatch causes GPU kernel ABI corruption during inference, producing a `freed pointer was not the last allocation` crash. Always use the metallib from `LocalPackages/mlx-swift/` — it is the only version-matched artifact for this build.

*(Add `--stream-experts` when running oversized MoE models like Qwen3.5 122B to bypass macOS virtual memory swapping and stream expert layers directly from NVMe.)*

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
- [flash-moe](https://github.com/danveloper/flash-moe) — Reference for SSD Expert Streaming

### 🙏 TurboQuant Credits

The TurboQuant KV cache compression implemented in `SwiftLM` is directly based on the following open-source work and research:

- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache)** — The primary reference for the C and Metal GPU implementation. The `turbo-wht.h` Fast Walsh-Hadamard kernel, WHT sign arrays (seed=42), Lloyd-Max centroid tables, and the `ggml-turbo-quant.c` quantize/dequantize logic were ported directly from this repository into our MLX C++ and Metal backend.

- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Python reference implementation used to validate the algorithm math, codebook construction (Lloyd's algorithm for N(0, 1/d)), and KV cache integration design.

- **TurboQuant Paper** — *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*, Zandieh et al., AISTATS/ICLR 2026. The two-stage PolarQuant + QJL algorithm described in Section 3 and Appendix A is the mathematical foundation of this implementation.

- **[amirzandieh/QJL](https://github.com/amirzandieh/QJL)** — Original Quantized Johnson-Lindenstrauss (QJL) 1-bit residual correction implementation by the paper authors.

**MIT License**
