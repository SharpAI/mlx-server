# ⚡️ SwiftLM

A blazingly fast, native Swift inference server that serves [MLX](https://github.com/ml-explore/mlx) models with a strict **OpenAI-compatible API**. 

No Python runtime, no Global Interpreter Lock (GIL), no unnecessary memory copies. Just bare-metal Apple Silicon performance compiled to a single binary.

<p align="center">
  <a href="https://youtu.be/E9vR5FREhMg"><img src="docs/mac_demo.gif" width="720" alt="SwiftLM Mac macOS demo" /></a>
</p>
<br>
<p align="center">
  <img src="docs/demo.gif" width="320" alt="SwiftBuddy iOS demo" />
</p>

---

## 🏁 Getting Started

### Fastest: Download Pre-built Binary

Download the latest release tarball from the [Releases page](https://github.com/SharpAI/SwiftLM/releases).
The archive is **self-contained** — `mlx.metallib` is bundled alongside the binary.

```bash
tar -xzf SwiftLM-<version>-macos-arm64.tar.gz
./SwiftLM --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 5413
```

### Build from Source

The build script handles everything: submodules, cmake, Metal kernel compilation, and the Swift build.

```bash
git clone --recursive https://github.com/SharpAI/SwiftLM
cd SwiftLM
./build.sh
```

This will:
1. Initialize git submodules
2. Install `cmake` via Homebrew (if not already installed)
3. Compile `mlx.metallib` from the Metal kernel sources
4. Build the `SwiftLM` binary in release mode

Then start the server (models download automatically if not cached):
```bash
.build/release/SwiftLM \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --port 5413
```

*(Add `--stream-experts` when running oversized MoE models to bypass macOS virtual memory swapping and stream expert layers directly from NVMe SSD.)*

## 📊 Performance: Gemma 4-26B on Apple Silicon

Benchmark results for `gemma-4-26b-a4b-it-4bit` (26B MoE, 4-bit) on M5 Pro 64 GB.

### Headline Numbers

| Configuration | 512 ctx | 40K ctx | 100K ctx |
|---|---|---|---|
| **Dense/Vanilla** | 33.0 tok/s · 23.4 GB | 20.2 tok/s · 57.0 GB | 15.7 tok/s · 56.7 GB |
| **SSD Stream** | 10.8 tok/s · **22.2 GB** | 10.4 tok/s · **24.2 GB** | 9.0 tok/s · **27.6 GB** |
| **TurboQuant** | 29.0 tok/s · 23.7 GB | 3.9 tok/s · 39.4 GB | 3.9 tok/s · 57.3 GB |
| **SSD + TurboQuant** | 11.4 tok/s · **22.0 GB** | 2.5 tok/s · **22.5 GB** | 1.6 tok/s · **22.3 GB** |

> Values shown as `generation speed · GPU memory allocated`

**Key takeaways:**
- 🚀 **Speed Doubled**: The newer MLX backend modifications have more than doubled raw `SSD Stream` inference speed (from 4.5 -> **10.8 tok/s**) while maintaining streaming stability.
- 📄 **40K context on 24 GB MacBook Pro**: SSD + TurboQuant effortlessly fits a 26B model in **22.5 GB** of memory footprint.
- 📚 **100K context on 24 GB MacBook Pro**: Due to hyper-efficient 3-bit KV compression paired with SSD weight streaming, you can process 100,000 tokens of context on a 24 GB machine — only utilizing **22.3 GB** total. (Previously required a 64 GB Mac Studio).

> Run `./run_benchmark.sh` to generate these metrics on your own device. (See **Benchmarks & Testing** below).

---

## 🚀 Features

- 🍎 **100% Native Apple Silicon**: Powered natively by Metal and Swift. 
- 🔌 **OpenAI-compatible**: Drop-in replacement for OpenAI SDKs (`/v1/chat/completions`, streaming, etc).
- 🧠 **Smart Model Routing**: Loads HuggingFace format models directly, with native Safetensors parsing.
- 👁️ **Vision-Language Models (VLM)**: Native multimodal vision processing natively on Metal via the `--vision` flag, supporting real-time base64 image parsing (e.g., Qwen2-VL, PaliGemma).
- 🎧 **Audio-Language Models (ALM)**: High-performance audio ingestion via the `--audio` flag, decoding OpenAI-spec `input_audio` payloads with AVFoundation WAV extraction.
- ⚡️ **TurboQuantization Integrated**: Custom low-level MLX Metal primitives that apply extremely fast quantization for KV caching out-of-the-box.
- 💾 **SSD Expert Streaming (10x)**: High-performance NVMe streaming that loads Mixture of Experts (MoE) layers directly from SSD to GPU — engineered by [@ericjlake](https://github.com/ericjlake), achieving **10x speedup** (0.58 → 5.91 tok/s) on 122B+ models with only ~10 GB resident memory. Uses cross-projection batching, concurrent pread (QD=24), asyncEval pipeline, and runtime top-k expert selection.
- 🔮 **Speculative Decoding**: Load a small draft model (e.g. 9B) alongside a large main model to generate candidate tokens and verify in bulk — accelerating in-RAM inference.
- 🎛️ **Granular Memory Control**: Integrated Layer Partitioning (`--gpu-layers`) and Wisdom Auto-Calibration for squeezing massive models into RAM.

---

## 🧠 Supported Models & Methodologies

`SwiftLM` dynamically maps Apple MLX primitives to standard HuggingFace architectures, enabling complete support for the latest frontier open-weights models across modalities (Text, Vision, Audio).

### Text (LLMs)
- **Gemma 4**: Fully supports both Dense (`gemma-4-e4b`) and Sparse Mixture of Experts (MoE) architectures (`gemma-4-26b`, `gemma-4-31b`).
- **Qwen 2.5 & 3**: Robust support for sliding window attention limits and custom RoPE scaling.
- **Mistral & Mixtral**: Out-of-the-box structural mappings.
- **Phi-3 & Phi-3.5**: Full 128k context parsing via Swift chunked-prefill.

### Vision (VLMs)
*Run with `--vision` flag.*
- **Qwen2-VL & Qwen3-VL**: Real-time positional bounding and Metal image scaling.
- **PaliGemma / LFM2-VL / Pixtral**: Base64 spatial decomposition.

### Audio (ALMs)
*Run with `--audio` flag.*
- **Qwen2-Audio (7B-Instruct)**: Deep multi-modal spectrogram processing via Swift audio interleaving.
- **Gemma-4 Audio Pipelines**: Ready for Audio-in/Text-out variants mapping `.audio_tower` extraction parameters natively off NVMe.

---

## 📱 SwiftBuddy — iOS App

A native iPhone & iPad companion app that downloads MLX models directly from HuggingFace and runs inference on-device via MLX Swift.

### Features
- **Tab UI**: Chat · Models · Settings
- **Live download progress** with speed indicator and circular progress ring
- **Model catalog**: Qwen3, Phi-3.5, Mistral, Llama — with on-device RAM fit indicators
- **HuggingFace search** — find any `mlx-community` model by name
- **Context-aware empty states** — downloading ring, loading spinner, idle prompt
- **iOS lifecycle hardened** — model unload only fires on true background (not notification banners); 30-second grace period on app-switch

> 📱 **Running live on iPhone 13 Pro (6 GB)** — no Python, no server, no GIL. Pure on-device MLX inference via Metal GPU.

### Build & Run (iOS)

```bash
cd SwiftBuddy
python3 generate_xcodeproj.py       # Generates SwiftBuddy.xcodeproj
open SwiftBuddy.xcodeproj
```

Then in Xcode:
1. Select the **SwiftBuddy** target → **Signing & Capabilities**
2. Set your **Team** (your Apple Developer account)
3. Select your iPhone as the run destination
4. ⌘R to build and run

> **Note for contributors**: The `.xcodeproj` is git-ignored (it contains your personal Team ID). Run `generate_xcodeproj.py` after cloning to regenerate it locally. Your Team ID is never committed.

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

## 💾 SSD Expert Streaming: 10x MoE Speedup

SwiftLM implements a **rewritten SSD expert streaming pipeline** (engineered by [Eric Lake](https://github.com/ericjlake)) that achieves 10x generation speedup for massive Mixture of Experts (MoE) models running on memory-constrained Apple Silicon. This enables running models like **Qwen3.5-122B** (69.6 GB) and **Qwen3.5-397B** (209 GB) on a **64 GB Mac** by streaming expert weights from NVMe SSD.

### Benchmark Results (M1 Ultra 64GB, Qwen3.5-122B-A10B-4bit)

| Configuration | tok/s | vs. Original | Notes |
|---|---|---|---|
| Original `--stream-experts` | 0.58 | baseline | Sequential pread, 1 NVMe queue |
| **This PR (top-k=8, full quality)** | **4.95** | **8.5×** | All 8 experts evaluated |
| **This PR (top-k=6, default)** | **5.20** | **9.0×** | Recommended default |
| **This PR (top-k=4, speed mode)** | **5.91** | **10.2×** | Best quality/speed tradeoff |
| **This PR (top-k=2, turbo mode)** | **6.52** | **11.2×** | Still coherent output |

> Memory stable at **~10.6 GB resident**, no swap activity. Tested over 200-token generation runs.

### The Approach: Small Model Helps Large Model

A novel aspect of this architecture is the **dual-model speculative decoding** pattern: a small draft model (e.g. Qwen3.5-9B at 73 tok/s) runs **entirely in RAM** while the large MoE model (e.g. 122B) streams experts from SSD. The draft model generates candidate tokens at high speed, and the main model verifies them in bulk — dramatically reducing the number of SSD-bound generation rounds needed.

> **Important finding:** Speculative decoding is **counterproductive for SSD-streaming MoE** specifically. The verify pass sends N+1 tokens, each routing to *different* experts — SSD I/O scales with the *union* of all positions' expert selections. Speculative decoding is therefore routed exclusively to **in-RAM models**.

### Optimization Techniques

1. **Cross-Projection Batching**: Collapses ~1,400 per-expert `eval()` calls down to ~48 per token by orchestrating gate/up/down projections together in `SwitchGLU`.
2. **Concurrent NVMe pread (QD=24)**: Replaces sequential pread with `DispatchQueue.concurrentPerform`, saturating the NVMe controller's queue depth (8 experts × 3 projections = 24 parallel reads).
3. **AsyncEval Pipeline with Speculative Pread**: Overlaps GPU compute with SSD I/O — uses previous-token routing to speculatively pre-load experts for the next token during the GPU async window (~70% hit rate). Only missed experts (~30%) require on-demand pread after routing sync.
4. **Persistent Metal Buffers**: Expert weight buffers are allocated once per `SwitchGLU` layer and reused across tokens, eliminating per-token allocation overhead.
5. **Runtime Top-K Expert Selection**: The `SWIFTLM_TOP_K` environment variable reduces the number of active experts per token at runtime without model recompilation — trading marginal quality for significant speed gains.

### Key Engineering Findings

| Finding | Detail |
|---|---|
| **GPU compute is the bottleneck** | At steady state, GPU compute is ~190ms of ~200ms per-token time. The OS page cache serves ~90% of expert reads from RAM. |
| **Don't cache experts in application memory** | An LRU expert cache *stole* from the OS page cache and regressed performance (4.84 → 4.01 tok/s). Let the kernel manage it. |
| **MambaCache requires checkpoint rollback** | Unlike attention KV caches (trim = decrement offset), Mamba's recurrent state integrates all history and cannot be partially undone. We implemented `checkpoint()`/`restore()` for speculative decoding on hybrid Attention+Mamba architectures (Qwen3.5). |

### Usage

```bash
# Standard SSD streaming (recommended, top-k=6):
SWIFTLM_TOP_K=6 SwiftLM --port 8002 \
  --model <path>/Qwen3.5-122B-A10B-4bit --stream-experts

# Speed mode (top-k=4):
SWIFTLM_TOP_K=4 SwiftLM --port 8002 \
  --model <path>/Qwen3.5-122B-A10B-4bit --stream-experts

# With speculative decoding (in-RAM models only):
SwiftLM --port 8002 \
  --model <path>/Qwen3.5-27B-4bit \
  --draft-model <path>/Qwen3.5-9B-4bit \
  --num-draft-tokens 4
```

---

## 🔀 Why We Forked Apple MLX

To achieve the extreme memory efficiency and speeds seen in **SSD Expert Streaming** and **Speculative Decoding**, `SwiftLM` relies on custom C++ primitives that bypass standard unified memory limits.

> [!NOTE]
> We maintain custom forks (`SharpAI/mlx` and `SharpAI/mlx-c`) to support **out-of-core memory-mapped execution**, streaming tensor blocks directly from the SSD (NVMe) to the GPU via custom Metal kernels (`ssd_streamer.mm` and `fence.air`). Official `ml-explore` repositories do not yet support this out-of-the-box.

For a detailed breakdown on repository architecture, upstream synchronization, our specific custom patches, and the specific indications for when we can safely revert to Apple's native upstream, read the full documentation: 
👉 **[Upstream MLX Synchronization & SSD Streaming Maintenance](.agents/workflows/mlx-upstream-sync.md)**

---

## 💻 Benchmarks & Testing

Run our automated benchmark suites via the interactive script:
```bash
./run_benchmark.sh
```

The script provides an interactive menu to select any model and run one of two automated testing suites:

### Test 1: Automated Context & Memory Profile (TPS & RAM matrix)
Tests generation speed (TPS) and granular Apple Metal GPU memory allocation across extreme context lengths (e.g., `512, 40000, 100000` tokens).
- Iterates over 4 configurations: Vanilla, SSD Streaming, TurboQuant, and SSD + TurboQuant.
- Generates a rich ANSI console visualization with bar charts and a configuration scoreboard.
- Saves the complete results matrix to `docs/profiling/profiling_results_<hostname>.md`.

### Test 2: Prompt Cache & Sliding Window Regression Test
Verifies the stability of the engine's KV prompt cache when interleaving long contexts with sliding window attention bounds.
- Automatically spins up an isolated background inference server instance.
- Generates a 5,000+ token mock JSON payload.
- Fires an extreme alternating sequence of 4 concurrent requests (`5537t` → `18t` → `5537t` → `Big Full Cache Hit`).
- Confirms the memory bounds remain stable without throwing $O(N^2)$ OS memory warnings, $OOM$ exceptions, or `SIGTRAP` errors.

### Throughput & Inference Memory Profile
Tested by rendering exactly 20 tokens under standard conversational evaluation (`--prefill-size 512`) to capture precise Token Generation (TPS) and Apple Metal memory footprint limits:

| Model | Time To First Token (s) | Generation Speed (tok/s) | Peak GPU Memory (GB) |
|---|---|---|---|
| `gemma-4-e2b-it-4bit` | 0.08s | 116.27 tok/s | 1.37 GB |
| `gemma-4-e4b-it-8bit` | 0.33s | 48.21 tok/s | 7.64 GB |
| `gemma-4-26b-a4b-it-4bit` | 0.14s | 85.49 tok/s | 13.46 GB |
| `gemma-4-31b-it-4bit` | 0.55s | 14.82 tok/s | 16.83 GB |

To run the automated suite on your machine for these models, execute:
```bash
python3 tests/run_4models_benchmark.py
```

> **🧠 How it works:** SwiftLM implements **Chunked Prefill** (controlled via `--prefill-size`, defaulting to 512). This is functionally equivalent to `llama.cpp`'s `--batch-size` parameter and mirrors the [`mlx-lm` Python library](https://github.com/ml-explore/mlx/tree/main/mlx_lm)'s reference implementation approach to preventing $O(N^2)$ Unified Memory over-allocation during massive sequence parsing.

> **⚠️ Quantization Disclaimer**: While heavier quantization shrinks the required memory footprint, **4-bit quantization** remains the strict production standard for MoE models. Our metrics indicated that aggressive 2-bit quantization heavily destabilizes JSON grammars—routinely producing broken keys like `\name\` instead of `"name"`—which systematically breaks OpenAI-compatible tool calling.

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
    "model": "gemma-4-26b-a4b-it-4bit",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are Aegis-AI, a local home security agent. Output strictly in JSON format."},
      {"role": "user", "content": "Clip 1: Delivery person drops package at 14:02. Clip 2: Delivery person walks away down driveway at 14:03. Do these clips represent the same security event? Output a JSON object with a `duplicate` boolean and a `reason` string."}
    ]
  }'
```
---

### Vision-Language Models (VLM)
To run a vision model (e.g., `mlx-community/Qwen2-VL-2B-Instruct-4bit`), launch SwiftLM with the `--vision` flag:
```bash
./.build/release/SwiftLM --model mlx-community/Qwen2-VL-2B-Instruct-4bit --vision
```

You can then pass standard OpenAI base64 encoded images directly. SwiftLM handles hardware spatial-mapping natively via Metal:
```bash
curl http://localhost:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the contents of this image."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."}}
        ]
      }
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
| `--vision` | `false` | Enable VLM (vision-language model) mode for image inputs |
| `--audio` | `false` | Enable ALM (audio-language model) mode for audio inputs |
| `--max-tokens` | `2048` | Max tokens limit per generation |
| `--prefill-size`| `512`  | Prompt prefill chunk size (micro-batching for long contexts) |
| `--top-p` | `1.0` | Default top-p nucleus sampling (overridable per-request) |
| `--top-k` | `50` | Default top-k sampling (0 disables, overridable per-request) |
| `--min-p` | `0.0` | Default min-p sampling threshold relative to the highest probability token (0 disables) |
| `--gpu-layers` | `model_default`| Restrict the amount of layers allocated to GPU hardware |
| `--stream-experts` | `false` | Enable SSD expert streaming for MoE models (10x speedup) |
| `--turbo-kv` | `false` | Enable TurboQuant 3-bit KV cache compression (activates after 2048 tokens, server-wide) |
| `--draft-model` | (none) | Draft model path/ID for speculative decoding (in-RAM models only) |
| `--num-draft-tokens` | `4` | Number of draft tokens per speculation round |

## 🔧 Per-Request API Parameters

In addition to the standard OpenAI fields (`temperature`, `top_p`, `max_tokens`, etc.), SwiftLM accepts the following **SwiftLM-specific** fields on `POST /v1/chat/completions`:

| Field | Type | Description |
|---|---|---|
| `kv_bits` | `int` (4 or 8) | Enable **MLX-native quantized KV cache** for this request. Uses `QuantizedKVCache` (standard group quantization) instead of `KVCacheSimple`. Separate from `--turbo-kv`. Reduces KV memory ~2–4× at mild quality cost. |
| `enable_thinking` | `bool` | Force-enable or disable chain-of-thought thinking blocks for Gemma-4 / Qwen3. |
| `kv_group_size` | `int` | Group size for `kv_bits` quantization (default: `64`). |
| `top_k` | `int` | Per-request top-k sampling override (0 = disabled). |
| `min_p` | `float` | Per-request min-p sampling threshold (0 = disabled). |
| `repetition_penalty` | `float` | Token repetition penalty (e.g. `1.15`). |

### `kv_bits` vs `--turbo-kv` — What's the difference?

| | `kv_bits` (per-request) | `--turbo-kv` (server flag) |
|---|---|---|
| **Scope** | Per-request, sent in JSON body | Server-wide, set at startup |
| **Algorithm** | MLX-native group quantization (4-bit / 8-bit) | Custom 3-bit PolarQuant + QJL Walsh-Hadamard |
| **Activation** | From token 0 | After 2048 tokens |
| **Memory savings** | ~2–4× vs FP16 | ~3.5× vs FP16 |
| **Use case** | Targeted memory reduction per conversation | Extreme long-context (100K+) compression |

### Example: Enable 4-bit KV cache per request
```bash
curl http://localhost:5413/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma-4-26b-a4b-it-4bit",
    "kv_bits": 4,
    "messages": [
      {"role": "user", "content": "Summarize the history of computing in 3 sentences."}
    ]
  }'
```

## 📦 Requirements

- macOS 14.0+
- Apple Silicon (M1/M2/M3/M4/M5)
- Xcode Command Line Tools
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

## 📖 The "Aha!" Moment

**The "2+2=4" Aha Moment**: During development, we encountered a severe "silent failure" where the model would successfully load and evaluate all 32 layers at high speed, but generate nothing but infinite whitespace. The model logits showed the correct *shape* but the wrong *magnitudes*. 

The breakthrough arrived when we realized the **embedding scale** was missing. The Gemma architecture requires scaling embedding outputs by `sqrt(hidden_size)`. For a hidden size of 2816, missing this meant every activation in the network was ~53x too small! By adding one single math operation:
`h = h * MLXArray(Float(config.hiddenSize).squareRoot())`

The model instantly woke up from "whispering" whitespace and successfully responded to `"What is 2+2?"` with a perfect `"2 + 2 equals 4."` — proving that the entire massive structural pipeline from Swift to Metal was working.

## 🙏 Acknowledgments & Credits

[![Awesome MLX](https://img.shields.io/badge/Awesome-MLX-blue?style=flat-square)](https://github.com/raullenchai/awesome-mlx)

`SwiftLM` leverages the powerful foundation of the Apple MLX community and relies heavily on the open-source ecosystem. While the custom C++ implementations, Metal optimizations, and high-performance pipeline architecture were engineered natively for this engine, we owe massive thanks to the following projects and contributors for their indispensable reference materials and underlying protocols:

### Contributors

- **[Eric Lake](https://github.com/ericjlake)** — Engineered the **SSD Expert Streaming 10x rewrite** ([PR #26](https://github.com/SharpAI/SwiftLM/pull/26)), achieving 10× generation speedup on 122B+ MoE models via cross-projection batching, concurrent NVMe pread (QD=24), asyncEval pipeline with speculative pread, and runtime top-k expert selection. Also implemented the **speculative decoding infrastructure** with `DraftModelRef`, dual-model loading, and **MambaCache checkpoint/restore** for hybrid Attention+Mamba architectures.

### Projects & References

- **[mlx-swift](https://github.com/ml-explore/mlx-swift)** — The core Apple MLX wrapper bringing Metal-accelerated operations into the Swift ecosystem.
- **[mlx-lm](https://github.com/ml-explore/mlx/tree/main/mlx_lm)** — The official Python language models implementation, serving as the core inspiration for our chunked-prefill architecture and attention manipulation logic.
- **[flash-moe](https://github.com/danveloper/flash-moe)** — Inspired the memory-mapped out-of-core SSD Expert Streaming mechanics that we implemented natively in SwiftLM.
- **[Hummingbird](https://github.com/hummingbird-project/hummingbird)** — The incredible event-driven Swift HTTP engine powering the OpenAI-compatible REST API.
- **[TurboQuant Paper](https://arxiv.org/abs/2504.19874)** — *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"* (Zandieh et al., AISTATS 2026). Provided the initial algorithmic framework for the dual-stage PolarQuant + QJL engine.
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache)** — Served as an invaluable reference architecture for the C and GPU quantization tables, guiding the development of our native `turbo-wht` Walsh-Hadamard kernels and custom Metal wrapper layers.
- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Essential Python validation logic used to certify the correctness of our manually constructed Lloyd-Max codebook generation math.
- **[amirzandieh/QJL](https://github.com/amirzandieh/QJL)** — The original 1-bit residual correction engine backing the paper, which informed our QJL error recovery in dot-product regimes.

---
**License**: MIT
