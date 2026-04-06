# ⚡️ SwiftLM

A blazingly fast, native Swift inference server that serves [MLX](https://github.com/ml-explore/mlx) models with a strict **OpenAI-compatible API**. 

No Python runtime, no Global Interpreter Lock (GIL), no unnecessary memory copies. Just bare-metal Apple Silicon performance compiled to a single binary.

<p align="center">
  <img src="docs/demo.gif" width="320" alt="SwiftLM Chat iOS demo" />
</p>

---

## 📊 Performance: Gemma 4-26B on Apple Silicon

Benchmark results for `gemma-4-26b-a4b-it-4bit` (26B MoE, 4-bit) on M5 Pro 64 GB.

### Headline Numbers

| Configuration | 512 ctx | 40K ctx | 100K ctx |
|---|---|---|---|
| **Dense/Vanilla** | 34 tok/s · 18.8 GB | 17 tok/s · 52.6 GB | 16 tok/s · 52.1 GB |
| **SSD Stream** | 4.5 tok/s · **7.7 GB** | 4.2 tok/s · 52.1 GB | 3.5 tok/s · 52.1 GB |
| **TurboQuant** | 34 tok/s · 18.6 GB | 7.0 tok/s · **35.0 GB** | 4.1 tok/s · **46.7 GB** |
| **SSD + TurboQuant** | 4.7 tok/s · **7.7 GB** | 2.1 tok/s · **22.7 GB** | 1.4 tok/s · **33.3 GB** |

> Values shown as `generation speed · GPU memory allocated`

**Key takeaways:**
- 🖥️ **8 GB Mac Mini**: SSD Stream runs a 26B model at **4.6 GB Active RAM**
- 📄 **40K context on 24 GB MacBook Pro**: SSD + TurboQuant fits in **22.7 GB**
- 📚 **100K context on 32 GB Mac Studio**: SSD + TurboQuant fits in **33.3 GB** — previously required 64 GB

Run the benchmark on your device:
```bash
./run_benchmark.sh
```

> The interactive script lets you pick a model and context sizes. Results are saved to `profiling_results_<hostname>.md` with a rich console visualization.

---

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

## 💻 Additional Benchmarks

### Prompt Cache & Sliding Window Regression Test
To verify the stability of the prompt cache when interleaving long contexts with sliding window attention (e.g. Gemma 4/Mistral 3), run this extreme test sequence:
```bash
# 1. Start the server with a large sliding-window MoE model
./SwiftLM --model mlx-community/gemma-4-26b-a4b-it-4bit --port 5431 --turbo-kv --stream-experts --ctx-size 16384

# 2. Run the 4-request sliding window regression test:
echo "=== Req 1 (Big 5537t) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
echo "=== Req 2 (Short 18t) ===" && curl -sS --max-time 60 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is today?"}],"max_tokens":30}' 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
echo "=== Req 3 (Big 5537t) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
echo "=== Req 4 (Big Full Cache Hit) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
echo "=== ALL 4 PASSED ==="
```
If you see `ALL 4 PASSED` without `SIGTRAP` or `curl: (52) Empty reply from server`, the memory bounds are stable.

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

## 📱 SwiftLM Chat — iOS App

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
The archive is **self-contained** — `mlx.metallib` is bundled alongside the binary.

```bash
tar -xzf SwiftLM-<version>-macos-arm64.tar.gz
./SwiftLM --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 5413
```

> **⚠️ Metal GPU Error?** If you see `Failed to load the default metallib`, make sure `mlx.metallib` is co-located with the `SwiftLM` binary.

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

Then run:
```bash
.build/release/SwiftLM \
  --model mlx-community/Qwen3.5-122B-A10B-4bit \
  --stream-experts \
  --port 5413
```

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
| `--prefill-size`| `512`  | Prompt prefill chunk size (micro-batching for long contexts) |
| `--gpu-layers` | `model_default`| Restrict the amount of layers allocated to GPU hardware |
| `--stream-experts` | `false` | Enable experimental SSD streaming for MoE model expert matrices |
| `--turbo-kv` | `false` | Enable TurboQuant 3-bit KV cache compression |

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

## 📄 Dependencies & License

Built entirely on the hard work of the Apple MLX community.
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [mlx-lm](https://github.com/ml-explore/mlx/tree/main/mlx_lm) — Python reference implementation for MLX Language Models (inspiration for prompt chunking architecture)
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) — Event-driven Swift HTTP server
- [flash-moe](https://github.com/danveloper/flash-moe) — Reference for SSD Expert Streaming

### 🙏 TurboQuant Credits

The TurboQuant KV cache compression implemented in `SwiftLM` is directly based on the following open-source work and research:

- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache)** — The primary reference for the C and Metal GPU implementation. The `turbo-wht.h` Fast Walsh-Hadamard kernel, WHT sign arrays (seed=42), Lloyd-Max centroid tables, and the `ggml-turbo-quant.c` quantize/dequantize logic were ported directly from this repository into our MLX C++ and Metal backend.

- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Python reference implementation used to validate the algorithm math, codebook construction (Lloyd's algorithm for N(0, 1/d)), and KV cache integration design.

- **TurboQuant Paper** — *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*, Zandieh et al., AISTATS/ICLR 2026. The two-stage PolarQuant + QJL algorithm described in Section 3 and Appendix A is the mathematical foundation of this implementation.

- **[amirzandieh/QJL](https://github.com/amirzandieh/QJL)** — Original Quantized Johnson-Lindenstrauss (QJL) 1-bit residual correction implementation by the paper authors.

**MIT License**
