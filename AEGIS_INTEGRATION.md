# 🛡️ Aegis-AI Integration Guide

`mlx-server` is designed to be a **completely transparent, drop-in replacement** for `llama-server` or any cloud VLM gateway within Aegis-AI, delivering dramatically faster zero-latency inference on Apple Silicon.

---

## 🚀 Quick Start for Aegis-AI

### 1. Download the Binary

Download the latest pre-built binary from the [Releases page](https://github.com/SharpAI/mlx-server/releases) — no Xcode required:

```bash
# Extract and make executable
tar -xzf mlx-server-*-macos-arm64.tar.gz
chmod +x mlx-server
```

### 2. Point Aegis-AI at the Server

In your `~/.aegis-ai/llm-config.json`, set the base URL to the mlx-server endpoint:

```json
{
  "provider": "local",
  "baseUrl": "http://127.0.0.1:5413/v1",
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit"
}
```

### 3. Launch the Sidecar

Aegis-AI should spin up `mlx-server` as a managed subprocess:

```bash
/path/to/mlx-server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 5413
```

The server will emit a machine-readable JSON ready event on stdout when it is ready to accept connections:

```json
{"event":"ready","port":5413,"model":"mlx-community/Qwen2.5-7B-Instruct-4bit","engine":"mlx","vision":false}
```

Aegis-AI should **wait for this event** before routing any requests to the server.

---

## 🧠 Running 122B+ MoE Models (Critical)

If you are running a Mixture of Experts (MoE) model — such as `Qwen3.5-122B-A10B` — you **must** pass the `--stream-experts true` flag.

```bash
/path/to/mlx-server \
  --model mlx-community/Qwen3.5-122B-A10B-4bit \
  --host 127.0.0.1 \
  --port 5413 \
  --stream-experts true
```

> [!CAUTION]
> **Without `--stream-experts true` on MoE models**, macOS will suffer a `Data Abort` kernel-level memory mapping fault when it attempts to load >100GB of weight tensors into Unified Memory simultaneously. The entire machine will freeze and require a hard reboot.

### Why `--stream-experts` Works

MoE models like Qwen3.5-122B have 122B *total* parameters, but only ~10B are **active** on any single forward pass. `mlx-server` exploits this sparsity:

- The 60GB+ of expert weight matrices are `mmap`'d directly from your NVMe SSD
- Only the **2-4 specific expert shards** selected by the router for the current token (~1.5MB each) are streamed into GPU RAM via a zero-copy DMA path
- The remaining experts stay on disk — never touching Unified Memory

The result: a 122B model running stably in ~21GB of RAM on a 64GB M5 Pro.

### Time-To-First-Token (TTFT) Expectations

Due to SSD streaming, TTFT is higher than a fully in-memory model. This is **expected and normal**:

| Prompt Length | Expected TTFT |
|---|---|
| Short (~100 tokens) | 5–15 seconds |
| Medium (~500 tokens) | 30–60 seconds |
| Long (1000+ tokens) | 1–3 minutes |

> [!TIP]
> **Aegis-AI Prompt Cache**: `mlx-server` automatically caches the KV state for repeated system prompts. After the first request with a given system prompt, subsequent requests with the same system prompt will skip the expensive prefill phase and start streaming almost immediately.

---

## 📡 API Reference

`mlx-server` is **fully OpenAI-compatible** — any client using the OpenAI SDK works without modification.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Server health, GPU memory stats, active request count |
| `/v1/models` | `GET` | List loaded models (OpenAI format) |
| `/v1/chat/completions` | `POST` | Chat completions — streaming and non-streaming |
| `/v1/completions` | `POST` | Legacy text completions |
| `/metrics` | `GET` | Prometheus-compatible metrics |

### Health Check

The `/health` endpoint returns detailed telemetry useful for Aegis-AI's system monitor:

```bash
curl http://127.0.0.1:5413/health
```

```json
{
  "status": "ok",
  "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
  "memory": {
    "active_mb": 21272,
    "peak_mb": 23500,
    "cache_mb": 4096,
    "total_system_mb": 65536,
    "gpu_architecture": "Apple M5 Pro"
  },
  "stats": {
    "requests_total": 42,
    "requests_active": 1,
    "tokens_generated": 18500,
    "avg_tokens_per_sec": 3.2
  }
}
```

### Streaming Chat Completion

```bash
curl http://127.0.0.1:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are Aegis-AI, a local home security agent. Always respond in JSON."},
      {"role": "user", "content": "Is the person in this clip a delivery courier?"}
    ]
  }'
```

---

## ⚙️ Full CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model ID or absolute local path |
| `--port` | `5413` | Port to listen on |
| `--host` | `127.0.0.1` | Host interface to bind |
| `--max-tokens` | `2048` | Max generation tokens per request |
| `--ctx-size` | *model default* | KV cache context window size |
| `--temp` | `0.6` | Default sampling temperature (0 = greedy) |
| `--top-p` | `1.0` | Nucleus sampling threshold |
| `--stream-experts` | `false` | **Enable SSD streaming for MoE models** |
| `--thinking` | `false` | Enable reasoning/thinking mode (Qwen3 etc.) |
| `--vision` | `false` | Enable VLM mode for image inputs |
| `--parallel` | `1` | Number of concurrent request slots |
| `--api-key` | *none* | Enable bearer token auth |
| `--cors` | *none* | Allowed CORS origin (`*` for all) |
| `--gpu-layers` | `auto` | Number of layers to run on GPU |
| `--mem-limit` | *system default* | Hard GPU memory cap in MB |
| `--prefill-size` | `512` | Prefill chunk size (lower if GPU watchdog triggers) |
| `--info` | `false` | Dry-run memory profiling report and exit |

---

## 🔍 Memory Behaviour Explained

On Apple Silicon, GPU and system RAM are the **same physical chips** (Unified Memory Architecture). `mlx-server` uses a layered strategy to fit the largest possible models:

| Model Size vs. RAM | Strategy | Notes |
|---|---|---|
| Fits in RAM (<85%) | `full_gpu` | All layers on GPU, maximum speed |
| Slightly over RAM | `swap_assisted` | macOS swap used, 2-4× slowdown |
| 2-4× over RAM | `layer_partitioned` | GPU/CPU split, use `--gpu-layers` |
| MoE > 2× RAM | `ssd_stream` | Use `--stream-experts true` |

You can always inspect the computed memory plan before loading a model:

```bash
mlx-server --model mlx-community/Qwen3.5-122B-A10B-4bit --info
```

---

## 📋 Requirements

- macOS 14.0+
- Apple Silicon (M1 / M2 / M3 / M4 / M5)
- Xcode Command Line Tools (for source builds only)

---

## 🔗 Resources

- [Main README](./README.md) — general usage and benchmarks
- [GitHub Releases](https://github.com/SharpAI/mlx-server/releases) — pre-built binaries
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — underlying MLX framework
