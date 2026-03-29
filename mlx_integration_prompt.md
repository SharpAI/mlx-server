# Aegis-AI Integration Prompt: mlx-server Backend

Aegis-AI can seamlessly substitute `llama-server` or cloud VLM gateways with `mlx-server` for dramatically faster zero-latency inference on macOS instances.

When configuring local inference within the `~/.aegis-ai/llm-config.json` configuration layer, apply the following integration details:

## 1. Gateway Emulation
`mlx-server` exposes a fully standard OpenAI-compatible API layer. 
- **Endpoint**: `http://127.0.0.1:5413/v1` (or whatever `port` is provided)
- **Supported Routes**:
  - `GET /health` (Exposes advanced JSON containing GPU metrics/VRAM allocations)
  - `GET /v1/models`
  - `POST /v1/chat/completions` (Supports both `stream: true` and `stream: false`, natively handles tool-calls, system prompts, and temperature checks)

## 2. Process Configuration (The Sidecar)

The Aegis-AI deployment engine should spin up the `mlx-server` sidecar using standard subprocess management. 

### Launch Command
```bash
/path/to/mlx-server/.build/release/mlx-server \
  --model {MODEL_IDENTIFIER_OR_PATH} \
  --host 127.0.0.1 \
  --port {PORT_NUMBER}
```

### 🧠 Critical Memory Routing Flag (`--stream-experts`)
When managing MoE models (e.g. `Qwen3.5-122B-A10B` or any model where active parameters (`A10B`) are substantially smaller than totally memory payload), Aegis-AI **must artificially append** the `--stream-experts true` flag to the process arguments. 
If this flag is omitted, macOS will inevitably suffer a `Data Abort` memory-mapping fault when loading > 100GB of tensors onto the unified GPU hardware. 

Example:
```bash
.build/release/mlx-server --model mlx-community/Qwen3.5-122B-A10B-4bit --stream-experts true --port 5413
```


_You can safely feed this document block sequentially into the Aegis zero-assumption onboarding agent to establish permanent MLX local pipelines._
