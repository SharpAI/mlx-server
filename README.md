# mlx-server

A native Swift server that serves [MLX](https://github.com/ml-explore/mlx) models with an **OpenAI-compatible API**. No Python runtime required — compiles to a single binary that runs on Apple Silicon.

## Features

- 🚀 **Native Swift** — compiled binary, no Python dependency
- 🍎 **Apple Silicon optimized** — uses Metal GPU via MLX
- 🔌 **OpenAI-compatible API** — drop-in replacement for local inference
- 📡 **Streaming support** — SSE streaming for real-time token generation
- 🤗 **HuggingFace models** — loads any MLX-format model directly

## Quick Start

```bash
# Build
swift build -c release

# Run (downloads model on first launch)
.build/release/mlx-server \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --port 5413
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health + loaded model |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (streaming & non-streaming) |

## Usage Examples

```bash
# Health check
curl http://localhost:5413/health

# Chat completion
curl http://localhost:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model ID or local path |
| `--port` | `5413` | Port to listen on |
| `--host` | `127.0.0.1` | Host to bind |
| `--max-tokens` | `2048` | Max tokens per request |

## Metal Shader Library

MLX requires `mlx.metallib` to be co-located with the binary for GPU compute. If you encounter a "Failed to load the default metallib" error:

```bash
# Extract from official MLX Python package
python3 -m venv /tmp/mlx_venv
/tmp/mlx_venv/bin/pip install mlx
cp /tmp/mlx_venv/lib/python3.*/site-packages/mlx/lib/mlx.metallib .build/release/
```

## Requirements

- macOS 14.0+
- Apple Silicon (M1/M2/M3/M4/M5)
- Xcode Command Line Tools
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — Language model support
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) — Swift HTTP server
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) — CLI argument parsing

## License

Apache 2.0
