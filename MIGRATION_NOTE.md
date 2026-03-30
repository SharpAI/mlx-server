# 📋 Migration Note: mlx-server → SwiftLM

> **Date**: 2026-03-30  
> **Affects**: All Aegis-AI components that reference the MLX inference sidecar

---

## What Changed

The MLX inference engine previously known as **`mlx-server`** has been renamed to **`SwiftLM`**.

This is a **rename only** — the API, ports, endpoints, and behaviour are 100% identical.

---

## Required Updates in Aegis-AI

### 1. Binary path / process spawn

| Before | After |
|---|---|
| `/path/to/mlx-server` | `/path/to/SwiftLM` |
| `.build/release/mlx-server` | `.build/release/SwiftLM` |

Update any hardcoded binary paths in:
- Engine registry / scheduler config
- IPC process spawn calls (`spawn('mlx-server', ...)`)
- systemd / launchd service definitions (if any)

### 2. GitHub release URL

| Before | After |
|---|---|
| `https://github.com/SharpAI/mlx-server/releases` | `https://github.com/SharpAI/SwiftLM/releases` |
| `mlx-server-*-macos-arm64.tar.gz` | `SwiftLM-*-macos-arm64.tar.gz` |

### 3. Wisdom cache directory

| Before | After |
|---|---|
| `~/.mlx-server/wisdom/` | `~/.swiftlm/wisdom/` |

If the user has existing calibration wisdom at `~/.mlx-server/wisdom/`, migrate it:

```bash
mkdir -p ~/.swiftlm
cp -r ~/.mlx-server/wisdom ~/.swiftlm/wisdom
```

### 4. Log prefix in console output

| Before | After |
|---|---|
| `[mlx-server]` | `[SwiftLM]` |

Update any log parsers or regex filters watching for `[mlx-server]` in stdout.

### 5. UI display name

Any UI label showing `"mlx-server"` or `"MLX Server"` should be updated to `"SwiftLM"`.

---

## What Did NOT Change

| Item | Value |
|---|---|
| Default port | `5413` (unchanged) |
| API endpoints | `/v1/chat/completions`, `/health`, `/v1/models` (unchanged) |
| CLI flags | All flags identical (`--model`, `--stream-experts`, `--turbo-kv`, etc.) |
| Ready event JSON | `{"event":"ready", ...}` (unchanged) |
| OpenAI compatibility | Full (unchanged) |
| Model format (MLX/HF) | Unchanged |

---

## No Action Required If…

- You are using the **OpenAI SDK** pointed at `http://127.0.0.1:5413/v1` — zero changes needed
- You are using **`AEGIS_INTEGRATION.md`** as your integration reference — it has already been updated

---

## Git History Note

All historical commits previously attributed to `Aegis-AI <aegis@sharpai.com>` have been rewritten to `Simba Zhang <solderzzc@gmail.com>`. If you have a local clone of the old `mlx-server` repo, you will need to re-clone `SwiftLM` or run:

```bash
git remote set-url origin https://github.com/SharpAI/SwiftLM.git
git fetch --all
git reset --hard origin/main
```
