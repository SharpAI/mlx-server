# SwiftLM API Response Formats

OpenAI-compatible API running at `http://127.0.0.1:5413`.
No authentication required when started without `--api-key`.

---

## 1. Non-Streaming Chat Completion

**Request:**
```bash
curl -X POST http://127.0.0.1:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 8192,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1711746000,
  "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The full generated text here..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 2048,
    "total_tokens": 2070
  }
}
```

**Capture:**
- Text: `.choices[0].message.content`
- Done signal: `.choices[0].finish_reason` → `"stop"` | `"length"` | `"tool_calls"`

---

## 2. Streaming Chat Completion (SSE)

**Request:**
```bash
curl -X POST http://127.0.0.1:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 8192,
    "stream": true
  }'
```

**Response (one line per token):**
```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":1711746000,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"The "},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"full "},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Capture pattern:**
1. Each line starts with `data: `
2. Strip `data: ` prefix → parse JSON
3. Accumulate `.choices[0].delta.content` (may be empty string on final chunk)
4. Stop when `.choices[0].finish_reason != null` OR line === `"data: [DONE]"`

---

## 3. Health Check

**Request:**
```bash
curl http://127.0.0.1:5413/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
  "memory": {
    "total_system_mb": 65536,
    "active_mb": 3300,
    "peak_mb": 4400,
    "cache_mb": 2085
  },
  "stats": {
    "requests_total": 1,
    "requests_active": 0,
    "avg_tokens_per_sec": 3.81,
    "tokens_generated": 2048
  }
}
```

**Ready check:** `.status === "ok"` and `.stats.requests_active === 0`

---

## 4. Prometheus Metrics

**Request:**
```bash
curl http://127.0.0.1:5413/metrics
```

**Response (plain text Prometheus format):**
```
mlx_server_requests_total 1
mlx_server_requests_active 0
mlx_server_tokens_generated_total 2048
mlx_server_tokens_per_second 3.81
mlx_server_memory_active_bytes 3462052656
mlx_server_memory_peak_bytes 4577196678
mlx_server_memory_cache_bytes 2185709558
mlx_server_uptime_seconds 652
```

**Generation in progress:** `mlx_server_requests_active > 0` AND `mlx_server_tokens_generated_total == 0` → still prefilling

---

## 5. Aegis-AI Integration (OpenAI SDK)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:5413/v1",
  apiKey: "none", // auth disabled on local server
});

// Non-streaming
const response = await client.chat.completions.create({
  model: "mlx-community/Qwen3.5-122B-A10B-4bit",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 8192,
});
const text = response.choices[0].message.content;
const finishReason = response.choices[0].finish_reason; // "stop" | "length"

// Streaming
const stream = await client.chat.completions.create({
  model: "mlx-community/Qwen3.5-122B-A10B-4bit",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 8192,
  stream: true,
});
let fullText = "";
for await (const chunk of stream) {
  fullText += chunk.choices[0]?.delta?.content ?? "";
  if (chunk.choices[0]?.finish_reason) break;
}
```

---

## 6. Available Parameters (per-request overrides)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model ID or local path |
| `messages` | array | required | Chat history |
| `max_tokens` | int | 2048 | Max tokens to generate |
| `stream` | bool | false | Enable SSE streaming |
| `temperature` | float | 0.6 | Sampling temperature (0=greedy) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `repetition_penalty` | float | disabled | Repetition penalty factor |
| `stop` | string[] | [] | Stop sequences |
| `response_format` | object | none | `{"type": "json_object"}` for JSON mode |
| `tools` | array | none | Tool/function definitions |
| `enable_thinking` | bool | false | Enable `<think>` reasoning tokens |

---

## 7. Benchmark Results (M5 Pro 64GB, 2026-03-29)

| Model | Strategy | GPU Footprint | tok/s |
|---|---|---|---|
| Qwen3.5-122B-A10B-4bit | SSD Streaming | 4.4 GB peak | **3.81** |
| Qwen2.5-0.5B-Instruct-4bit | Full GPU | 0.3 GB | ~100 |
