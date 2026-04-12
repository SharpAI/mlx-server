#!/bin/bash
# test-audio.sh — ALM Integration tests for SwiftLM
#
# Usage:
#   ./tests/test-audio.sh [binary_path] [port]

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15413}"
HOST="127.0.0.1"
MODEL="mlx-community/gemma-4-e4b-it-4bit" # CI Small ALM (ungated, Any-to-Any)
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[test-audio]${NC} $*"; }
pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        log "Stopping server (PID $SERVER_PID)"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Start server ─────────────────────────────────────────────────────
log "Starting server: $BINARY --model $MODEL --port $PORT --audio"
"$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" --audio &
SERVER_PID=$!

log "Waiting for server to be ready (this may take a while on first run)..."
MAX_WAIT=600  # 10 minutes for model download
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf "$URL/health" >/dev/null 2>&1; then
        log "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "Error: Server did not become ready in ${MAX_WAIT}s"
    exit 1
fi

# ── Test ALM ──────────────────────────────────────────────────────────
mkdir -p /tmp/audio_test

cat << 'EOF' > /tmp/audio_test/gen.py
import wave, struct, math
with wave.open('/tmp/audio_test/test.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    for i in range(16000):
        v = int(math.sin(i * 440.0 * 2.0 * math.pi / 16000.0) * 10000.0)
        w.writeframes(struct.pack('<h', v))
EOF
python3 /tmp/audio_test/gen.py

BASE64_AUDIO=$(base64 -i /tmp/audio_test/test.wav | tr -d '\n')

cat <<EOF > /tmp/audio_test/payload.json
{"model":"$MODEL","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"Transcribe this audio strictly."},{"type":"input_audio","input_audio":{"data":"${BASE64_AUDIO}","format":"wav"}}]}]}
EOF

COMPLETION=$(curl -sf -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"/tmp/audio_test/payload.json")

if echo "$COMPLETION" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    CONTENT=$(echo "$COMPLETION" | jq -r '.choices[0].message.content')
    pass "ALM successfully processed audio file. Output: \"$CONTENT\""
else
    fail "ALM completion failed: $COMPLETION"
    exit 1
fi

rm -rf /tmp/audio_test
exit 0
