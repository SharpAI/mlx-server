#!/bin/bash
# test-vision.sh — VLM Integration tests for SwiftLM
#
# Usage:
#   ./tests/test-vision.sh [binary_path] [port]

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15413}"
HOST="127.0.0.1"
MODEL="mlx-community/Qwen2-VL-2B-Instruct-4bit" # CI Small VLM
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[test-vision]${NC} $*"; }
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
log "Starting server: $BINARY --model $MODEL --port $PORT --vision"
"$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" --vision &
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

# ── Test VLM ──────────────────────────────────────────────────────────
mkdir -p /tmp/vision_test
curl -sL "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Statue_of_Liberty%2C_NY.jpg/800px-Statue_of_Liberty%2C_NY.jpg" -o /tmp/vision_test/statue.jpg
BASE64_IMG=$(base64 -i /tmp/vision_test/statue.jpg | tr -d '\n')

COMPLETION=$(curl -sf -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":100,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is this?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${BASE64_IMG}\"}}]}]}")

if echo "$COMPLETION" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    CONTENT=$(echo "$COMPLETION" | jq -r '.choices[0].message.content')
    pass "VLM successfully analyzed statue of liberty image. Output: \"$CONTENT\""
else
    fail "VLM completion failed: $COMPLETION"
    exit 1
fi

rm -rf /tmp/vision_test
exit 0
