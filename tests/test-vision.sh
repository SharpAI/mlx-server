#!/bin/bash
# test-vision.sh — VLM Integration tests for SwiftLM
#
# Usage:
#   ./tests/test-vision.sh [binary_path] [port]

set -eo pipefail

BINARY="${1:-.build/release/SwiftLM}"
BASE_PORT="${2:-15413}"
HOST="127.0.0.1"
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

wait_for_server() {
    local url="$1"
    local max_wait=600
    log "Waiting for server to be ready (this may take a while on first run)..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf "$url/health" >/dev/null 2>&1; then
            log "Server ready after ${i}s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: Server process died"
            return 1
        fi
        sleep 1
    done
    echo "Error: Server did not become ready in ${max_wait}s"
    return 1
}

run_case() {
    local model="$1"
    local port="$2"
    local use_vision_flag="$3"
    local url="http://${HOST}:${port}"
    local extra_args=()

    if [ "$use_vision_flag" = "yes" ]; then
        extra_args+=(--vision)
    fi

    log "Starting server: $BINARY --model $model --port $port ${extra_args[*]:-}"
    "$BINARY" --model "$model" --port "$port" --host "$HOST" "${extra_args[@]}" &
    SERVER_PID=$!

    wait_for_server "$url"

    local completion
    completion=$(curl -sf -X POST "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"max_tokens\":100,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is the image?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${BASE64_IMG}\"}}]}]}")

    if echo "$completion" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        local content
        content=$(echo "$completion" | jq -r '.choices[0].message.content')
        pass "$model returned VLM output: \"$content\""
    else
        fail "$model VLM completion failed: $completion"
        return 1
    fi

    cleanup
    SERVER_PID=""
}

mkdir -p /tmp/vision_test
# 28x28 black PNG (requires multiple of 28 for Qwen2-VL patch embedder)
BASE64_IMG="iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAGUlEQVR4nO3BMQEAAADCoPVPbQdvoAAA6DQJTAABRMAOLAAAAABJRU5ErkJggg=="

run_case "mlx-community/Qwen2-VL-2B-Instruct-4bit" "$BASE_PORT" "yes"
run_case "LiquidAI/LFM2.5-VL-450M-MLX-4bit" "$((BASE_PORT + 1))" "no"

rm -rf /tmp/vision_test
exit 0
