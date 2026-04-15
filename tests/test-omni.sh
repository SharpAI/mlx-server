#!/bin/bash
# test-omni.sh — Omni E2E Validation Tests (using real media assets)

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15413}"
HOST="127.0.0.1"
MODEL="mlx-community/gemma-4-e4b-it-4bit"
URL="http://${HOST}:${PORT}"
START_SERVER=${START_SERVER:-1}

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[test-omni]${NC} $*"; }
pass() { echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { echo -e "  ${RED}❌ FAIL${NC}: $*"; exit 1; }

check_transcription_match() {
    local actual_text="$1"
    local expected_text="$2"
    python3 - "$actual_text" "$expected_text" <<'PY'
import difflib, re, sys
actual = sys.argv[1]
expected = sys.argv[2]
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
actual_n = normalize(actual)
expected_n = normalize(expected)
actual_words = actual_n.split()
expected_words = expected_n.split()
expected_prefix_n = " ".join(expected_words[:len(actual_words)]).strip()
full_ratio = difflib.SequenceMatcher(None, actual_n, expected_n).ratio()
prefix_ratio = difflib.SequenceMatcher(None, actual_n, expected_prefix_n).ratio() if actual_n else 0.0
prefix_exact = bool(actual_words) and actual_words == expected_words[:len(actual_words)]
if actual_n == expected_n or prefix_exact or prefix_ratio >= 0.85 or full_ratio >= 0.90:
    print("ok")
else:
    print(f"fail:{full_ratio:.3f}:{actual_n}:{expected_n}")
PY
}

cleanup() {
    if [ "$START_SERVER" == "1" ] && [ -n "${SERVER_PID:-}" ]; then
        log "Stopping server (PID $SERVER_PID)"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ "$START_SERVER" == "1" ]; then
    log "Starting server: $BINARY --model $MODEL --port $PORT --vision --audio"
    "$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" --vision --audio &
    SERVER_PID=$!

    log "Waiting for server to be ready..."
    MAX_WAIT=600
    for i in $(seq 1 "$MAX_WAIT"); do
        if curl -sf "$URL/health" >/dev/null 2>&1; then
            log "Server ready after ${i}s"
            break
        fi
        sleep 1
    done
fi

EXPECTED_TRANSCRIPT="Security alert. A brown and white dog has been detected on the camera. Please send assistance to the front gate immediately."
IMG_PATH="tests/fixtures/omni/dog.jpg"
AUDIO_PATH="tests/fixtures/omni/alert.wav"

if [ ! -f "$IMG_PATH" ] || [ ! -f "$AUDIO_PATH" ]; then
    fail "Required fixture assets not found in tests/fixtures/omni/"
fi

BASE64_IMG=$(base64 -i "$IMG_PATH" | tr -d '\n')
BASE64_AUDIO=$(base64 -i "$AUDIO_PATH" | tr -d '\n')

cat <<PAYLOAD > /tmp/omni_payload.json
{
  "model": "$MODEL",
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG}"}},
        {"type": "text", "text": "First describe the image in one sentence. Then transcribe the spoken words from the audio clip verbatim. The audio clip is present and contains speech."},
        {"type": "input_audio", "input_audio": {"data": "${BASE64_AUDIO}", "format": "wav"}}
      ]
    }
  ]
}
PAYLOAD

log "Sending benchmark payload..."
COMPLETION=$(curl -sf -X POST "$URL/v1/chat/completions" -H "Content-Type: application/json" -d @/tmp/omni_payload.json)

if ! echo "$COMPLETION" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    fail "Omni completion failed/empty: $COMPLETION"
fi

RAW_CONTENT=$(echo "$COMPLETION" | jq -r '.choices[0].message.content')
# Strip any thinking
CONTENT=$(echo "$RAW_CONTENT" | sed -E 's/<\|channel\|>thought.*<channel\|>//g')

# Extract last paragraph as transcript
TRANSCRIPT=$(echo "$CONTENT" | awk -v RS="" 'END {print}')

MATCH=$(check_transcription_match "$TRANSCRIPT" "$EXPECTED_TRANSCRIPT")

if [[ "$MATCH" == "ok" ]]; then
    pass "Transcription perfectly matched or strongly aligned!"
    log "Output: $CONTENT"
else
    fail "Transcription did not match sufficiently.\nExpected: $EXPECTED_TRANSCRIPT\nGot: $TRANSCRIPT\nMatch Data: $MATCH"
fi

rm -f /tmp/omni_payload.json
exit 0
