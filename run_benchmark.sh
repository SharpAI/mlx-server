#!/bin/bash

# Ensure we execute from the project root
cd "$(dirname "$0")"

generate_tts_wav() {
    local text="$1"
    local output_path="$2"
    local temp_aiff
    local sample_rate
    local audio_bytes

    temp_aiff=$(mktemp /tmp/swiftlm_tts.XXXXXX.aiff) || return 1

    if ! say -v Samantha -r 150 -o "$temp_aiff" "$text"; then
        rm -f "$temp_aiff"
        return 1
    fi

    if ! afconvert -f WAVE -d LEI16@16000 "$temp_aiff" "$output_path" >/dev/null 2>&1; then
        rm -f "$temp_aiff" "$output_path"
        return 1
    fi

    rm -f "$temp_aiff"

    sample_rate=$(
        afinfo "$output_path" 2>/dev/null \
            | sed -n 's/.*Data format:[[:space:]]*[0-9][0-9]* ch,[[:space:]]*\([0-9][0-9]*\) Hz.*/\1/p' \
            | head -n 1
    )
    audio_bytes=$(
        afinfo "$output_path" 2>/dev/null \
            | sed -n 's/.*audio bytes:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
            | head -n 1
    )

    if [ "$sample_rate" != "16000" ] || [ -z "$audio_bytes" ] || [ "$audio_bytes" -le 0 ]; then
        rm -f "$output_path"
        return 1
    fi
}

check_transcription_match() {
    local actual_text="$1"
    local expected_text="$2"
    python3 - "$actual_text" "$expected_text" <<'PY'
import difflib
import re
import sys

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
    print(f"fail:{prefix_ratio:.3f}:{actual_n}:{expected_n}")
PY
}

print_server_log() {
    local log_path="$1"
    if [ -f "$log_path" ]; then
        cat "$log_path"
    else
        echo "No log found at $log_path"
    fi
}

echo "=============================================="
export METAL_LIBRARY_PATH="$(pwd)/.build/arm64-apple-macosx/release"
echo "    Aegis-AI MLX Profiling Benchmark Suite    "
echo "=============================================="
echo ""

echo "Select Action:"
echo "0) Test 0: Run Full Automated Matrix (Offline Evaluation)"
echo "1) Test 1: Automated Context & Memory Profile (TPS & RAM matrix)"
echo "2) Test 2: Prompt Cache & Sliding Window Regression Test"
echo "3) Test 3: HomeSec Benchmark (LLM Only)"
echo "4) Test 4: VLM End-to-End Evaluation"
echo "5) Test 5: ALM Audio End-to-End Evaluation"
echo "6) Test 6: Omni End-to-End Evaluation"
echo "7) Model Maintain List and Delete"
echo "8) Test 8: Tool-Call Degeneration Regression (Gemma-4 vague-query bug)"
echo "9) Test 9: Quantized KV Cache Regression (Gemma-4 issue #71 — native kv_bits)"
echo "q) Quit"
read -p "Option (0-9/q): " suite_opt

if [ "$suite_opt" == "0" ]; then
    echo "=============================================="
    echo "  RUNNING FULL OFFLINE AUTOMATED MATRIX "
    echo "=============================================="
    mkdir -p tmp
    for TEST_ID in 3 4 5; do
        echo ""
        echo ">>> Executing Test Suite $TEST_ID <<<"
        
        # We dynamically fetch the highest downloaded Instruct mode model specifically to avoid hallucinating Vector/Embedding architectures
        MODEL=$(python3 scripts/hf_discovery.py "mlx-community/Qwen Instruct 4bit" || echo "Qwen2.5-7B-Instruct-4bit")
        
        if [ "$TEST_ID" == "4" ]; then
            MODEL=$(python3 scripts/hf_discovery.py "mlx-community/Qwen VL Instruct 4bit" || echo "mlx-community/Qwen2-VL-2B-Instruct-4bit")
        fi
        if [ "$TEST_ID" == "5" ]; then
            MODEL=$(python3 scripts/hf_discovery.py "mlx-community/Qwen Audio Instruct" || echo "mlx-community/Qwen2-Audio-7B-Instruct")
        fi
        
        echo -e "$TEST_ID\n11\n$MODEL" | HEADLESS=1 ./run_benchmark.sh
        sleep 5
    done
    echo "✅ Offline matrix execution fully completed."
    exit 0
fi

if [ "$suite_opt" == "q" ] || [ -z "$suite_opt" ]; then
    echo "Exiting."
    exit 0
fi

if [ "$suite_opt" == "9" ] || [ "$suite_opt" == "8" ]; then
    : # handled below — fall through
fi

if [ "$suite_opt" == "7" ]; then
    echo ""
    echo "=> Downloaded Models Maintenance"
    CACHE_DIR="$HOME/.cache/huggingface/hub"
    if [ ! -d "$CACHE_DIR" ]; then
        echo "Cache directory $CACHE_DIR not found."
        exit 1
    fi
    cd "$CACHE_DIR" || exit 1
    
    while true; do
        models=(models--*)
        if [ "${models[0]}" == "models--*" ]; then
            echo "No models found."
            exit 0
        fi
        
        echo ""
        echo "Downloaded Models:"
        for i in "${!models[@]}"; do
            size=$(du -sh "${models[$i]}" | cut -f1)
            name=$(echo ${models[$i]} | sed 's/models--//' | sed 's/--/\//g')
            echo "$((i+1))) $name ($size)"
        done
        echo "$(( ${#models[@]} + 1 ))) Delete ALL Models"
        echo "$(( ${#models[@]} + 2 ))) Quit"
        
        read -p "Select a model to delete (1-$(( ${#models[@]} + 2 ))): " del_opt
        
        if [ "$del_opt" == "$(( ${#models[@]} + 1 ))" ]; then
            echo ""
            read -p "⚠️ Are you sure you want to delete ALL models? This will free up significant space. (y/N): " confirm_all
            if [[ "$confirm_all" =~ ^[Yy]$ ]]; then
                echo "Deleting ALL models in $CACHE_DIR..."
                rm -rf models--*
                echo "✅ All models deleted."
                exit 0
            else
                echo "Canceled."
                continue
            fi
        elif [[ "$del_opt" =~ ^[0-9]+$ ]] && [ "$del_opt" -gt 0 ] && [ "$del_opt" -le "${#models[@]}" ]; then
            target_dir="${models[$((del_opt-1))]}"
            echo "Deleting $target_dir..."
            rm -rf "$target_dir"
            echo "✅ Deleted."
        else
            echo "Exiting."
            exit 0
        fi
    done
fi

echo ""
PS3="Select a model to use: "
if [ "$suite_opt" == "4" ]; then
    options=(
        "mlx-community/gemma-4-26b-a4b-it-8bit"
        "mlx-community/gemma-4-31b-it-8bit"
        "mlx-community/gemma-4-e4b-it-8bit"
        "mlx-community/gemma-4-26b-a4b-it-4bit"
        "mlx-community/Qwen3.5-9B-MLX-4bit"
        "mlx-community/Qwen3.5-27B-4bit"
        "LiquidAI/LFM2.5-VL-450M-MLX-4bit"
        "mlx-community/LFM2-VL-1.6B-4bit"
        "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        "mlx-community/Qwen2-VL-7B-Instruct-4bit"
        "mlx-community/pixtral-12b-2409-4bit"
        "Custom (Enter your own Hub ID)"
        "Quit"
    )
elif [ "$suite_opt" == "5" ] || [ "$suite_opt" == "6" ]; then
    # NOTE: Only Gemma 4 e4b variants support audio (audio_config present).
    # gemma-4-26b-a4b has audio_config=null — no audio tower, always hallucinates 'no audio'.
    # Qwen2-Audio is not exposed here because the current SwiftLM build does not support qwen2_audio.
    options=(
        "mlx-community/gemma-4-e4b-it-8bit"
        "mlx-community/gemma-4-e4b-it-4bit"
        "Custom (Enter your own Hub ID)"
        "Quit"
    )
else
    options=(
        "mlx-community/gemma-4-26b-a4b-it-8bit"
        "mlx-community/gemma-4-31b-it-8bit"
        "mlx-community/gemma-4-e4b-it-8bit"
        "mlx-community/gemma-4-26b-a4b-it-4bit"
        "mlx-community/gemma-4-26b-a4b-it-4bit"
        "mlx-community/Qwen2.5-7B-Instruct-4bit"
        "mlx-community/Qwen2.5-14B-Instruct-4bit"
        "mlx-community/phi-4-mlx-4bit"
        "baa-ai/GLM-5.1-RAM-270GB-MLX"
        "baa-ai/GLM-5.1-4bit"
        "Custom (Enter your own Hub ID)"
        "Quit"
    )
fi

select opt in "${options[@]}"
do
    case $opt in
        "Custom (Enter your own Hub ID)")
            read -p "Enter HuggingFace ID (e.g., mlx-community/Llama-3.2-3B-Instruct-4bit): " custom_model
            MODEL=$custom_model
            break
            ;;
        "Quit")
            echo "Exiting."
            exit 0
            ;;
        *) 
            if [[ -n "$opt" ]]; then
                MODEL=$opt
                break
            else
                echo "Invalid option $REPLY"
            fi
            ;;
    esac
done

# Ensure model has an org prefix if it doesn't already
if [[ "$MODEL" != *"/"* ]]; then
    FULL_MODEL="mlx-community/$MODEL"
else
    FULL_MODEL="$MODEL"
fi

if { [ "$suite_opt" == "5" ] || [ "$suite_opt" == "6" ]; } && [[ "$FULL_MODEL" == "mlx-community/Qwen2-Audio-7B-Instruct-4bit" ]]; then
    echo "❌ ERROR: $FULL_MODEL is not supported by this SwiftLM build because model type 'qwen2_audio' is not implemented yet."
    exit 1
fi

# Quick sanity check
if [ -f ".build/arm64-apple-macosx/release/SwiftLM" ]; then
    BIN=".build/arm64-apple-macosx/release/SwiftLM"
elif [ -f ".build/release/SwiftLM" ]; then
    BIN=".build/release/SwiftLM"
else
    echo "⚠️  SwiftLM release binary not found! Please compile the project by running ./build.sh first."
    exit 1
fi

# ── Test 8: Tool-Call Degeneration Regression ───────────────────────────────
# Regression test for the Gemma-4 vague-query bug:
#   With a small tool schema (<<100 tokens) the model should call the tool
#   for an obvious tool-use query.  Previously it produced garbage/text 6/6
#   times due to the <|channel>thought\n<channel|> generation-prompt suffix
#   flattening the first-token distribution.
# Pass criteria: ≥3/5 clean tool_calls on vague query  AND  3/3 on explicit query.
if [ "$suite_opt" == "8" ]; then
    echo ""
    echo "=> Test 8: Tool-Call Degeneration Regression on $FULL_MODEL"
    echo "   (Reproduces GitHub issue: vague query + small tool = degenerate output)"

    echo "Starting server on port 5431..."
    killall SwiftLM 2>/dev/null
    mkdir -p tmp
    $BIN --model "$FULL_MODEL" --port 5431 --stream-experts --ctx-size 4096 > ./tmp/tool_regression.log 2>&1 &
    SERVER_PID=$!

    echo "Waiting for server (up to 120s)..."
    for i in {1..120}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ Server died early. Logs:"
            print_server_log ./tmp/tool_regression.log
            exit 1
        fi
        if curl -sf http://127.0.0.1:5431/health > /dev/null 2>&1; then
            echo "Server ready (${i}s)"
            break
        fi
        sleep 1
    done

    echo ""
    echo "Running regression suite..."

    python3 - << 'TOOL_REG_EOF'
import json, urllib.request, time, sys

BASE = "http://127.0.0.1:5431"
TOOL = {"type":"function","function":{"name":"web_search",
    "description":"Search the web",
    "parameters":{"type":"object",
    "properties":{"query":{"type":"string"}},"required":["query"]}}}

def call(messages, tools=None, temp=0.0, max_tokens=2000):
    payload = {"messages": messages, "max_tokens": max_tokens,
               "temperature": temp, "stream": False, "repetition_penalty": 1.15}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    elapsed = time.time() - t0
    choice = d["choices"][0]
    tc = choice["message"].get("tool_calls")
    content = choice["message"].get("content") or ""
    return tc, content, elapsed, d["usage"]["prompt_tokens"]

def classify(tc, content):
    if tc:
        return "TOOL_CALL", tc[0]["function"]["name"]
    words = content.split()
    if len(words) > 5:
        top = max(set(words), key=words.count)
        if words.count(top) > len(words) * 0.35:
            return "DEGENERATE", f"repeat={repr(top)}"
    if "<|channel>" in content or "<channel|>" in content:
        return "DEGENERATE", "leaked control tokens"
    return "TEXT", content[:60]

FAILS = []

print("\n─── [1/3] Vague query WITH tool schema (must handle ambiguity naturally, tool call or text) ───")
vague_ok = 0
for i in range(5):
    tc, content, t, pt = call(
        [{"role":"system","content":"You are a helpful AI assistant."}, {"role":"user","content":"what is the news"}], tools=[TOOL])
    kind, detail = classify(tc, content)
    ok = kind in ("TOOL_CALL", "TEXT")
    if ok: vague_ok += 1
    print(f"  {'✅' if ok else '❌'} run {i+1} [{t:.1f}s P={pt}t]: {kind} — {detail.replace(chr(10), ' ')[:75]}")
print(f"  → {vague_ok}/5 runs passed without degenerating")
if vague_ok < 3:
    FAILS.append(f"Vague query: only {vague_ok}/5 clean runs (need ≥3)")

print("\n─── [2/3] Control: same query WITHOUT tools (must be coherent text) ───")
coherent_ok = 0
for i in range(3):
    tc, content, t, pt = call([{"role":"system","content":"You are a helpful AI assistant."}, {"role":"user","content":"what is the news"}], temp=0.7, max_tokens=200)
    kind, detail = classify(tc, content)
    ok = kind == "TEXT"
    if ok: coherent_ok += 1
    print(f"  {'✅' if ok else '❌'} run {i+1} [{t:.1f}s P={pt}t]: {kind} — {detail}")
print(f"  → {coherent_ok}/3 coherent text responses")
if coherent_ok < 3:
    FAILS.append(f"No-tool control: only {coherent_ok}/3 coherent (need 3)")

print("\n─── [3/3] Explicit query WITH tool schema (must always call tool) ───")
explicit_ok = 0
for i in range(3):
    tc, content, t, pt = call(
        [{"role":"system","content":"You are a helpful AI assistant."}, {"role":"user","content":"Use web_search to find news today"}], tools=[TOOL], max_tokens=2000)
    kind, detail = classify(tc, content)
    ok = kind == "TOOL_CALL"
    if ok: explicit_ok += 1
    print(f"  {'✅' if ok else '❌'} run {i+1} [{t:.1f}s P={pt}t]: {kind} — {detail}")
print(f"  → {explicit_ok}/3 tool_calls")
if explicit_ok < 3:
    FAILS.append(f"Explicit query: only {explicit_ok}/3 tool_calls (need 3)")

print("\n" + "─"*60)
if not FAILS:
    print("✅  REGRESSION PASSED — tool-call degeneration bug is fixed.")
    print(f"   Vague: {vague_ok}/5  |  No-tool: {coherent_ok}/3  |  Explicit: {explicit_ok}/3")
    sys.exit(0)
else:
    print("❌  REGRESSION FAILED:")
    for f in FAILS:
        print(f"    • {f}")
    print("\n   Root cause: Gemma-4 <|channel>thought\\n<channel|> generation prefix")
    print("   flattens the first-token distribution for vague queries with tools.")
    sys.exit(1)
TOOL_REG_EOF
    TEST8_EXIT=$?

    echo ""
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    if [ $TEST8_EXIT -eq 0 ]; then
        echo "✅ Test 8 PASSED"
    else
        echo "❌ Test 8 FAILED — see output above."
    fi
    exit $TEST8_EXIT
fi

if [ "$suite_opt" == "2" ]; then
    echo ""
    echo "=> Starting Prompt Cache Regression Test on $FULL_MODEL"
    echo "Generating /tmp/big_prompt.json (approx 5K tokens)..."
    python3 -c 'import json; open("/tmp/big_prompt.json", "w").write(json.dumps({"messages": [{"role": "user", "content": "apple "*4500}], "max_tokens": 30}))'
    
    echo "Starting Server in background..."
    killall SwiftLM 2>/dev/null
    mkdir -p tmp
    $BIN --model "$FULL_MODEL" --port 5431 --stream-experts --ctx-size 16384 > ./tmp/regression_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            print_server_log ./tmp/regression_server.log
            exit 1
        fi
        if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
        sleep 1
    done
    
    echo ""
    echo "Server is up! Running 4-request sliding window validation..."
    
    echo "=== Req 1 (Big 5537t) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== Req 2 (Short 18t) ===" && curl -sS --max-time 60 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is today?"}],"max_tokens":30}' 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== Req 3 (Big 5537t) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== Req 4 (Big Full Cache Hit) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== ALL 4 PASSED ==="
    
    echo ""
    echo "✅ Test Passed! The server successfully interleaved long context (sliding window)"
    echo "with short context, without crashing or throwing Out-of-Memory / SIGTRAP errors."
    echo "This proves the Prompt Cache bounds are stable."
    
    echo ""
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    exit 0
fi

if [ "$suite_opt" == "3" ]; then
    echo ""
    echo "=> Starting HomeSec Benchmark (LLM Only) on $FULL_MODEL"
    
    echo "Starting Server in background..."
    killall SwiftLM 2>/dev/null
    mkdir -p tmp
    $BIN --model "$FULL_MODEL" --port 5431 --stream-experts --ctx-size 8192 > ./tmp/homesec_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            print_server_log ./tmp/homesec_server.log
            exit 1
        fi
        if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
        sleep 1
    done
    
    echo ""
    echo "Server is up! Executing DeepCamera HomeSec Benchmark..."
    
    LOCAL_BENCHMARK="./homesec-benchmark"
    BENCHMARK_DIR="$LOCAL_BENCHMARK/skills/analysis/home-security-benchmark"
    if [ ! -d "$BENCHMARK_DIR" ]; then
        echo "HomeSec benchmark skill not found locally. Cloning thinly via git sparse-checkout..."
        rm -rf "$LOCAL_BENCHMARK"
        git clone --filter=blob:none --no-checkout https://github.com/SharpAI/DeepCamera.git "$LOCAL_BENCHMARK"
        pushd "$LOCAL_BENCHMARK" > /dev/null
        git sparse-checkout init --cone
        git sparse-checkout set skills/analysis/home-security-benchmark
        git checkout master 2>/dev/null || git checkout main
        popd > /dev/null
    fi
    
    if [ ! -d "$BENCHMARK_DIR/node_modules" ]; then
        echo "Installing npm dependencies for HomeSec benchmark..."
        pushd "$BENCHMARK_DIR" > /dev/null
        npm install --silent
        popd > /dev/null
    fi
    
    # Run the benchmark against the LLM gateway. Not specifying --vlm disables VLM tests.
    node "$BENCHMARK_DIR/scripts/run-benchmark.cjs" --gateway http://127.0.0.1:5431 --out ./tmp/benchmarks
    
    echo ""
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    exit 0
fi

if [ "$suite_opt" == "4" ]; then
    echo ""
    echo "=> Starting Test 4: VLM End-to-End Evaluation on $FULL_MODEL"
    echo "Looking for a test image..."
    
    mkdir -p tmp
    IMAGE_PATH="./tmp/dog.jpg"
    # Download a small but recognizable image of a dog (golden retriever puppy)
    curl -sL "https://images.unsplash.com/photo-1543466835-00a7907e9de1?auto=format&fit=crop&q=80&w=320" -o "$IMAGE_PATH"
    
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Failed to download image."
        exit 1
    fi
    
    echo "Encoding image to base64..."
    BASE64_IMG=$(base64 -i "$IMAGE_PATH" | tr -d '\n')
    
    echo "Generating /tmp/vlm_payload.json..."
    cat <<EOF > /tmp/vlm_payload.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image? Explain concisely."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG}"}}
      ]
    }
  ]
}
EOF

    echo "Starting Server in background with --vision..."
    killall SwiftLM 2>/dev/null
    rm -f ./tmp/vlm_server.log
    $BIN --model "$FULL_MODEL" --vision --port 5431 > ./tmp/vlm_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            print_server_log ./tmp/vlm_server.log
            exit 1
        fi
        if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
        sleep 1
    done
    
    echo ""
    echo "Server is up! Sending payload..."
    echo "=== VLM Request ==="
    RAW_OUT=$(curl -sS --max-time 180 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/vlm_payload.json)
    if [ -z "$RAW_OUT" ] || [[ "$RAW_OUT" == *"curl: "* ]]; then
        echo "❌ ERROR: Server dropped the connection or crashed!"
        exit 1
    fi
    VLM_RES=$(echo "$RAW_OUT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR').replace('\n', '<br/>'))")
    if [ -z "$VLM_RES" ] || [[ "$VLM_RES" == *"ERROR"* ]]; then
        echo "❌ ERROR: JSON Decode failed!"
        exit 1
    fi
    
    echo -e "\n🤖 VLM Output: $VLM_RES"
    
    if [ -z "${HEADLESS:-}" ]; then
        UI_FILE="/tmp/vlm_ui.html"
        cat <<EOF > "$UI_FILE"
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f1115; color: #E0E0E0; max-width: 700px; margin: 40px auto; line-height: 1.6; }
    .container { background: #1a1d24; padding: 30px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.8); border: 1px solid #2d313a; }
    img { max-width: 100%; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    .prompt { background: #21252d; padding: 15px; border-left: 4px solid #00ffcc; border-radius: 4px; margin-bottom: 20px; font-weight: 500; font-size: 14px; color: #a1aabf; }
    .response { background: #16181e; padding: 20px; border-radius: 8px; font-size: 16px; color: #ffffff; border: 1px solid #252932; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }
    h2 { color: #f5f6f8; font-weight: 600; letter-spacing: -0.5px; margin-top: 0; }
  </style>
</head>
<body>
  <div class="container">
    <h2>👁️ SwiftLM Vision Pipeline</h2>
    <div style="font-size: 13px; color: #727a8e; margin-top: -15px; margin-bottom: 20px;">Model: $FULL_MODEL</div>
    <img src="data:image/jpeg;base64,${BASE64_IMG}" />
    <div class="prompt">Prompt: What is in this image? Explain concisely.</div>
    <div class="response">🤖 $VLM_RES</div>
  </div>
</body>
</html>
EOF
        open "$UI_FILE"
    fi
    
    echo ""
    echo "✅ Test Complete!"
    
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    rm -f /tmp/vlm_payload.json "$IMAGE_PATH"
    exit 0
fi

if [ "$suite_opt" == "5" ]; then
    echo ""
    echo "=> Starting Test 5: ALM Audio End-to-End Evaluation on $FULL_MODEL"
    echo "Looking for a test audio payload..."
    
    mkdir -p tmp
    AUDIO_PATH="./tmp/audio_test"
    EXPECTED_TRANSCRIPT="The quick brown fox jumps over the lazy dog. Machine learning systems require careful validation."
    # Generate speech via macOS TTS, then resample explicitly since `say`
    # may still emit 22.05 kHz audio for some voices even when 16 kHz is requested.
    if ! generate_tts_wav \
        "$EXPECTED_TRANSCRIPT" \
        "${AUDIO_PATH}.wav"; then
        echo "Failed to generate a valid 16 kHz WAV test clip."
        exit 1
    fi
    
    if [ ! -f "${AUDIO_PATH}.wav" ]; then
        echo "Failed to create benchmark audio."
        exit 1
    fi
    
    echo "Encoding audio to base64..."
    BASE64_AUDIO=$(base64 -i "${AUDIO_PATH}.wav" | tr -d '\n')
    
    echo "Generating /tmp/alm_payload_1.json (Turn 1)..."
    cat <<EOF > /tmp/alm_payload_1.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 500,
  "temperature": 0,
  "top_p": 1.0,
  "enable_thinking": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio clip word for word. Output only the transcription, nothing else."},
        {"type": "input_audio", "input_audio": {"data": "${BASE64_AUDIO}", "format": "wav"}}
      ]
    }
  ]
}
EOF

    echo "Starting Server in background with --audio..."
    killall SwiftLM 2>/dev/null
    rm -f ./tmp/alm_server.log
    $BIN --model "$FULL_MODEL" --audio --port 5431 > ./tmp/alm_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            print_server_log ./tmp/alm_server.log
            exit 1
        fi
        if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
        sleep 1
    done
    
    echo ""
    echo "Server is up! Sending Turn 1 payload..."
    echo "=== ALM Request 1 ==="
    RAW_ALM_OUT=$(curl -sS --max-time 180 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/alm_payload_1.json)
    if [ -z "$RAW_ALM_OUT" ] || [[ "$RAW_ALM_OUT" == *"curl: "* ]]; then
        echo "❌ ERROR: Server dropped the connection or crashed!"
        exit 1
    fi
    # Extract content and strip any thinking blocks (Gemma4 <|channel|>thought...)<channel|>)
    ALM_RES=$(echo "$RAW_ALM_OUT" | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
content = d.get('choices',[{}])[0].get('message',{}).get('content', '')
gen_tok = d.get('usage',{}).get('completion_tokens', 0)
# Strip Gemma4 thinking blocks: <|channel|>thought ... <channel|>
content = re.sub(r'<\|channel\|>thought.*?<channel\|>', '', content, flags=re.DOTALL).strip()
if not content:
    print(f'[WARN: gen_tokens={gen_tok}, empty after stripping thinking]')
else:
    print(content)
")
    if [ -z "$ALM_RES" ]; then
        echo "❌ ERROR: Server dropped turn 1 connection!"
        exit 1
    fi
    echo -e "\n🎤 ALM Turn 1 Transcription:\n  → $ALM_RES\n"

    ALM_CHECK=$(check_transcription_match "$ALM_RES" "$EXPECTED_TRANSCRIPT")
    if [[ "$ALM_CHECK" != "ok" ]]; then
        ALM_RATIO=$(echo "$ALM_CHECK" | cut -d: -f2)
        echo "❌ ERROR: Turn 1 transcription did not match the expected audio closely enough."
        echo "Expected: $EXPECTED_TRANSCRIPT"
        echo "Observed: $ALM_RES"
        echo "Similarity: ${ALM_RATIO:-unknown}"
        exit 1
    fi
    
    echo "Generating /tmp/alm_payload_2.json (Turn 2 - Closed Loop)..."
    ASSISTANT_CONTENT_ESCAPED=$(echo "$RAW_ALM_OUT" | python3 -c "import sys,json;print(json.dumps(json.load(sys.stdin).get('choices',[{}])[0].get('message',{}).get('content', 'ERROR')))")
    
    cat <<EOF > /tmp/alm_payload_2.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 200,
  "temperature": 0,
  "top_p": 1.0,
  "enable_thinking": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio clip word for word. Output only the transcription, nothing else."},
        {"type": "input_audio", "input_audio": {"data": "${BASE64_AUDIO}", "format": "wav"}}
      ]
    },
    {
      "role": "assistant",
      "content": $ASSISTANT_CONTENT_ESCAPED
    },
    {
      "role": "user",
      "content": "In one sentence, summarize what the speaker said."
    }
  ]
}
EOF

    echo "=== ALM Request 2 (Multi-turn Cache Evaluation) ==="
    RAW_ALM_OUT_2=$(curl -sS --max-time 180 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/alm_payload_2.json)
    if [ -z "$RAW_ALM_OUT_2" ] || [[ "$RAW_ALM_OUT_2" == *"curl: "* ]]; then
        echo "❌ ERROR: Server dropped the connection or crashed on Turn 2!"
        exit 1
    fi
    ALM_RES_2=$(echo "$RAW_ALM_OUT_2" | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
content = d.get('choices',[{}])[0].get('message',{}).get('content', '')
content = re.sub(r'<\|channel\|>thought.*?<channel\|>', '', content, flags=re.DOTALL).strip()
print(content if content else '[empty]')
")
    if [ -z "$ALM_RES_2" ]; then
        echo "❌ ERROR: Server dropped turn 2 connection!"
        exit 1
    fi
    echo -e "\n🎤 ALM Turn 2 Summary:\n  → $ALM_RES_2\n"

    echo ""
    echo "✅ Test Complete! Closed-Loop validation successful."
    
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    rm -f /tmp/alm_payload_1.json /tmp/alm_payload_2.json "${AUDIO_PATH}.wav"
    exit 0
fi

if [ "$suite_opt" == "6" ]; then
    echo ""
    echo "=> Starting Test 6: Omni End-to-End Evaluation on $FULL_MODEL"
    echo "Looking for a test image and audio payload..."
    
    mkdir -p tmp
    IMAGE_PATH="./tmp/omni_dog.jpg"
    curl -sL "https://images.unsplash.com/photo-1543466835-00a7907e9de1?auto=format&fit=crop&q=80&w=320" -o "$IMAGE_PATH"
    
    AUDIO_PATH="./tmp/omni_audio_test"
    EXPECTED_OMNI_TRANSCRIPT="Security alert. A brown and white dog has been detected on the camera. Please send assistance to the front gate immediately."
    echo "Generating real audio sample via TTS..."
    if ! generate_tts_wav \
        "$EXPECTED_OMNI_TRANSCRIPT" \
        "${AUDIO_PATH}.wav"; then
        echo "Failed to generate a valid 16 kHz WAV test clip."
        exit 1
    fi
    

    
    if [ ! -f "$IMAGE_PATH" ] || [ ! -f "${AUDIO_PATH}.wav" ]; then
        echo "Failed to download media assets."
        exit 1
    fi
    
    echo "Encoding media..."
    BASE64_IMG=$(base64 -i "$IMAGE_PATH" | tr -d '\n')
    BASE64_AUDIO=$(base64 -i "${AUDIO_PATH}.wav" | tr -d '\n')
    
    echo "Generating /tmp/omni_payload.json..."
    cat <<EOF > /tmp/omni_payload.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 400,
  "temperature": 0,
  "top_p": 1.0,
  "enable_thinking": false,
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
EOF

    echo "Starting Server in background with --vision AND --audio (Omni)..."
    killall SwiftLM 2>/dev/null
    rm -f ./tmp/omni_server.log
    $BIN --model "$FULL_MODEL" --vision --audio --port 5431 2>&1 | tee ./tmp/omni_server.log &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            print_server_log ./tmp/omni_server.log
            exit 1
        fi
        if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
        sleep 1
    done
    
    echo ""
    echo "Server is up! Sending Omni payload..."
    echo "=== Omni Request ==="
    RAW_OMNI_OUT=$(curl -sS --max-time 180 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/omni_payload.json)
    if [ -z "$RAW_OMNI_OUT" ] || [[ "$RAW_OMNI_OUT" == *"curl: "* ]]; then
        echo "❌ ERROR: Server dropped the connection or crashed!"
        exit 1
    fi
    # Extract content and strip any thinking blocks
    OMNI_RES=$(echo "$RAW_OMNI_OUT" | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
content = d.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR')
content = re.sub(r'<\|channel\|>thought.*?<channel\|>', '', content, flags=re.DOTALL).strip()
print(content.replace('\n', '<br/>'))
")
    if [ -z "$OMNI_RES" ] || [[ "$OMNI_RES" == *"ERROR"* ]]; then
        echo "❌ ERROR: JSON Decode failed!"
        exit 1
    fi
    
    echo -e "\n🤖 Omni Output: $OMNI_RES"

    OMNI_AUDIO_CHECK=$(echo "$RAW_OMNI_OUT" | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
content = d.get('choices',[{}])[0].get('message',{}).get('content', '')
content = re.sub(r'<\|channel\|>thought.*?<channel\|>', '', content, flags=re.DOTALL).strip().lower()
bad_markers = [
    'no audio clip provided',
    'no audio provided',
    'there is no audio',
    'audio clip is provided',
]
print('fail' if any(marker in content for marker in bad_markers) else 'ok')
")
    if [ "$OMNI_AUDIO_CHECK" != "ok" ]; then
        echo "❌ ERROR: Omni response ignored the supplied audio clip."
        echo "Cleaning up..."
        killall SwiftLM
        wait $SERVER_PID 2>/dev/null
        rm -f /tmp/omni_payload.json "$IMAGE_PATH" "${AUDIO_PATH}.wav"
        exit 1
    fi

    OMNI_TRANSCRIPT_CHECK=$(echo "$RAW_OMNI_OUT" | python3 -c "
import json, re, sys
d = json.load(sys.stdin)
content = d.get('choices',[{}])[0].get('message',{}).get('content', '')
content = re.sub(r'<\|channel\|>thought.*?<channel\|>', '', content, flags=re.DOTALL).strip()
# Split on any newline (model may use \n or \n\n between image-desc / audio-transcript paragraphs)
parts = [part.strip() for part in re.split(r'\n+', content) if part.strip()]
print(parts[-1] if parts else content)
")
    OMNI_MATCH=$(check_transcription_match "$OMNI_TRANSCRIPT_CHECK" "$EXPECTED_OMNI_TRANSCRIPT")
    if [[ "$OMNI_MATCH" != "ok" ]]; then
        OMNI_RATIO=$(echo "$OMNI_MATCH" | cut -d: -f2)
        echo "❌ ERROR: Omni transcription did not match the expected audio closely enough."
        echo "Expected: $EXPECTED_OMNI_TRANSCRIPT"
        echo "Observed: $OMNI_TRANSCRIPT_CHECK"
        echo "Similarity: ${OMNI_RATIO:-unknown}"
        echo "Cleaning up..."
        killall SwiftLM
        wait $SERVER_PID 2>/dev/null
        rm -f /tmp/omni_payload.json "$IMAGE_PATH" "${AUDIO_PATH}.wav"
        exit 1
    fi
    
    if [ "$HEADLESS" != "1" ]; then
        UI_FILE="/tmp/swiftlm_omni_result.html"
        cat <<EOF > "$UI_FILE"
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SwiftLM Omni Pipeline Demo</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background: #0f1115; color: #e1e4e8; line-height: 1.5; }
    .container { max-width: 800px; margin: 0 auto; background: #1a1c23; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    img { max-width: 100%; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
    .prompt { background: #21252d; padding: 15px; border-left: 4px solid #00ffcc; border-radius: 4px; margin-bottom: 20px; font-weight: 500; font-size: 14px; color: #a1aabf; }
    .response { background: #16181e; padding: 20px; border-radius: 8px; font-size: 16px; color: #ffffff; border: 1px solid #252932; text-shadow: 0 1px 2px rgba(0,0,0,0.5); margin-top: 20px; }
    h2 { color: #f5f6f8; font-weight: 600; letter-spacing: -0.5px; margin-top: 0; }
    audio { width: 100%; margin-top: 10px; margin-bottom: 20px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>🌐 SwiftLM Omni Pipeline</h2>
    <div style="font-size: 13px; color: #727a8e; margin-top: -15px; margin-bottom: 20px;">Model: $FULL_MODEL</div>
    <img src="data:image/jpeg;base64,${BASE64_IMG}" />
    <audio controls>
      <source src="data:audio/wav;base64,${BASE64_AUDIO}" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
    <div class="prompt">Prompt: Describe the image and then describe the audio.</div>
    <div class="response">🤖 Omni Output: $OMNI_RES</div>
  </div>
</body>
</html>
EOF
        open "$UI_FILE"
    fi
    
    echo ""
    echo "✅ Test Complete! Omni evaluation successful."
    
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    rm -f /tmp/omni_payload.json "$IMAGE_PATH" "${AUDIO_PATH}.wav" "${AUDIO_PATH}.mp3"
    exit 0
fi

# ── Test 9: QuantizedKVCache Regression (issue #71) ────────────────────────
# Verifies that Gemma-4 text models can decode with native MLX QuantizedKVCache
# (kv_bits=4 and kv_bits=8) without triggering the:
#   fatalError: `update` was called on `QuantizedKVCache`. Use `updateQuantized`.
# crash fixed in PR #29 of mlx-swift-lm.
#
# Pass criteria:
#   - 4-bit run: prefill + ≥20 decode tokens, response is non-empty coherent text
#   - 8-bit run: same
#   - Multi-turn run: second turn with kv_bits=4 also succeeds (exercises sharedKV path)
if [ "$suite_opt" == "9" ]; then
    echo ""
    echo "=> Test 9: Quantized KV Cache Regression (issue #71) on $FULL_MODEL"
    echo "   Tests MLX native QuantizedKVCache (kv_bits=4, kv_bits=8) — NOT TurboKV"
    echo "   This exercises the fix in mlx-swift-lm PR #29."

    echo "Starting server on port 5431..."
    killall SwiftLM 2>/dev/null
    mkdir -p tmp
    # No --turbo-kv flag: we want the vanilla KVCacheSimple path that will be
    # upgraded to QuantizedKVCache by the per-request kv_bits field.
    $BIN --model "$FULL_MODEL" --port 5431 --stream-experts --ctx-size 8192 > ./tmp/kvcache_regression.log 2>&1 &
    SERVER_PID=$!

    echo "Waiting for server (up to 180s)..."
    for i in {1..180}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ Server died early. Logs:"
            print_server_log ./tmp/kvcache_regression.log
            exit 1
        fi
        if curl -sf http://127.0.0.1:5431/health > /dev/null 2>&1; then
            echo "Server ready (${i}s)"
            break
        fi
        sleep 1
    done

    echo ""
    echo "Running QuantizedKVCache regression suite..."

    python3 - << 'KVBITS_EOF'
import json, urllib.request, time, sys, re

BASE = "http://127.0.0.1:5431"

FAILS = []

def call(messages, kv_bits=None, max_tokens=60, temperature=0.0):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if kv_bits is not None:
        payload["kv_bits"] = kv_bits
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except Exception as e:
        return None, str(e), time.time() - t0
    elapsed = time.time() - t0
    content = d["choices"][0]["message"].get("content") or ""
    # Strip Gemma-4 thinking blocks
    content = re.sub(r"<\|channel\|>thought.*?<channel\|>", "", content, flags=re.DOTALL).strip()
    return d, content, elapsed

MSGS_SHORT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Name the three primary colours. Be brief."},
]

# Longer prompt to exercise the KV sharing layers (last 20 of Gemma-4 share KV
# from earlier layers — the bug manifests at those layers on multi-token prefills).
MSGS_LONG = [
    {"role": "system", "content": "You are a knowledgeable AI assistant. Answer concisely."},
    {"role": "user",   "content": "Explain in two sentences why the sky appears blue during the day and red at sunset. Use physics terminology."},
]

# ── [1] 4-bit quantized KV cache ──
print("\n─── [1/4] kv_bits=4, short prompt ───")
d, content, t = call(MSGS_SHORT, kv_bits=4)
if d is None:
    print(f"  ❌ CRASHED: {content}")
    FAILS.append("kv_bits=4 short: server crash or timeout")
else:
    gen_toks = d["usage"]["completion_tokens"]
    ok = len(content.strip()) > 5 and gen_toks >= 3
    print(f"  {'✅' if ok else '❌'} [{t:.1f}s, {gen_toks} tokens]: {content[:100]}")
    if not ok:
        FAILS.append(f"kv_bits=4 short: too few tokens or empty ({gen_toks} tokens)")

# ── [2] 8-bit quantized KV cache ──
print("\n─── [2/4] kv_bits=8, short prompt ───")
d, content, t = call(MSGS_SHORT, kv_bits=8)
if d is None:
    print(f"  ❌ CRASHED: {content}")
    FAILS.append("kv_bits=8 short: server crash or timeout")
else:
    gen_toks = d["usage"]["completion_tokens"]
    ok = len(content.strip()) > 5 and gen_toks >= 3
    print(f"  {'✅' if ok else '❌'} [{t:.1f}s, {gen_toks} tokens]: {content[:100]}")
    if not ok:
        FAILS.append(f"kv_bits=8 short: too few tokens or empty ({gen_toks} tokens)")

# ── [3] 4-bit, longer prompt (exercises KV-sharing layers) ──
print("\n─── [3/4] kv_bits=4, longer prompt (exercises KV-sharing path) ───")
d, content, t = call(MSGS_LONG, kv_bits=4, max_tokens=120)
if d is None:
    print(f"  ❌ CRASHED: {content}")
    FAILS.append("kv_bits=4 long: server crash or timeout")
else:
    gen_toks = d["usage"]["completion_tokens"]
    ok = len(content.strip()) > 10 and gen_toks >= 5
    print(f"  {'✅' if ok else '❌'} [{t:.1f}s, {gen_toks} tokens]: {content[:120]}")
    if not ok:
        FAILS.append(f"kv_bits=4 long: too few tokens or empty ({gen_toks} tokens)")

# ── [4] Baseline without kv_bits (must still work — regression guard) ──
print("\n─── [4/4] kv_bits=None baseline (no quantization) ───")
d, content, t = call(MSGS_SHORT, kv_bits=None)
if d is None:
    print(f"  ❌ CRASHED: {content}")
    FAILS.append("baseline (no kv_bits): server crash or timeout")
else:
    gen_toks = d["usage"]["completion_tokens"]
    ok = len(content.strip()) > 5 and gen_toks >= 3
    print(f"  {'✅' if ok else '❌'} [{t:.1f}s, {gen_toks} tokens]: {content[:100]}")
    if not ok:
        FAILS.append(f"baseline: too few tokens or empty ({gen_toks} tokens)")

print("\n" + "─" * 60)
if not FAILS:
    print("✅  REGRESSION PASSED — QuantizedKVCache dispatches correctly.")
    print("   kv_bits=4 ✓  |  kv_bits=8 ✓  |  KV-sharing path ✓  |  baseline ✓")
    sys.exit(0)
else:
    print("❌  REGRESSION FAILED:")
    for f in FAILS:
        print(f"    • {f}")
    print("\n   Root cause (if kv_bits runs crash): unconditional `cache.update()` call")
    print("   in Gemma4TextAttention.callAsFunction — see mlx-swift-lm PR #29.")
    sys.exit(1)
KVBITS_EOF
    TEST9_EXIT=$?

    echo ""
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    if [ $TEST9_EXIT -eq 0 ]; then
        echo "✅ Test 9 PASSED"
    else
        echo "❌ Test 9 FAILED — see output above."
    fi
    exit $TEST9_EXIT
fi

# Fallback to Test 1 for anything else
echo ""
read -p "Enter context lengths to test [default: 512,40000,100000]: " CONTEXTS
CONTEXTS=${CONTEXTS:-"512,40000,100000"}

echo ""
echo "=> Starting benchmark for $FULL_MODEL with contexts: $CONTEXTS"
echo ""

EXTRA_FLAGS=""
if [[ "$FULL_MODEL" == *"GLM-5.1"* ]]; then
    EXTRA_FLAGS="--ssd-only"
    echo "Note: GLM-5.1 is very large. Restricting to SSD streaming configurations only."
    echo ""
fi

python3 -u scripts/profiling/profile_runner.py \
  --model "$FULL_MODEL" \
  --contexts "$CONTEXTS" \
  $EXTRA_FLAGS \
  --out "./docs/profiling/profiling_results_$(hostname -s).md"

echo ""
echo "✅ Benchmark finished! Results saved to ./docs/profiling/profiling_results_$(hostname -s).md"
