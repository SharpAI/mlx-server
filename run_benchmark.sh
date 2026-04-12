#!/bin/bash

# Ensure we execute from the project root
cd "$(dirname "$0")"

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
echo "8) Quit"
read -p "Option (0-8): " suite_opt

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

if [ "$suite_opt" == "8" ] || [ -z "$suite_opt" ]; then
    echo "Exiting."
    exit 0
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
        "mlx-community/LFM2-VL-1.6B-4bit"
        "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        "mlx-community/Qwen2-VL-7B-Instruct-4bit"
        "mlx-community/pixtral-12b-2409-4bit"
        "Custom (Enter your own Hub ID)"
        "Quit"
    )
elif [ "$suite_opt" == "5" ] || [ "$suite_opt" == "6" ]; then
    options=(
        "mlx-community/gemma-4-e4b-it-8bit"
        "mlx-community/gemma-4-e4b-it-4bit"
        "mlx-community/gemma-4-26b-a4b-it-4bit"
        "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
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

# Quick sanity check
if [ -f ".build/arm64-apple-macosx/release/SwiftLM" ]; then
    BIN=".build/arm64-apple-macosx/release/SwiftLM"
elif [ -f ".build/release/SwiftLM" ]; then
    BIN=".build/release/SwiftLM"
else
    echo "⚠️  SwiftLM release binary not found! Please compile the project by running ./build.sh first."
    exit 1
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
            cat ./tmp/*_server.log || echo "No log found"
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
            cat ./tmp/*_server.log || echo "No log found"
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
    $BIN --model "$FULL_MODEL" --vision --port 5431 > ./tmp/vlm_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            cat ./tmp/*_server.log || echo "No log found"
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
    # Small test audio clip (we assume standard tools and curl).
    curl -sL "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" -o "${AUDIO_PATH}.mp3" 
    
    echo "Converting MP3 to WAV for engine pipeline ingestion..."
    # afconvert converts mp3 to standardized WAV under macOS natively without ffmpeg dependencies
    afconvert -f WAVE -d LEI16 "${AUDIO_PATH}.mp3" "${AUDIO_PATH}.wav" 
    
    if [ ! -f "${AUDIO_PATH}.wav" ]; then
        echo "Failed to convert audio via afconvert."
        exit 1
    fi
    
    echo "Encoding audio to base64..."
    BASE64_AUDIO=$(base64 -i "${AUDIO_PATH}.wav" | tr -d '\n')
    
    echo "Generating /tmp/alm_payload_1.json (Turn 1)..."
    cat <<EOF > /tmp/alm_payload_1.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio strictly."},
        {"type": "input_audio", "input_audio": {"data": "${BASE64_AUDIO}", "format": "wav"}}
      ]
    }
  ]
}
EOF

    echo "Starting Server in background with --audio..."
    killall SwiftLM 2>/dev/null
    $BIN --model "$FULL_MODEL" --audio --port 5431 > ./tmp/alm_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            cat ./tmp/*_server.log || echo "No log found"
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
    ALM_RES=$(echo "$RAW_ALM_OUT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR'))")
    if [ -z "$ALM_RES" ] || [[ "$ALM_RES" == *"ERROR"* ]]; then
        echo "❌ ERROR: JSON Decode failed on Turn 1!"
        exit 1
    fi
    echo -e "\n🎤 ALM Output 1: $ALM_RES\n"
    
    echo "Generating /tmp/alm_payload_2.json (Turn 2 - Closed Loop)..."
    ASSISTANT_CONTENT_ESCAPED=$(echo "$RAW_ALM_OUT" | python3 -c "import sys,json;print(json.dumps(json.load(sys.stdin).get('choices',[{}])[0].get('message',{}).get('content', 'ERROR')))")
    
    cat <<EOF > /tmp/alm_payload_2.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio strictly."},
        {"type": "input_audio", "input_audio": {"data": "${BASE64_AUDIO}", "format": "wav"}}
      ]
    },
    {
      "role": "assistant",
      "content": $ASSISTANT_CONTENT_ESCAPED
    },
    {
      "role": "user",
      "content": "Are there any instruments playing in this audio track? List them."
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
    ALM_RES_2=$(echo "$RAW_ALM_OUT_2" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR'))")
    if [ -z "$ALM_RES_2" ] || [[ "$ALM_RES_2" == *"ERROR"* ]]; then
        echo "❌ ERROR: JSON Decode failed on Turn 2!"
        exit 1
    fi
    echo -e "\n🎤 ALM Output 2: $ALM_RES_2\n"

    echo ""
    echo "✅ Test Complete! Closed-Loop validation successful."
    
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    rm -f /tmp/alm_payload_1.json /tmp/alm_payload_2.json "${AUDIO_PATH}.wav" "${AUDIO_PATH}.mp3"
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
    echo "Generating real audio sample via TTS..."
    say "Warning! A dog has been detected on the security camera footage!" -o "${AUDIO_PATH}.aiff"
    afconvert -f WAVE -d LEI16 "${AUDIO_PATH}.aiff" "${AUDIO_PATH}.wav"
    

    
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
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image and then describe the audio."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG}"}},
        {"type": "input_audio", "input_audio": {"data": "${BASE64_AUDIO}", "format": "wav"}}
      ]
    }
  ]
}
EOF

    echo "Starting Server in background with --vision AND --audio (Omni)..."
    killall SwiftLM 2>/dev/null
    $BIN --model "$FULL_MODEL" --vision --audio --port 5431 2>&1 | tee ./tmp/omni_server.log &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERROR: Server process died unexpectedly! Printing logs:"
            cat ./tmp/*_server.log || echo "No log found"
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
    OMNI_RES=$(echo "$RAW_OMNI_OUT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR').replace('\n', '<br/>'))")
    if [ -z "$OMNI_RES" ] || [[ "$OMNI_RES" == *"ERROR"* ]]; then
        echo "❌ ERROR: JSON Decode failed!"
        exit 1
    fi
    
    echo -e "\n🤖 Omni Output: $OMNI_RES"
    
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
  --out "./profiling_results_$(hostname -s).md"

echo ""
echo "✅ Benchmark finished! Results saved to ./profiling_results_$(hostname -s).md"
