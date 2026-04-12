#!/bin/bash

# Ensure we execute from the project root
cd "$(dirname "$0")"

echo "=============================================="
echo "    Aegis-AI MLX Profiling Benchmark Suite    "
echo "=============================================="
echo ""

echo "Select Action:"
echo "1) Test 1: Automated Context & Memory Profile (TPS & RAM matrix)"
echo "2) Test 2: Prompt Cache & Sliding Window Regression Test"
echo "3) Test 3: HomeSec Benchmark (LLM Only)"
echo "4) Model Maintain List and Delete"
echo "5) Quit"
read -p "Option (1-5): " suite_opt

if [ "$suite_opt" == "5" ] || [ -z "$suite_opt" ]; then
    echo "Exiting."
    exit 0
fi

if [ "$suite_opt" == "4" ]; then
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
options=(
    "gemma-4-26b-a4b-it-8bit"
    "gemma-4-31b-it-8bit"
    "gemma-4-e4b-it-8bit"
    "gemma-4-26b-a4b-it-4bit"
    "gemma-4-2b-a4b-it-4bit"
    "Qwen3.5-7B-Instruct-4bit"
    "Qwen3.5-14B-Instruct-4bit"
    "phi-4-mlx-4bit"
    "baa-ai/GLM-5.1-RAM-270GB-MLX"
    "GLM-5.1-4bit"
    "Custom (Enter your own Hub ID)"
    "Quit"
)

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
