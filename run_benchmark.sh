#!/bin/bash

# Ensure we execute from the project root
cd "$(dirname "$0")"

echo "=============================================="
echo "    Aegis-AI MLX Profiling Benchmark Suite    "
echo "=============================================="
echo ""

PS3="Select a model to benchmark (1-7): "
options=(
    "gemma-4-26b-a4b-it-4bit"
    "gemma-4-2b-a4b-it-4bit"
    "Qwen2.5-7B-Instruct-4bit"
    "Qwen2.5-14B-Instruct-4bit"
    "phi-4-mlx-4bit"
    "Custom (Enter your own Hub ID)"
    "Quit"
)

select opt in "${options[@]}"
do
    case $opt in
        "Custom (Enter your own Hub ID)")
            read -p "Enter HuggingFace ID (e.g., Llama-3.2-3B-Instruct-4bit): " custom_model
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

echo ""
read -p "Enter context lengths to test [default: 512,40000,100000]: " CONTEXTS
CONTEXTS=${CONTEXTS:-"512,40000,100000"}

echo ""
echo "=> Starting benchmark for $MODEL with contexts: $CONTEXTS"
echo ""

# Quick sanity check
if [ ! -f ".build/arm64-apple-macosx/release/SwiftLM" ] && [ ! -f ".build/release/SwiftLM" ]; then
    echo "⚠️  SwiftLM release binary not found! Please compile the project by running ./build.sh first."
    exit 1
fi

python3 -u scripts/profiling/profile_runner.py \
  --model "$MODEL" \
  --contexts "$CONTEXTS" \
  --out "./profiling_results_$(hostname -s).md"

echo ""
echo "✅ Benchmark finished! Results saved to ./profiling_results_$(hostname -s).md"
