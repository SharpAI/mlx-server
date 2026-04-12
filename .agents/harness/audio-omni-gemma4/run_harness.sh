#!/bin/bash
# .agents/harness/audio-omni-gemma4/run_harness.sh
# Long-run harness for validating Gemma 4 Any-to-Any Integration 
# Ensure SwiftLM binary is accessible prior to executing.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
WORKSPACE_DIR="$REPO_ROOT"
LOG_DIR="$REPO_ROOT/.agents/harness/audio-omni-gemma4/runs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/harness_$TIMESTAMP.log"

echo "=========================================="
echo " Gemma 4 Omni (Any-to-Any) Harness Loop"
echo "=========================================="
echo "Initiating build..."

cd "$WORKSPACE_DIR"
swift build -c release > "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ [FAILED] Harness Compilation Terminated. See $LOG_FILE"
    exit 1
fi
echo "✅ [SUCCESS] Compiled SwiftLM"

# Check if model exists (mlx-community/gemma-4-e4b-it-4bit)
MODEL_NAME="mlx-community/gemma-4-e4b-it-4bit"
echo "Initializing Omni Benchmark via SwiftBuddy"

cat << EOF > "$LOG_DIR/omni_test_$TIMESTAMP.json"
{
  "messages": [
    {
      "role": "user",
      "content": "<|audio|> Please transcribe what you hear."
    }
  ],
  "model": "$MODEL_NAME",
  "mock_audio": true 
}
EOF

echo "Running Integration Pipeline against Omni Mock Generator..."

# Trigger the Omni Evaluation Test (Test 6) and select the 4bit Gemma model (Option 2) automatically
echo -e "6\n2\n" | HEADLESS=1 ./run_benchmark.sh >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ [FAILED] Benchmark Test completely failed or crashed. See $LOG_FILE"
    exit 1
fi

echo "✅ [SUCCESS] Harness execution completed perfectly."
echo "View diagnostic logs at $LOG_FILE"
