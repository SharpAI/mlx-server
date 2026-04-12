#!/bin/bash
# .agents/harness/audio-stft-pipeline/run_harness.sh
# Long-run harness for validating raw audio payload decoding natively via AVFoundation & MLX AudioProcessor

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
WORKSPACE_DIR="$REPO_ROOT"
LOG_DIR="$REPO_ROOT/.agents/harness/audio-stft-pipeline/runs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/harness_$TIMESTAMP.log"

echo "=========================================="
echo " Audio STFT Extraction Pipeline TDD Loop"
echo "=========================================="
echo "Initiating environment setup..."

cd "$WORKSPACE_DIR"

# Ensure we have a sample audio file
AUDIO_PATH="./tmp/stft_test_sample.wav"
mkdir -p tmp
if [ ! -f "$AUDIO_PATH" ]; then
    echo "Downloading and generating test audio..."
    curl -sL "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" -o "./tmp/stft_test_sample.mp3"
    afconvert -f WAVE -d LEI16 "./tmp/stft_test_sample.mp3" "$AUDIO_PATH"
fi

echo "Compiling test executable..."
swift build -c release > "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ [FAILED] Harness Compilation Terminated. See $LOG_FILE"
    exit 1
fi
echo "✅ [SUCCESS] Compiled SwiftLM"

echo "Executing STFT Validation Test..."
# We will use swift run with a specific target if available, or just use a custom script
# Assuming swift run SwiftLM --test-stft "$AUDIO_PATH" or a similar diagnostic flag exists
# For now, we utilize the integrated diagnostic script execution block natively via `swift read` or custom executable

# For our plan, we'll execute an isolated script target:
swift run -c release SwiftLMTestSTFT "$AUDIO_PATH" >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ [FAILED] STFT Benchmark Test completely failed or crashed. See $LOG_FILE"
    exit 1
fi

echo "✅ [SUCCESS] Harness execution completed correctly."
echo "View diagnostic logs at $LOG_FILE"
