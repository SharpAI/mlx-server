#!/usr/bin/env bash
# Fast repeatable harness for SwiftBuddy model-loading and recovery work.
# Runs focused cache/config tests inside an isolated HF_HOME so the user's real
# ~/.cache/huggingface cache is never modified.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT="${SWIFTBUDDY_MODEL_HARNESS_LOG_DIR:-$ROOT_DIR/.build/swiftbuddy-model-loading-harness}"
MODE="quick"

usage() {
    cat <<'USAGE'
Usage: scripts/debugging/model_loading_recovery_harness.sh [--quick|--xcode|--full]

Modes:
  --quick  Run syntax checks and focused ModelStorage cache/config tests.
  --xcode  Run quick mode, then build the SwiftBuddy Xcode target.
  --full   Run quick mode, Xcode build, and focused SwiftPM model lifecycle tests.

Environment:
  SWIFTBUDDY_MODEL_HARNESS_LOG_DIR  Override log directory.
  KEEP_HARNESS_CACHE=1              Preserve isolated HF_HOME after run.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) MODE="quick" ;;
        --xcode) MODE="xcode" ;;
        --full) MODE="full" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

mkdir -p "$LOG_ROOT"
HARNESS_HOME="$(mktemp -d "$LOG_ROOT/hf-home.XXXXXX")"
export HF_HOME="$HARNESS_HOME"

cleanup() {
    if [[ "${KEEP_HARNESS_CACHE:-0}" != "1" ]]; then
        rm -rf "$HARNESS_HOME"
    else
        echo "Preserved isolated HF_HOME: $HARNESS_HOME"
    fi
}
trap cleanup EXIT

run_step() {
    local name="$1"
    shift
    local log_file="$LOG_ROOT/${name}.log"
    echo "==> $name"
    echo "    log: $log_file"
    (cd "$ROOT_DIR" && "$@") >"$log_file" 2>&1 || {
        local status=$?
        echo "FAILED: $name (exit $status)" >&2
        tail -80 "$log_file" >&2 || true
        exit "$status"
    }
}

echo "SwiftBuddy model-loading recovery harness"
echo "mode:    $MODE"
echo "root:    $ROOT_DIR"
echo "logs:    $LOG_ROOT"
echo "HF_HOME: $HF_HOME"

run_step bash_syntax bash -n scripts/debugging/model_loading_recovery_harness.sh
run_step focused_storage_tests swift test --filter ModelStorageCacheTests

if [[ "$MODE" == "xcode" || "$MODE" == "full" ]]; then
    run_step xcode_swiftbuddy_build xcodebuild \
        -project SwiftBuddy/SwiftBuddy.xcodeproj \
        -scheme SwiftBuddy \
        -destination 'platform=macOS,arch=arm64' \
        build
fi

if [[ "$MODE" == "full" ]]; then
    run_step model_lifecycle_tests swift test --filter ModelLifecycleTests
fi

echo "Harness completed successfully. Logs are in $LOG_ROOT"
