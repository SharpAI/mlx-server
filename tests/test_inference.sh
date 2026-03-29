#!/bin/bash
echo "=== Starting inference at $(date) ==="
START=$(python3 -c 'import time; print(time.time())')
curl -sf --max-time 600 -X POST http://127.0.0.1:5413/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3.5-122B-A10B-4bit","messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}' > curl_out.txt
RC=$?
END=$(python3 -c 'import time; print(time.time())')
DUR=$(python3 -c "print(f'{$END - $START:.2f}')")
echo "=== Done at $(date) ==="
echo "cURL Exit code: $RC"
echo "Duration: $DUR seconds"
cat curl_out.txt
echo ""
