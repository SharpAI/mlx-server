#!/bin/bash
BIN="./.build/arm64-apple-macosx/release/SwiftLM"
FULL_MODEL="mlx-community/gemma-4-e4b-it-4bit"
IMAGE_PATH="/tmp/test_image.jpg"
BASE64_IMG=$(base64 -i "$IMAGE_PATH" | tr -d '\n')
cat <<JSON_EOF > /tmp/vlm_payload.json
{
  "model": "$FULL_MODEL",
  "max_tokens": 100,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMG}"}}
      ]
    }
  ]
}
JSON_EOF

killall SwiftLM 2>/dev/null
$BIN --model "$FULL_MODEL" --vision --port 5431 > /tmp/vlm_test_server.log 2>&1 &
sleep 5
for i in {1..300}; do
    if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
    sleep 1
done

RAW_OUT=$(curl -sS --max-time 180 http://127.0.0.1:5431/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/vlm_payload.json)

echo "$RAW_OUT" | python3 -c "import sys,json;d=json.load(sys.stdin);print('vlm out:', d.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR').replace('\n', ' '))"
