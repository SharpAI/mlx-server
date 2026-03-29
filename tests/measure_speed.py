import time
import urllib.request
import json

print("Starting inference request...", flush=True)
start = time.time()
try:
    data = json.dumps({
        "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5
    }).encode('utf-8')
    req = urllib.request.Request(
        "http://127.0.0.1:5413/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        result = response.read().decode('utf-8')
        duration = time.time() - start
        print(f"Status Code: {response.status}")
        print(f"Response: {result}")
        print(f"Duration: {duration:.2f} seconds")
except Exception as e:
    duration = time.time() - start
    print(f"Request failed: {e}")
    print(f"Failed after {duration:.2f} seconds")
