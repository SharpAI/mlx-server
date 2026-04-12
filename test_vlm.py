import subprocess
import time
import requests
import json
import base64
import sys
import os

# The model to test (can be overridden via first command-line argument)
MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
if len(sys.argv) > 1:
    MODEL = sys.argv[1]

# 1. Download a proper 256x256 test image to bypass ViT 32px constraints!
IMAGE_PATH = "vlm_test_image.jpg"
if not os.path.exists(IMAGE_PATH):
    print("[0] Downloading valid validation image (256x256) to disk...")
    # Get a dependable small structural image from Wikimedia (the standard Lena or just a solid 256x256 pattern)
    img_data = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png/256px-Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png").content
    with open(IMAGE_PATH, 'wb') as f:
        f.write(img_data)

with open(IMAGE_PATH, 'rb') as f:
    encoded_image = base64.b64encode(f.read()).decode('utf-8')

print(f"\n[1] Spawning SwiftLM VLM instance for: {MODEL}...")
process = subprocess.Popen(
    ["./.build/release/SwiftLM", "--model", MODEL, "--vision"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print("[2] Waiting for server initialization...")

ready = False
for line in process.stdout:
    print(f"Server: {line.strip()}")
    # Stop looking for startup text once it hits the 'Ready' bound
    if "Listening on" in line or "Ready" in line:
        ready = True
        break
    if "error" in line.lower() or "fatal" in line.lower():
        print("Server encountered a fatal load error. It likely does not support Vision!")
        process.terminate()
        sys.exit(1)

if not ready:
    print("Server failed to start cleanly.")
    process.terminate()
    sys.exit(1)

print("\n[3] Server is live! Firing multi-modal Vision API Request...")
payload = {
    # Using 'vlm-test' overrides standard logic inside OpenAI SDKs since the actual model is resolved by the server
    "model": "vlm-test", 
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What animal is depicted in this image? Respond with purely the name of the animal."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ],
    "max_tokens": 15,
    "temperature": 0.0
}

try:
    print("    -> Waiting for model inference (spatial map + completion)...")
    response = requests.post("http://127.0.0.1:5413/v1/chat/completions", json=payload, timeout=300)
    print("\n[4] Output Received:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request failed: {e}")

print("\n[5] Tearing down server...")
process.terminate()
process.wait()
print("Pipeline Successfully Completed!")
