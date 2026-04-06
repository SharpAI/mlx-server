import argparse
import subprocess
import time
import urllib.request
import urllib.error
import json
import re
import signal
import sys
import os

CONFIGS = [
    {"name": "Dense/Vanilla", "flags": []},
    {"name": "SSD Stream", "flags": ["--stream-experts"]},
    {"name": "TurboQuant", "flags": ["--turbo-kv"]},
    {"name": "SSD + TurboQuant", "flags": ["--stream-experts", "--turbo-kv"]}
]

SWIFTLM_PATH = ".build/arm64-apple-macosx/release/SwiftLM"

def poll_health(port=5413, timeout=30):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(url)
            if r.getcode() == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def get_gpu_alloc_gb():
    """Query Apple GPU driver for total allocated system memory via ioreg.
    This value CAN exceed physical RAM — it includes memory swapped to SSD.
    It is the TRUE memory demand of the model + KV cache."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=5
        )
        alloc_match = re.search(r'"Alloc system memory"=(\d+)', result.stdout)
        in_use_match = re.search(r'"In use system memory"=(\d+)', result.stdout)
        alloc_gb = int(alloc_match.group(1)) / (1024**3) if alloc_match else 0
        in_use_gb = int(in_use_match.group(1)) / (1024**3) if in_use_match else 0
        return alloc_gb, in_use_gb
    except:
        return 0, 0

def make_request_stream(prompt_len, max_tokens, port=5413):
    prompt = "apple " * int(prompt_len * 0.75)
    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    ttft = None
    start = time.time()
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    payload = line[6:]
                    # Skip prefill heartbeat SSE chunks — only count real generation tokens
                    if "prefill_progress" in payload or "prefill" in payload:
                        continue
                    if ttft is None:
                        ttft = time.time() - start
                    tokens += 1
            total_time = time.time() - start
            gen_time = total_time - ttft if ttft else 0
            tps = (tokens - 1) / gen_time if gen_time > 0 and tokens > 1 else 0
            return True, ttft, tps
    except Exception as e:
        print(f"Request failed: {e}")
        return False, 0, 0

def extract_base_memory(log_path):
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Memory strategy: FULL GPU" in line:
                    m = re.search(r"\(([0-9.]+)GB model", line)
                    if m: return f"{m.group(1)} GB"
    except: pass
    return "N/A"

def extract_os_ram(log_path):
    """Get the last OS_RAM value from the server log (post-generation preferred)."""
    try:
        with open(log_path, 'r') as f:
            log_data = f.read()
            # Prefer post-generation ("slot done") over prefill
            post_vals = re.findall(r"slot done.*?OS_RAM=([0-9.]+)", log_data)
            if post_vals:
                return post_vals[-1]
            prefill_vals = re.findall(r"prefill done.*?OS_RAM=([0-9.]+)", log_data)
            if prefill_vals:
                return prefill_vals[-1]
    except: pass
    return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Aegis-AI Physical Model Profiler")
    parser.add_argument("--model", required=True, help="Model ID (e.g. gemma-4-26b-a4b-it-4bit)")
    parser.add_argument("--out", default="./profiling_results.md", help="Output markdown file path")
    parser.add_argument("--contexts", default="512", help="Comma-separated list of context lengths to test (e.g. 512,40000,100000)")
    args = parser.parse_args()
    
    # SwiftLM handles model downloading natively via HubApi.
    # Just pass the model ID directly — prepend mlx-community/ if no org is specified.
    model_id = args.model if "/" in args.model else f"mlx-community/{args.model}"

    
    context_sizes = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    results = []
    
    subprocess.run(["killall", "SwiftLM"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # Capture baseline GPU alloc before any model is loaded
    baseline_alloc, _ = get_gpu_alloc_gb()
    print(f"Baseline GPU alloc (no model): {baseline_alloc:.1f} GB")
    
    for config in CONFIGS:
        print(f"\n==============================================")
        print(f"--- Profiling {args.model} [{config['name']}] ---")
        print(f"==============================================")
        
        log_path = "./tmp/profile_server.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd = [SWIFTLM_PATH, "--model", model_id] + config["flags"]
        
        with open(log_path, "w") as root_log:
            server_proc = subprocess.Popen(cmd, stdout=root_log, stderr=subprocess.STDOUT)
        
        if not poll_health(timeout=60):
            print("Server failed to start.")
            server_proc.terminate()
            continue
            
        static_mem = extract_base_memory(log_path)
        
        for ctx_size in context_sizes:
            print(f"\n>> Running {ctx_size}-token context test (max generation ~20)...")
            ok, ttft, tps = make_request_stream(prompt_len=ctx_size, max_tokens=20)
            
            # Wait for server to flush post-generation logs
            time.sleep(1)
            
            os_ram = extract_os_ram(log_path)
            
            # Query Apple GPU driver for the TOTAL allocated memory (physical + swapped)
            gpu_alloc, gpu_in_use = get_gpu_alloc_gb()
            
            if ok:
                results.append({
                    "config": config["name"],
                    "context": ctx_size,
                    "ttft": f"{ttft:.2f}",
                    "tps": f"{tps:.2f}",
                    "static_mem": static_mem,
                    "os_ram": os_ram,
                    "gpu_alloc": f"{gpu_alloc:.1f}",
                    "gpu_in_use": f"{gpu_in_use:.1f}",
                })
                print(f"  TTFT={ttft:.2f}s  TPS={tps:.2f}  OS_RAM={os_ram}GB  GPU_Alloc={gpu_alloc:.1f}GB  GPU_InUse={gpu_in_use:.1f}GB")
            else:
                print(f"  FAILED / OOM")
                
        server_proc.send_signal(signal.SIGTERM)
        server_proc.wait(timeout=20)
        time.sleep(3)  # Let OS reclaim memory before next config
        
    with open(args.out, "w") as f:
        f.write(f"### `{args.model}` — Context & Memory Profile\n\n")
        f.write(f"Context depths tested: {args.contexts}\n\n")
        f.write("| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['config']} | {r['context']} | {r['ttft']}s | {r['tps']} tok/s | {r['static_mem']} | {r['os_ram']} GB | {r['gpu_alloc']} GB |\n")
        
        f.write(f"\n> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).\n")
        f.write(f"> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.\n")
            
    print(f"\nDone. Matrix saved to {args.out}")

if __name__ == "__main__":
    main()
