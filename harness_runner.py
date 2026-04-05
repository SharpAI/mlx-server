import os
import subprocess
import time
import urllib.request
import urllib.error
import json
import re

MODELS = [
    "gemma-4-e4b-it-8bit",
    "gemma-4-26b-a4b-it-4bit",
    "gemma-4-31b-it-8bit"
]

CONFIGS = [
    {"name": "Dense/Vanilla", "flags": []},
    {"name": "SSD Stream", "flags": ["--stream-experts"]},
    {"name": "TurboQuant", "flags": ["--turbo-kv"]},
    {"name": "SSD + TurboQuant", "flags": ["--stream-experts", "--turbo-kv"]}
]

SWIFTLM_PATH = ".build/arm64-apple-macosx/release/SwiftLM"
DEEPCAMERA_SCRIPT = "/Users/simba/workspace/DeepCamera/skills/analysis/home-security-benchmark/scripts/run-benchmark.cjs"
RESULTS_FILE = "./benchmark_matrix_report.md"

def poll_health(port=5413, timeout=120):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(url)
            if r.getcode() == 200:
                print("Server is healthy!")
                return True
        except urllib.error.URLError:
            pass
        time.sleep(2)
    return False

def run_harness():
    print("Clearing background SwiftLM processes...")
    subprocess.run(["killall", "SwiftLM"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    results = []
    
    for model in MODELS:
        is_dense = "e4b" in model or "e2b" in model
        
        for config in CONFIGS:
            if is_dense and "--stream-experts" in config["flags"]:
                print(f"Skipping {config['name']} for dense model {model}")
                continue

            print(f"\n=============================================")
            print(f"Testing {model} | Config: {config['name']}")
            print(f"=============================================")
            
            model_path = f"/Users/simba/.aegis-ai/models/mlx_models/mlx-community/{model}"
            cmd = [SWIFTLM_PATH, "--model", model_path] + config["flags"]
            print(f"Starting SwiftLM: {' '.join(cmd)}")
            
            server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            if not poll_health():
                print(f"ERR: Server failed to start for {model}")
                server_proc.terminate()
                continue
            
            print("Running DeepCamera HomeSec-Bench...")
            bench_cmd = ["node", DEEPCAMERA_SCRIPT, "--llm", "http://127.0.0.1:5413"]
            bench_cwd = os.path.dirname(os.path.dirname(DEEPCAMERA_SCRIPT))
            bench_proc = subprocess.Popen(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=bench_cwd)
            
            success_rate = "N/A"
            average_tps = "N/A"
            
            for line in bench_proc.stdout:
                line = line.strip()
                if "Overall Success Rate:" in line:
                    match = re.search(r"Overall Success Rate:\s*([^\s]+)\s*\((.*?)\)", line)
                    if match:
                        success_rate = match.group(1)
                elif "Average Generation Speed:" in line:
                    match = re.search(r"Average Generation Speed:\s*([0-9.]+)", line)
                    if match:
                        average_tps = match.group(1)
            
            bench_proc.wait()
            
            server_proc.send_signal(subprocess.signal.SIGTERM)
            
            peak_ram = "N/A"
            for line in server_proc.stdout:
                if "Peak Memory" in line:
                    match = re.search(r"Peak Memory:\s*([0-9.]+)\s*GB", line)
                    if match:
                        peak_ram = match.group(1)
            
            server_proc.wait()
            
            res = {
                "model": model,
                "config": config["name"],
                "success_rate": success_rate,
                "average_tps": average_tps,
                "peak_ram": peak_ram
            }
            results.append(res)
            print(res)
            time.sleep(5)
            
    with open(RESULTS_FILE, "w") as f:
        f.write("### Benchmark Results (HomeSec-Bench)\n\n")
        f.write("| Model | Configuration | Success Rate | Avg TPS | Peak Memory (GB) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| `{r['model']}` | {r['config']} | {r['success_rate']} | {r['average_tps']} tok/s | {r['peak_ram']} GB |\n")

    print(f"\n[DONE] Matrix complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_harness()
