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

def get_physical_ram_gb():
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(result.stdout.strip()) / (1024**3)
    except:
        return 0

def get_hf_model_size_gb(model_id):
    try:
        url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as r:
            tree = json.loads(r.read().decode())
            total_bytes = sum(f.get('size', 0) for f in tree if f.get('path', '').endswith('.safetensors'))
            if total_bytes > 0: return total_bytes / (1024**3)
    except: pass
    
    try:
        url = f"https://huggingface.co/api/models/{model_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read().decode())
            if "safetensors" in data and "total" in data["safetensors"]:
                return data["safetensors"]["total"] / (1024**3)
    except: pass
    return 0.0

def get_hf_cache_bytes(model_id):
    """Scan the HuggingFace cache directory for total downloaded bytes for a model."""
    home = os.path.expanduser("~")
    folder_name = "models--" + model_id.replace("/", "--")
    
    cache_dirs = [
        os.path.join(home, ".cache/huggingface/hub", folder_name),
        os.path.join(home, "Library/Caches/huggingface/hub", folder_name),
    ]
    
    total = 0
    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            continue
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
                except:
                    pass
    return total

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

def poll_health(port=5413, timeout=30, model_id="", model_size_gb=0, check_overcommit_log=None, baseline_alloc=0, requires_dense_memory=False):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    total_bytes = int(model_size_gb * 1024**3) if model_size_gb > 0 else 0
    spin_idx = 0
    last_bytes = 0
    last_time = time.time()
    last_speed = 0.0
    downloading = False
    
    while time.time() - start < timeout:
        # ── Monitor download progress via filesystem ──
        if total_bytes > 0 and model_id:
            current_bytes = get_hf_cache_bytes(model_id)
            now = time.time()
            
            dt = now - last_time
            if dt >= 0.5:
                speed = (current_bytes - last_bytes) / dt / (1024**2) if dt > 0 else 0
                if speed > 0:
                    last_speed = speed
                last_bytes = current_bytes
                last_time = now
            
            pct = min(current_bytes / total_bytes * 100, 100) if total_bytes > 0 else 0
            downloaded_gb = current_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            
            if pct < 99.5 and downloaded_gb > 0.1:
                downloading = True
                bar_len = 25
                filled = int(pct / 100 * bar_len)
                bar_str = "=" * max(0, filled - 1) + (">" if filled > 0 else "") + " " * (bar_len - filled)
                spin_idx = (spin_idx + 1) % len(SPINNER)
                speed_str = f"{last_speed:.1f} MB/s" if last_speed > 0 else "..."
                
                sys.stdout.write(f"\r  {SPINNER[spin_idx]} Download: [{bar_str}] {pct:5.1f}%  {downloaded_gb:.1f} / {total_gb:.1f} GB  | {speed_str}   ")
                sys.stdout.flush()
                start = time.time()  # Reset timeout — download is active
            elif downloading and pct >= 99.5:
                sys.stdout.write(f"\r  ✅ Download complete: {downloaded_gb:.1f} GB{' ' * 50}\n")
                sys.stdout.flush()
                downloading = False
        
        # ── Fallback overcommitment check from server log ──
        if requires_dense_memory and check_overcommit_log and os.path.exists(check_overcommit_log):
            try:
                with open(check_overcommit_log, "r") as f:
                    for line in f:
                        m = re.search(r"\(([0-9.]+)GB model\)", line)
                        if m:
                            model_gb = float(m.group(1))
                            phys_ram_gb = get_physical_ram_gb()
                            if phys_ram_gb > 0:
                                demand = model_gb + baseline_alloc
                                if demand > phys_ram_gb * 1.30:
                                    if downloading:
                                        sys.stdout.write("\n")
                                    print(f"\n  [Abort] Configuration requires {demand:.1f}GB. Exceeds physical RAM ({phys_ram_gb:.1f}GB) by >30%.")
                                    return False, True
            except: pass

        try:
            r = urllib.request.urlopen(url)
            if r.getcode() == 200:
                if downloading:
                    sys.stdout.write(f"\r  ✅ Model loaded!{' ' * 60}\n")
                    sys.stdout.flush()
                return True, False
        except:
            pass
        time.sleep(1)
        
    if downloading:
        sys.stdout.write("\n")
    return False, False

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
                m = re.search(r"\(([0-9.]+)GB model\)", line)
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
    
    model_size_gb = get_hf_model_size_gb(model_id)
    if model_size_gb > 0:
        print(f"Model Framework Size: {model_size_gb:.1f} GB (via Hugging Face API)")
    else:
        print("Model Framework Size: Unknown (failed to fetch from API)")
    
    for config in CONFIGS:
        print(f"\n==============================================")
        print(f"--- Profiling {args.model} [{config['name']}] ---")
        print(f"==============================================")
        
        requires_dense_memory = "--stream-experts" not in config["flags"]
        
        # 1) PRE-BOOT Check: If we know the size from HF API, skip early to avoid freezing the system!
        if requires_dense_memory and model_size_gb > 0:
            phys_ram_gb = get_physical_ram_gb()
            if phys_ram_gb > 0:
                demand = model_size_gb + baseline_alloc
                if demand > phys_ram_gb * 1.30:
                    print(f"  [Abort] Early pre-boot check shows config requires {demand:.1f}GB demand.")
                    print(f"  This exceeds physical RAM ({phys_ram_gb:.1f}GB) by >30%.")
                    print(f"  > Skipping {config['name']} to protect system stability.")
                    continue
        
        log_path = "./tmp/profile_server.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd = [SWIFTLM_PATH, "--model", model_id] + config["flags"]
        
        with open(log_path, "w") as root_log:
            server_proc = subprocess.Popen(cmd, stdout=root_log, stderr=subprocess.STDOUT)
        
        requires_dense_memory = "--stream-experts" not in config["flags"]
        is_healthy, overcommitted = poll_health(
            port=5413, 
            timeout=120,
            model_id=model_id,
            model_size_gb=model_size_gb,
            check_overcommit_log=log_path, 
            baseline_alloc=baseline_alloc, 
            requires_dense_memory=requires_dense_memory
        )
        
        if not is_healthy:
            if not overcommitted:
                print("Server failed to start.")
            server_proc.terminate()
            server_proc.wait(timeout=5)
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
        
    # ── Write markdown report ──
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
    
    # ── Console visualization ──
    if results:
        print_visualization(results, args.model, baseline_alloc)


# ══════════════════════════════════════════════════════════════════════════════
#  Console Visualization
# ══════════════════════════════════════════════════════════════════════════════

# ANSI color codes
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Foreground
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    # Background
    BG_BLUE = "\033[44m"
    BG_MAG  = "\033[45m"

CONFIG_COLORS = {
    "Dense/Vanilla":    C.BLUE,
    "SSD Stream":       C.CYAN,
    "TurboQuant":       C.MAGENTA,
    "SSD + TurboQuant": C.GREEN,
}

def bar(value, max_val, width=30, fill="█", empty="░", color=""):
    if max_val <= 0:
        filled = 0
    else:
        filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return f"{color}{fill * filled}{C.DIM}{empty * (width - filled)}{C.RESET}"

def print_visualization(results, model_name, baseline_alloc):
    W = 72  # box width

    print()
    print(f"{C.BOLD}{C.CYAN}{'═' * W}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'  BENCHMARK RESULTS':^{W}}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * W}{C.RESET}")
    print(f"{C.DIM}  Model: {model_name}  |  Baseline GPU: {baseline_alloc:.1f} GB{C.RESET}")
    print(f"{C.CYAN}{'─' * W}{C.RESET}")

    # Group results by context size
    ctx_sizes = sorted(set(r["context"] for r in results))

    # ── 1) Generation Speed (TPS) ──
    print(f"\n{C.BOLD}  ⚡ Generation Speed (tokens/sec) — higher is better{C.RESET}")
    print(f"{C.DIM}  {'─' * (W - 4)}{C.RESET}")
    
    all_tps = [float(r["tps"]) for r in results if r["tps"] != "N/A"]
    max_tps = max(all_tps) if all_tps else 1

    for ctx in ctx_sizes:
        ctx_results = [r for r in results if r["context"] == ctx]
        ctx_label = f"{ctx:,} tokens"
        print(f"\n  {C.BOLD}{C.WHITE}{ctx_label}{C.RESET}")
        for r in ctx_results:
            tps_val = float(r["tps"])
            color = CONFIG_COLORS.get(r["config"], "")
            label = f"    {r['config']:<20}"
            b = bar(tps_val, max_tps, width=28, color=color)
            val_str = f"{C.BOLD}{tps_val:>6.1f}{C.RESET} tok/s"
            # Highlight the best TPS per context group
            best_in_ctx = max(float(x["tps"]) for x in ctx_results)
            crown = f" {C.YELLOW}★{C.RESET}" if tps_val == best_in_ctx and len(ctx_results) > 1 else ""
            print(f"{label} {b} {val_str}{crown}")

    # ── 2) Time to First Token (TTFT) ──
    print(f"\n{C.BOLD}  ⏱  Time to First Token (seconds) — lower is better{C.RESET}")
    print(f"{C.DIM}  {'─' * (W - 4)}{C.RESET}")
    
    all_ttft = [float(r["ttft"]) for r in results if r["ttft"] != "N/A"]
    max_ttft = max(all_ttft) if all_ttft else 1

    for ctx in ctx_sizes:
        ctx_results = [r for r in results if r["context"] == ctx]
        ctx_label = f"{ctx:,} tokens"
        print(f"\n  {C.BOLD}{C.WHITE}{ctx_label}{C.RESET}")
        for r in ctx_results:
            ttft_val = float(r["ttft"])
            color = CONFIG_COLORS.get(r["config"], "")
            label = f"    {r['config']:<20}"
            b = bar(ttft_val, max_ttft, width=28, color=color)
            val_str = f"{C.BOLD}{ttft_val:>7.2f}{C.RESET}s"
            best_in_ctx = min(float(x["ttft"]) for x in ctx_results)
            crown = f" {C.YELLOW}★{C.RESET}" if ttft_val == best_in_ctx and len(ctx_results) > 1 else ""
            print(f"{label} {b} {val_str}{crown}")

    # ── 3) GPU Memory Demand ──
    print(f"\n{C.BOLD}  💾 GPU Memory Allocated (GB) — lower is better{C.RESET}")
    print(f"{C.DIM}  {'─' * (W - 4)}{C.RESET}")
    
    all_gpu = [float(r["gpu_alloc"]) for r in results if r["gpu_alloc"] != "N/A"]
    max_gpu = max(all_gpu) if all_gpu else 1

    for ctx in ctx_sizes:
        ctx_results = [r for r in results if r["context"] == ctx]
        ctx_label = f"{ctx:,} tokens"
        print(f"\n  {C.BOLD}{C.WHITE}{ctx_label}{C.RESET}")
        for r in ctx_results:
            gpu_val = float(r["gpu_alloc"])
            color = CONFIG_COLORS.get(r["config"], "")
            label = f"    {r['config']:<20}"
            b = bar(gpu_val, max_gpu, width=28, color=color)
            val_str = f"{C.BOLD}{gpu_val:>6.1f}{C.RESET} GB"
            best_in_ctx = min(float(x["gpu_alloc"]) for x in ctx_results)
            crown = f" {C.YELLOW}★{C.RESET}" if gpu_val == best_in_ctx and len(ctx_results) > 1 else ""
            print(f"{label} {b} {val_str}{crown}")

    # ── 4) Summary scoreboard ──
    print(f"\n{C.CYAN}{'─' * W}{C.RESET}")
    print(f"{C.BOLD}  🏆 Configuration Ranking (by avg TPS across all contexts){C.RESET}")
    print(f"{C.DIM}  {'─' * (W - 4)}{C.RESET}")

    config_avg = {}
    for cfg_name in set(r["config"] for r in results):
        tps_vals = [float(r["tps"]) for r in results if r["config"] == cfg_name]
        config_avg[cfg_name] = sum(tps_vals) / len(tps_vals) if tps_vals else 0

    ranked = sorted(config_avg.items(), key=lambda x: x[1], reverse=True)
    medals = ["🥇", "🥈", "🥉", "  "]
    
    for i, (cfg_name, avg_tps) in enumerate(ranked):
        medal = medals[min(i, 3)]
        color = CONFIG_COLORS.get(cfg_name, "")
        avg_gpu = sum(float(r["gpu_alloc"]) for r in results if r["config"] == cfg_name) / max(1, len([r for r in results if r["config"] == cfg_name]))
        print(f"  {medal} {color}{C.BOLD}{cfg_name:<22}{C.RESET}  avg {avg_tps:>5.1f} tok/s  |  avg {avg_gpu:>5.1f} GB GPU")

    print(f"\n{C.CYAN}{'═' * W}{C.RESET}")
    print()


if __name__ == "__main__":
    main()

