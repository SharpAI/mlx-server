#!/usr/bin/env python3
"""
mtp_bench.py — Test 13 MTP Speculative Decoding Benchmark
Uses the pre-built SwiftLM HTTP server (same approach as profile_runner.py / Test 1).
Boots the server for each config, sends real HTTP streaming requests with a properly-sized
dummy prompt to stress the KV cache, measures generation TPS (isolated from prefill/TTFT),
and prints a summary table.

Usage:
  python3 scripts/profiling/mtp_bench.py \
    --model mlx-community/gemma-4-26b-a4b-it-4bit \
    --contexts 512,40000,100000

Configs tested per context:
  - Vanilla                    (no flags)
  - Vanilla + MTP              (--mtp --num-mtp-tokens 4)
  - Vanilla + TurboQuant       (--turbo-kv)
  - Vanilla + MTP + TurboQuant (--mtp --num-mtp-tokens 4 --turbo-kv)
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

SWIFTLM_PATH = ".build/arm64-apple-macosx/release/SwiftLM"
PORT = 5430

CONFIGS = [
    {"name": "Vanilla",                     "flags": []},
    {"name": "Vanilla + MTP",               "flags": ["--mtp", "--num-mtp-tokens", "4"]},
    {"name": "Vanilla + TurboQuant",        "flags": ["--turbo-kv"]},
    {"name": "Vanilla + MTP + TurboQuant",  "flags": ["--mtp", "--num-mtp-tokens", "4", "--turbo-kv"]},
]

def get_gpu_alloc_gb():
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=5
        )
        alloc_match = re.search(r'"Alloc system memory"=(\d+)', result.stdout)
        in_use_match = re.search(r'"In use system memory"=(\d+)', result.stdout)
        alloc_gb = int(alloc_match.group(1)) / (1024**3) if alloc_match else 0.0
        in_use_gb = int(in_use_match.group(1)) / (1024**3) if in_use_match else 0.0
        return alloc_gb, in_use_gb
    except:
        return 0.0, 0.0

def extract_os_ram(log_path):
    try:
        with open(log_path, 'r') as f:
            log_data = f.read()
            post_vals = re.findall(r"slot done.*?OS_RAM=([0-9.]+)", log_data)
            if post_vals:
                return post_vals[-1]
            prefill_vals = re.findall(r"prefill done.*?OS_RAM=([0-9.]+)", log_data)
            if prefill_vals:
                return prefill_vals[-1]
    except:
        pass
    return "N/A"

def poll_health(server_proc, port, timeout=300):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    spinner = ["|", "/", "-", "\\"]
    spin_idx = 0
    while time.time() < deadline:
        if server_proc.poll() is not None:
            return False
        try:
            r = urllib.request.urlopen(url, timeout=2)
            if r.getcode() == 200:
                sys.stdout.write(f"\r  ✅ Model loaded!{' ' * 40}\n")
                sys.stdout.flush()
                return True
        except:
            pass
        spin_idx = (spin_idx + 1) % len(spinner)
        sys.stdout.write(f"\r  {spinner[spin_idx]} Waiting for model to load...")
        sys.stdout.flush()
        time.sleep(1)
    return False

def make_warmup_request(port):
    """
    Fire a short dummy request to prime Metal shader compilation.
    Without this, the first timed request carries ~1s of JIT overhead
    (visible as inflated TTFT on the vanilla 512-token run).
    """
    prompt = "apple " * 200  # ~200 tokens — enough to trigger all kernels
    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0.0,
        "stream": False,
    }).encode('utf-8')
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    try:
        urllib.request.urlopen(req, timeout=120)
    except Exception:
        pass  # warmup failures are non-fatal


def make_request_stream(prompt_len, max_tokens, port):
    """
    Send a chat completion request with a dummy prompt of `prompt_len` approximate tokens.
    Returns (ok, ttft_s, gen_tps, peak_gpu_in_use_gb, os_ram_gb).
    Measures TTFT separately from generation TPS so that long prefills don't distort the speed.
    """
    # Same approach as profile_runner.py: "apple " repeated to fill context
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

    peak_in_use = [0.0]
    poller_stop = threading.Event()

    def _poll_gpu():
        while not poller_stop.is_set():
            _, in_use = get_gpu_alloc_gb()
            if in_use > peak_in_use[0]:
                peak_in_use[0] = in_use
            poller_stop.wait(timeout=0.5)

    poller = threading.Thread(target=_poll_gpu, daemon=True)
    poller.start()

    ttft = None
    start = time.time()
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    payload = line[6:]
                    if "prefill_progress" in payload or "prefill" in payload:
                        continue
                    if ttft is None:
                        ttft = time.time() - start
                    tokens += 1
        total_time = time.time() - start
        gen_time = total_time - (ttft or 0)
        tps = (tokens - 1) / gen_time if gen_time > 0 and tokens > 1 else 0
        poller_stop.set()
        poller.join(timeout=2)
        return True, ttft, tps, peak_in_use[0]
    except Exception as e:
        print(f"\n  ❌ Request failed: {e}")
        poller_stop.set()
        poller.join(timeout=2)
        return False, 0, 0, 0.0

def main():
    parser = argparse.ArgumentParser(description="Gemma-4 MTP Speculative Decoding Benchmark (Test 13)")
    parser.add_argument("--model", required=True, help="Model HF ID")
    parser.add_argument("--contexts", default="512,40000,100000", help="Comma-separated context lengths")
    parser.add_argument("--max-tokens", type=int, default=60, help="Tokens to generate per run")
    args = parser.parse_args()

    model_id = args.model if "/" in args.model else f"mlx-community/{args.model}"
    context_sizes = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]

    bin_path = SWIFTLM_PATH
    if not os.path.exists(bin_path):
        alt = ".build/release/SwiftLM"
        if os.path.exists(alt):
            bin_path = alt
        else:
            print(f"❌ SwiftLM binary not found at {SWIFTLM_PATH}. Run ./build.sh first.")
            sys.exit(1)

    subprocess.run(["killall", "SwiftLM"], stderr=subprocess.DEVNULL)
    time.sleep(2)

    summary = []  # list of dicts

    for config in CONFIGS:
        print(f"\n{'='*62}")
        print(f"  Config: {config['name']}")
        print(f"{'='*62}")

        log_path = "./tmp/mtp_bench_server.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        cmd = [bin_path, "--model", model_id, "--port", str(PORT)] + config["flags"]
        print(f"  Starting: {' '.join(cmd[-4:])}")

        with open(log_path, "w") as log_f:
            server_proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

        is_healthy = poll_health(server_proc, PORT, timeout=600)
        if not is_healthy:
            print(f"  ❌ Server failed to start for config: {config['name']}")
            server_proc.terminate()
            server_proc.wait(timeout=10)
            for ctx in context_sizes:
                summary.append({"config": config["name"], "context": ctx,
                                 "ttft": "FAIL", "tps": "FAIL",
                                 "gpu_alloc": "N/A", "gpu_in_use_peak": "N/A", "os_ram": "N/A"})
            continue

        # Prime Metal shader compilation so first timed run isn't inflated by JIT overhead.
        sys.stdout.write("  🔥 Warming up Metal shaders...")
        sys.stdout.flush()
        make_warmup_request(PORT)
        sys.stdout.write(" done\n")
        sys.stdout.flush()

        for ctx in context_sizes:
            print(f"\n  >> Context={ctx} tokens (generating {args.max_tokens} tokens)...")
            ok, ttft, tps, peak_in_use = make_request_stream(
                prompt_len=ctx, max_tokens=args.max_tokens, port=PORT
            )
            time.sleep(1)  # let server flush logs
            os_ram = extract_os_ram(log_path)
            gpu_alloc, _ = get_gpu_alloc_gb()

            if ok:
                ttft_s = f"{ttft:.2f}" if ttft is not None else "N/A"
                print(f"     TTFT={ttft_s}s  TPS={tps:.1f}  OS_RAM={os_ram}GB  GPU_Alloc={gpu_alloc:.1f}GB  GPU_InUse(peak)={peak_in_use:.1f}GB")
                summary.append({
                    "config": config["name"], "context": ctx,
                    "ttft": ttft_s,
                    "tps": f"{tps:.1f}",
                    "gpu_alloc": f"{gpu_alloc:.1f}",
                    "gpu_in_use_peak": f"{peak_in_use:.1f}",
                    "os_ram": os_ram,
                })
            else:
                print(f"     ⚠️  [OOM/Crash] Request failed at context={ctx}")
                summary.append({"config": config["name"], "context": ctx,
                                 "ttft": "OOM", "tps": "OOM",
                                 "gpu_alloc": "N/A", "gpu_in_use_peak": "N/A", "os_ram": "N/A"})

        server_proc.send_signal(signal.SIGKILL)
        server_proc.wait(timeout=20)
        print("\n  [Teardown] Waiting 12s for macOS to reclaim GPU heap...")
        time.sleep(12)

    # ── Summary Table ────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  🏆 Gemma-4 MTP Speculative Decoding Summary")
    print(f"  Model: {model_id}")
    print(f"{'─'*80}")
    header = f"  {'Context':<10} | {'Configuration':<32} | {'TPS':>7} | {'TTFT':>6} | {'OS RAM':>8} | {'GPU Peak':>9}"
    print(header)
    print(f"  {'-'*78}")
    for row in summary:
        os_ram = f"{row['os_ram']} GB" if row['os_ram'] != 'N/A' else 'N/A'
        gpu_peak = f"{row['gpu_in_use_peak']} GB" if row['gpu_in_use_peak'] != 'N/A' else 'N/A'
        tps = f"{row['tps']} tok/s" if row['tps'] not in ('FAIL','OOM') else row['tps']
        ttft = f"{row['ttft']}s" if row['ttft'] not in ('FAIL','OOM','N/A') else row['ttft']
        print(f"  {str(row['context']):<10} | {row['config']:<32} | {tps:>10} | {ttft:>6} | {os_ram:>8} | {gpu_peak:>9}")
    print(f"{'─'*80}")

if __name__ == "__main__":
    main()
