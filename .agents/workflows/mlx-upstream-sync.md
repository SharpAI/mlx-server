---
description: How to synchronize Apple MLX ecosystem updates into SharpAI forks and triage SSD-streaming bugs
---

# Upstream MLX Synchronization & SSD Streaming Maintenance

This workflow documents the architecture for maintaining Apple MLX forks within the SharpAI repository ecosystem, executing upstream synchronization, and resolving bugs within the `ssd_streamer` custom extensions.

## 1. Ecosystem Architecture

The `mlx-server` repository now cleanly references the upstream Swift layer `SharpAI/mlx-swift` via Swift Package Manager (`SPM`).

```
mlx-server (SharpAI/SwiftLM)
│
└── SPM Dependency: SharpAI/mlx-swift (The Swift wrapper wrapper)
    ├── .gitmodules
    │   ├── submodules/mlx   -> https://github.com/SharpAI/mlx   (Branch: main)
    │   └── submodules/mlx-c -> https://github.com/SharpAI/mlx-c (Branch: main)
```

**Never bundle C++ source files directly into `mlx-swift`.** All Apple core Engine updates and C-wrapper modifications MUST be executed in the `SharpAI/mlx` and `SharpAI/mlx-c` forks respectively.

## 2. Upstream Feature Verification & Integration Flow

When Apple releases new features to `ml-explore/mlx` or `ml-explore/mlx-c`, follow this systematic process to verify, integrate, and validate the changes before bringing them into the SharpAI ecosystem.

### 2.1 Double-Checking Upstream Features

Before syncing, verify if Apple's upstream actually fulfills all your custom requirements (which informs whether you should safely drop your custom patches):

1. **Review Upstream Logging/Releases:** Actively monitor the [Apple MLX Releases page](https://github.com/ml-explore/mlx/releases) or the `main` commit history for mentions of "quantization", "streaming", "memory-mapped operations", or "out-of-core inference".
2. **Examine Target C++ Kernels:**
   - Look primarily in `mlx/backend/metal/` and `mlx/core/`.
   - Has upstream Apple added an equivalent to `moe_stream_op.cpp` natively?
   - Do the Metal shaders in `mlx/backend/metal/kernels/` natively introduce block execution / memory-mapped loading primitives similar to our `ssd_streamer.mm` and `fence.air` logic?
3. **Check Exported C-APIs:** Look at `mlx/c/ops.h` and `mlx/c/fast.h` in `ml-explore/mlx-c`. If Apple has added official C-bindings for out-of-core tensor operations, you can securely begin stripping out the custom SharpAI C++ bridging codebase.

### 2.2 Integration Flow

If Apple's features are highly beneficial (e.g., core Metal optimizations) but do not explicitly replace our SSD streaming, we need to pull their features *while maintaining* the SharpAI SSD kernels.

1. **Pull Upstream to SharpAI forks**:
   ```bash
   git clone https://github.com/SharpAI/mlx && cd mlx
   git remote add upstream https://github.com/ml-explore/mlx
   git fetch upstream
   
   # Rebase Apple's latest main directly under our custom SSD commits
   git rebase upstream/main
   # Resolve any merge conflicts specifically around `fast.cpp` or Make/CMake builds
   git push -f origin main
   ```
2. Execute the identical rebasing process for `SharpAI/mlx-c`, monitoring `mlx_c/ops.cpp`.
3. In `SharpAI/mlx-swift`, update the submodule pointers to mount your freshly rebased commits:
   ```bash
   cd LocalPackages/mlx-swift
   git submodule update --remote --recursive
   git commit -am "chore: sync latest Apple MLX components and re-graft SSD patches"
   git push origin main
   ```

### 2.3 Validation Flow

Do not deploy binary updates to the inference engine without executing the extreme validation matrix.

1. **Clean Re-Build:** Always execute a destructive cache wipe before a Metal compilation test.
   ```bash
   # In mlx-server framework
   rm -rf .build
   ./build.sh
   ```
2. **Swift API Layer Verification:** Run the test suites within your wrapper to certify that the Swift `->` C `->` C++ bindings remain structurally unified.
   ```bash
   cd LocalPackages/mlx-swift
   swift test
   ```
3. **Extreme Context Benchmarking (The Harness):**
   - Run the dedicated `/run-benchmark` workflow from the root `mlx-server` directory (utilizing `run_benchmark.sh` or `profile_runner.py`).
   - Specifically target models invoking >32k token contexts. High prompt generation latency, GPU thrashing, or hard Out-of-Memory (OOM) faults directly indicate that the Metal barrier (`fence.air`) or `ssd_streamer.mm` broke silently during the git rebase.

## 3. Triaging SSD-Stream Bugs

The SSD streaming kernels introduce custom memory synchronization routines (`ssd_streamer.h`, `ssd_streamer.mm`) that interact with Apple's core MLX framework (`mlx/core/moe_stream_op.cpp`). 

**Triage Protocol:**
- **Crash in Metal Execution (`fence.air`, `moe_stream.metal`)**: Identify if Apple's upstream Metal API (`mlx/backend/metal/device.h`) changed rendering assumptions. Navigate to `SharpAI/mlx` and patch `mlx/backend/metal/ssd_streamer.mm`.
- **C-API Mapping Errors (`fast.cpp`, `ops.cpp`)**: Swift throws errors linking to underlying kernels. Navigate to `SharpAI/mlx-c` and ensure `mlx/c/ops.cpp` cleanly wraps the updated arguments from `SharpAI/mlx`'s `moe_stream_op.h`.
- **Memory Leaks/High Swap Usage**: Typically arises if the `fence.air` streaming barrier lacks synchronization with the newly upstreamed Apple thread-pool executors.

## 4. Retiring the Fork (When to Drop)

> [!WARNING]
> The ultimate goal is to delete the `SharpAI/mlx` and `SharpAI/mlx-c` forks and point `SharpAI/mlx-swift` directly to `ml-explore/mlx` natively.

**Indications for Dropping the Fork:**
1. Apple officially merges Turbo Quant framework into `ml-explore/mlx/fast/turbo_quant.h` or equivalent upstream PR.
2. Apple natively supports out-of-core SSD context offloading (e.g., streaming inference blocks directly from Non-Volatile Memory to GPU) in `ml-explore/mlx/backend/metal/`.
3. If Apple's `moe_stream_op` native implementations match or exceed the latency speedups provided by your custom `ssd_streamer.mm`.

If any of these conditions are met, simply rewrite `SharpAI/mlx-swift/.gitmodules` back to `https://github.com/ml-explore/mlx` and delete your Github forks!

## 5. SharpAI Custom Patches Inventory (vs. Upstream ml-explore)

As of **April 2026**, the following specific features exist ONLY in our custom forks. Knowing precisely *what* we added is the key to knowing exactly *when* we can revert to Apple's native upstream (`ml-explore`).

### 🛠️ In `SharpAI/mlx` (C++ Engine)
*Compared to `ml-explore/mlx:main`*
1. `feat: custom ssd-streaming kernels and custom MLX I/O fast loaders`
   - Added `moe_stream_op` primitives enabling SSD flash streaming (out-of-core execution).
2. `fix(metal): align moe_stream_op add_temporary signature with latest apple upstream`
   - Custom extensions needed maintaining against newer MLX memory-pool updates.
3. `fix(metal): add default initialization loop for bound encoder contexts in async`
   - Patched `device.cpp` so thread pool reassignments by Swift's async engine don't result in fatal runtime aborts due to missing context dictionaries.

### 🛠️ In `SharpAI/mlx-c` (C-API Bridge)
*Compared to `ml-explore/mlx-c:main`*
1. `chore: rebase SharpAI custom ops onto latest Apple MLX-C upstream to fix fft/dequantize signatures`
2. `fix(ops): align c wrappers with mlx 0.30.0+ upstream signatures for dequantize, qqmm, and fft`
3. `fix(fft): restore Shape type for fft methods n parameter` & `fix(fft): remove invalid norm from fftshift calls`
   - Resolves signature drift and struct mismatches linking the new C++ API modifications down to Swift C headers.

### 🛠️ In `SharpAI/mlx-swift` (Swift Wrappers)
*Compared to `ml-explore/mlx-swift:main`*
1. `Restoration of missing MLX custom extensions including C-API and Swift bridge` & `Update custom C++ kernel patches for SSD Streaming`
   - Recreated Swift integrations bridging into out-of-core functionality.
2. `chore: isolate SharpAI custom MLX/MLX-C engines into dedicated GitHub forks`
   - Submodule remotes internally pinned from `ml-explore` tracking links to `SharpAI` ecosystem forks.
3. `fix(build): bump cxxLanguageStandard to .gnucxx20 for Apple MLX upstream compatibility`
   - Custom `Package.swift` override explicitly permitting C++20 standard since upstream didn't upgrade constraints simultaneously.
4. `fix(mlx): build steel_conv_3d C++ string for Cmlx target`
   - Added missing header dependencies specifically isolated by recent upstream migrations.
5. `fix(jit): update generated mlx c++ metal headers and fix fast.h signature to match fast.cpp`
   - Recompiled Metal header string buffers internally inside `mlx-generated` ensuring `affine_qmm_t_splitk` and other functions are dynamically injected at runtime.
