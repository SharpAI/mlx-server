# SwiftBuddy Model Loading Recovery Plan

## Goal

Make SwiftBuddy's model lifecycle deterministic and recoverable for large MLX/Hugging Face models, especially partially downloaded or corrupted SSD-streamed MoE models such as `mlx-community/Qwen3.5-122B-A10B-4bit`.

The finished flow should feel like a local LLM studio:

- Selecting a missing model starts a visible download lifecycle.
- A verified local model loads into MLX exactly once.
- A corrupted model produces an actionable recovery prompt, not a silent retry loop.
- Context-window controls come from the model's real `config.json`, not a fallback while the model is unavailable.
- Repeated fixes can be validated with a fast harness before running the full app build.

## Current Failure Modes

1. `InferenceEngine.load(modelId:)` mixes download, verification, MLX loading, UI state, corruption recovery, and post-download behavior.
2. A failed or missing model can start `downloadManager.startDownload(modelId:)` and then return without a clear follow-up path to load after download completes.
3. `deleteCorruptedAndRedownload()` can immediately call back into `load(modelId:)`, which may re-enter the same mixed flow while old state is still visible.
4. Context UI falls back to `8192` when `config.json` is unavailable, which makes settings and bottom-right context readouts look authoritative when they are only unknown.
5. The corruption UI exists in multiple places but the desired decision point should be a single explicit modal: delete and re-download, or choose another model.
6. Root-level scratch scripts from previous diagnosis should not become the long-term test harness.

## Target Architecture

Introduce an explicit lifecycle around these states:

1. `missing`
2. `downloading`
3. `verifying`
4. `loading`
5. `ready`
6. `corrupted`
7. `failed`

This can be implemented incrementally without a large rewrite:

- Keep `ModelState` for UI compatibility in the first pass.
- Add small helper methods inside `InferenceEngine` to split the flow:
  - `loadVerifiedModel(modelId:)`
  - `downloadThenLoad(modelId:)`
  - `markCorrupted(modelId:message:)`
  - `recoverCorruptedModel(action:)`
- Make `ModelDownloadManager.startDownload` publish completion/failure in a way `InferenceEngine` can await.
- Keep `ModelStorage` as the source of truth for cache structure, refs, snapshots, and config parsing.

## Implementation Phases

### Phase 1: Harness And Fixtures

Create a fast harness that runs with an isolated `HF_HOME` and never modifies the user's real cache.

Coverage:

- `readMaxContextLength` reads nested `text_config.max_position_embeddings` for Qwen-style configs.
- `verifyModelIntegrity` rejects missing shards listed in `model.safetensors.index.json`.
- `verifyModelIntegrity` accepts minimal complete multi-shard fixtures.
- `isDownloaded` rejects `.incomplete` files and broken/incomplete snapshots.
- The harness can optionally run SwiftBuddy's Xcode build after focused tests pass.

### Phase 2: Storage Cleanup

Refine `ModelStorage` so all snapshot resolution uses one deterministic helper:

- Resolve `refs/main` first.
- Fall back to `snapshots/main` if it exists.
- Fall back to exactly one non-hidden snapshot only when refs are absent.
- Avoid selecting arbitrary hidden or partial directories.
- Remove unnecessary `try?` around non-throwing `URL.resolvingSymlinksInPath()`.

Acceptance:

- Focused cache tests pass.
- `swift test --filter ModelStorageCacheTests` has no new warnings from these files.

### Phase 3: Download/Load Orchestration

Split `InferenceEngine.load(modelId:)` into clear paths:

- If verified: load model only.
- If missing/incomplete: start or resume download and remain in a download state.
- If caller requested auto-load after download: await the task, verify, then load.
- If verification fails after download: mark corrupted and prompt.

Acceptance:

- Selecting a missing model shows download progress immediately.
- When the download completes successfully, the app either loads automatically or clearly asks the user to load, depending on the originating action.
- No path silently falls back to `8192` for an unavailable model.

### Phase 4: Corruption Recovery UI

Replace scattered corruption affordances with one primary recovery prompt.

Prompt actions:

- `Delete & Re-download`: unload, clear MLX/SSD streaming state, delete cache, start fresh download.
- `Choose Another Model`: dismiss corruption, open the unified model management UI.
- `Cancel`: keep error state without retrying.

Acceptance:

- Any SSD streaming error containing `pread`, `safetensors`, or `SSDStreamingError` sets `corruptedModelId` and shows the modal.
- Dismissing the modal does not erase the inline error unless the user explicitly chooses another model or recovery.
- Deletion failure is shown as a hard error and does not start a re-download.

### Phase 5: Context Controls

Model context UI should use a three-value model:

- Unknown: no verified config yet; hide context counter and use conservative settings labels.
- Known: show exact `max_position_embeddings` from config.
- Active: show active token estimate against known max context.

Acceptance:

- Qwen-style configs display `262144` when the config is present.
- Settings `Max Tokens` slider max is based on known model context only after the model/config is verified.
- Bottom-right context counter hides when no model is ready or context is unknown.

### Phase 6: Full Validation

Run the harness in layers:

```bash
scripts/debugging/model_loading_recovery_harness.sh --quick
scripts/debugging/model_loading_recovery_harness.sh --xcode
scripts/debugging/model_loading_recovery_harness.sh --full
```

Acceptance:

- Focused Swift tests pass.
- Xcode SwiftBuddy build succeeds.
- SwiftPM build/test path remains healthy.
- Manual app flow can reproduce corruption recovery without crashing or silently retrying.

## Notes From Other LLM Studio Patterns

The useful pattern from local LLM studio download flows is not a specific UI shape; it is lifecycle separation:

- Deduplicate active downloads by model ID.
- Treat backend/file download completion separately from runtime model load.
- Persist enough download state that UI can rediscover active or incomplete downloads.
- Surface recovery choices at the point of failure, rather than burying them in a status strip.

SwiftBuddy already has pieces of this through `ModelDownloadManager`, `activeDownloads`, and `incompleteDownloads`; the remaining work is making `InferenceEngine` consume that lifecycle instead of partially recreating it.
