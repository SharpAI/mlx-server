# SwiftBuddy Persistent Test Harness

This directory is the **single source of truth** for continuous TDD loops on the SwiftBuddy application. Any agent session (human or AI) can resume work by reading this spec, executing the loop, and writing results back.

## Quick Start

1. Read the relevant `features.md` to find all 🔲 TODO items
2. Read the matching `acceptance.md` for the exact pass/fail contract
3. Follow the **Agent Loop Protocol** defined in `harness_config.md`
4. Write results to the `runs/` directory after each execution

## Harnesses

| Harness | Path | Scope | Features |
|---------|------|-------|----------|
| Memory Handling | `memory/` | JSON extraction from LLM output. ExtractionService resilience. | 9 ✅ |
| Model Management | `model-management/` | HuggingFace search, MLX filtering, UI state correctness. | — |
| MemPalace Parity | `mempalace-parity/` | Feature parity with [milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) (v3.0.0). | — |
| **VLM Pipeline** | `vlm/` | Vision-Language Model loading, image parsing, multimodal inference, registry completeness. | 12 🔲 |
| **Audio Pipeline** | `audio/` | Audio input/output: mel spectrograms, Whisper STT, multimodal fusion, TTS vocoder. | 20 🔲 |

## File Conventions

- `features.md` — Feature registry with status tracking (🔲 TODO / 🔄 WIP / ✅ PASS / ❌ FAIL)
- `acceptance.md` — Per-feature completion criteria with exact input→output contracts
- `fixtures/` — Pre-written test inputs and mock data (standalone files, not hardcoded)
- `runs/` — Append-only timestamped run logs

## Rules

1. **Never delete run logs.** They are append-only history.
2. **Always update `features.md`** after a test passes or fails.
3. **Write the test before the implementation** (TDD).
4. **One feature at a time.** Don't batch-implement without verifying each.
5. **If a test fails after a code change**, fix the code, don't weaken the test.
