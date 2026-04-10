# Audio Model Support Roadmap

SwiftLM currently has **zero audio support**. There is no audio input parsing, no speech encoder, no TTS output, and no `--audio` CLI flag.

This document outlines the implementation plan for adding audio capabilities alongside the existing LLM and VLM pipelines.

---

## Current State

| Modality | SwiftLM | `mlx-swift-lm` (upstream) | Ecosystem |
|---|---|---|---|
| **Text (LLM)** | ✅ Full | ✅ Full | Mature |
| **Vision (VLM)** | ✅ 14 architectures | ✅ 14 architectures | Mature |
| **Audio (ALM)** | ❌ None | ❌ None | Emerging — `mlx-audio-swift` exists as a separate SPM package |

> [!NOTE]
> The upstream Apple `ml-explore/mlx-swift-lm` library does **not** have an `MLXALM` module.
> Audio on MLX Swift is currently handled by the community `Blaizzy/mlx-audio-swift` package,
> which is a **separate** SPM dependency — not part of the LM pipeline.

---

## Why Audio Matters

Several next-generation models ship with native audio encoders embedded in their `config.json`:

| Model | `audio_config.model_type` | Audio Encoder | Notes |
|---|---|---|---|
| **Gemma 4** | `gemma4_audio` | 12-layer conformer, 1024-dim | Built-in audio tower alongside vision |
| **Qwen3-Omni** | `qwen3_omni_audio` | Whisper-style encoder | Unified text/vision/audio/speech |
| **Qwen3-ASR** | `whisper` | Whisper encoder | Speech-to-text specialist |

These models natively expect audio tokens alongside text and image tokens. Without audio support, SwiftLM silently drops the audio modality and falls back to text-only — losing a core capability.

---

## Implementation Plan

### Phase 1 — Audio Input Pipeline (Foundation)

**Goal**: Accept audio data via the OpenAI-compatible API and convert it to mel spectrograms.

| Component | Description | Estimated Effort |
|---|---|---|
| **CLI flag** | Add `--audio` flag (or auto-detect from `config.json` `audio_config`) | 1 day |
| **API endpoint** | Extend `/v1/chat/completions` to accept `input_audio` content parts (base64 WAV/PCM) | 1-2 days |
| **Mel spectrogram** | Implement FFT → mel filterbank conversion in Swift (or integrate `Accelerate.framework` vDSP) | 2-3 days |
| **Audio tokenizer** | Convert mel spectrograms into audio token embeddings via the model's audio encoder | 2-3 days |

### Phase 2 — Speech-to-Text (STT) Models

**Goal**: Run Whisper-class ASR models natively in SwiftLM.

| Model Family | `model_type` | Notes | Est. Effort |
|---|---|---|---|
| **Whisper** | `whisper` | Most popular ASR model. Reference exists in `mlx-audio-swift`. | ~3-4 days |
| **Qwen3-ASR** | `qwen3_asr` | Alibaba's speech recognition. | ~2-3 days |

### Phase 3 — Multimodal Audio Integration

**Goal**: Enable models that fuse text + vision + audio (like Gemma 4's full multimodal config).

| Model Family | Audio Tower | Notes | Est. Effort |
|---|---|---|---|
| **Gemma 4** | `gemma4_audio` (conformer) | Already have text LLM. Need audio encoder + fusion with existing vision projector. | ~4-5 days |
| **Qwen3-Omni** | Whisper-based | Full omni-modal: text + vision + audio + speech output. | ~5-7 days |

### Phase 4 — Text-to-Speech (TTS) Output

**Goal**: Generate speech audio from model output tokens.

| Component | Description | Estimated Effort |
|---|---|---|
| **Audio decoding** | Convert output audio tokens → waveform via vocoder (e.g., CosyVoice, Kokoro) | 3-5 days |
| **Streaming output** | Stream PCM/WAV audio chunks over the HTTP API as they are generated | 2-3 days |
| **API endpoint** | Add `/v1/audio/speech` endpoint (OpenAI TTS compatibility) | 1-2 days |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   SwiftLM Server                     │
│                                                      │
│  /v1/chat/completions                                │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Text Input  │  │ Image Input  │  │ Audio Input  │ │
│  │ (tokenizer) │  │ (CIImage →   │  │ (WAV → Mel → │ │
│  │             │  │  ViT embed)  │  │  Conformer)  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘ │
│         │               │                 │          │
│         └───────────┬────┘─────────────────┘          │
│                     ▼                                │
│         ┌───────────────────────┐                    │
│         │  Multimodal Projector │                    │
│         │  (token interleaving) │                    │
│         └───────────┬───────────┘                    │
│                     ▼                                │
│         ┌───────────────────────┐                    │
│         │   Language Model      │                    │
│         │   (Transformer)       │                    │
│         └───────────┬───────────┘                    │
│                     ▼                                │
│         ┌───────────────────────┐                    │
│         │  Output Router        │                    │
│         │  Text │ Audio tokens  │                    │
│         └───┬──────────┬────────┘                    │
│             ▼          ▼                             │
│         [Text out]  [Vocoder → WAV]                  │
└──────────────────────────────────────────────────────┘
```

---

## Shared Infrastructure Required

| Component | Status | Notes |
|---|---|---|
| Base64 decoding pipeline | ✅ Exists | Already handles images; extend for `audio/wav` MIME type |
| `CIImage` → MLXArray | ✅ Exists | Vision-specific; audio needs mel → MLXArray equivalent |
| OpenAI content parts parser | ✅ Exists | Supports `text` and `image_url`; needs `input_audio` support |
| Metal GPU acceleration | ✅ Exists | FFT/mel can use `Accelerate.framework` vDSP on CPU, encoder runs on Metal |
| `--vision` flag pattern | ✅ Exists | Same pattern for `--audio` or unified `--multimodal` |

---

## Integration Strategy

> [!IMPORTANT]
> **Option A: Build native** — Implement audio processing directly in SwiftLM using `Accelerate.framework` for FFT/mel and our own MLX encoder modules.
>
> **Option B: Integrate `mlx-audio-swift`** — Add `Blaizzy/mlx-audio-swift` as an SPM dependency for proven STT/TTS implementations, then wire into SwiftLM's server pipeline.
>
> **Recommendation**: Start with **Option A** for the audio input pipeline (mel spectrogram is straightforward with vDSP), then evaluate **Option B** for TTS output where vocoder complexity is high.

---

## Priority vs. VLM Roadmap

Audio support is **lower priority** than completing the VLM roadmap (Phase 1 VLM ports: Gemma 4 vision, Llama 4, Mistral 4, etc.). However, since Gemma 4 natively bundles both `vision_config` and `audio_config`, the Gemma 4 VLM port is the natural entry point for audio — both modalities can be developed together.

### Suggested Sequencing
1. ✅ Complete VLM Phase 1 (Gemma 4 vision, Llama 4, Mistral 4)
2. 🔜 Audio Phase 1 (mel spectrogram pipeline + API input support)
3. 🔜 Audio Phase 2 (Whisper STT)
4. 🔜 Audio Phase 3 (Gemma 4 multimodal fusion — vision + audio)
5. 🔮 Audio Phase 4 (TTS output)
