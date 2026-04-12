# Gemma 4 Omni (USM) Audio Harness

This harness tracks the TDD lifecycle for porting Google's Universal Speech Model (USM) architecture natively to Apple Silicon via MLX Swift.

## Phase 1: MLX Swift Conformer Architecture
- [ ] Implement `Gemma4AudioConfiguration` with `subsampling_conv_channels`, `attention_chunk_size`
- [ ] Implement `SubsampleConvProjection` with dual GLU/Conv scaling.
- [ ] Implement `ConformerConvModule` mapped as `lconv1d` with `linear_start` and `linear_end`.
- [ ] Implement `MacaronFFN` layers (`feed_forward1`, `feed_forward2`) with `ffw_layer_1` and `ffw_layer_2` (ClippedLinears/Linears).
- [ ] Implement `ConformerBlock` tracking exact norm structures (`norm_out`, `norm_pre_attn`, `norm_post_attn`).
- [ ] Implement `Gemma4AudioModel` encapsulating `subsample_conv_projection` and `output_proj`.

## Phase 2: Feature Extraction Pipeline
- [ ] Scaffold `extractMelSpectrogram()` in `AudioProcessing.swift` or equivalent module to produce `[1, 80, SeqLen]` tensors.
- [ ] Write STFT windowing tests against an open source DSP reference vector.

## Phase 3: Graph Integration
- [ ] Update `Gemma4VL.swift` to instantiate `audioTower`.
- [ ] Define weight sanitization maps for `"audio_tower"` weight aliases in `sanitize(weights:)` method.
- [ ] Extend `prepareInputsForMultimodal()` to ingest `scaledAudioFeatures` via `maskedScatter()`.

## Phase 4: E2E Verification
- [ ] Load `mlx-community/gemma-4-e4b-it-8bit` using Omni Mode in test server.
- [ ] End-to-end verification via Swift Buddy Omni Audio suite payload.
