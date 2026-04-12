# VLM (Vision-Language Model) — Feature Registry

## Scope
SwiftLM must reliably load VLM models, parse multimodal image+text requests via the OpenAI-compatible API, route images through the vision encoder, and return valid completions. This harness validates the entire VLM pipeline end-to-end.

## Source Locations

| Component | Location |
|---|---|
| VLM model registry | `mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift` |
| VLM model implementations | `mlx-swift-lm/Libraries/MLXVLM/Models/` |
| Image extraction from API | `Sources/SwiftLM/Server.swift` (`extractImages()`) |
| CLI `--vision` flag | `Sources/SwiftLM/SwiftLM.swift` |
| Test validation script | `test_vlm.py` |

## Features

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 1 | `--vision` flag loads VLM instead of LLM | ✅ DONE | `testVLM_VisionFlagLoadsVLMFactory` | 2026-04-10 |
| 2 | Base64 data URI image extraction from multipart content | ✅ DONE | `testVLM_Base64ImageExtraction` | 2026-04-10 |
| 3 | HTTP URL image extraction from multipart content | ✅ DONE | `testVLM_HTTPURLImageExtraction` | 2026-04-10 |
| 4 | Reject request with no image when model requires one | ✅ DONE | `testVLM_RejectMissingImage` | 2026-04-10 |
| 5 | Text-only fallback when VLM receives no image | ✅ DONE | `testVLM_TextOnlyFallback` | 2026-04-10 |
| 6 | Valid JSON response from Qwen2-VL with real image | ✅ DONE | `testVLM_Qwen2VLEndToEnd` | 2026-04-10 |
| 7 | Image too small for ViT patch size returns graceful error | ✅ DONE | `testVLM_ImageTooSmallError` | 2026-04-10 |
| 8 | Multiple images in single message are all processed | ✅ DONE | `testVLM_MultipleImagesInMessage` | 2026-04-10 |
| 9 | VLM model type registry covers all 14 supported types | ✅ DONE | `testVLM_TypeRegistryCompleteness` | 2026-04-10 |
| 10 | VLM processor type registry covers all 14 supported types | ✅ DONE | `testVLM_ProcessorRegistryCompleteness` | 2026-04-10 |
| 11 | Unsupported model_type returns clear error (not crash) | ✅ DONE | `testVLM_UnsupportedModelType` | 2026-04-10 |
| 12 | Gemma 3 VLM loads and produces output | ✅ DONE | `testVLM_Gemma3EndToEnd` | 2026-04-10 |
