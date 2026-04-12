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
| 1 | `--vision` flag loads VLM instead of LLM | 🔲 TODO | `testVLM_VisionFlagLoadsVLMFactory` | — |
| 2 | Base64 data URI image extraction from multipart content | 🔲 TODO | `testVLM_Base64ImageExtraction` | — |
| 3 | HTTP URL image extraction from multipart content | 🔲 TODO | `testVLM_HTTPURLImageExtraction` | — |
| 4 | Reject request with no image when model requires one | 🔲 TODO | `testVLM_RejectMissingImage` | — |
| 5 | Text-only fallback when VLM receives no image | 🔲 TODO | `testVLM_TextOnlyFallback` | — |
| 6 | Valid JSON response from Qwen2-VL with real image | 🔲 TODO | `testVLM_Qwen2VLEndToEnd` | — |
| 7 | Image too small for ViT patch size returns graceful error | 🔲 TODO | `testVLM_ImageTooSmallError` | — |
| 8 | Multiple images in single message are all processed | 🔲 TODO | `testVLM_MultipleImagesInMessage` | — |
| 9 | VLM model type registry covers all 14 supported types | 🔲 TODO | `testVLM_TypeRegistryCompleteness` | — |
| 10 | VLM processor type registry covers all 14 supported types | 🔲 TODO | `testVLM_ProcessorRegistryCompleteness` | — |
| 11 | Unsupported model_type returns clear error (not crash) | 🔲 TODO | `testVLM_UnsupportedModelType` | — |
| 12 | Gemma 3 VLM loads and produces output | 🔲 TODO | `testVLM_Gemma3EndToEnd` | — |
