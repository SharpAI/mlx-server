# VLM (Vision-Language Model) — Acceptance Criteria

Each feature below defines the exact input→output contract. A test passes **only** if the output matches the expectation precisely.

---

### Feature 1: `--vision` flag loads VLM instead of LLM
- **Input**: Launch SwiftLM with `--model mlx-community/Qwen2-VL-2B-Instruct-4bit --vision`
- **Expected**: Server log contains `Loading VLM (vision-language model)`
- **FAIL if**: Server loads as LLM or crashes on startup

### Feature 2: Base64 data URI image extraction
- **Input**: Message content part with `{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}`
- **Expected**: `extractImages()` returns a non-empty `[UserInput.Image]` array with a valid `CIImage`
- **FAIL if**: Returns empty array, crashes, or corrupts image data

### Feature 3: HTTP URL image extraction
- **Input**: Message content part with `{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}`
- **Expected**: `extractImages()` returns a valid image downloaded from the URL
- **FAIL if**: Returns empty array or fails silently

### Feature 4: Reject request with no image when model requires one
- **Input**: POST `/v1/chat/completions` with text-only content to a VLM server
- **Expected**: Response contains appropriate error or processes as text-only (model-dependent)
- **FAIL if**: Server crashes or returns HTTP 500

### Feature 5: Text-only fallback
- **Input**: POST text-only message to VLM server
- **Expected**: Server processes the request using only the language model (no vision encoder invoked)
- **FAIL if**: Server crashes or returns an image-required error for models that support text-only

### Feature 6: Qwen2-VL end-to-end inference
- **Input**: POST with a 256×256 test image (cat from Wikimedia) and prompt "What animal is in this image?"
- **Expected**: Response JSON has `choices[0].message.content` containing a non-empty string
- **FAIL if**: Response is an error, empty content, or HTTP timeout
- **Fixture**: `fixtures/vlm_test_image.jpg` (256×256 Wikimedia cat image)

### Feature 7: Image too small for ViT patch size
- **Input**: POST with a 1×1 pixel image to Qwen2-VL
- **Expected**: Response is a graceful JSON error: `imageProcessingFailure` with descriptive message
- **FAIL if**: Server crashes, returns HTTP 500, or hangs

### Feature 8: Multiple images in single message
- **Input**: POST with two `image_url` parts in the same message
- **Expected**: `extractImages()` returns an array with 2 images
- **FAIL if**: Only first image is extracted, or second is silently dropped

### Feature 9: VLM type registry completeness
- **Input**: Enumerate all keys in `VLMTypeRegistry.shared`
- **Expected**: Registry contains all 14 model types: `paligemma`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_5`, `qwen3_5_moe`, `idefics3`, `gemma3`, `smolvlm`, `fastvlm`, `llava_qwen2`, `pixtral`, `mistral3`, `lfm2_vl`, `lfm2-vl`, `glm_ocr`
- **FAIL if**: Any registered type is missing

### Feature 10: VLM processor type registry completeness
- **Input**: Enumerate all keys in `VLMProcessorTypeRegistry.shared`
- **Expected**: Registry contains matching processor for each model type
- **FAIL if**: A model type has no corresponding processor

### Feature 11: Unsupported model_type returns clear error
- **Input**: Attempt to load a model with `model_type: "nonexistent_model"`
- **Expected**: Throws `ModelFactoryError.unsupportedModelType("nonexistent_model")`
- **FAIL if**: Crashes, returns nil silently, or throws a different error type

### Feature 12: Gemma 3 VLM end-to-end
- **Input**: POST with 256×256 test image to Gemma 3 VLM server
- **Expected**: Response JSON has `choices[0].message.content` containing a non-empty string
- **FAIL if**: Model fails to load, crashes during inference, or returns empty content
- **NOTE**: Requires `mlx-community/gemma-3-4b-it-qat-4bit` to be cached locally
