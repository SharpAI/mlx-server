# Audio Model — Acceptance Criteria

Each feature below defines the exact input→output contract. A test passes **only** if the output matches the expectation precisely.

---

## Phase 1 — Audio Input Pipeline

### Feature 1: `--audio` CLI flag accepted
- **Input**: Launch SwiftLM with `--audio` flag
- **Expected**: Flag is parsed without error; server starts (may warn "no audio model loaded" if no model specified)
- **FAIL if**: Flag causes argument parsing error or crash

### Feature 2: Base64 WAV data URI extraction
- **Input**: Message content part with `{"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}}`
- **Expected**: `extractAudio()` returns valid PCM sample data
- **FAIL if**: Returns nil, crashes, or silently ignores the audio part

### Feature 3: WAV header parsing
- **Input**: 16-bit, 16kHz, mono WAV file (44-byte header + PCM data)
- **Expected**: Parser extracts: `sampleRate=16000`, `channels=1`, `bitsPerSample=16`, `dataOffset=44`
- **FAIL if**: Any header field is wrong, or parser crashes on valid WAV

### Feature 4: Mel spectrogram generation
- **Input**: 1 second of 440Hz sine wave at 16kHz sample rate (16000 samples)
- **Expected**: Output is a 2D MLXArray with shape `[80, N]` where N = number of frames
- **FAIL if**: Output shape is wrong, values are all zero, or function crashes
- **NOTE**: Use `Accelerate.framework` vDSP FFT for efficiency

### Feature 5: Mel spectrogram dimensions
- **Input**: 30 seconds of audio at 16kHz
- **Expected**: Output shape matches Whisper's expected `[80, 3000]` (80 mel bins, 3000 frames for 30s)
- **FAIL if**: Frame count doesn't match Whisper's hop_length=160 convention

### Feature 6: Long audio chunking
- **Input**: 90 seconds of audio
- **Expected**: Audio is split into 3 x 30-second chunks, each producing `[80, 3000]` mel spectrograms
- **FAIL if**: Single oversized tensor is created, or chunks overlap/drop samples

### Feature 7: Silent audio handling
- **Input**: 1 second of all-zero PCM samples
- **Expected**: Returns valid mel spectrogram (all low-energy values); no crash, no division-by-zero
- **FAIL if**: Function crashes, returns NaN, or throws

---

## Phase 2 — Speech-to-Text (STT)

### Feature 8: Whisper model type registered
- **Input**: Check `ALMTypeRegistry.shared` for key `"whisper"`
- **Expected**: Registry contains a valid model creator for `"whisper"`
- **FAIL if**: Key not found or creator returns nil

### Feature 9: Whisper encoder output
- **Input**: `[80, 3000]` mel spectrogram tensor
- **Expected**: Encoder returns hidden states tensor of shape `[1, 1500, encoder_dim]`
- **FAIL if**: Output shape is wrong or values are all zero

### Feature 10: Whisper decoder output
- **Input**: Encoder hidden states + start-of-transcript token
- **Expected**: Decoder generates a token ID sequence terminated by end-of-transcript
- **FAIL if**: Returns empty sequence, hangs, or crashes

### Feature 11: Transcription endpoint
- **Input**: POST `/v1/audio/transcriptions` with base64 WAV body
- **Expected**: Response JSON: `{"text": "..."}`
- **FAIL if**: Endpoint returns 404, 500, or malformed JSON

### Feature 12: Transcription accuracy
- **Input**: Known fixture WAV of "the quick brown fox"
- **Expected**: `text` field contains words matching the spoken content (fuzzy match acceptable)
- **FAIL if**: Completely wrong transcription or empty text
- **Fixture**: `fixtures/quick_brown_fox.wav`

---

## Phase 3 — Multimodal Audio Fusion

### Feature 13: Gemma 4 audio_config parsed
- **Input**: Gemma 4 `config.json` with `audio_config.model_type: "gemma4_audio"`
- **Expected**: Configuration struct correctly populates audio encoder fields (hidden_size=1024, num_hidden_layers=12, num_attention_heads=8)
- **FAIL if**: Audio config is nil or fields are zero/default

### Feature 14: Audio token interleaving
- **Input**: Text tokens `[101, 102]` + audio embeddings `[A1, A2, A3]` + `boa_token_id=255010` + `eoa_token_id=255011`
- **Expected**: Combined sequence: `[101, 102, 255010, A1, A2, A3, 255011]`
- **FAIL if**: Audio tokens are appended instead of interleaved at correct position

### Feature 15: Audio token boundaries
- **Input**: Audio segment with known `boa_token_id` and `eoa_token_id`
- **Expected**: `boa` token appears immediately before first audio embedding; `eoa` token appears immediately after last
- **FAIL if**: Boundary tokens are missing, duplicated, or in wrong position

### Feature 16: Trimodal request (text + vision + audio)
- **Input**: POST with text prompt + base64 image + base64 WAV audio
- **Expected**: All three modalities are parsed, encoded, and fused without crash; model produces output
- **FAIL if**: Any modality is silently dropped, or server crashes

---

## Phase 4 — Text-to-Speech (TTS) Output

### Feature 17: TTS endpoint accepts input
- **Input**: POST `/v1/audio/speech` with `{"input": "Hello world", "voice": "default"}`
- **Expected**: Response status 200 with `Content-Type: audio/wav`
- **FAIL if**: Returns 404, 500, or non-audio content type

### Feature 18: Vocoder output
- **Input**: Sequence of audio output tokens from language model
- **Expected**: Vocoder produces PCM waveform with valid sample values (not all zero, not NaN)
- **FAIL if**: Output is silence, contains NaN, or has wrong sample rate

### Feature 19: Valid WAV output
- **Input**: Generated PCM from vocoder
- **Expected**: Output has valid 44-byte WAV header with correct `sampleRate`, `bitsPerSample`, `dataSize`
- **FAIL if**: Header is malformed, file size doesn't match header, or file is not playable

### Feature 20: Streaming TTS output
- **Input**: POST `/v1/audio/speech` with `"stream": true`
- **Expected**: Response is chunked transfer-encoding with progressive PCM/WAV chunks
- **FAIL if**: Entire response is buffered before sending, or chunks have invalid boundaries
