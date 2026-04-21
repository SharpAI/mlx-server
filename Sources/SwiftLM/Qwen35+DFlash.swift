// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Bridge: Qwen35 models conform to DFlashTargetModel
//
// The dflash* methods are defined on Qwen35TextModel/Qwen35Model in the
// MLXLLM module. This file adds the DFlashTargetModel protocol conformance
// so the DFlash runtime can use them generically.

import DFlash
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Qwen35TextModel + DFlashTargetModel

extension Qwen35TextModel: DFlashTargetModel {}

// MARK: - Qwen35Model + DFlashTargetModel

extension Qwen35Model: DFlashTargetModel {}
