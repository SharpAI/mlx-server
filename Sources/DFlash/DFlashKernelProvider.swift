// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file

import Foundation
import MLX

/// Provider for DFlash specialized kernels.
public protocol DFlashKernelProvider: Sendable {
    func gatedDeltaKernelWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray,
        state: MLXArray, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray)
}

/// Registry to allow models to use DFlash kernels without module circular dependencies.
public struct DFlashKernelRegistry: Sendable {
    public nonisolated(unsafe) static var provider: DFlashKernelProvider? = nil
}
