// SSDStreamingRecovery.swift — Graceful handling of SSD expert streaming errors
// Lives in MLXInferenceCore so SwiftBuddy can reference these types directly,
// independent of which mlx-swift-lm package version is resolved.

import Foundation

/// Error surfaced when SSD expert streaming encounters a corrupted, truncated,
/// or incomplete safetensors file during pread I/O.
public struct SSDStreamingError: Error, LocalizedError {
    public let underlyingError: Error

    public init(underlyingError: Error) {
        self.underlyingError = underlyingError
    }

    public var errorDescription: String? {
        "MLX SSD Streaming Error: \(underlyingError.localizedDescription). The model safetensors file may be corrupted, truncated, or incomplete. Try re-downloading the model."
    }
}

/// Global error latch for SSD streaming errors that occur inside non-throwing
/// `callAsFunction` paths (e.g. SwitchGLU expert streaming).
///
/// Pattern:
///   1. `ThreadSafeError.check()` posts to this latch instead of calling fatalError.
///   2. The generation loop checks `throwIfSet()` after each token.
///   3. `InferenceEngine.generate()` catches the thrown error and surfaces it in the UI.
public final class SSDStreamingErrorLatch: @unchecked Sendable {
    public static let shared = SSDStreamingErrorLatch()
    private let lock = NSLock()
    private var _error: Error?

    /// Record an error (first-wins semantics).
    public func set(_ error: Error) {
        lock.withLock {
            if _error == nil { _error = error }
        }
    }

    /// Consume and return the recorded error, resetting the latch.
    /// Returns nil if no error was recorded.
    public func consume() -> Error? {
        lock.withLock {
            let e = _error
            _error = nil
            return e
        }
    }

    /// Throw the recorded error if one exists, then clear it.
    public func throwIfSet() throws {
        if let error = consume() {
            throw error
        }
    }
}
