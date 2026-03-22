// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mlx-server",
    platforms: [.macOS(.v14)],
    dependencies: [
        // Apple MLX Swift — core inference engine (Apple-maintained, tagged releases)
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        // Apple's LLM library built on MLX Swift (SharpAI fork)
        // Pinned to main branch for Qwen3.5 support (PRs #97, #120, #129, #133, #135 — not yet in a release tag)
        .package(url: "https://github.com/SharpAI/mlx-swift-lm", branch: "main"),
        // HuggingFace tokenizers + model download
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.2.0")),
        // Lightweight HTTP server (Apple-backed Swift server project)
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        // Async argument parser (for CLI flags: --model, --port)
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "mlx-server",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/mlx-server",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
