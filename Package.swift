// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "mlx-swift-turboquant",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/joelnishanth/mlx-swift-lm", branch: "feature/turboquant"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.2"),
    ],
    targets: [
        .executableTarget(
            name: "TurboQuantBench",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/TurboQuantBench"
        ),
    ]
)
