// swift-tools-version: 6.2
// ernie-image-swift — Swift/MLX mirror of baidu/ERNIE-Image-Turbo (Apache-2.0):
// the lightweight textToImage for lower-tier clients. 8B single-stream DiT +
// Mistral-3B text encoder (second-to-last hidden) + Flux2VAE (REUSED from
// lens-mlx-swift, parity-locked 120 dB). Turbo: 8 steps, guidance 1.0 (no CFG).
// Reference = diffusers 0.38 ErnieImagePipeline + mflux main (MLX); spec at
// /Volumes/DEV_ARCHIVE/ernie-image-mlx/PORTING-SPEC.md; goldens at
// VideoResearch/ernie-image-models/goldens.

import PackageDescription

let package = Package(
    name: "ErnieImage",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "ErnieImage", targets: ["ErnieImage"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Flux2VAE (decoder) — parity-locked in the Lens port; reused as-is.
        .package(path: "../lens-mlx-swift"),
    ],
    targets: [
        .target(
            name: "ErnieImage",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Lens", package: "lens-mlx-swift"),
            ],
            path: "Sources/ErnieImage"
        ),
        .testTarget(
            name: "ErnieImageTests",
            dependencies: ["ErnieImage"],
            path: "Tests/ErnieImageTests"
        ),
    ]
)
