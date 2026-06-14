# ernie-image-swift

A Swift/MLX port of [baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)
(Apache-2.0) plus its MLXEngine **`textToImage`** package — the lightweight, lower-tier T2I backer:
fast, vivid 1024²-class generation at a fraction of the full-tier footprint.

- **`ErnieImage`** — the standalone inference port: 8B single-stream DiT (8 steps, guidance 1.0,
  no CFG) + Mistral-3B text encoder (second-to-last hidden, YaRN rope factor 16) + **Flux2VAE from
  the neutral [flux2-vae-mlx-swift](https://github.com/xocialize/flux2-vae-mlx-swift) package**
  (shared with Lens, parity-locked ~120 dB). Reference = diffusers 0.38 `ErnieImagePipeline` + mflux (MLX).
- **`MLXErnieImage`** — the thin MLXEngine wrapper (`ErnieImagePackage`, PackageID
  `ernie-image-turbo`): the canonical `T2IRequest`/`T2IResponse` surface, license declaration,
  requirements manifest, and PNG artifact encoding. The **second `textToImage` backer** alongside
  Lens, selected by `PackageID`.

## Parity

Validated against PyTorch fp32 goldens: scheduler exact · DiT step-0 0.9999996 (fp32-CPU) ·
encoder ids exact + YaRN inv_freq exact + hidden 0.9999969 · VAE 64 dB · e2e eye-verified. The
in-app int4 render is eye-verified at 1024² (~24 s; ~7.4 GB resident / ~15 GB peak on an M-series
Max). bf16 ≈ 22 GB resident.

## Use

```swift
import MLXErnieImage
import MLXToolKit

// int4 (lower tier): pass quantizedPath; bf16: leave it nil.
let package = ErnieImagePackage(configuration: .init(
    snapshotPath:  "<root>/ERNIE-Image-Turbo",
    quantizedPath: "<root>/ERNIE-Image-Turbo-4bit"))   // nil → bf16 from the snapshot
try await package.load()
let response = try await package.run(T2IRequest(
    prompt: "a lighthouse on a stormy coast at dusk",
    width: 1024, height: 1024, steps: 8, seed: 42)) as! T2IResponse
// response.image: canonical Image (.png)
```

Behavior notes: Turbo is distilled at guidance 1.0, so `guidanceScale` is ignored; dimensions are
rounded to the nearest /16. In int4 mode only the DiT + text encoder come from `quantizedPath`; the
tokenizer and VAE are still read from the bf16 `snapshotPath`, so both must be readable.

Pair with [ernie-pe-swift](https://github.com/xocialize/ernie-pe-swift) (the ERNIE Prompt Enhancer,
an `llm` package) to expand a brief prompt before rendering — the enhancer is t2i-agnostic.

## Status / consuming this package

The **MLX-converted** weights are not yet on the Hub (the PyTorch source `baidu/ERNIE-Image-Turbo`
is; these wrappers need MLX weights and do **not** download — they read a local snapshot) — set
`snapshotPath`/`quantizedPath`). The package itself is public + version-tagged on
github.com/xocialize: add by tagged URL
`.package(url: "https://github.com/xocialize/ernie-image-swift", from: "0.1.0")`. It consumes the
neutral **`flux2-vae-mlx-swift`** (FLUX.2 VAE, shared with Lens — **not** a dependency on lens; the
VAE was extracted into its own package per the no-model-package-depends-on-another rule) and the
engine contract **`mlx-engine-swift`** (`MLXToolKit`) as tagged-URL net dependencies, so it builds
standalone with no local sibling checkouts.

Apache-2.0 (weights) · MIT (port code).
