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

### GPU numerics: VAE decode route (2026-09-24)

The bf16 FLUX.2 decoder has 32 3×3 convs inside mlx's Metal Winograd conv2d window. The window
covers 3×3 convs with stride 1 and groups 1, where C%32 == O%32 == 0, C+O ≥ 256 and N·H·W ≥ 4096.
In bf16 that path loses about 5.8e-2 relL2 per conv, 35× the bf16 rounding floor.

`ErnieImagePackage` now sets `vae.convRoute = .fp32Winograd`, which runs just those convs in fp32
(flux2-vae-mlx-swift ≥ 0.2.0). Measured on the Lens 512² golden against its PyTorch fp32 decode:

| Route | PSNR | Max error | Extra decode time at 1024² |
|---|---|---|---|
| Before (raw Winograd) | 48.7 dB | 0.17 (22 of 255 levels) | — |
| `.fp32Winograd` (now) | 56.0 dB | 0.037 (5 levels) | +186 ms |
| `.conv3d` (for reference) | 57.3 dB | — | +677 ms |

Set `FLUX2VAE_CONV_ROUTE=winograd` to restore the old path for an A/B comparison.

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

## Run-phase progress (v0.2.2, AB-A-0139)

The package reports into the engine's `RunProgress` plane (contract 1.18.0), so a host shows
`denoise 5/8` instead of an unexplained "running". The phases, in order:

| phase | covers | steps? |
|---|---|---|
| `encode` | Mistral-3B conditioning, including the per-request encoder load unless it is kept resident | no |
| `denoise` | the flow-matching Euler loop: once at loop entry with no step, then after each completed step | **yes** — `step i/N`, 1-based |
| `decode` | FLUX.2 VAE decode — one eval, so no per-chunk counts | no |
| `postprocess` | PNG encode (reported by the wrapper, after the model is done) | no |

The core stays engine-free: `ErnieImage` reports into its own `ErnieImageProgress` task-local sink, and
`MLXErnieImage` forwards those events into `RunProgress` (`ErnieImageProgressBridge`). Nothing bound ⇒ no-op, so the
CLI and the parity gates run unchanged. A host binds a sink around `engine.run` (contract 1.51.0)
or reads `engine.runProgress`.

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
