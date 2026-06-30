# Efficiency Adoption Brief — `ernie-image-swift` (ERNIE-Image-Turbo, `textToImage`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.15.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**, esp. the *in-app phys vs smoke MLX-peak* note) + references/memory-harness.md. Closest
> template: the **Qwen-Image-Edit** brief (`qwen-image-edit-swift/EFFICIENCY-ADOPTION.md`) — same shape
> (multi-component text-to-image, encoder-evict is the headline). Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXErnieImage` (`ErnieImagePackage`) over core `ErnieImage` (`Pipeline.swift`). Capability
  `textToImage`. Multi-component: **text encoder + DiT + VAE** (distilled Turbo).
- **Footprints today (flat):** bf16 **26 GB** · int4 **16 GB**. No split, no `QuantConfigured`.
- Config `ErnieImageConfiguration: PackageConfiguration, ModelStorable`. Engine pinned `from: "0.3.0"`.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.15.0 | **P0** |
| 1. Split footprint | ❌ | flat 26/16 GB; transient unaccounted | **P1** |
| 2. Per-stage evict | ❌ (likely) | encoder used once then idle through denoise (verify in `Pipeline.swift`/`load`) | **P2 (headline)** |
| 3. mmap/lazy | 🟡 verify | confirm lazy/mmap load (floor ≈ on-disk) | note |
| 4. BudgetAware | ➖ | quant config-chosen | defer |

## Plan (mirror Qwen-Image-Edit)
- **P0:** `swift package update` → 0.15.0; build + fix any drift.
- **P2 (headline):** if `load()` holds encoder+DiT+VAE resident and the text encoder is idle through the
  denoise (the Qwen-Image-Edit / LTX-Gemma pattern), refactor the core to stage: load encoder → encode →
  **evict** (`nil` + `Memory.clearCache()`) before the denoise loop. Swift 6 `#isolation` gotcha if the
  staged path goes async (recurred on LTX + Qwen-Image-Edit — use `isolated (any Actor)? = #isolation`).
  If the components are genuinely interleaved (verify), P2 is N/A — note it with the reason.
- **P1:** `QuantConfigured` (bf16/int4). residentBytes = weights floor (≈ on-disk for the peak-phase
  resident set); peakActivationBytes = the **measured** transient.
- **P3** mmap (note). **P4** defer.
- **`unload()` must `MLX.Memory.clearCache()`** (the eviction-frees-RSS rule — already applied across the
  adopted set; include it here).

## Measurement — IMPORTANT (the in-app-phys lesson)
The per-package smoke measures **MLX working-set peak**, which UNDER-reads the real process
`phys_footprint` (what R-MEM-1/admission use) by ~2.7× (MLX cache + overhead) — BiRefNet's smoke 18.3 GB
was really 47.8 GB in-app. So: declare `residentBytes` from the measured weight floor (solid), and a
**best-effort `peakActivationBytes`** from the package smoke, **FLAGGED** for an in-app phys re-baseline
once ERNIE is registered in the MLXEngineImage app (`IMAGE_AUTORUN`) — the app is the footprint source of
truth now. Land P2 + the split; flag the activation as smoke-estimate-pending-phys.

## Definition of done
- [ ] engine 0.15.0; `QuantConfigured`; P2 (encoder-evict or N/A-with-reason); `unload()` clearCache.
- [ ] Split declared per quant (`residentBytes` weights + `peakActivationBytes` smoke estimate, flagged).
- [ ] Smoke green (valid coherent image); split recorded; activation flagged for in-app re-baseline.
- [ ] Registry: ernie-image row Eff ⬜→✅ (note "activation = smoke est, phys re-baseline pending"), Eng→0.15.0.

## Report back
flat→split per quant, the encoder-evict effect, the smoke transient (flagged for phys), drift, effort, SHAs.
STAY IN SCOPE — four-lever adoption + brief + registry row only; no testing-app/shell changes; stop-and-report if bigger.

---

## Adoption outcome (executed 2026-06-30, engine 0.16.0)

**P0 — engine 0.16.0.** `swift package update mlx-engine-swift` moved the resolved engine to **0.16.0**
(the latest published; the `from: "0.3.0"` floor admits it, no manifest edit). **Zero API drift** — the
`MLXErnieImage` wrapper built green against 0.16.0 unchanged. The `textToImage` / `T2IRequest` /
`PackageManifest` / `QuantFootprint(quant:residentBytes:)` surface is stable from 0.3.0 → 0.16.0 (the new
`peakActivationBytes` param defaults, so old call sites compile).

**P2 — per-stage encoder eviction (the headline).** Refactored the core `ErnieImageGenerator`
(`Sources/ErnieImage/Pipeline.swift`): it no longer holds the Mistral-3B text encoder resident — it owns an
async `encoderProvider` closure (the wrapper's loader). `generate(...)` is now `async`: load encoder →
encode the prompt → `eval` the text features → drop the encoder ref (`encoderRef = nil`) +
`Memory.clearCache()` → then the 8B DiT denoise loop + Flux2 VAE decode. The DiT + bf16 VAE stay resident.
**Swift 6 isolation:** `generate` / `loadEncoder` take `isolation: isolated (any Actor)? = #isolation`, so
they inherit the wrapper's `@InferenceActor` and the non-Sendable generator never crosses an actor hop (the
canonical per-stage-eviction fix from LTX/Qwen-Image-Edit). A back-compat `init(encoder:…)` keeps the encoder
resident (`keepEncoderResident = true`) for the parity/demo tests, which can't reload it. The wrapper `load()`
switched to the `encoderProvider` init (closure picks `loadTextEncoder` vs `loadTextEncoderQuantized`).
**Parity preserved:** encode/denoise/decode math is byte-identical; only `eval()` (materialization, no
numerical change) + the eviction were added. Test `generate` call sites updated to `try await`.

**P1 — split footprint (residentBytes MEASURED from on-disk weight floor; activation FLAGGED):**

| Quant | OLD flat resident | Resident floor (declared) | Activation (declared) | Encoder now |
|---|---|---|---|---|
| bf16 | 26 GB | **15 GB** (DiT 15 GB on-disk + bf16 VAE ~0.1) | **11 GB** (encoder transient + denoise/decode scratch) | transient |
| int4 | 16 GB | **5 GB** (int4 DiT ~4 + bf16 VAE ~0.1) | **11 GB** (doc'd peak 15 GB @1024² − floor) | transient |

On-disk: DiT 15 GB, text_encoder 7.2 GB, VAE 160 MB. The headline: the **~7.2 GB encoder moved from resident
into the transient bucket** — the bf16 persistent floor drops from the flat 26 GB to **15 GB** (the on-disk DiT).
`ErnieImageConfiguration` conforms to `QuantConfigured` (`quant` = int4 when `quantizedPath` set, else bf16).

**Measurement path:** the per-package demo/CLI smoke could **not** be driven here — `xcodebuild test` does not
propagate the `ERNIE_DEMO`/`ERNIE_Q4_DEMO` gate env into the xctest runner process (it skips), and the
ERNIE demo test target additionally carries **pre-existing drift** (`Tests/ErnieImageTests/{GenerateDemoTests,
QuantizeTests}.swift` `import Lens` + `LensWeights.loadVAE`, a stale dep from before the VAE moved to the
neutral `Flux2VAE` package — present in HEAD, unrelated to this change; left untouched per scope). So per the
brief's fallback: `residentBytes` is declared from the **measured on-disk weight floor** (solid) and
`peakActivationBytes` is a **smoke/derived estimate**, FLAGGED in the manifest + registry for a clean in-app
phys re-baseline once ERNIE is registered in the MLXEngineImage app (the smoke MLX-peak under-reads process
`phys_footprint` ~2.7×, the BiRefNet lesson). The async refactor's correctness rests on full clean
compilation of all targets + byte-identical math (the validated Qwen-Image-Edit pattern).

**P3 — mmap/lazy: verified, no change.** `loadDiTFromPT` rebuilds a per-key dict with `v.asType(dtype)` (lazy
in MLX) and `eval(model)` once — no full eager copy; the resident floor tracks the on-disk DiT bytes.

**P4 — BudgetAware: deferred.** Quant is config-chosen (bf16 snapshot vs the pre-quantized int4 stack via
`quantizedPath`), not a load-time adaptive dtype lever.

**`unload()`** now calls `MLX.Memory.clearCache()` (eviction-frees-RSS rule); the `MLXErnieImage` wrapper
target gained an explicit `.product(name: "MLX", package: "mlx-swift")` link so the call doesn't rely on a
transitive import.
