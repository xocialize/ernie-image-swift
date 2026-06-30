// ERNIE-Image-Turbo generation pipeline — mirror of diffusers ErnieImagePipeline
// (PE-off path): encode (second-to-last hidden) -> denoise (no CFG at guidance 1.0)
// -> bn de-norm + unpatchify + Flux2 decode (Lens decodePackedLatents).

import Foundation
import Flux2VAE
import MLX
import MLXRandom
import Tokenizers

/// End-to-end ERNIE-Image-Turbo generation.
///
/// **Per-stage residency (efficiency contract 1.14.0).** The Mistral-3B text encoder
/// (~6 GB bf16) is used ONCE per request to encode the prompt, then sits idle through the
/// entire DiT denoise loop and the VAE decode — the heaviest, longest phase. So the
/// generator does NOT hold the encoder resident: it owns an async `encoderProvider` (the
/// wrapper's loader), loads the encoder on demand, encodes, then **evicts it (`nil` +
/// `Memory.clearCache()`) before the denoise peak**. Only the 8B DiT (the resident floor +
/// the activation peak) and the small VAE stay resident. Tradeoff: the encoder re-loads per
/// request (cheap encode vs. expensive denoise) — a `keepEncoderResident` flag covers
/// big-RAM tiers (and the parity tests, which can't reload it).
public final class ErnieImageGenerator {
    /// Lazy loader for the Mistral-3B text encoder. Invoked per request, then evicted
    /// before the denoise peak (unless `keepEncoderResident`).
    public let encoderProvider: () async throws -> ErnieTextEncoder
    public let transformer: ErnieImageTransformer2DModel
    public let vae: Flux2VAE
    public let tokenizer: any Tokenizers.Tokenizer
    /// Keep the encoder resident across requests (skip per-request evict+reload). Default
    /// `false` = evict-between-stages, the memory-citizen default; `true` on big-RAM tiers.
    public let keepEncoderResident: Bool

    /// Hot encoder when `keepEncoderResident` is set (avoids the reload each request).
    private var residentEncoder: ErnieTextEncoder?

    /// Staged init: the encoder is loaded on demand via `encoderProvider`, not held resident.
    public init(
        encoderProvider: @escaping () async throws -> ErnieTextEncoder,
        transformer: ErnieImageTransformer2DModel,
        vae: Flux2VAE, tokenizer: any Tokenizers.Tokenizer,
        keepEncoderResident: Bool = false
    ) {
        self.encoderProvider = encoderProvider
        self.transformer = transformer
        self.vae = vae
        self.tokenizer = tokenizer
        self.keepEncoderResident = keepEncoderResident
    }

    /// Back-compat init from an already-loaded encoder. The encoder is kept resident (the
    /// pre-staged behavior) since the caller has no way to reload it. Prefer the
    /// `encoderProvider` init to get per-stage eviction.
    public convenience init(
        encoder: ErnieTextEncoder, transformer: ErnieImageTransformer2DModel,
        vae: Flux2VAE, tokenizer: any Tokenizers.Tokenizer
    ) {
        self.init(
            encoderProvider: { encoder }, transformer: transformer, vae: vae,
            tokenizer: tokenizer, keepEncoderResident: true)
    }

    /// Obtain the text encoder for this request. Reuses the hot encoder when
    /// `keepEncoderResident`, otherwise loads a fresh one via `encoderProvider`.
    private func loadEncoder(isolation: isolated (any Actor)? = #isolation) async throws
        -> ErnieTextEncoder
    {
        if keepEncoderResident, let residentEncoder { return residentEncoder }
        let encoder = try await encoderProvider()
        if keepEncoderResident { residentEncoder = encoder }
        return encoder
    }

    /// Drop the encoder's weights before the denoise peak. A no-op when keeping it resident;
    /// otherwise nils the caller's last strong reference and clears the buffer cache,
    /// reclaiming the ~6 GB before the DiT denoise loop.
    private func evictEncoder(_ encoder: inout ErnieTextEncoder?) {
        guard !keepEncoderResident else { return }
        encoder = nil           // release the encoder's MLXArrays (last strong ref)
        Memory.clearCache()     // return the freed buffers to the OS before denoise
    }

    /// Generate at (width, height) — both divisible by 16. Returns interleaved RGB8.
    public func generate(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 8,
        seed: UInt64 = 0,
        progress: ((Int, Int) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> (pixels: [UInt8], width: Int, height: Int) {
        // PER-STAGE EVICTION: load the Mistral-3B encoder, encode, force-materialize the
        // text features (`eval`), then drop the encoder + clear the cache BEFORE the denoise
        // loop so the ~6 GB encoder is not co-resident with the DiT activation peak.
        var encoderRef: ErnieTextEncoder? = try await loadEncoder()
        let ids = tokenizer.encode(text: prompt)
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)
        let text = encoderRef!(inputIds)
        eval(text)              // materialize off the encoder graph
        evictEncoder(&encoderRef)  // reclaim the ~6 GB before the DiT denoise peak
        let dtype = text.dtype

        let (lh, lw) = (height / 16, width / 16)
        let key = MLXRandom.key(seed)
        var latents = MLXRandom.normal([1, 128, lh, lw], key: key).asType(dtype)

        let sigmas = ErnieScheduler.sigmas(steps: steps)
        for i in 0..<steps {
            let t = MLXArray([sigmas[i] * ErnieScheduler.numTrainTimesteps])
            let pred = transformer(hiddenStates: latents, timestep: t, text: text)
            latents = latents + (sigmas[i + 1] - sigmas[i]) * pred
            eval(latents)
            progress?(i + 1, steps)
        }

        // Lens Flux2VAE: bn de-norm in packed space + unpatchify + decode.
        if ProcessInfo.processInfo.environment["ERNIE_MEM_TRACE"] == "1" {
            eval(latents)
            print("[mem] pre-decode peak \(GPU.peakMemory / 1_000_000) MB")
            GPU.resetPeakMemory()
        }
        // bf16 decode matches the Python reference (mflux runs the Flux2 VAE bf16
        // internally) and cuts the decode high-water ~2x vs fp32 — the lever that
        // fits the 4-bit variant on lower-tier working sets. bn de-norm stays fp32
        // inside the VAE (loadVAE pins bn stats fp32).
        let decoded = vae.decodePackedLatents(latents)  // (B,3,H,W)
        if ProcessInfo.processInfo.environment["ERNIE_MEM_TRACE"] == "1" {
            eval(decoded)
            print("[mem] decode-only peak \(GPU.peakMemory / 1_000_000) MB")
        }
        let img = clip((decoded + 1) * 127.5, min: 0, max: 255).asType(.uint8)
        let hwc = img[0].transposed(1, 2, 0)
        eval(hwc)
        return (hwc.asArray(UInt8.self), width, height)
    }
}
