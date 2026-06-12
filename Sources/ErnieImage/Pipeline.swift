// ERNIE-Image-Turbo generation pipeline — mirror of diffusers ErnieImagePipeline
// (PE-off path): encode (second-to-last hidden) -> denoise (no CFG at guidance 1.0)
// -> bn de-norm + unpatchify + Flux2 decode (Lens decodePackedLatents).

import Foundation
import Flux2VAE
import MLX
import MLXRandom
import Tokenizers

public final class ErnieImageGenerator {
    public let encoder: ErnieTextEncoder
    public let transformer: ErnieImageTransformer2DModel
    public let vae: Flux2VAE
    public let tokenizer: any Tokenizers.Tokenizer

    public init(
        encoder: ErnieTextEncoder, transformer: ErnieImageTransformer2DModel,
        vae: Flux2VAE, tokenizer: any Tokenizers.Tokenizer
    ) {
        self.encoder = encoder
        self.transformer = transformer
        self.vae = vae
        self.tokenizer = tokenizer
    }

    /// Generate at (width, height) — both divisible by 16. Returns interleaved RGB8.
    public func generate(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 8,
        seed: UInt64 = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let ids = tokenizer.encode(text: prompt)
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)
        let text = encoder(inputIds)
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
