// S4 gate: Lens Flux2VAE (reused) vs the ERNIE vae_decode golden.
// The ERNIE pipeline de-norms in PACKED space (bn stats, eps 1e-5) then
// channel-major-unpatchifies then decodes — exactly Lens's decodePackedLatents
// convention; this gate arbitrates any eps/packing doubt.
//
// Run: ERNIE_PARITY=1 swift test --filter VAEGateTests

import Foundation
import Lens
import MLX
import XCTest

@testable import ErnieImage

final class VAEGateTests: XCTestCase {
    func testDecodeGolden() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_PARITY"] == "1", "ERNIE_PARITY=1")

        let goldens = URL(
            fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/goldens")
        let modelDir = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo")

        let gold = try MLX.loadArrays(url: goldens.appendingPathComponent("vae_decode.safetensors"))
        let vae = try LensWeights.loadVAE(directory: modelDir.appendingPathComponent("vae"))

        // The golden was produced from PACKED latents; decodePackedLatents does the
        // bn de-norm + unpatchify + decode in one move. Feed the raw packed noise.
        let noise = try MLX.loadArrays(
            url: goldens.appendingPathComponent("latents.safetensors"))["noise"]!
        let decoded = vae.decodePackedLatents(noise.asType(.float32))
        let ref = gold["decoded"]!
        XCTAssertEqual(decoded.shape, ref.shape)

        let diff = decoded - ref
        let mse = mean(diff * diff)
        let psnr = (10 * log10(MLXArray(Float(4)) / mse)).item(Float.self)
        print("ERNIE VAE decode PSNR: \(psnr) dB")
        XCTAssertGreaterThanOrEqual(psnr, 60)
    }
}
