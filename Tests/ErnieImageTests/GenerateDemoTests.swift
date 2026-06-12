// S5 e2e: full Swift Turbo render (eye gate). Saves ~/Desktop/ernie-swift-demo.png.
// Run: ERNIE_DEMO=1 swift test --filter GenerateDemoTests

import CoreGraphics
import Foundation
import ImageIO
import Lens
import MLX
import Tokenizers
import UniformTypeIdentifiers
import XCTest

@testable import ErnieImage

final class GenerateDemoTests: XCTestCase {
    func testTurboDemo() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_DEMO"] == "1", "ERNIE_DEMO=1")
        let modelDir = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo")

        let encoder = try ErnieImageWeights.loadTextEncoder(
            directory: modelDir.appendingPathComponent("text_encoder"))
        let transformer = try ErnieImageWeights.loadDiTFromPT(
            directory: modelDir.appendingPathComponent("transformer"))
        let vae = try LensWeights.loadVAE(
            directory: modelDir.appendingPathComponent("vae"))
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelDir.appendingPathComponent("tokenizer"))
        let generator = ErnieImageGenerator(
            encoder: encoder, transformer: transformer, vae: vae, tokenizer: tokenizer)

        let start = Date()
        let (pixels, w, h) = try generator.generate(
            prompt: "A red fox standing in tall golden grass at sunset, photorealistic wildlife photography",
            steps: 8, seed: 42,
            progress: { s, t in print("step \(s)/\(t)") })
        print("generated \(w)x\(h) in \(Date().timeIntervalSince(start))s")

        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/ernie-swift-demo.png")
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: w * h * 4)
        for i in 0..<(w * h) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        let dest = CGImageDestinationCreateWithURL(
            out as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        print("saved \(out.path)")
    }
}
