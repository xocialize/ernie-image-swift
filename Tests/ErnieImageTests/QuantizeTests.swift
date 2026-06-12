// P4b: convert bf16 -> 4-bit repos, then render with the quantized stack (eye gate).
//
// Convert: ERNIE_CONVERT=1 swift test --filter QuantizeTests/testConvert
// Render:  ERNIE_Q4_DEMO=1 swift test --filter QuantizeTests/testQuantizedDemo

import Foundation
import Lens
import MLX
import Tokenizers
import XCTest

@testable import ErnieImage

final class QuantizeTests: XCTestCase {
    static let modelDir = URL(
        fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo")
    static let q4Dir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo-4bit")

    func testConvert() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_CONVERT"] == "1", "ERNIE_CONVERT=1")

        let dit = try ErnieImageWeights.loadDiTFromPT(
            directory: Self.modelDir.appendingPathComponent("transformer"))
        try ErnieImageWeights.saveQuantized(
            model: dit, directory: Self.q4Dir.appendingPathComponent("transformer-4bit"),
            keepHi: ErnieImageWeights.ditKeepHi)
        print("[converted] transformer-4bit")

        let enc = try ErnieImageWeights.loadTextEncoder(
            directory: Self.modelDir.appendingPathComponent("text_encoder"))
        try ErnieImageWeights.saveQuantized(
            model: enc, directory: Self.q4Dir.appendingPathComponent("text_encoder-4bit"))
        print("[converted] text_encoder-4bit")
    }

    func testQuantizedDemo() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_Q4_DEMO"] == "1", "ERNIE_Q4_DEMO=1")

        let encoder = try ErnieImageWeights.loadTextEncoderQuantized(
            directory: Self.q4Dir.appendingPathComponent("text_encoder-4bit"))
        let transformer = try ErnieImageWeights.loadDiTQuantized(
            directory: Self.q4Dir.appendingPathComponent("transformer-4bit"))
        let vae = try LensWeights.loadVAE(
            directory: Self.modelDir.appendingPathComponent("vae"))
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: Self.modelDir.appendingPathComponent("tokenizer"))
        let generator = ErnieImageGenerator(
            encoder: encoder, transformer: transformer, vae: vae, tokenizer: tokenizer)

        let start = Date()
        let (pixels, w, h) = try generator.generate(
            prompt: "A red fox standing in tall golden grass at sunset, photorealistic wildlife photography",
            steps: 8, seed: 42)
        print("q4 generated \(w)x\(h) in \(Date().timeIntervalSince(start))s; peak \(GPU.peakMemory / 1_000_000) MB")

        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/ernie-swift-q4-demo.png")
        try GenerateDemoTests.writePNGHelper(pixels: pixels, width: w, height: h, to: out)
        print("saved \(out.path)")
    }
}
