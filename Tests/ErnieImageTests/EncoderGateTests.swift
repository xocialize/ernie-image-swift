// S3 gates: YaRN inv_freq exact vs dump -> tokenizer ids exact -> true-SDPA q/k
// (max_abs) -> second-to-last hidden (cosine).
//
// Run: ERNIE_PARITY=1 [ERNIE_FP32_CPU=1] swift test --filter EncoderGateTests

import Foundation
import MLX
import Tokenizers
import XCTest

@testable import ErnieImage

final class EncoderGateTests: XCTestCase {
    static let goldens = URL(
        fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/goldens")
    static let modelDir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo")

    func testYarnInvFreqExact() throws {
        let rope = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder_rope.safetensors"))
        let ref = rope["inv_freq"]!.asArray(Float.self)
        let ours = ErnieYarnRope.invFreq()
        XCTAssertEqual(ours.count, ref.count)
        for (a, b) in zip(ours, ref) {
            XCTAssertEqual(a, b, accuracy: max(1e-6, abs(b) * 1e-5))
        }
        XCTAssertEqual(
            rope["attention_scaling"]!.asArray(Float.self)[0], 1.0, accuracy: 1e-6)
    }

    func testEncoderStages() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_PARITY"] == "1", "ERNIE_PARITY=1")
        let fp32 = ProcessInfo.processInfo.environment["ERNIE_FP32_CPU"] == "1"
        if fp32 { Device.setDefault(device: Device(.cpu)) }
        let dtype: DType = fp32 ? .float32 : .bfloat16

        let enc = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder.safetensors"))
        let sdpa = try MLX.loadArrays(
            url: Self.goldens.appendingPathComponent("encoder_sdpa.safetensors"))
        let metaData = try Data(
            contentsOf: Self.goldens.appendingPathComponent("goldens_meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let prompt = meta["prompt"] as! String

        // 1. Tokenizer: ids exact.
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: Self.modelDir.appendingPathComponent("tokenizer"))
        let ids = tokenizer.encode(text: prompt)
        let refIds = enc["input_ids"]!.asArray(Float.self).map { Int($0) }
        XCTAssertEqual(ids, refIds, "token ids differ from golden")

        // 2. Model: layer-0 post-rope q/k vs the TRUE SDPA capture (fp32 regime).
        let model = try ErnieImageWeights.loadTextEncoder(
            directory: Self.modelDir.appendingPathComponent("text_encoder"), dtype: dtype)
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)
        let embeds = model.embedTokens(inputIds)
        let (cosT, sinT) = ErnieYarnRope.cosSin(length: ids.count)
        let l0 = model.layers[0]
        let (q, k) = l0.attention.ropedQK(l0.inputNorm(embeds), cos: cosT, sin: sinT)
        func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
            max(abs(a.asType(.float32) - b)).item(Float.self)
        }
        print("q max_abs: \(maxAbs(q, sdpa["q"]!))  k max_abs: \(maxAbs(k, sdpa["k"]!))")

        // 3. Full forward: second-to-last hidden cosine.
        let hidden = model(inputIds)
        let a = hidden.asType(.float32).flattened()
        let b = enc["hidden_secondlast"]!.flattened()
        let cos = (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)) + 1e-12)).item(Float.self)
        print("hidden[-2] cosine: \(cos)")
        XCTAssertGreaterThanOrEqual(cos, fp32 ? 0.9999 : 0.995)
    }
}
