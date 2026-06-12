// S2 gate: DiT step-0 vs the PT fp32 golden (Turbo: single branch, no CFG).
// Regimes per house calibration: fp32 CPU >= 0.9999; bf16 GPU >= 0.9985.
//
// Run: ERNIE_PARITY=1 [ERNIE_FP32_CPU=1] swift test --filter DiTGateTests

import Foundation
import MLX
import XCTest

@testable import ErnieImage

final class DiTGateTests: XCTestCase {
    func testStep0() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_PARITY"] == "1", "ERNIE_PARITY=1")
        let fp32 = ProcessInfo.processInfo.environment["ERNIE_FP32_CPU"] == "1"
        if fp32 { Device.setDefault(device: Device(.cpu)) }
        let dtype: DType = fp32 ? .float32 : .bfloat16

        let goldens = URL(
            fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/goldens")
        let modelDir = URL(
            fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo")

        let dit = try MLX.loadArrays(url: goldens.appendingPathComponent("dit_step0.safetensors"))
        let lat = try MLX.loadArrays(url: goldens.appendingPathComponent("latents.safetensors"))

        let model = try ErnieImageWeights.loadDiTFromPT(
            directory: modelDir.appendingPathComponent("transformer"), dtype: dtype)

        let pred = model(
            hiddenStates: lat["noise"]!.asType(dtype),
            timestep: dit["timestep"]!.asType(.float32),
            text: dit["text_bth"]!.asType(dtype))
        let ours = pred.asType(.float32).flattened()
        let ref = dit["pred"]!.flattened()
        let cos = (sum(ours * ref) / (sqrt(sum(ours * ours)) * sqrt(sum(ref * ref)) + 1e-12))
            .item(Float.self)
        let mae = mean(abs(ours - ref)).item(Float.self)
        print("dit step0: cosine \(cos)  mae \(mae)")
        XCTAssertGreaterThanOrEqual(cos, fp32 ? 0.9999 : 0.9985)
    }
}
