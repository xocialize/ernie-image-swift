// CancellationTests.swift — ERNIE-Image-Turbo through the engine's CAN gate (offline, no
// MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled: the entry checkpoint
// (`try Task.checkCancellation()` as the FIRST act of run(), before notLoaded validation)
// fires before weights are touched, so a stub configuration suffices. CAN-3 is the document
// of record for the checkpoint cadence — all checkpoints live in the core
// (ErnieImageGenerator.generate, Sources/ErnieImage/Pipeline.swift), a throwing seam that
// rethrows CancellationError unchanged:
//   - post-encode seam (Mistral-3B encode done, encoder evicted, before denoise);
//   - denoise/step — `try Task.checkCancellation()` at the top of the denoise loop;
//   - pre-decode seam (before the monolithic Flux2VAE decode — ONE MLX eval, no chunk loop,
//     so no per-chunk decode cadence is claimed).
// No catch blocks exist on the run() path — nothing to launder (CAN-2).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXErnieImage

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = ErnieImagePackage(configuration: ErnieImageConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // 11 GB declared peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: ErnieImagePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: ErnieImagePackage.manifest,
            posture: .cadence([
                // Per-denoise-step try Task.checkCancellation() in ErnieImageGenerator
                // .generate's denoise loop; post-encode + pre-decode seams bracket it
                // (single forwards — seams, not recurring units).
                // RunProgress-evidenced (AB-A-0139): the same loop seam reports `.denoise` i/N.
                .init(phase: .denoise, unit: .step, reportsRunProgress: true),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
