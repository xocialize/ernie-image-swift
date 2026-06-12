// P3 gate 1: scheduler vs the recorded PT sigmas/timesteps (exact-constants rule:
// gate values come from the live reference dump, never from memory).

import Foundation
import MLX
import XCTest

@testable import ErnieImage

final class SchedulerGateTests: XCTestCase {
    func testSigmasMatchGolden() throws {
        let goldens = URL(
            fileURLWithPath: "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/goldens")
        let gold = try MLX.loadArrays(
            url: goldens.appendingPathComponent("scheduler.safetensors"))
        let refSigmas = gold["sigmas"]!.asArray(Float.self)
        let refTimesteps = gold["timesteps"]!.asArray(Float.self)

        let ours = ErnieScheduler.sigmas(steps: 8)
        XCTAssertEqual(ours.count, refSigmas.count)
        for (a, b) in zip(ours, refSigmas) {
            XCTAssertEqual(a, b, accuracy: 1e-6)
        }
        let ts = ErnieScheduler.timesteps(steps: 8)
        XCTAssertEqual(ts.count, refTimesteps.count)
        for (a, b) in zip(ts, refTimesteps) {
            XCTAssertEqual(a, b, accuracy: 1e-3)
        }
    }
}
