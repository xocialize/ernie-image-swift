// RunProgressTests.swift — ERNIE-Image's end of the V2 run-phase progress plane (AB-A-0139): the
// core reports into its own engine-free `ErnieImageProgress` sink, and the wrapper forwards those
// events into the engine's `RunProgress`. Offline — no kernels, no weights:
// `ErnieImageProgressBridge.forward` IS the wrapper's entire mapping. The denoise loop shares
// `generate()` with the Mistral encoder load, so the loop itself is exercised only by a live run
// with weights.

import ErnieImage
import Foundation
import MLXToolKit
import XCTest

@testable import MLXErnieImage

final class RunProgressTests: XCTestCase {

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [RunPhaseReport] = []
        func append(_ r: RunPhaseReport) { lock.lock(); storage.append(r); lock.unlock() }
        var items: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return storage }
    }

    /// Bind both task-locals the way `run()` does — the engine's sink outermost, the core's
    /// bridge inside it — and return what the engine plane observed.
    private func forwarded(_ body: () -> Void) -> [RunPhaseReport] {
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            ErnieImageProgress.$sink.withValue(ErnieImageProgressBridge.forward, operation: body)
        }
        return box.items
    }

    // MARK: - forwarding

    func testCoreEventsReachTheEnginePlane() {
        let seen = forwarded {
            ErnieImageProgress.report(.encode)
            ErnieImageProgress.report(.denoise)
            ErnieImageProgress.report(.denoise, step: 1, totalSteps: 8)
            ErnieImageProgress.report(.denoise, step: 8, totalSteps: 8)
            ErnieImageProgress.report(.decode)
        }
        XCTAssertEqual(seen.map(\.phase), [.encode, .denoise, .denoise, .denoise, .decode])
        XCTAssertEqual(seen[2], RunPhaseReport(phase: .denoise, step: 1, totalSteps: 8))
        XCTAssertEqual(seen[3].step, 8)
        XCTAssertNil(seen[0].step)   // phases without steps carry no invented counts
        XCTAssertNil(seen[1].step)   // nor does the loop-entry report
    }

    /// The forward is a raw-value pass-through, which is only correct while the core's names
    /// ARE the canonical constants. Rename a core phase and this fails instead of shipping a
    /// non-canonical phase name to consumers.
    func testCorePhaseNamesMatchCanonicalRunPhases() {
        XCTAssertEqual(RunPhase(rawValue: ErnieImageProgress.Phase.encode.rawValue), .encode)
        XCTAssertEqual(RunPhase(rawValue: ErnieImageProgress.Phase.denoise.rawValue), .denoise)
        XCTAssertEqual(RunPhase(rawValue: ErnieImageProgress.Phase.decode.rawValue), .decode)
    }

    // MARK: - scoping

    func testUnboundCoreSinkIsANoOp() {
        // The CLI and the parity gates run with nothing bound — reporting must stay safe there.
        ErnieImageProgress.report(.denoise, step: 1, totalSteps: 8)
        // And a core event outside the wrapper's binding must not reach a bound engine plane.
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            ErnieImageProgress.report(.decode)
        }
        XCTAssertTrue(box.items.isEmpty)
    }
}
