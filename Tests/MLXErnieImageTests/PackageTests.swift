// Engine conformance: manifest + (gated) load -> T2IRequest -> PNG -> unload,
// plus the multi-package coexistence path (selection by PackageID).
//
// Run: ERNIE_PKG=1 swift test --filter PackageTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXErnieImage

final class PackageTests: XCTestCase {
    func testManifest() {
        let m = ErnieImagePackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .textToImage)
        XCTAssertEqual(m.surfaces[0].name, "ernie-image-turbo")
        XCTAssertEqual(m.license.weightLicense, .apache2)
    }

    func testLoadRunUnload() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ERNIE_PKG"] == "1", "ERNIE_PKG=1")

        let package = ErnieImagePackage(configuration: .init())
        try await package.load()

        let request = T2IRequest(
            prompt: "A lighthouse on a rocky coast at dawn, dramatic clouds, photorealistic",
            steps: 8, seed: 7)
        let start = Date()
        let response = try await package.run(request)
        guard let t2i = response as? T2IResponse else { return XCTFail("wrong response type") }
        print("package render: \(t2i.image.width ?? 0)x\(t2i.image.height ?? 0) in \(Date().timeIntervalSince(start))s")
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/ernie-t2i-package.png")
        try t2i.image.data.write(to: out)
        print("saved \(out.path)")

        await package.unload()
    }
}
