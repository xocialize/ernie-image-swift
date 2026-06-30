// MLXEngine `textToImage` package over the ErnieImage core — the lower-tier T2I
// module (coexists with Lens via the multi-package registry; apps select by
// PackageID "ernie-image-turbo" vs "lens-t2i", or setDefault per device tier).
//
// ERNIE-Image-Turbo (Apache-2.0): Mistral-3B text conditioning (second-to-last
// hidden, YaRN rope) -> 8B single-stream zero-CFG DiT (8 steps) -> Flux2VAE
// (reused from the Lens port). Swift core parity-locked vs PT goldens (DiT
// 0.9999996 fp32 · encoder 0.9999969 · VAE 64 dB · e2e 19.5 s @1024² bf16).

import CoreGraphics
import Foundation
import ImageIO
import Flux2VAE
import MLX
import MLXToolKit
import ErnieImage
import Tokenizers
import UniformTypeIdentifiers

/// Init-time configuration (C9): the Turbo snapshot root + generation defaults.
public struct ErnieImageConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Snapshot root with `transformer/`, `text_encoder/`, `vae/`, `tokenizer/`.
    public var snapshotPath: String
    /// Converted 4-bit repo root (`transformer-4bit/`, `text_encoder-4bit/`); when set,
    /// load() uses the quantized stack (the lower-tier variant: ~7.4 GB resident).
    public var quantizedPath: String?
    public var defaultSteps: Int
    public var modelsRootDirectory: URL?

    /// The selected DiT tier: int4 when the quantized stack is configured, else bf16. Lets
    /// the memory governor charge the matching split `QuantFootprint` rather than guessing
    /// largest-that-fits. The Mistral-3B encoder is a per-request transient (evicted before
    /// the denoise peak — see the generator), not part of the resident floor.
    public var quant: Quant { quantizedPath != nil ? .int4 : .bf16 }

    public init(
        snapshotPath: String =
            "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo",
        quantizedPath: String? = nil,
        defaultSteps: Int = 8,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.quantizedPath = quantizedPath
        self.defaultSteps = defaultSteps
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case snapshotPath, quantizedPath, defaultSteps }
}

public enum ErnieImagePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "ERNIE snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class ErnieImagePackage: ModelPackage {
    public typealias Configuration = ErnieImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "baidu/ERNIE-Image-Turbo", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (efficiency contract 1.14.0). Per-stage eviction (P2):
                // the Mistral-3B text encoder loads per request and is evicted (`nil` +
                // Memory.clearCache()) before the denoise peak — a TRANSIENT, not a
                // resident, in BOTH tiers. Only the 8B DiT + bf16 VAE stay resident.
                //   bf16: resident floor = DiT bf16 15 GB (on-disk) + bf16 VAE ~0.1 GB
                //     ≈ 15 GB. activation ≈ 11 GB = the transient encoder load (7.2 GB on
                //     disk) + DiT denoise/VAE-decode scratch (old flat 26 GB folded the
                //     encoder into the resident floor).
                //   int4: resident floor = int4 DiT ~4 GB (keep-hi linears stay bf16) +
                //     bf16 VAE ~0.1 GB ≈ 5 GB (encoder int4 ~1.8 GB is now a transient,
                //     not co-resident); inference peak 15.0 GB @1024² (decode conv scratch
                //     dominates; tiled decode tracked) → activation ≈ 11 GB. 9.4 GB peak
                //     @512² — the 16 GB tier runs the 4-bit variant at ≤640².
                // [residentBytes = measured on-disk weight floor (solid). peakActivationBytes
                //  is a smoke/derived estimate (encoder transient + the documented pre-split
                //  peak − floor); the smoke MLX-peak under-reads process phys_footprint ~2.7×
                //  (BiRefNet lesson) — FLAGGED for a clean in-app phys re-baseline once ERNIE
                //  is registered in the MLXEngineImage app (IMAGE_AUTORUN).]
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 15_000_000_000,
                        peakActivationBytes: 11_000_000_000),
                    QuantFootprint(
                        quant: .int4, residentBytes: 5_000_000_000,
                        peakActivationBytes: 11_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil  // memory admission gates; no chip-tier floor beyond it
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "ernie-image-turbo",
                    summary: "ERNIE-Image-Turbo 8B single-stream T2I (distilled, 8-step, "
                        + "no CFG): fast vivid 1024²-class generation at a fraction of the "
                        + "footprint of the full-tier T2I models.",
                    modes: []
                )
            ]
        )
    }

    private let configuration: Configuration
    private var generator: ErnieImageGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        let snapshot = URL(fileURLWithPath: configuration.snapshotPath)
        guard FileManager.default.fileExists(
            atPath: snapshot.appendingPathComponent("transformer").path)
        else { throw ErnieImagePackageError.unreadableSnapshot(snapshot.path) }

        // Per-stage residency (efficiency contract 1.14.0): the DiT + VAE stay resident;
        // the Mistral-3B text encoder is loaded per request and evicted before the denoise
        // peak (see ErnieImageGenerator). The encoder loader is captured as a closure so the
        // ~6 GB encoder is never co-resident with the DiT denoise activation peak.
        let transformer: ErnieImageTransformer2DModel
        let encoderProvider: () async throws -> ErnieTextEncoder
        if let quantizedPath = configuration.quantizedPath {
            let q4 = URL(fileURLWithPath: quantizedPath)
            transformer = try ErnieImageWeights.loadDiTQuantized(
                directory: q4.appendingPathComponent("transformer-4bit"))
            let encDir = q4.appendingPathComponent("text_encoder-4bit")
            encoderProvider = {
                try ErnieImageWeights.loadTextEncoderQuantized(directory: encDir)
            }
        } else {
            transformer = try ErnieImageWeights.loadDiTFromPT(
                directory: snapshot.appendingPathComponent("transformer"))
            let encDir = snapshot.appendingPathComponent("text_encoder")
            encoderProvider = {
                try ErnieImageWeights.loadTextEncoder(directory: encDir)
            }
        }
        // bf16 VAE decode matches the Python reference's internal regime and halves
        // the decode high-water vs fp32.
        let vae = try Flux2VAEWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"), dtype: .bfloat16)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer"))
        generator = ErnieImageGenerator(
            encoderProvider: encoderProvider, transformer: transformer,
            vae: vae, tokenizer: tokenizer)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Turbo is distilled at guidance 1.0 — no CFG path; guidanceScale is ignored.
        // Dimensions must be /16 (the released buckets are 1024² + six aspects).
        let width = ((t2i.width ?? 1024) / 16) * 16
        let height = ((t2i.height ?? 1024) / 16) * 16
        let (pixels, w, h) = try await generator.generate(
            prompt: t2i.prompt,
            width: width,
            height: height,
            steps: t2i.steps ?? configuration.defaultSteps,
            seed: t2i.seed ?? 0,
            progress: nil)

        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))
    }

    /// Interleaved RGB8 → PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw ErnieImagePackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw ErnieImagePackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw ErnieImagePackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ErnieImagePackageError.pngEncode }
        return out as Data
    }
}

extension ErnieImagePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(ErnieImagePackage.self)
    }
}
