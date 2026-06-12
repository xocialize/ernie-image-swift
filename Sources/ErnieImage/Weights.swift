// Weight loading for the ERNIE-Image DiT (diffusers `transformer/` checkpoint).
// Renames: adaLN_modulation.1. -> adaLN_modulation. (Sequential(SiLU, Linear));
// final_norm.linear. -> final_norm_linear.; x_embedder.proj (Conv2d k=1, 4D
// (O,I,1,1)) -> x_embedder_proj Linear (O,I).

import Foundation
import MLX
import MLXNN

public enum ErnieImageWeights {

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw ErnieImageError.loading("no .safetensors under \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            merged.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        return merged
    }

    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: "adaLN_modulation.1.", with: "adaLN_modulation.")
            .replacingOccurrences(of: "final_norm.linear.", with: "final_norm_linear.")
            .replacingOccurrences(of: "x_embedder.proj.", with: "x_embedder_proj.")
    }

    public static func loadDiTFromPT(directory: URL, dtype: DType = .bfloat16) throws
        -> ErnieImageTransformer2DModel
    {
        let model = ErnieImageTransformer2DModel()
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) {
            var v = v
            if k == "x_embedder.proj.weight" {  // Conv2d k=1 (O,I,1,1) -> Linear (O,I)
                v = v.reshaped(v.dim(0), v.dim(1))
            }
            weights[sanitizeDiTKey(k)] = v.asType(dtype)
        }
        try verifyAndLoad(model: model, weights: weights, label: "ErnieDiT")
        return model
    }

    /// Two-way strict load (workspace discipline).
    static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw ErnieImageError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. "
                    + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw ErnieImageError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. "
                    + unused.prefix(4).joined(separator: ", "))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }
}

// MARK: - Quantized repos (4-bit lower-tier variant)

extension ErnieImageWeights {
    /// Quantize every Linear except the quality-sensitive glue (modulation, time
    /// embedding, final projection, patch embed for the DiT; nothing excluded in the
    /// text encoder except the embedding, which stays a non-Linear module anyway).
    static let ditKeepHi = ["adaLN_modulation", "time_embedding", "final_norm_linear",
                            "final_linear", "x_embedder_proj", "text_proj"]

    /// Convert a loaded model's Linears to 4-bit and save (flattened params +
    /// quantization config) in `directory`.
    public static func saveQuantized(
        model: Module, directory: URL, groupSize: Int = 64, bits: Int = 4,
        keepHi: [String] = []
    ) throws {
        quantize(model: model, groupSize: groupSize, bits: bits) { path, module in
            guard module is Linear else { return false }
            return !keepHi.contains { path.contains($0) }
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: flat, url: directory.appendingPathComponent("weights.safetensors"))
        let config: [String: Any] = [
            "quantization": ["group_size": groupSize, "bits": bits, "keep_hi": keepHi]
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted])
            .write(to: directory.appendingPathComponent("config.json"))
    }

    static func loadQuantizedRepo(model: Module, directory: URL, label: String) throws {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let cfg = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let q = cfg["quantization"] as? [String: Any],
              let groupSize = q["group_size"] as? Int, let bits = q["bits"] as? Int
        else { throw ErnieImageError.loading("\(label): unreadable quantization config") }
        let keepHi = (q["keep_hi"] as? [String]) ?? []
        let weights = try MLX.loadArrays(
            url: directory.appendingPathComponent("weights.safetensors"))
        quantize(model: model, groupSize: groupSize, bits: bits) { path, module in
            guard module is Linear else { return false }
            guard weights["\(path).scales"] != nil else { return false }
            return !keepHi.contains { path.contains($0) }
        }
        try verifyAndLoad(model: model, weights: weights, label: label)
    }

    /// Load the DiT from a converted 4-bit repo (`transformer-4bit/`).
    public static func loadDiTQuantized(directory: URL) throws -> ErnieImageTransformer2DModel {
        let model = ErnieImageTransformer2DModel()
        try loadQuantizedRepo(model: model, directory: directory, label: "ErnieDiT(4bit)")
        return model
    }

    /// Load the text encoder from a converted 4-bit repo (`text_encoder-4bit/`).
    public static func loadTextEncoderQuantized(directory: URL) throws -> ErnieTextEncoder {
        let model = ErnieTextEncoder()
        try loadQuantizedRepo(model: model, directory: directory, label: "ErnieTextEncoder(4bit)")
        return model
    }
}
