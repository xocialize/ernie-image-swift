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
