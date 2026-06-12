// ERNIE-Image text encoder — the Mistral3 (Ministral-3B) backbone, text-only path.
//
// Reference: transformers 5.12 Mistral3Model as driven by diffusers ErnieImagePipeline:
// plain tokenizer (add_special_tokens) -> 26 causal layers -> SECOND-TO-LAST hidden
// state (= the last layer's output, pre-final-norm; HF appends the post-norm state
// as hidden_states[-1]).
//
// RoPE is **YaRN** (theta 1e6, factor 16, beta 32/1, original_max 16384, mscale 1.0
// => attention_scaling 1.0) — pinned by true-SDPA-input capture (pitfall #26; the
// config buries it under rope_parameters and mflux's port reads as plain rope).
// The Swift YaRN inv_freq is gated EXACTLY against the dumped resolved values.
//
// Architecture (text_config): 26L / hidden 3072 / 32 q-heads, 8 kv-heads, head_dim
// 128 (explicit — NOT hidden/heads) / ffn 9216 SwiGLU / RMSNorm eps 1e-5 / NO biases
// / vocab 131072. The checkpoint also carries a vision tower + projector (unused by
// the t2i pipeline) — skipped at load.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum ErnieYarnRope {
    public struct Params {
        public var theta: Float = 1_000_000
        public var headDim: Int = 128
        public var factor: Float = 16
        public var betaFast: Float = 32
        public var betaSlow: Float = 1
        public var originalMaxPositions: Float = 16384
        public init() {}
    }

    /// HF _compute_yarn_parameters: ramp-blended interpolation/extrapolation inv_freq.
    public static func invFreq(_ p: Params = Params()) -> [Float] {
        let half = p.headDim / 2
        func correctionDim(_ numRotations: Float) -> Float {
            Float(p.headDim) * log(p.originalMaxPositions / (numRotations * 2 * .pi))
                / (2 * log(p.theta))
        }
        let low = floor(correctionDim(p.betaFast))
        let high = ceil(correctionDim(p.betaSlow))
        let lowC = max(low, 0)
        let highC = min(high, Float(half - 1))

        var out = [Float](repeating: 0, count: half)
        for i in 0..<half {
            let posFreq = pow(p.theta, Float(2 * i) / Float(p.headDim))
            let extrapolation = 1.0 / posFreq
            let interpolation = 1.0 / (p.factor * posFreq)
            // linear ramp -> mask = 1 - ramp (clamped 0...1)
            var ramp = (Float(i) - lowC) / max(highC - lowC, 0.001)
            ramp = min(max(ramp, 0), 1)
            let mask = 1 - ramp
            out[i] = interpolation * (1 - mask) + extrapolation * mask
        }
        return out
    }

    /// cos/sin tables for positions 0..<length: (length, headDim), duplicated halves.
    public static func cosSin(length: Int, _ p: Params = Params()) -> (MLXArray, MLXArray) {
        let inv = MLXArray(invFreq(p))
        let pos = MLXArray(0..<length).asType(.float32)
        let freqs = pos[0..., .newAxis] * inv[.newAxis, 0...]  // (L, half)
        let emb = concatenated([freqs, freqs], axis: -1)       // (L, headDim)
        return (cos(emb), sin(emb))  // attention_scaling = 1.0
    }

    /// Non-interleaved rotate-half application. x: (B, H, L, D); cos/sin: (L, D).
    static func apply(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let x1 = x[.ellipsis, ..<(d / 2)]
        let x2 = x[.ellipsis, (d / 2)...]
        let rotated = concatenated([-x2, x1], axis: -1)
        let c = cosT[.newAxis, .newAxis].asType(x.dtype)
        let s = sinT[.newAxis, .newAxis].asType(x.dtype)
        return x * c + rotated * s
    }
}

final class ErnieTextAttention: Module {
    let heads = 32
    let kvHeads = 8
    let headDim = 128

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(hidden: Int) {
        self._wq.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = wq(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = ErnieYarnRope.apply(q, cos: cos, sin: sin)
        k = ErnieYarnRope.apply(k, cos: cos, sin: sin)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)),
            mask: .causal)
        return wo(out.transposed(0, 2, 1, 3).reshaped(b, l, heads * headDim))
    }

    /// Post-rope q/k for the true-SDPA-input gate.
    func ropedQK(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = wq(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = ErnieYarnRope.apply(q, cos: cos, sin: sin)
        k = ErnieYarnRope.apply(k, cos: cos, sin: sin)
        return (q, k)
    }
}

final class ErnieTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, ffn: Int) {
        self._gate.wrappedValue = Linear(hidden, ffn, bias: false)
        self._up.wrappedValue = Linear(hidden, ffn, bias: false)
        self._down.wrappedValue = Linear(ffn, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

public final class ErnieTextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: ErnieTextAttention
    @ModuleInfo(key: "mlp") var mlp: ErnieTextMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(hidden: Int, ffn: Int, eps: Float) {
        self._attention.wrappedValue = ErnieTextAttention(hidden: hidden)
        self._mlp.wrappedValue = ErnieTextMLP(hidden: hidden, ffn: ffn)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._postNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), cos: cos, sin: sin)
        return h + mlp(postNorm(h))
    }
}

public final class ErnieTextEncoder: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [ErnieTextLayer]
    /// Final norm — loaded (key consumed) but NOT applied: the pipeline takes
    /// hidden_states[-2] = the last layer's pre-norm output.
    @ModuleInfo(key: "norm") var finalNorm: RMSNorm

    public init(hidden: Int = 3072, ffn: Int = 9216, numLayers: Int = 26,
                vocab: Int = 131_072, eps: Float = 1e-5) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            ErnieTextLayer(hidden: hidden, ffn: ffn, eps: eps)
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        super.init()
    }

    /// ids (B, T) -> the pipeline's "second-to-last hidden state".
    ///
    /// HF appends each hidden state BEFORE the layer runs (embeddings first) and the
    /// post-norm state last — so hidden_states[-2] is the INPUT to the final layer
    /// = the output after numLayers-1 (25) layers. The 26th layer and the final norm
    /// are dead weight for the encoder use (verified: running all 26 gives cosine
    /// -0.04 vs the golden; 25 matches).
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var h = embedTokens(ids)
        let (cosT, sinT) = ErnieYarnRope.cosSin(length: ids.dim(1))
        for layer in layers.dropLast() {
            h = layer(h, cos: cosT, sin: sinT)
        }
        return h
    }
}

extension ErnieImageWeights {
    /// Load the text backbone from `text_encoder/` (prefix `language_model.model.`);
    /// the vision tower + multimodal projector are unused by the t2i pipeline.
    public static func loadTextEncoder(directory: URL, dtype: DType = .bfloat16) throws
        -> ErnieTextEncoder
    {
        let model = ErnieTextEncoder()
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) {
            if k.hasPrefix("vision_tower.") || k.hasPrefix("multi_modal_projector.") {
                continue
            }
            guard k.hasPrefix("language_model.model.") else {
                throw ErnieImageError.loading("unexpected key prefix: \(k)")
            }
            weights[String(k.dropFirst("language_model.model.".count))] = v.asType(dtype)
        }
        try verifyAndLoad(model: model, weights: weights, label: "ErnieTextEncoder")
        return model
    }
}
