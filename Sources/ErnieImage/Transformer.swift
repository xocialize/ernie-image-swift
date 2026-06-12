// ERNIE-Image denoising transformer — Swift/MLX port.
//
// Isomorphic to diffusers 0.38 transformer_ernie_image.py (ErnieImageTransformer2DModel;
// the key/naming oracle), cross-checked against mflux main's ernie_transformer (MLX).
// Single-stream DiT, batch-first internally (reference runs [S, B, H]; equivalent math).
//
// Conventions that differ from the Lens/Qwen family — all from the reference:
//   - sequence order [IMAGE, text]; image tokens take rope axis-0 position = text_len
//     (a constant), grid y/x on axes 1/2; text takes axis-0 = arange(T), 0, 0.
//   - rope theta 256, axes [32,48,48]; angles duplicated INTERLEAVED [θ0,θ0,θ1,θ1,…]
//     with NON-interleaved rotate-half (Megatron bshd, rotary_interleaved=False).
//   - SHARED AdaLN: one 6-way modulation from the time embedding, reused by all layers
//     (blocks own no modulation weights); modulation arithmetic in fp32.
//   - FF: linear_fc2(up_proj(x) * gelu(gate_proj(x))) — EXACT gelu, gate on the gelu side.
//   - time proj: 4096-dim sinusoidal, flip_sin_to_cos=false, shift 0; timestep = sigma*1000.
//   - final AdaLNContinuous: scale/shift = linear(c) directly (NO SiLU), order (scale, shift).

import Foundation
import MLX
import MLXFast
import MLXNN

public enum ErnieImageError: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .loading(let m): return "ErnieImage loading error: \(m)"
        case .invalidInput(let m): return "ErnieImage input error: \(m)"
        }
    }
}

// MARK: - Embeddings & RoPE

/// Sinusoidal timestep proj — diffusers Timesteps(hidden, flip_sin_to_cos=false,
/// downscale_freq_shift=0): [sin, cos] halves over hidden/2 frequencies.
func ernieTimestepProj(_ timesteps: MLXArray, dim: Int) -> MLXArray {
    let halfDim = dim / 2
    var exponent = -log(Float(10000)) * MLXArray(0..<halfDim).asType(.float32)
    exponent = exponent / Float(halfDim)  // downscale_freq_shift = 0
    let emb = timesteps[0..., .newAxis].asType(.float32) * exp(exponent)[.newAxis, 0...]
    return concatenated([sin(emb), cos(emb)], axis: -1)
}

public final class ErnieTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(dim: Int) {
        self._linear1.wrappedValue = Linear(dim, dim)
        self._linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// 3-axis rope table: ids (B, S, 3) -> angles (B, S, 1, headDim) with the
/// interleaved [θ0,θ0,θ1,θ1,…] duplication of the reference.
func ernieRopeAngles(ids: MLXArray, axesDim: [Int], theta: Float) -> MLXArray {
    var parts: [MLXArray] = []
    for (i, dim) in axesDim.enumerated() {
        let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim)
        let omega = 1.0 / pow(MLXArray(theta), scale)  // (dim/2,)
        let pos = ids[.ellipsis, i].asType(.float32)   // (B, S)
        parts.append(pos[.ellipsis, .newAxis] * omega) // (B, S, dim/2)
    }
    let emb = concatenated(parts, axis: -1)            // (B, S, headDim/2)
    let dup = stacked([emb, emb], axis: -1)            // (B, S, headDim/2, 2)
    let shape = emb.shape
    return dup.reshaped(shape[0], shape[1], 1, shape[2] * 2)  // (B, S, 1, headDim)
}

/// Megatron-style non-interleaved rotation with interleaved-duplicated angles.
/// x: (B, S, H, D); angles: (B, S, 1, D).
func ernieApplyRope(_ x: MLXArray, angles: MLXArray) -> MLXArray {
    let cosA = cos(angles).asType(x.dtype)
    let sinA = sin(angles).asType(x.dtype)
    let d = x.dim(-1)
    let x1 = x[.ellipsis, ..<(d / 2)]
    let x2 = x[.ellipsis, (d / 2)...]
    let rotated = concatenated([-x2, x1], axis: -1)
    return x * cosA + rotated * sinA
}

// MARK: - Block

/// Gated FF: linear_fc2(up(x) * gelu_exact(gate(x))) — all bias-free.
public final class ErnieFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    public init(hidden: Int, ffnHidden: Int) {
        self._gateProj.wrappedValue = Linear(hidden, ffnHidden, bias: false)
        self._upProj.wrappedValue = Linear(hidden, ffnHidden, bias: false)
        self._fc2.wrappedValue = Linear(ffnHidden, hidden, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(upProj(x) * gelu(gateProj(x)))  // EXACT gelu (erf), per the reference F.gelu
    }
}

/// Self-attention: bias-free QKV + out, RMSNorm QK (eps 1e-6), rope, plain SDPA.
public final class ErnieAttention: Module {
    let heads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    public init(hidden: Int, heads: Int, eps: Float = 1e-6) {
        self.heads = heads
        self.headDim = hidden / heads
        self._toQ.wrappedValue = Linear(hidden, hidden, bias: false)
        self._toK.wrappedValue = Linear(hidden, hidden, bias: false)
        self._toV.wrappedValue = Linear(hidden, hidden, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toOut.wrappedValue = [Linear(hidden, hidden, bias: false)]
        super.init()
    }

    /// x: (B, S, H*D); ropeAngles: (B, S, 1, D); mask: additive (B,1,1,S) or nil.
    public func callAsFunction(
        _ x: MLXArray, ropeAngles: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        var q = toQ(x).reshaped(b, s, heads, headDim)
        var k = toK(x).reshaped(b, s, heads, headDim)
        let v = toV(x).reshaped(b, s, heads, headDim)
        q = normQ(q)
        k = normK(k)
        q = ernieApplyRope(q, angles: ropeAngles)
        k = ernieApplyRope(k, angles: ropeAngles)

        let scale = 1.0 / sqrt(Float(headDim))
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            mask.map { .array($0) } ?? .none
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3), scale: scale, mask: maskMode)
        return toOut[0](out.transposed(0, 2, 1, 3).reshaped(b, s, heads * headDim))
    }
}

/// One single-stream block. Owns NO modulation weights — the shared 6-way AdaLN
/// params arrive from the top level. Modulation arithmetic in fp32 (reference).
public final class ErnieTransformerBlock: Module {
    @ModuleInfo(key: "adaLN_sa_ln") var saNorm: RMSNorm
    @ModuleInfo(key: "self_attention") var attention: ErnieAttention
    @ModuleInfo(key: "adaLN_mlp_ln") var mlpNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: ErnieFeedForward

    public init(hidden: Int, heads: Int, ffnHidden: Int, eps: Float = 1e-6) {
        self._saNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._attention.wrappedValue = ErnieAttention(hidden: hidden, heads: heads, eps: eps)
        self._mlpNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._mlp.wrappedValue = ErnieFeedForward(hidden: hidden, ffnHidden: ffnHidden)
        super.init()
    }

    /// temb: 6 tensors (B, 1, H) [shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP].
    public func callAsFunction(
        _ x: MLXArray, ropeAngles: MLXArray, temb: [MLXArray], mask: MLXArray?
    ) -> MLXArray {
        let dtype = x.dtype
        var h = saNorm(x).asType(.float32) * (1 + temb[1]) + temb[0]
        let attnOut = attention(h.asType(dtype), ropeAngles: ropeAngles, mask: mask)
        var out = x + (temb[2] * attnOut.asType(.float32)).asType(dtype)
        h = mlpNorm(out).asType(.float32) * (1 + temb[4]) + temb[3]
        out = out + (temb[5] * mlp(h.asType(dtype)).asType(.float32)).asType(dtype)
        return out
    }
}

// MARK: - Top-level model

public final class ErnieImageTransformer2DModel: Module {
    public let hiddenSize: Int
    public let heads: Int
    public let headDim: Int
    public let patchSize: Int
    public let outChannels: Int
    let ropeTheta: Float
    let ropeAxesDim: [Int]

    // x_embedder.proj is a kernel-1 Conv2d == per-pixel Linear; loaded as Conv2d
    // weights reshaped to Linear (O, I) at load time.
    @ModuleInfo(key: "x_embedder_proj") var xEmbedder: Linear
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ModuleInfo(key: "time_embedding") var timeEmbedding: ErnieTimestepEmbedding
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear  // Sequential(SiLU, Linear) -> .1
    @ModuleInfo(key: "layers") var layers: [ErnieTransformerBlock]
    @ModuleInfo(key: "final_norm_linear") var finalNormLinear: Linear
    @ModuleInfo(key: "final_linear") var finalLinear: Linear

    let finalLayerNorm: LayerNorm  // affine-less; plain property (no params)

    public init(
        hiddenSize: Int = 4096,
        numAttentionHeads: Int = 32,
        numLayers: Int = 36,
        ffnHiddenSize: Int = 12288,
        inChannels: Int = 128,
        outChannels: Int = 128,
        patchSize: Int = 1,
        textInDim: Int = 3072,
        ropeTheta: Float = 256,
        ropeAxesDim: [Int] = [32, 48, 48],
        eps: Float = 1e-6
    ) {
        self.hiddenSize = hiddenSize
        self.heads = numAttentionHeads
        self.headDim = hiddenSize / numAttentionHeads
        self.patchSize = patchSize
        self.outChannels = outChannels
        self.ropeTheta = ropeTheta
        self.ropeAxesDim = ropeAxesDim

        self._xEmbedder.wrappedValue = Linear(inChannels * patchSize * patchSize, hiddenSize)
        self._textProj.wrappedValue = Linear(textInDim, hiddenSize, bias: false)
        self._timeEmbedding.wrappedValue = ErnieTimestepEmbedding(dim: hiddenSize)
        self._adaLNModulation.wrappedValue = Linear(hiddenSize, 6 * hiddenSize)
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            ErnieTransformerBlock(
                hidden: hiddenSize, heads: numAttentionHeads, ffnHidden: ffnHiddenSize, eps: eps)
        }
        self._finalNormLinear.wrappedValue = Linear(hiddenSize, 2 * hiddenSize)
        self._finalLinear.wrappedValue = Linear(
            hiddenSize, patchSize * patchSize * outChannels)
        self.finalLayerNorm = LayerNorm(dimensions: hiddenSize, eps: eps, affine: false)
        super.init()
    }

    /// - Parameters:
    ///   - hiddenStates: (B, 128, h, w) packed latents (spatial layout).
    ///   - timestep: (B,) — sigma * 1000.
    ///   - text: (B, T, textInDim) — single prompt, unpadded (B=1 supported).
    /// - Returns: (B, 128, h, w) prediction.
    public func callAsFunction(
        hiddenStates: MLXArray, timestep: MLXArray, text: MLXArray
    ) -> MLXArray {
        let (b, c, h, w) = (hiddenStates.dim(0), hiddenStates.dim(1),
                            hiddenStates.dim(2), hiddenStates.dim(3))
        precondition(patchSize == 1, "patch_size 1 (per the released config)")
        let nImg = h * w
        let tLen = text.dim(1)

        // Patch embed (kernel-1 conv == linear over the channel dim, NHWC order).
        let imgTokens = xEmbedder(
            hiddenStates.transposed(0, 2, 3, 1).reshaped(b, nImg, c))
        let textTokens = textProj(text)
        var x = concatenated([imgTokens, textTokens], axis: 1)  // [IMAGE, text]

        // Position ids: image -> (text_len, y, x); text -> (arange(T), 0, 0).
        let grid = MLXArray(
            (0..<h).flatMap { y in (0..<w).flatMap { xi in [Float(tLen), Float(y), Float(xi)] } },
            [1, nImg, 3])
        let textIds = MLXArray(
            (0..<tLen).flatMap { [Float($0), 0, 0] }, [1, tLen, 3])
        let ids = broadcast(concatenated([grid, textIds], axis: 1), to: [b, nImg + tLen, 3])
        let ropeAngles = ernieRopeAngles(ids: ids, axesDim: ropeAxesDim, theta: ropeTheta)

        // Shared AdaLN from the time embedding (fp32 modulation values).
        let proj = ernieTimestepProj(timestep.asType(.float32), dim: hiddenSize)
        let cond = timeEmbedding(proj.asType(hiddenStates.dtype))
        let mods = split(
            adaLNModulation(silu(cond)).asType(.float32), parts: 6, axis: -1
        ).map { $0[0..., .newAxis, 0...] }  // each (B, 1, H)

        for layer in layers {
            x = layer(x, ropeAngles: ropeAngles, temb: mods, mask: nil)
        }

        // Final AdaLN-continuous: scale/shift = linear(cond) directly (no SiLU).
        let scaleShift = split(finalNormLinear(cond), parts: 2, axis: -1)
        let (scale, shift) = (scaleShift[0], scaleShift[1])
        x = finalLayerNorm(x) * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...]

        let patches = finalLinear(x)[0..., ..<nImg]  // (B, nImg, outChannels)
        return patches.reshaped(b, h, w, outChannels).transposed(0, 3, 1, 2)
    }
}
