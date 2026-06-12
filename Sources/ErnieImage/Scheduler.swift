// ERNIE-Image scheduler — FlowMatchEuler with a STATIC shift (4.0), exponential
// time-shift form. Mirrors diffusers 0.38: pipeline passes linspace(1, 0, N+1)[:-1]
// sigmas; set_timesteps applies sigma' = shift*s / (1 + (shift-1)*s) (identical to
// exp(mu)/(exp(mu) + (1/s - 1)) with mu = ln(shift)); timesteps = sigmas * 1000;
// a trailing 0 closes the Euler step. Gated against goldens/scheduler.safetensors.

import Foundation

public enum ErnieScheduler {
    public static let shift: Float = 4.0
    public static let numTrainTimesteps: Float = 1000

    /// Shifted sigmas for `steps`, with the trailing 0 (count = steps + 1).
    public static func sigmas(steps: Int) -> [Float] {
        var out = (0..<steps).map { i -> Float in
            let s = 1.0 - Float(i) / Float(steps)  // linspace(1, 0, steps+1)[:-1]
            return shift * s / (1 + (shift - 1) * s)
        }
        out.append(0)
        return out
    }

    /// Timestep values fed to the DiT (sigma * 1000), one per step.
    public static func timesteps(steps: Int) -> [Float] {
        sigmas(steps: steps).dropLast().map { $0 * numTrainTimesteps }
    }
}
