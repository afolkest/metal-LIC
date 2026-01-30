import Foundation

/// Builds the discrete Hann kernel and populates `LicParams` for the GPU.
public enum LICKernel {

    public enum Error: Swift.Error {
        /// `L` (kernel half-length) must be > 0.
        case invalidL
        /// `h` (step size) must be > 0.
        case invalidH
    }

    /// Builds a discrete symmetric Hann kernel per spec Section 6.
    ///
    /// - Parameters:
    ///   - L: Kernel half-length in pixels (must be > 0).
    ///   - h: Integration step size in pixels (must be > 0, default 1.0).
    ///   - eps2: Zero-vector guard threshold (default 1e-12).
    ///   - edgeGainStrength: Mask edge gain strength (default 0 = disabled).
    ///   - edgeGainPower: Mask edge gain exponent (default 2).
    ///   - domainEdgeGainStrength: Domain edge gain strength (default 0 = disabled).
    ///   - domainEdgeGainPower: Domain edge gain exponent (default 2).
    /// - Returns: Populated `LicParams` and the kernel weights array.
    public static func build(
        L: Float,
        h: Float = 1.0,
        eps2: Float = 1e-12,
        edgeGainStrength: Float = 0,
        edgeGainPower: Float = 2,
        domainEdgeGainStrength: Float = 0,
        domainEdgeGainPower: Float = 2
    ) throws -> (params: LicParams, weights: [Float]) {
        guard L > 0 else { throw Error.invalidL }
        guard h > 0 else { throw Error.invalidH }

        // steps = round(L / h), ties away from zero (Section 6.1)
        let steps = UInt32(floor(L / h + 0.5))

        // Edge case: steps == 0 → degenerate single-sample kernel (Section 6.1)
        if steps == 0 {
            let params = LicParams(
                h: h, eps2: eps2,
                steps: 0, kmid: 0, kernelLen: 1,
                fullSum: 1.0, centerWeight: 1.0,
                edgeGainStrength: edgeGainStrength,
                edgeGainPower: edgeGainPower,
                domainEdgeGainStrength: domainEdgeGainStrength,
                domainEdgeGainPower: domainEdgeGainPower
            )
            return (params, [1.0])
        }

        let kernelLen = 2 * steps + 1
        let kmid = kernelLen / 2

        // Build Hann window: kernel[i] = 0.5 * (1 + cos(π * s_i / L))
        // where s_i = (i - kmid) * h
        var weights = [Float](repeating: 0, count: Int(kernelLen))
        var fullSum: Float = 0

        for i in 0..<Int(kernelLen) {
            let si = Float(i - Int(kmid)) * h
            if abs(si) <= L {
                let w = 0.5 * (1.0 + cosf(Float.pi * si / L))
                weights[i] = w
                fullSum += w
            }
        }

        let centerWeight = weights[Int(kmid)]

        let params = LicParams(
            h: h, eps2: eps2,
            steps: steps, kmid: kmid, kernelLen: kernelLen,
            fullSum: fullSum, centerWeight: centerWeight,
            edgeGainStrength: edgeGainStrength,
            edgeGainPower: edgeGainPower,
            domainEdgeGainStrength: domainEdgeGainStrength,
            domainEdgeGainPower: domainEdgeGainPower
        )

        return (params, weights)
    }
}
