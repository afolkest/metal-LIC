import Foundation

/// Spec-exact CPU reference implementation of Line Integral Convolution.
/// Uses Float64 accumulation for maximum precision (exceeds GPU's Float32).
public enum LICReferenceCPU {

    // MARK: - Bilinear sampling

    /// Bilinear sample a single-channel image at pixel coordinates.
    /// Matches Metal's linear clamp-to-edge sampler with normalizedCoordinates = false.
    public static func sampleInput(
        _ image: [Float], width: Int, height: Int,
        x: Double, y: Double
    ) -> Double {
        let fx = x - 0.5
        let fy = y - 0.5
        let i0 = Int(floor(fx)), j0 = Int(floor(fy))
        let sx = fx - Double(i0), sy = fy - Double(j0)

        func px(_ i: Int, _ j: Int) -> Double {
            Double(image[max(0, min(height - 1, j)) * width + max(0, min(width - 1, i))])
        }

        return (1 - sx) * (1 - sy) * px(i0, j0)
             + sx       * (1 - sy) * px(i0 + 1, j0)
             + (1 - sx) * sy       * px(i0, j0 + 1)
             + sx       * sy       * px(i0 + 1, j0 + 1)
    }

    /// Bilinear sample a 2-channel vector field at pixel coordinates.
    public static func sampleVector(
        _ field: [SIMD2<Float>], width: Int, height: Int,
        x: Double, y: Double
    ) -> (Double, Double) {
        let fx = x - 0.5, fy = y - 0.5
        let i0 = Int(floor(fx)), j0 = Int(floor(fy))
        let sx = fx - Double(i0), sy = fy - Double(j0)

        func px(_ i: Int, _ j: Int) -> (Double, Double) {
            let v = field[max(0, min(height - 1, j)) * width + max(0, min(width - 1, i))]
            return (Double(v.x), Double(v.y))
        }
        let p00 = px(i0, j0), p10 = px(i0+1, j0), p01 = px(i0, j0+1), p11 = px(i0+1, j0+1)
        return (
            (1-sx)*(1-sy)*p00.0 + sx*(1-sy)*p10.0 + (1-sx)*sy*p01.0 + sx*sy*p11.0,
            (1-sx)*(1-sy)*p00.1 + sx*(1-sy)*p10.1 + (1-sx)*sy*p01.1 + sx*sy*p11.1
        )
    }

    // MARK: - Helpers

    static func sampleDirection(
        _ field: [SIMD2<Float>], width: Int, height: Int,
        x: Double, y: Double, eps2: Double
    ) -> (dir: (Double, Double), valid: Bool) {
        let raw = sampleVector(field, width: width, height: height, x: x, y: y)
        let len2 = raw.0 * raw.0 + raw.1 * raw.1
        if !len2.isFinite { return ((0, 0), false) }
        if len2 < eps2 { return ((0, 0), true) }
        let inv = 1.0 / len2.squareRoot()
        return ((raw.0 * inv, raw.1 * inv), true)
    }

    static func isOutside(_ x: Double, _ y: Double, _ W: Double, _ H: Double) -> Bool {
        x < 0.5 || x > W - 0.5 || y < 0.5 || y > H - 0.5
    }

    // MARK: - LIC

    /// Run spec-exact LIC on CPU. Output is Float32 (not float16) to preserve reference precision.
    public static func run(
        input: [Float],
        vectorField: [SIMD2<Float>],
        width: Int, height: Int,
        params: LicParams,
        kernelWeights: [Float],
        mask: [UInt8]? = nil
    ) -> [Float] {
        let W = Double(width), H = Double(height)
        let h = Double(params.h)
        let eps2 = Double(params.eps2)
        let steps = Int(params.steps)
        let kmid = Int(params.kmid)
        let fullSum = Double(params.fullSum)
        let centerWeight = Double(params.centerWeight)

        var output = [Float](repeating: 0, count: width * height)

        for py in 0..<height {
            for px in 0..<width {
                let x0 = Double(px) + 0.5, y0 = Double(py) + 0.5

                // Starting pixel mask check (Section 9)
                let startingMasked = mask.map { $0[py * width + px] != 0 } ?? false
                if startingMasked {
                    let cs = sampleInput(input, width: width, height: height, x: x0, y: y0)
                    output[py * width + px] = Float(fullSum != 0 ? fullSum * cs : cs)
                    continue
                }

                // Center sample
                let centerSample = sampleInput(input, width: width, height: height, x: x0, y: y0)
                var value = centerWeight * centerSample
                var usedSum = centerWeight
                var hitDomainEdge = false
                var hitMaskEdge = false

                for dir in 0..<2 {
                    let sign: Double = dir == 0 ? 1 : -1
                    var x = x0, y = y0

                    for step in 1...max(steps, 1) {
                        if step > steps { break }

                        let (v, valid) = sampleDirection(vectorField, width: width, height: height,
                                                         x: x, y: y, eps2: eps2)
                        if !valid { break }
                        let vx = v.0 * sign, vy = v.1 * sign

                        // RK2 midpoint
                        let x1 = x + 0.5 * h * vx, y1 = y + 0.5 * h * vy
                        if isOutside(x1, y1, W, H) { hitDomainEdge = true; break }

                        let (v1, v1Valid) = sampleDirection(vectorField, width: width, height: height,
                                                            x: x1, y: y1, eps2: eps2)
                        if !v1Valid { break }
                        let v1x = v1.0 * sign, v1y = v1.1 * sign

                        // Full step
                        let xn = x + h * v1x, yn = y + h * v1y
                        if isOutside(xn, yn, W, H) { hitDomainEdge = true; break }

                        // Mask check
                        if let mask = mask {
                            let mi = max(0, min(width - 1, Int(floor(xn))))
                            let mj = max(0, min(height - 1, Int(floor(yn))))
                            if mask[mj * width + mi] != 0 { hitMaskEdge = true; break }
                        }

                        // Sample & accumulate
                        let k = dir == 0 ? kmid + step : kmid - step
                        let w = Double(kernelWeights[k])
                        let s = sampleInput(input, width: width, height: height, x: xn, y: yn)
                        value += w * s
                        usedSum += w
                        x = xn; y = yn
                    }
                }

                // Boundary processing (Section 9)
                let needsBoundary = usedSum > centerWeight && usedSum < fullSum
                let applyMaskEdge = hitMaskEdge && !startingMasked

                if needsBoundary && (applyMaskEdge || hitDomainEdge) {
                    let sf = min(1.0, max(0.0,
                        (usedSum - centerWeight) / (fullSum - centerWeight)))
                    value *= fullSum / usedSum

                    if applyMaskEdge && params.edgeGainStrength > 0 {
                        let t = min(1.0, max(0.0, (fullSum - usedSum) / fullSum))
                        value *= 1 + Double(params.edgeGainStrength) * pow(t, Double(params.edgeGainPower)) * sf
                    }
                    if hitDomainEdge && params.domainEdgeGainStrength > 0 {
                        let t = min(1.0, max(0.0, (fullSum - usedSum) / fullSum))
                        value *= 1 + Double(params.domainEdgeGainStrength) * pow(t, Double(params.domainEdgeGainPower)) * sf
                    }
                }

                output[py * width + px] = Float(value)
            }
        }
        return output
    }
}
