import simd

enum VectorFieldGenerator {

    /// Generates a vector field for the given preset.
    /// Returns a flat array of SIMD2<Float> in row-major order.
    static func generate(preset: Int, width: Int, height: Int) -> [SIMD2<Float>] {
        var field = [SIMD2<Float>](repeating: .zero, count: width * height)
        let w = Float(width)
        let h = Float(height)

        for y in 0..<height {
            for x in 0..<width {
                // Normalized coordinates centered at (0, 0)
                let nx = (Float(x) + 0.5) / w * 2.0 - 1.0
                let ny = (Float(y) + 0.5) / h * 2.0 - 1.0

                let vec: SIMD2<Float>
                switch preset {
                case 1: vec = vortex(nx, ny)
                case 2: vec = saddle(nx, ny)
                case 3: vec = radial(nx, ny)
                case 4: vec = diagonal(nx, ny)
                case 5: vec = doubleVortex(nx, ny)
                default: vec = vortex(nx, ny)
                }

                // Normalize to direction-only (unit length or zero)
                let len = length(vec)
                field[y * width + x] = len > 1e-8 ? vec / len : .zero
            }
        }

        return field
    }

    // MARK: - Presets

    /// 1. Counter-clockwise vortex centered at origin
    private static func vortex(_ x: Float, _ y: Float) -> SIMD2<Float> {
        SIMD2<Float>(-y, x)
    }

    /// 2. Saddle (hyperbolic) flow
    private static func saddle(_ x: Float, _ y: Float) -> SIMD2<Float> {
        SIMD2<Float>(x, -y)
    }

    /// 3. Radial outward flow
    private static func radial(_ x: Float, _ y: Float) -> SIMD2<Float> {
        SIMD2<Float>(x, y)
    }

    /// 4. Uniform diagonal flow (lower-left to upper-right)
    private static func diagonal(_ x: Float, _ y: Float) -> SIMD2<Float> {
        SIMD2<Float>(1.0, 1.0)
    }

    /// 5. Double vortex: two counter-rotating vortices
    private static func doubleVortex(_ x: Float, _ y: Float) -> SIMD2<Float> {
        // Left vortex center at (-0.4, 0), counter-clockwise
        let lx = x + 0.4, ly = y
        let left = SIMD2<Float>(-ly, lx)

        // Right vortex center at (0.4, 0), clockwise
        let rx = x - 0.4, ry = y
        let right = SIMD2<Float>(ry, -rx)

        // Blend by inverse distance (smooth transition)
        let dL = max(sqrt(lx * lx + ly * ly), 0.01)
        let dR = max(sqrt(rx * rx + ry * ry), 0.01)
        let wL = 1.0 / (dL * dL)
        let wR = 1.0 / (dR * dR)

        return (left * wL + right * wR) / (wL + wR)
    }
}
