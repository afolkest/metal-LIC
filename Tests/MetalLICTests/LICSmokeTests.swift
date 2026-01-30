import XCTest
import Metal
@testable import MetalLIC

/// End-to-end GPU smoke tests: build kernel, encode, dispatch, read back.
final class LICSmokeTests: XCTestCase {

    var device: MTLDevice!
    var encoder: LICEncoder!
    var commandQueue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MetalLIC.device
        try XCTSkipIf(device == nil, "No Metal device")
        encoder = try LICEncoder(device: device)
        commandQueue = device.makeCommandQueue()!
    }

    // MARK: - Helpers

    /// Fill an r32Float texture with values from a closure (x, y) -> Float.
    private func fillR32Float(_ texture: MTLTexture,
                              _ fill: (Int, Int) -> Float) {
        let w = texture.width, h = texture.height
        var data = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                data[y * w + x] = fill(x, y)
            }
        }
        texture.replace(
            region: MTLRegionMake2D(0, 0, w, h),
            mipmapLevel: 0,
            withBytes: &data,
            bytesPerRow: w * MemoryLayout<Float>.stride)
    }

    /// Fill an rg32Float texture with values from a closure (x, y) -> (u, v).
    private func fillRG32Float(_ texture: MTLTexture,
                               _ fill: (Int, Int) -> (Float, Float)) {
        let w = texture.width, h = texture.height
        var data = [Float](repeating: 0, count: w * h * 2)
        for y in 0..<h {
            for x in 0..<w {
                let (u, v) = fill(x, y)
                data[(y * w + x) * 2]     = u
                data[(y * w + x) * 2 + 1] = v
            }
        }
        texture.replace(
            region: MTLRegionMake2D(0, 0, w, h),
            mipmapLevel: 0,
            withBytes: &data,
            bytesPerRow: w * 2 * MemoryLayout<Float>.stride)
    }

    /// Read back an r16Float texture as [Float].
    private func readR16Float(_ texture: MTLTexture) -> [Float] {
        let w = texture.width, h = texture.height
        var raw = [UInt16](repeating: 0, count: w * h)
        texture.getBytes(
            &raw,
            bytesPerRow: w * MemoryLayout<UInt16>.stride,
            from: MTLRegionMake2D(0, 0, w, h),
            mipmapLevel: 0)
        return raw.map { float16BitsToFloat($0) }
    }

    /// Simple deterministic hash-based noise (not random, reproducible).
    private func noise(x: Int, y: Int, seed: UInt32 = 0) -> Float {
        var h = seed &+ UInt32(x) &* 374761393 &+ UInt32(y) &* 668265263
        h = (h ^ (h >> 13)) &* 1274126177
        h = h ^ (h >> 16)
        return Float(h & 0xFFFF) / 65535.0
    }

    // MARK: - Smoke test: uniform field

    func testUniformField_producesSmoothedOutput() throws {
        let size = 64
        let L: Float = 10
        let (params, weights) = try LICKernel.build(L: L)

        let inputTex = try encoder.makeInputTexture(width: size, height: size)
        let vecTex   = try encoder.makeVectorFieldTexture(width: size, height: size)
        let outTex   = try encoder.makeOutputTexture(width: size, height: size)

        // White noise input
        fillR32Float(inputTex) { x, y in self.noise(x: x, y: y) }

        // Uniform rightward vector field: V = (1, 0)
        fillRG32Float(vecTex) { _, _ in (1.0, 0.0) }

        // Dispatch
        let cb = commandQueue.makeCommandBuffer()!
        try encoder.encode(commandBuffer: cb,
                           params: params,
                           kernelWeights: weights,
                           inputTexture: inputTex,
                           vectorField: vecTex,
                           outputTexture: outTex)
        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertEqual(cb.status, .completed, "Command buffer should complete without error")

        // Read back
        let output = readR16Float(outTex)

        // Basic sanity: output should not be all zeros
        let nonZero = output.filter { $0 > 0 }.count
        XCTAssertGreaterThan(nonZero, output.count / 2,
                             "Most pixels should have non-zero LIC output")

        // LIC with uniform field should smooth along X but not Y.
        // Variance along rows (X) should be lower than variance along columns (Y).
        // Exclude border pixels (size of kernel half-length) to avoid boundary truncation.
        let w = size, h = size
        let border = Int(params.steps)
        var rowVariances = [Float]()
        for y in border..<(h - border) {
            let row = (border..<(w - border)).map { output[y * w + $0] }
            let mean = row.reduce(0, +) / Float(row.count)
            let variance = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(row.count)
            rowVariances.append(variance)
        }
        var colVariances = [Float]()
        for x in border..<(w - border) {
            let col = (border..<(h - border)).map { output[$0 * w + x] }
            let mean = col.reduce(0, +) / Float(col.count)
            let variance = col.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(col.count)
            colVariances.append(variance)
        }
        let avgRowVar = rowVariances.reduce(0, +) / Float(rowVariances.count)
        let avgColVar = colVariances.reduce(0, +) / Float(colVariances.count)
        XCTAssertLessThan(avgRowVar, avgColVar,
                          "Row variance (along flow) should be less than column variance")
    }

    // MARK: - Smoke test: vortex field

    func testVortexField_producesNonZeroOutput() throws {
        let size = 64
        let (params, weights) = try LICKernel.build(L: 8)

        let inputTex = try encoder.makeInputTexture(width: size, height: size)
        let vecTex   = try encoder.makeVectorFieldTexture(width: size, height: size)
        let outTex   = try encoder.makeOutputTexture(width: size, height: size)

        // White noise input
        fillR32Float(inputTex) { x, y in self.noise(x: x, y: y, seed: 42) }

        // Vortex field: V = (-dy, dx) where dx,dy relative to center
        let cx = Float(size) / 2, cy = Float(size) / 2
        fillRG32Float(vecTex) { x, y in
            let dx = Float(x) + 0.5 - cx
            let dy = Float(y) + 0.5 - cy
            return (-dy, dx)
        }

        let cb = commandQueue.makeCommandBuffer()!
        try encoder.encode(commandBuffer: cb,
                           params: params,
                           kernelWeights: weights,
                           inputTexture: inputTex,
                           vectorField: vecTex,
                           outputTexture: outTex)
        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertEqual(cb.status, .completed)

        let output = readR16Float(outTex)
        let nonZero = output.filter { $0 > 0 }.count
        XCTAssertGreaterThan(nonZero, output.count / 2,
                             "Vortex LIC should produce non-zero output")

        // Check output range: raw weighted sum of [0,1] input bounded by full_sum
        let maxVal = output.max() ?? 0
        let minVal = output.min() ?? 0
        XCTAssertGreaterThanOrEqual(minVal, 0,
                                    "Output should be non-negative")
        XCTAssertLessThanOrEqual(maxVal, params.fullSum * 1.05,
                                 "Output should not exceed full_sum (got \(maxVal), full_sum=\(params.fullSum))")
    }

    // MARK: - Determinism

    func testDeterminism_identicalOutputs() throws {
        let size = 32
        let (params, weights) = try LICKernel.build(L: 5)

        let inputTex = try encoder.makeInputTexture(width: size, height: size)
        let vecTex   = try encoder.makeVectorFieldTexture(width: size, height: size)
        let outTex1  = try encoder.makeOutputTexture(width: size, height: size)
        let outTex2  = try encoder.makeOutputTexture(width: size, height: size)

        fillR32Float(inputTex) { x, y in self.noise(x: x, y: y, seed: 7) }
        fillRG32Float(vecTex) { x, y in
            let dx = Float(x) + 0.5 - 16
            let dy = Float(y) + 0.5 - 16
            return (-dy, dx)
        }

        // Run twice
        for outTex in [outTex1, outTex2] {
            let cb = commandQueue.makeCommandBuffer()!
            try encoder.encode(commandBuffer: cb,
                               params: params,
                               kernelWeights: weights,
                               inputTexture: inputTex,
                               vectorField: vecTex,
                               outputTexture: outTex)
            cb.commit()
            cb.waitUntilCompleted()
        }

        let out1 = readR16Float(outTex1)
        let out2 = readR16Float(outTex2)
        XCTAssertEqual(out1, out2, "Identical inputs must produce identical outputs")
    }

    // MARK: - Debug modes compile and run

    func testDebugModes_runWithoutCrash() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 4)

        let inputTex = try encoder.makeInputTexture(width: size, height: size)
        let vecTex   = try encoder.makeVectorFieldTexture(width: size, height: size)

        fillR32Float(inputTex) { x, y in self.noise(x: x, y: y) }
        fillRG32Float(vecTex) { _, _ in (1.0, 0.0) }

        // Test all debug modes (1 = step count, 2 = boundary, 3 = support ratio)
        for mode: UInt32 in [1, 2, 3] {
            let config = LICPipelineConfig(debugMode: mode)
            try encoder.buildPipeline(for: config)

            let outTex = try encoder.makeOutputTexture(width: size, height: size)
            let cb = commandQueue.makeCommandBuffer()!
            try encoder.encode(commandBuffer: cb,
                               params: params,
                               kernelWeights: weights,
                               inputTexture: inputTex,
                               vectorField: vecTex,
                               outputTexture: outTex,
                               config: config)
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed,
                           "Debug mode \(mode) should complete without error")
        }
    }
}

// MARK: - Float16 conversion

/// Convert a float16 bit pattern to Float.
private func float16BitsToFloat(_ bits: UInt16) -> Float {
    let sign     = UInt32((bits >> 15) & 0x1)
    let exponent = UInt32((bits >> 10) & 0x1F)
    let mantissa = UInt32(bits & 0x3FF)

    if exponent == 0 {
        if mantissa == 0 { return sign == 0 ? 0.0 : -0.0 }
        // Denormalized
        var f = Float(mantissa) / 1024.0
        f *= pow(2.0, -14.0)
        return sign == 0 ? f : -f
    }
    if exponent == 31 {
        if mantissa == 0 { return sign == 0 ? Float.infinity : -Float.infinity }
        return Float.nan
    }
    let f32Exponent = exponent + 112  // rebias: -15 + 127 = 112
    let f32Bits = (sign << 31) | (f32Exponent << 23) | (mantissa << 13)
    return Float(bitPattern: f32Bits)
}
