import XCTest
import Metal
@testable import MetalLIC

/// Small-grid invariant tests (validation plan Section 1).
/// Covers zero-vector handling, NaN/inf termination, mask starting-pixel,
/// and mask stop-on-enter semantics.
final class LICInvariantTests: XCTestCase {

    var device: MTLDevice!
    var encoder: LICEncoder!
    var commandQueue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MetalLIC.device
        try XCTSkipIf(device == nil, "No Metal device")
        encoder = try LICEncoder(device: device)
        commandQueue = device.makeCommandQueue()!
    }

    // MARK: - Zero-vector handling

    /// Zero vector field with constant input: every step samples the starting pixel
    /// (no streamline advance). Output should equal full_sum * constant for ALL pixels,
    /// including edges and corners (position never moves, so no domain boundary hit).
    func testZeroVector_constantInput_outputEqualsFullSum() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 5)
        let config = LICPipelineConfig()

        let input = [Float](repeating: 1.0, count: size * size)
        let field = [SIMD2<Float>](repeating: .zero, count: size * size)

        let gpuResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights, config: config)

        let cpuResult = LICReferenceCPU.run(
            input: input, vectorField: field,
            width: size, height: size,
            params: params, kernelWeights: weights)

        let expected = params.fullSum
        let tol = expected * 0.005  // f16 tolerance

        for y in 0..<size {
            for x in 0..<size {
                let idx = y * size + x
                XCTAssertEqual(gpuResult[idx], expected, accuracy: tol,
                    "GPU zero-vec (\(x),\(y)): expected \(expected), got \(gpuResult[idx])")
                XCTAssertEqual(cpuResult[idx], expected, accuracy: tol * 0.1,
                    "CPU zero-vec (\(x),\(y)): expected \(expected), got \(cpuResult[idx])")
            }
        }
    }

    // MARK: - NaN/inf handling

    /// NaN vector field: integration terminates immediately in both directions.
    /// Output = center_weight * center_sample. No NaN in output, no boundary hit.
    func testNaNVector_terminatesCleanly_outputIsCenterWeightOnly() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 5)
        let config = LICPipelineConfig()

        let input = [Float](repeating: 1.0, count: size * size)
        let nanVec = SIMD2<Float>(Float.nan, Float.nan)
        let field = [SIMD2<Float>](repeating: nanVec, count: size * size)

        let gpuResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights, config: config)

        let cpuResult = LICReferenceCPU.run(
            input: input, vectorField: field,
            width: size, height: size,
            params: params, kernelWeights: weights)

        let expected = params.centerWeight
        let tol = max(expected * 0.01, 1e-4)

        for y in 0..<size {
            for x in 0..<size {
                let idx = y * size + x
                XCTAssertFalse(gpuResult[idx].isNaN,
                    "GPU NaN-vec (\(x),\(y)) output must not be NaN")
                XCTAssertEqual(gpuResult[idx], expected, accuracy: tol,
                    "GPU NaN-vec (\(x),\(y)): expected center_weight \(expected), got \(gpuResult[idx])")
                XCTAssertFalse(cpuResult[idx].isNaN,
                    "CPU NaN-vec (\(x),\(y)) output must not be NaN")
                XCTAssertEqual(cpuResult[idx], expected, accuracy: tol * 0.1,
                    "CPU NaN-vec (\(x),\(y)): expected center_weight \(expected), got \(cpuResult[idx])")
            }
        }
    }

    /// Inf vector field: same behavior as NaN — terminate before sampling.
    func testInfVector_terminatesCleanly_outputIsCenterWeightOnly() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 5)
        let config = LICPipelineConfig()

        let input = [Float](repeating: 1.0, count: size * size)
        let infVec = SIMD2<Float>(Float.infinity, 0)
        let field = [SIMD2<Float>](repeating: infVec, count: size * size)

        let gpuResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights, config: config)

        let cpuResult = LICReferenceCPU.run(
            input: input, vectorField: field,
            width: size, height: size,
            params: params, kernelWeights: weights)

        let expected = params.centerWeight
        let tol = max(expected * 0.01, 1e-4)

        for y in 0..<size {
            for x in 0..<size {
                let idx = y * size + x
                XCTAssertFalse(gpuResult[idx].isNaN,
                    "GPU inf-vec (\(x),\(y)) output must not be NaN")
                XCTAssertEqual(gpuResult[idx], expected, accuracy: tol,
                    "GPU inf-vec (\(x),\(y)): expected center_weight \(expected), got \(gpuResult[idx])")
                XCTAssertFalse(cpuResult[idx].isNaN,
                    "CPU inf-vec (\(x),\(y)) output must not be NaN")
                XCTAssertEqual(cpuResult[idx], expected, accuracy: tol * 0.1,
                    "CPU inf-vec (\(x),\(y)): expected center_weight \(expected), got \(cpuResult[idx])")
            }
        }
    }

    // MARK: - Mask: starting-pixel

    /// A masked starting pixel returns full_sum * center_sample, bypassing integration.
    /// Uses noise input so the mask shortcut (full_sum * local_sample) is distinguishable
    /// from normal LIC (weighted average along streamline with varying samples).
    func testMaskStartingPixel_returnsFullSumTimesCenterSample() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 5)
        let config = LICPipelineConfig(maskEnabled: true)
        try encoder.buildPipeline(for: config)

        // Noise input — critical: varying values make the mask shortcut produce
        // a different result than normal LIC integration would.
        var rng = SplitMix64(seed: 77777)
        let input = (0..<size * size).map { _ in Float.random(in: 0...1, using: &rng) }
        let field = [SIMD2<Float>](repeating: SIMD2<Float>(1, 0), count: size * size)

        // Left half masked
        var mask = [UInt8](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<8 { mask[y * size + x] = 1 }
        }

        let gpuResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights,
            config: config, mask: mask)

        let cpuResult = LICReferenceCPU.run(
            input: input, vectorField: field,
            width: size, height: size,
            params: params, kernelWeights: weights,
            mask: mask)

        // Each masked pixel should equal full_sum * its local input value.
        let tol = params.fullSum * 0.005  // f16 tolerance

        for y in 0..<size {
            for x in 0..<8 {
                let idx = y * size + x
                let expectedMasked = params.fullSum * input[idx]
                XCTAssertEqual(gpuResult[idx], expectedMasked, accuracy: tol,
                    "GPU masked (\(x),\(y)): expected \(expectedMasked), got \(gpuResult[idx])")
                XCTAssertEqual(cpuResult[idx], expectedMasked, accuracy: tol * 0.1,
                    "CPU masked (\(x),\(y)): expected \(expectedMasked), got \(cpuResult[idx])")
            }
        }

        // Run without mask to verify the shortcut produced different output than
        // normal LIC would — proves the mask path was actually taken.
        let noMaskConfig = LICPipelineConfig()
        let noMaskResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights, config: noMaskConfig)

        // Interior masked pixel (x=4) far from domain edges: normal LIC averages
        // along the streamline, which differs from full_sum * local_sample with noise.
        let probeIdx = 8 * size + 4  // pixel (4, 8)
        let maskedVal = gpuResult[probeIdx]
        let noMaskVal = noMaskResult[probeIdx]
        XCTAssertNotEqual(maskedVal, noMaskVal,
            "Masked shortcut must differ from normal LIC with varying input")
    }

    // MARK: - Mask: stop-on-enter

    /// Streamlines entering a masked region stop and may trigger mask edge gain.
    /// Use noise input + rightward field + masked column; verify GPU matches CPU.
    func testMaskStopOnEnter_gpuMatchesCPU() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 5,
            edgeGainStrength: 0.3, edgeGainPower: 2)
        let config = LICPipelineConfig(maskEnabled: true, edgeGainsEnabled: true)
        try encoder.buildPipeline(for: config)

        var rng = SplitMix64(seed: 99999)
        let input = (0..<size * size).map { _ in Float.random(in: 0...1, using: &rng) }
        let field = [SIMD2<Float>](repeating: SIMD2<Float>(1, 0), count: size * size)

        // Mask column x = 10
        var mask = [UInt8](repeating: 0, count: size * size)
        for y in 0..<size { mask[y * size + 10] = 1 }

        let gpuResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights,
            config: config, mask: mask)

        let cpuResult = LICReferenceCPU.run(
            input: input, vectorField: field,
            width: size, height: size,
            params: params, kernelWeights: weights,
            mask: mask)

        var maxErr: Float = 0
        var meanErr: Float = 0
        for i in 0..<(size * size) {
            let err = abs(gpuResult[i] - cpuResult[i])
            maxErr = max(maxErr, err)
            meanErr += err
        }
        meanErr /= Float(size * size)

        print("--- Mask Stop-on-Enter ---")
        print("maxErr: \(maxErr), meanErr: \(meanErr), full_sum: \(params.fullSum)")

        XCTAssertLessThan(meanErr, params.fullSum * 0.002,
            "Mask stop-on-enter: mean GPU/CPU error too large (\(meanErr))")
        XCTAssertLessThan(maxErr, params.fullSum * 0.01,
            "Mask stop-on-enter: max GPU/CPU error too large (\(maxErr))")

        // Verify the mask actually stopped the streamline: run the same inputs
        // without a mask and confirm the mask-adjacent pixel changed.
        let noMaskConfig = LICPipelineConfig(edgeGainsEnabled: true)
        try encoder.buildPipeline(for: noMaskConfig)
        let noMaskResult = try runGPU(
            input: input, field: field,
            width: size, height: size,
            params: params, weights: weights, config: noMaskConfig)

        // Pixel (9, 8): with mask, forward streamline is blocked at x=10.
        // Without mask, it continues freely. Output must differ.
        let adjacentIdx = 8 * size + 9
        XCTAssertNotEqual(gpuResult[adjacentIdx], noMaskResult[adjacentIdx],
            "Mask-adjacent pixel must differ from no-mask output (proves mask stopped streamline)")
    }

    // MARK: - GPU execution helper

    private func runGPU(
        input: [Float], field: [SIMD2<Float>],
        width: Int, height: Int,
        params: LicParams, weights: [Float],
        config: LICPipelineConfig,
        mask: [UInt8]? = nil
    ) throws -> [Float] {
        let inputTex = try encoder.makeInputTexture(width: width, height: height)
        let vecTex = try encoder.makeVectorFieldTexture(width: width, height: height)
        let outTex = try encoder.makeOutputTexture(width: width, height: height)

        input.withUnsafeBufferPointer { ptr in
            inputTex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: width * MemoryLayout<Float>.stride)
        }

        let packed = field.flatMap { [Float($0.x), Float($0.y)] }
        packed.withUnsafeBufferPointer { ptr in
            vecTex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: width * 2 * MemoryLayout<Float>.stride)
        }

        var maskTex: MTLTexture? = nil
        if let maskData = mask {
            maskTex = try encoder.makeMaskTexture(width: width, height: height)
            maskData.withUnsafeBufferPointer { ptr in
                maskTex!.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: ptr.baseAddress!,
                    bytesPerRow: width * MemoryLayout<UInt8>.stride)
            }
        }

        let cb = commandQueue.makeCommandBuffer()!
        try encoder.encode(
            commandBuffer: cb, params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: outTex, maskTexture: maskTex,
            config: config)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed,
            "GPU failed: \(cb.error?.localizedDescription ?? "unknown")")

        return readR16Float(outTex)
    }

    // MARK: - Readback

    private func readR16Float(_ texture: MTLTexture) -> [Float] {
        let w = texture.width, h = texture.height
        var raw = [UInt16](repeating: 0, count: w * h)
        texture.getBytes(&raw,
                         bytesPerRow: w * MemoryLayout<UInt16>.stride,
                         from: MTLRegionMake2D(0, 0, w, h),
                         mipmapLevel: 0)
        return raw.map { float16BitsToFloat($0) }
    }
}

// MARK: - Float16 conversion

private func float16BitsToFloat(_ bits: UInt16) -> Float {
    let sign     = UInt32((bits >> 15) & 0x1)
    let exponent = UInt32((bits >> 10) & 0x1F)
    let mantissa = UInt32(bits & 0x3FF)
    if exponent == 0 {
        if mantissa == 0 { return sign == 0 ? 0.0 : -0.0 }
        var f = Float(mantissa) / 1024.0
        f *= pow(2.0, -14.0)
        return sign == 0 ? f : -f
    }
    if exponent == 31 {
        if mantissa == 0 { return sign == 0 ? Float.infinity : -Float.infinity }
        return Float.nan
    }
    let f32Exponent = exponent + 112
    let f32Bits = (sign << 31) | (f32Exponent << 23) | (mantissa << 13)
    return Float(bitPattern: f32Bits)
}

// MARK: - Deterministic RNG

private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}
