import XCTest
import Metal
@testable import MetalLIC

/// Tests for LICDispatcher: pipelined multi-buffered dispatch.
final class LICDispatcherTests: XCTestCase {

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

    private func makeNoiseInput(width: Int, height: Int, seed: UInt64 = 42) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<width * height).map { _ in Float(rng.next() & 0xFFFF) / 65535.0 }
    }

    private func makeUniformField(width: Int, height: Int,
                                  direction: SIMD2<Float> = SIMD2(1, 0)) -> [SIMD2<Float>] {
        [SIMD2<Float>](repeating: direction, count: width * height)
    }

    private func makeVortexField(width: Int, height: Int) -> [SIMD2<Float>] {
        let cx = Float(width) / 2, cy = Float(height) / 2
        return (0..<height).flatMap { y in
            (0..<width).map { x in
                let dx = Float(x) - cx, dy = Float(y) - cy
                let len = sqrt(dx * dx + dy * dy) + 1e-6
                return SIMD2<Float>(-dy / len, dx / len)
            }
        }
    }

    private func uploadTextures(
        input: [Float], field: [SIMD2<Float>],
        width: Int, height: Int
    ) throws -> (inputTex: MTLTexture, vecTex: MTLTexture) {
        let inputTex = try encoder.makeInputTexture(width: width, height: height)
        let vecTex = try encoder.makeVectorFieldTexture(width: width, height: height)

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

        return (inputTex, vecTex)
    }

    private func readR16Float(_ texture: MTLTexture) -> [Float] {
        let w = texture.width, h = texture.height
        var raw = [UInt16](repeating: 0, count: w * h)
        texture.getBytes(&raw,
                         bytesPerRow: w * MemoryLayout<UInt16>.stride,
                         from: MTLRegionMake2D(0, 0, w, h),
                         mipmapLevel: 0)
        return raw.map { float16BitsToFloat($0) }
    }

    /// Run LIC serially (single command buffer, wait for completion).
    private func runSerial(
        params: LicParams, weights: [Float],
        inputTex: MTLTexture, vecTex: MTLTexture, outTex: MTLTexture,
        config: LICPipelineConfig = LICPipelineConfig(),
        iterations: Int = 1
    ) throws {
        let cb = commandQueue.makeCommandBuffer()!
        try encoder.encodeMultiPass(
            commandBuffer: cb, params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: outTex, config: config, iterations: iterations)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)
    }

    // MARK: - Warm-up

    /// warmUp() completes without error and produces valid output.
    func testWarmUp_completesSuccessfully() throws {
        let size = 32
        let (params, weights) = try LICKernel.build(L: 5)

        let input = makeNoiseInput(width: size, height: size)
        let field = makeUniformField(width: size, height: size)
        let (inputTex, vecTex) = try uploadTextures(
            input: input, field: field, width: size, height: size)
        let outTex = try encoder.makeOutputTexture(width: size, height: size)

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 3)

        try dispatcher.warmUp(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: outTex)

        let output = readR16Float(outTex)
        let nonZero = output.filter { $0 > 0 }.count
        XCTAssertGreaterThan(nonZero, output.count / 2,
                             "Warm-up should produce non-zero output")
    }

    // MARK: - Single-pass correctness

    /// Pipelined single-pass dispatch matches serial dispatch exactly.
    func testSinglePass_pipelinedMatchesSerial() throws {
        let size = 32
        let (params, weights) = try LICKernel.build(L: 10)

        let input = makeNoiseInput(width: size, height: size)
        let field = makeVortexField(width: size, height: size)
        let (inputTex, vecTex) = try uploadTextures(
            input: input, field: field, width: size, height: size)

        // Serial reference
        let serialOut = try encoder.makeOutputTexture(width: size, height: size)
        try runSerial(params: params, weights: weights,
                      inputTex: inputTex, vecTex: vecTex, outTex: serialOut)
        let serialResult = readR16Float(serialOut)

        // Pipelined dispatch
        let pipelinedOut = try encoder.makeOutputTexture(width: size, height: size)
        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 3)

        try dispatcher.dispatch(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: pipelinedOut)
        dispatcher.waitForAllFrames()

        let pipelinedResult = readR16Float(pipelinedOut)

        // Must be bit-identical (same GPU, same inputs, same pipeline)
        for i in 0..<serialResult.count {
            XCTAssertEqual(pipelinedResult[i], serialResult[i],
                "Pixel \(i): serial=\(serialResult[i]), pipelined=\(pipelinedResult[i])")
        }
    }

    // MARK: - Multi-pass isolation

    /// Multi-pass (iterations=2) through dispatcher produces correct results,
    /// verifying per-frame ping-pong textures don't alias.
    func testMultiPass_pipelinedMatchesSerial() throws {
        let size = 32
        let iterations = 2
        let (params, weights) = try LICKernel.build(L: 5)

        let input = makeNoiseInput(width: size, height: size)
        let field = makeUniformField(width: size, height: size, direction: SIMD2(0.7, 0.7))
        let (inputTex, vecTex) = try uploadTextures(
            input: input, field: field, width: size, height: size)

        // Serial reference
        let serialOut = try encoder.makeOutputTexture(width: size, height: size)
        try runSerial(params: params, weights: weights,
                      inputTex: inputTex, vecTex: vecTex, outTex: serialOut,
                      iterations: iterations)
        let serialResult = readR16Float(serialOut)

        // Pipelined dispatch
        let pipelinedOut = try encoder.makeOutputTexture(width: size, height: size)
        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 3)

        try dispatcher.dispatch(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: pipelinedOut, iterations: iterations)
        dispatcher.waitForAllFrames()

        let pipelinedResult = readR16Float(pipelinedOut)

        for i in 0..<serialResult.count {
            XCTAssertEqual(pipelinedResult[i], serialResult[i],
                "Multi-pass pixel \(i): serial=\(serialResult[i]), pipelined=\(pipelinedResult[i])")
        }
    }

    // MARK: - Multiple frames in flight

    /// Dispatch multiple frames concurrently and verify all complete correctly.
    func testMultipleFrames_allCompleteSuccessfully() throws {
        let size = 32
        let frameCount = 10
        let (params, weights) = try LICKernel.build(L: 5)

        let input = makeNoiseInput(width: size, height: size)
        let field = makeVortexField(width: size, height: size)
        let (inputTex, vecTex) = try uploadTextures(
            input: input, field: field, width: size, height: size)

        // Per-frame output textures (must be distinct to avoid write hazards)
        let outputTextures = try (0..<frameCount).map { _ in
            try encoder.makeOutputTexture(width: size, height: size)
        }

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 3)

        var completedFrames = 0
        let lock = NSLock()

        for i in 0..<frameCount {
            try dispatcher.dispatch(
                params: params, kernelWeights: weights,
                inputTexture: inputTex, vectorField: vecTex,
                outputTexture: outputTextures[i]) { cb in
                    XCTAssertEqual(cb.status, .completed)
                    lock.lock()
                    completedFrames += 1
                    lock.unlock()
                }
        }

        dispatcher.waitForAllFrames()
        XCTAssertEqual(completedFrames, frameCount,
                       "All \(frameCount) frames should complete")

        // Verify all frames produced identical output (same inputs)
        let reference = readR16Float(outputTextures[0])
        for i in 1..<frameCount {
            let result = readR16Float(outputTextures[i])
            for j in 0..<reference.count {
                XCTAssertEqual(result[j], reference[j],
                    "Frame \(i) pixel \(j) differs from frame 0")
            }
        }
    }

    // MARK: - Determinism

    /// Same dispatch sequence produces identical results across two runs.
    func testDeterminism_repeatedDispatchesMatch() throws {
        let size = 32
        let (params, weights) = try LICKernel.build(L: 10)

        let input = makeNoiseInput(width: size, height: size)
        let field = makeVortexField(width: size, height: size)
        let (inputTex, vecTex) = try uploadTextures(
            input: input, field: field, width: size, height: size)

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 3)

        // Run 1
        let out1 = try encoder.makeOutputTexture(width: size, height: size)
        try dispatcher.dispatch(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: out1)
        dispatcher.waitForAllFrames()
        let result1 = readR16Float(out1)

        // Run 2
        let out2 = try encoder.makeOutputTexture(width: size, height: size)
        try dispatcher.dispatch(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: out2)
        dispatcher.waitForAllFrames()
        let result2 = readR16Float(out2)

        for i in 0..<result1.count {
            XCTAssertEqual(result1[i], result2[i],
                "Pixel \(i) differs between runs: \(result1[i]) vs \(result2[i])")
        }
    }

    // MARK: - waitForAllFrames reusability

    /// Dispatcher remains usable after waitForAllFrames.
    func testWaitForAllFrames_dispatcherReusableAfterWait() throws {
        let size = 16
        let (params, weights) = try LICKernel.build(L: 5)

        let input = makeNoiseInput(width: size, height: size)
        let field = makeUniformField(width: size, height: size)
        let (inputTex, vecTex) = try uploadTextures(
            input: input, field: field, width: size, height: size)

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 2)

        // First batch
        let out1 = try encoder.makeOutputTexture(width: size, height: size)
        try dispatcher.dispatch(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: out1)
        dispatcher.waitForAllFrames()

        // Second batch after wait — dispatcher should still work
        let out2 = try encoder.makeOutputTexture(width: size, height: size)
        try dispatcher.dispatch(
            params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: out2)
        dispatcher.waitForAllFrames()

        let result1 = readR16Float(out1)
        let result2 = readR16Float(out2)
        for i in 0..<result1.count {
            XCTAssertEqual(result1[i], result2[i])
        }
    }

    // MARK: - Multi-pass with multiple in-flight frames

    /// Dispatch multiple multi-pass frames with distinct inputs to stress ping-pong isolation.
    /// Using per-frame noise seeds ensures cross-frame ping-pong aliasing would produce
    /// detectable corruption (identical inputs would mask it).
    func testMultiPass_multipleInFlight_pingPongIsolation() throws {
        let size = 32
        let frameCount = 6
        let iterations = 2
        let (params, weights) = try LICKernel.build(L: 5)

        let field = makeVortexField(width: size, height: size)

        // Per-frame distinct inputs and serial references.
        let seeds: [UInt64] = (0..<frameCount).map { UInt64($0) + 1 }
        var serialResults: [[Float]] = []
        var inputTextures: [MTLTexture] = []
        let vecTex = try encoder.makeVectorFieldTexture(width: size, height: size)
        let packedField = field.flatMap { [Float($0.x), Float($0.y)] }
        packedField.withUnsafeBufferPointer { ptr in
            vecTex.replace(
                region: MTLRegionMake2D(0, 0, size, size),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: size * 2 * MemoryLayout<Float>.stride)
        }

        for seed in seeds {
            let input = makeNoiseInput(width: size, height: size, seed: seed)
            let inputTex = try encoder.makeInputTexture(width: size, height: size)
            input.withUnsafeBufferPointer { ptr in
                inputTex.replace(
                    region: MTLRegionMake2D(0, 0, size, size),
                    mipmapLevel: 0,
                    withBytes: ptr.baseAddress!,
                    bytesPerRow: size * MemoryLayout<Float>.stride)
            }
            inputTextures.append(inputTex)

            let serialOut = try encoder.makeOutputTexture(width: size, height: size)
            try runSerial(params: params, weights: weights,
                          inputTex: inputTex, vecTex: vecTex, outTex: serialOut,
                          iterations: iterations)
            serialResults.append(readR16Float(serialOut))
        }

        // Pipelined: dispatch all frames with multi-pass
        let outputTextures = try (0..<frameCount).map { _ in
            try encoder.makeOutputTexture(width: size, height: size)
        }

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue, maxInFlight: 3)

        for i in 0..<frameCount {
            try dispatcher.dispatch(
                params: params, kernelWeights: weights,
                inputTexture: inputTextures[i], vectorField: vecTex,
                outputTexture: outputTextures[i], iterations: iterations)
        }
        dispatcher.waitForAllFrames()

        // Each frame must match its own serial reference
        for i in 0..<frameCount {
            let result = readR16Float(outputTextures[i])
            for j in 0..<serialResults[i].count {
                XCTAssertEqual(result[j], serialResults[i][j],
                    "Frame \(i) pixel \(j): serial=\(serialResults[i][j]), pipelined=\(result[j])")
            }
        }
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
