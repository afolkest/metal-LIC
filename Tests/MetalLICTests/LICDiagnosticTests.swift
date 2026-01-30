import XCTest
import Metal
@testable import MetalLIC

/// Diagnostic tests to verify shader output correctness.
final class LICDiagnosticTests: XCTestCase {

    var device: MTLDevice!
    var encoder: LICEncoder!
    var commandQueue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MetalLIC.device
        try XCTSkipIf(device == nil, "No Metal device")
        encoder = try LICEncoder(device: device)
        commandQueue = device.makeCommandQueue()!
    }

    /// Constant input + uniform field → output should equal full_sum * constant.
    /// This is the simplest possible correctness check.
    func testConstantInput_uniformField_outputEqualsFullSumTimesConstant() throws {
        let size = 32
        let L: Float = 5
        let (params, weights) = try LICKernel.build(L: L)

        let inputTex = try encoder.makeInputTexture(width: size, height: size)
        let vecTex   = try encoder.makeVectorFieldTexture(width: size, height: size)
        let outTex   = try encoder.makeOutputTexture(width: size, height: size)

        // Constant input = 1.0
        let inputData = [Float](repeating: 1.0, count: size * size)
        inputTex.replace(region: MTLRegionMake2D(0, 0, size, size),
                         mipmapLevel: 0,
                         withBytes: inputData,
                         bytesPerRow: size * MemoryLayout<Float>.stride)

        // Uniform rightward field
        let vecData = [Float](repeating: 0, count: size * size * 2)
        var vecMut = vecData
        for i in 0..<(size * size) {
            vecMut[i * 2]     = 1.0  // u
            vecMut[i * 2 + 1] = 0.0  // v
        }
        vecTex.replace(region: MTLRegionMake2D(0, 0, size, size),
                       mipmapLevel: 0,
                       withBytes: vecMut,
                       bytesPerRow: size * 2 * MemoryLayout<Float>.stride)

        let cb = commandQueue.makeCommandBuffer()!
        try encoder.encode(commandBuffer: cb,
                           params: params,
                           kernelWeights: weights,
                           inputTexture: inputTex,
                           vectorField: vecTex,
                           outputTexture: outTex)
        cb.commit()
        cb.waitUntilCompleted()

        // Read back
        let output = readR16Float(outTex)

        // Interior pixel (far from boundaries): should equal full_sum * 1.0
        let center = output[16 * size + 16]
        let expected = params.fullSum
        // float16 tolerance: ~0.1% relative error at this magnitude
        XCTAssertEqual(center, expected, accuracy: expected * 0.01,
                       "Center pixel with constant input should equal full_sum (\(expected)), got \(center)")

        // Edge pixel (x=0): truncated streamline, should be less than full_sum
        let edge = output[16 * size + 0]
        XCTAssertLessThan(edge, expected,
                          "Edge pixel should have truncated kernel (less than full_sum)")
        XCTAssertGreaterThan(edge, 0,
                             "Edge pixel should still be positive")

        // Print some diagnostics
        print("--- Constant Input Diagnostic ---")
        print("full_sum: \(params.fullSum), steps: \(params.steps), kernel_len: \(params.kernelLen)")
        print("center pixel (16,16): \(center)")
        print("edge pixel (0,16): \(edge)")
        print("row 16, first 10 values: \((0..<10).map { output[16 * size + $0] })")
    }

    /// Read back r16Float as [Float].
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
