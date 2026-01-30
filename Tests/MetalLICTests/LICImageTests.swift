import XCTest
import Metal
import CoreGraphics
import ImageIO
@testable import MetalLIC

/// Generates LIC images for visual inspection and CPU/GPU comparison.
final class LICImageTests: XCTestCase {

    static let outputDir: String = {
        let dir = NSTemporaryDirectory() + "MetalLIC_images"
        try? FileManager.default.createDirectory(atPath: dir,
                                                  withIntermediateDirectories: true)
        return dir
    }()

    var device: MTLDevice!
    var encoder: LICEncoder!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "No Metal device")
        encoder = try LICEncoder(device: device)
    }

    // MARK: - Image generation

    func testGenerateImages() throws {
        let size = 512
        let L: Float = 30
        let (params, weights) = try LICKernel.build(L: L)

        // Edge gains enabled so boundary processing matches CPU reference
        let config = LICPipelineConfig(edgeGainsEnabled: true)
        try encoder.buildPipeline(for: config)

        // Deterministic noise
        var rng = SplitMix64(seed: 12345)
        let noise = (0..<size * size).map { _ in Float.random(in: 0...1, using: &rng) }

        let fields: [(String, [SIMD2<Float>])] = [
            ("uniform",  makeUniformField(width: size, height: size, dx: 1, dy: 0)),
            ("diagonal", makeUniformField(width: size, height: size, dx: 1, dy: 1)),
            ("vortex",   makeVortexField(width: size, height: size)),
            ("saddle",   makeSaddleField(width: size, height: size)),
            ("radial",   makeRadialField(width: size, height: size)),
        ]

        writePNG(noise, width: size, height: size, normalize: 1.0, name: "input_noise")

        print("\n=== LIC Image Generation ===")
        print("Output: \(Self.outputDir)")
        print("Size: \(size)x\(size), L=\(L), h=\(params.h), steps=\(params.steps), full_sum=\(params.fullSum)\n")

        for (name, field) in fields {
            // CPU reference
            let cpuStart = CFAbsoluteTimeGetCurrent()
            let cpuResult = LICReferenceCPU.run(
                input: noise, vectorField: field,
                width: size, height: size,
                params: params, kernelWeights: weights)
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            writePNG(cpuResult, width: size, height: size,
                     normalize: params.fullSum, name: "cpu_\(name)")

            // GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            let gpuResult = try runGPU(
                noise: noise, field: field,
                width: size, height: size,
                params: params, weights: weights, config: config)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

            writePNG(gpuResult, width: size, height: size,
                     normalize: params.fullSum, name: "gpu_\(name)")

            // Diff (amplified 50x for visibility)
            let diff = zip(cpuResult, gpuResult).map { abs($0 - $1) }
            let maxDiff = diff.max() ?? 0
            let meanDiff = diff.reduce(0, +) / Float(diff.count)
            writePNG(diff, width: size, height: size,
                     normalize: params.fullSum / 50.0, name: "diff_\(name)")

            let pad = name.padding(toLength: 10, withPad: " ", startingAt: 0)
            print(String(format: "%@  CPU %.2fs  GPU %.3fs  maxErr %.4f  meanErr %.6f",
                         pad as NSString, cpuTime, gpuTime, maxDiff, meanDiff))

            // Boundary-renormalized pixels can diverge due to float32 vs float64
            // position tracking (one extra/fewer step changes used_sum, which
            // renormalization amplifies). Mean error is the better metric.
            XCTAssertLessThan(meanDiff, params.fullSum * 0.001,
                              "\(name): GPU/CPU mean error too large")
        }

        print("\nImages: \(Self.outputDir)")
    }

    // MARK: - GPU execution

    private func runGPU(
        noise: [Float], field: [SIMD2<Float>],
        width: Int, height: Int,
        params: LicParams, weights: [Float],
        config: LICPipelineConfig
    ) throws -> [Float] {
        let inputTex = try encoder.makeInputTexture(width: width, height: height)
        let vectorTex = try encoder.makeVectorFieldTexture(width: width, height: height)
        let outputTex = try encoder.makeOutputTexture(width: width, height: height)

        noise.withUnsafeBufferPointer { ptr in
            inputTex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: width * MemoryLayout<Float>.stride)
        }

        let packed = field.flatMap { [Float($0.x), Float($0.y)] }
        packed.withUnsafeBufferPointer { ptr in
            vectorTex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: width * 2 * MemoryLayout<Float>.stride)
        }

        let cb = device.makeCommandQueue()!.makeCommandBuffer()!
        try encoder.encode(
            commandBuffer: cb, params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vectorTex,
            outputTexture: outputTex, config: config)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed,
                       "GPU failed: \(cb.error?.localizedDescription ?? "unknown")")

        return readR16Float(outputTex, width: width, height: height)
    }

    // MARK: - Readback

    private func readR16Float(_ tex: MTLTexture, width: Int, height: Int) -> [Float] {
        var raw = [UInt16](repeating: 0, count: width * height)
        tex.getBytes(&raw,
                     bytesPerRow: width * MemoryLayout<UInt16>.stride,
                     from: MTLRegionMake2D(0, 0, width, height),
                     mipmapLevel: 0)
        return raw.map { float16BitsToFloat($0) }
    }

    private func float16BitsToFloat(_ bits: UInt16) -> Float {
        let sign: UInt32 = UInt32(bits >> 15) & 1
        let exp:  UInt32 = UInt32(bits >> 10) & 0x1F
        let frac: UInt32 = UInt32(bits) & 0x3FF
        if exp == 0 {
            if frac == 0 { return sign == 1 ? -0.0 : 0.0 }
            var f = Float(frac) / 1024.0
            f *= powf(2.0, -14.0)
            return sign == 1 ? -f : f
        }
        if exp == 31 { return frac == 0 ? (sign == 1 ? -.infinity : .infinity) : .nan }
        let f32exp = exp + 112
        let bits32 = (sign << 31) | (f32exp << 23) | (frac << 13)
        return Float(bitPattern: bits32)
    }

    // MARK: - Vector field generators

    private func makeUniformField(width: Int, height: Int, dx: Float, dy: Float) -> [SIMD2<Float>] {
        let len = sqrtf(dx * dx + dy * dy)
        let v = SIMD2<Float>(dx / len, dy / len)
        return [SIMD2<Float>](repeating: v, count: width * height)
    }

    private func makeVortexField(width: Int, height: Int) -> [SIMD2<Float>] {
        let cx = Float(width) / 2, cy = Float(height) / 2
        return (0..<height).flatMap { j in
            (0..<width).map { i in
                let x = Float(i) - cx + 0.5, y = Float(j) - cy + 0.5
                let len = sqrtf(x * x + y * y)
                return len < 1e-6 ? .zero : SIMD2<Float>(-y / len, x / len)
            }
        }
    }

    private func makeSaddleField(width: Int, height: Int) -> [SIMD2<Float>] {
        let cx = Float(width) / 2, cy = Float(height) / 2
        return (0..<height).flatMap { j in
            (0..<width).map { i in
                let x = Float(i) - cx + 0.5, y = Float(j) - cy + 0.5
                let len = sqrtf(x * x + y * y)
                return len < 1e-6 ? .zero : SIMD2<Float>(x / len, -y / len)
            }
        }
    }

    private func makeRadialField(width: Int, height: Int) -> [SIMD2<Float>] {
        let cx = Float(width) / 2, cy = Float(height) / 2
        return (0..<height).flatMap { j in
            (0..<width).map { i in
                let x = Float(i) - cx + 0.5, y = Float(j) - cy + 0.5
                let len = sqrtf(x * x + y * y)
                return len < 1e-6 ? .zero : SIMD2<Float>(x / len, y / len)
            }
        }
    }

    // MARK: - PNG output

    private func writePNG(_ data: [Float], width: Int, height: Int,
                          normalize: Float, name: String) {
        let bytes: [UInt8] = data.map { px in
            UInt8(max(0, min(255, (px / normalize) * 255 + 0.5)))
        }
        let cfData = Data(bytes) as CFData
        guard let provider = CGDataProvider(data: cfData),
              let image = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                  space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                  provider: provider,
                  decode: nil, shouldInterpolate: false,
                  intent: .defaultIntent)
        else { return }

        let url = URL(fileURLWithPath: Self.outputDir + "/\(name).png")
        guard let dest = CGImageDestinationCreateWithURL(
                  url as CFURL, "public.png" as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }
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
