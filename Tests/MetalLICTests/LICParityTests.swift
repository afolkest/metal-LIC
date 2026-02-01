import XCTest
import Metal
import CoreGraphics
import ImageIO
import Foundation
@testable import MetalLIC

/// bryLIC parity checks: SSIM, histogram distance, error heatmaps.
/// All metrics are advisory — no hard assertions. Differences are expected
/// due to algorithmic divergence (RK2 vs pixel-crossing, bilinear vs nearest-neighbor).
///
/// Prerequisites:
///   python3 scripts/generate_brylic_reference.py
///
/// Run with:
///   RUN_PARITY=1 swift test --filter LICParityTests
///
/// Run a single scene:
///   RUN_PARITY=1 swift test --filter LICParityTests/testParity_vortex
final class LICParityTests: XCTestCase {

    // MARK: - Paths

    static let fixtureDir: String = {
        let src = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()   // MetalLICTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // package root
        return src
            .appendingPathComponent("Tests/MetalLICTests/Fixtures/brylic_reference")
            .path
    }()

    static let outputDir: String = {
        let src = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let dir = src
            .appendingPathComponent("output/parity")
            .path
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        return dir
    }()

    // MARK: - State

    var device: MTLDevice!
    var encoder: LICEncoder!
    var commandQueue: MTLCommandQueue!
    var metadata: ParityMetadata!

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_PARITY"] != nil,
            "Set RUN_PARITY=1 to run bryLIC parity checks")

        let fm = FileManager.default
        try XCTSkipUnless(
            fm.fileExists(atPath: Self.fixtureDir + "/metadata.json"),
            "bryLIC fixtures not found. Run: python3 scripts/generate_brylic_reference.py")

        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "No Metal device")
        encoder = try LICEncoder(device: device)
        commandQueue = device.makeCommandQueue()!

        metadata = try loadMetadata()
    }

    // MARK: - Per-scene tests (baseline)

    func testParity_uniform()       throws { try runScene("uniform") }
    func testParity_vortex()        throws { try runScene("vortex") }
    func testParity_saddle()        throws { try runScene("saddle") }
    func testParity_radial()        throws { try runScene("radial") }
    func testParity_vortex_3pass()  throws { try runScene("vortex_3pass") }
    func testParity_vortex_masked() throws { try runScene("vortex_masked") }

    // MARK: - Per-scene tests (stress)

    func testParity_vortex_2pass()       throws { try runScene("vortex_2pass") }
    func testParity_vortex_edge_gain()   throws { try runScene("vortex_edge_gain") }
    func testParity_radial_domain_gain() throws { try runScene("radial_domain_gain") }
    func testParity_vortex_both_gains()  throws { try runScene("vortex_both_gains") }
    func testParity_radial_unnorm()      throws { try runScene("radial_unnorm") }
    func testParity_shear()              throws { try runScene("shear") }
    func testParity_zero_patch()         throws { try runScene("zero_patch") }
    func testParity_nan_patch()          throws { try runScene("nan_patch") }

    func testParity_summary() throws {
        var results: [(String, ParityResult)] = []
        for scene in metadata.scenes {
            let result = try computeParity(scene)
            results.append((scene.name, result))
        }
        printSummaryTable(results)
    }

    // MARK: - Core logic

    private func runScene(_ sceneName: String) throws {
        guard let scene = metadata.scenes.first(where: { $0.name == sceneName }) else {
            XCTFail("Scene '\(sceneName)' not found in metadata")
            return
        }
        let result = try computeParity(scene)
        printSceneReport(sceneName, result)
        writeComparisonImages(sceneName, result)
    }

    private func computeParity(_ scene: SceneConfig) throws -> ParityResult {
        let width = metadata.width
        let height = metadata.height

        // Build params with per-scene edge gain settings
        let hasEdgeGains = scene.edge_gain_strength > 0
            || scene.domain_edge_gain_strength > 0
        let (params, weights) = try LICKernel.build(
            L: metadata.L, h: metadata.h,
            edgeGainStrength: scene.edge_gain_strength,
            edgeGainPower: scene.edge_gain_power,
            domainEdgeGainStrength: scene.domain_edge_gain_strength,
            domainEdgeGainPower: scene.domain_edge_gain_power)

        let config = LICPipelineConfig(
            maskEnabled: scene.mask,
            edgeGainsEnabled: hasEdgeGains)

        // Load inputs from fixtures
        let noise = try loadFloat32("noise.bin", count: width * height)
        let u = try loadFloat32("field_\(scene.field)_u.bin", count: width * height)
        let v = try loadFloat32("field_\(scene.field)_v.bin", count: width * height)

        // Load bryLIC output
        let brylicOutput = try loadFloat32("brylic_\(scene.name).bin", count: width * height)

        // Build interleaved vector field for Metal
        let field: [SIMD2<Float>] = (0..<width * height).map {
            SIMD2<Float>(u[$0], v[$0])
        }

        // Load mask if needed
        var mask: [UInt8]? = nil
        if scene.mask {
            mask = try loadUInt8("mask_circular.bin", count: width * height)
        }

        // Run metal-LIC GPU
        let metalOutput: [Float]
        if scene.iterations > 1 {
            metalOutput = try runGPUMultiPass(
                input: noise, field: field,
                width: width, height: height,
                params: params, weights: weights,
                config: config, mask: mask,
                iterations: scene.iterations)
        } else {
            metalOutput = try runGPU(
                input: noise, field: field,
                width: width, height: height,
                params: params, weights: weights,
                config: config, mask: mask)
        }

        // Normalize both to [0, 1] using the joint max of both outputs.
        // For multi-pass, output scales as ~fullSum^iterations, so a single
        // fullSum divisor produces values >> 1 and makes error stats meaningless.
        let fullSum = params.fullSum
        let jointMax = max(
            metalOutput.max() ?? 1,
            brylicOutput.max() ?? 1,
            Float.leastNormalMagnitude)
        let metalNorm = metalOutput.map { $0 / jointMax }
        let brylicNorm = brylicOutput.map { $0 / jointMax }

        // Compute metrics on [0, 1]-normalized outputs
        let ssim = computeSSIM(metalNorm, brylicNorm, width: width, height: height)
        let histDist = computeHistogramChiSquared(metalNorm, brylicNorm)
        let errorStats = computeErrorStats(metalNorm, brylicNorm)
        let errorMap = zip(metalNorm, brylicNorm).map { abs($0 - $1) }

        return ParityResult(
            ssim: ssim,
            histogramDistance: histDist,
            meanAbsError: errorStats.mean,
            maxAbsError: errorStats.max,
            p95AbsError: errorStats.p95,
            p99AbsError: errorStats.p99,
            metalOutput: metalOutput,
            brylicOutput: brylicOutput,
            errorMap: errorMap,
            fullSum: fullSum,
            normFactor: jointMax)
    }

    // MARK: - GPU execution

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

    private func runGPUMultiPass(
        input: [Float], field: [SIMD2<Float>],
        width: Int, height: Int,
        params: LicParams, weights: [Float],
        config: LICPipelineConfig,
        mask: [UInt8]? = nil,
        iterations: Int
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
        try encoder.encodeMultiPass(
            commandBuffer: cb, params: params, kernelWeights: weights,
            inputTexture: inputTex, vectorField: vecTex,
            outputTexture: outTex, maskTexture: maskTex,
            config: config, iterations: iterations)
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

    // MARK: - SSIM

    /// Mean SSIM using 11x11 uniform window (Wang et al. 2004).
    /// Inputs should be normalized to approximately [0, 1].
    private func computeSSIM(
        _ a: [Float], _ b: [Float],
        width: Int, height: Int,
        windowSize: Int = 11
    ) -> Float {
        let L: Float = 1.0
        let C1: Float = (0.01 * L) * (0.01 * L)
        let C2: Float = (0.03 * L) * (0.03 * L)
        let halfWin = windowSize / 2
        let winArea = Float(windowSize * windowSize)

        var ssimSum: Double = 0
        var count = 0

        for y in halfWin..<(height - halfWin) {
            for x in halfWin..<(width - halfWin) {
                var sumA: Float = 0, sumB: Float = 0
                var sumA2: Float = 0, sumB2: Float = 0, sumAB: Float = 0

                for wy in -halfWin...halfWin {
                    for wx in -halfWin...halfWin {
                        let idx = (y + wy) * width + (x + wx)
                        let va = a[idx], vb = b[idx]
                        sumA += va;  sumB += vb
                        sumA2 += va * va;  sumB2 += vb * vb
                        sumAB += va * vb
                    }
                }

                let muA = sumA / winArea
                let muB = sumB / winArea
                let sigA2 = max(0, sumA2 / winArea - muA * muA)
                let sigB2 = max(0, sumB2 / winArea - muB * muB)
                let sigAB = sumAB / winArea - muA * muB

                let num = (2 * muA * muB + C1) * (2 * sigAB + C2)
                let den = (muA * muA + muB * muB + C1) * (sigA2 + sigB2 + C2)

                ssimSum += Double(num / den)
                count += 1
            }
        }

        return count > 0 ? Float(ssimSum / Double(count)) : 0
    }

    // MARK: - Histogram chi-squared

    /// Chi-squared distance between 256-bin histograms of two images.
    /// Lower is better; 0 = identical distributions.
    private func computeHistogramChiSquared(
        _ a: [Float], _ b: [Float], bins: Int = 256
    ) -> Float {
        let n = Float(a.count)
        let aMin = a.min() ?? 0, aMax = a.max() ?? 1
        let bMin = b.min() ?? 0, bMax = b.max() ?? 1
        let globalMin = min(aMin, bMin)
        let globalMax = max(aMax, bMax)
        let range = globalMax - globalMin
        guard range > 0 else { return 0 }

        var histA = [Float](repeating: 0, count: bins)
        var histB = [Float](repeating: 0, count: bins)

        for val in a {
            let bin = min(bins - 1, max(0, Int((val - globalMin) / range * Float(bins))))
            histA[bin] += 1.0 / n
        }
        for val in b {
            let bin = min(bins - 1, max(0, Int((val - globalMin) / range * Float(bins))))
            histB[bin] += 1.0 / n
        }

        var chi2: Float = 0
        for i in 0..<bins {
            let sum = histA[i] + histB[i]
            if sum > 0 {
                let diff = histA[i] - histB[i]
                chi2 += diff * diff / sum
            }
        }
        return chi2
    }

    // MARK: - Error statistics

    private struct ErrorStats {
        let mean: Float
        let max: Float
        let p95: Float
        let p99: Float
    }

    private func computeErrorStats(_ a: [Float], _ b: [Float]) -> ErrorStats {
        let errors = zip(a, b).map { abs($0 - $1) }
        let sorted = errors.sorted()
        let n = errors.count
        guard n > 0 else {
            return ErrorStats(mean: 0, max: 0, p95: 0, p99: 0)
        }

        let mean = errors.reduce(0, +) / Float(n)
        let maxErr = sorted[n - 1]
        let p95 = sorted[min(n - 1, Int(0.95 * Float(n)))]
        let p99 = sorted[min(n - 1, Int(0.99 * Float(n)))]

        return ErrorStats(mean: mean, max: maxErr, p95: p95, p99: p99)
    }

    // MARK: - Fixture loading

    private func loadFloat32(_ name: String, count: Int) throws -> [Float] {
        let path = Self.fixtureDir + "/" + name
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let expected = count * MemoryLayout<Float>.size
        XCTAssertEqual(data.count, expected,
            "Size mismatch for \(name): expected \(expected), got \(data.count)")
        return data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }

    private func loadUInt8(_ name: String, count: Int) throws -> [UInt8] {
        let path = Self.fixtureDir + "/" + name
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        XCTAssertEqual(data.count, count,
            "Size mismatch for \(name): expected \(count), got \(data.count)")
        return Array(data)
    }

    private func loadMetadata() throws -> ParityMetadata {
        let path = Self.fixtureDir + "/metadata.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(ParityMetadata.self, from: data)
    }

    // MARK: - PNG output

    private func writeComparisonImages(_ sceneName: String, _ result: ParityResult) {
        let width = metadata.width
        let height = metadata.height
        let normFactor = result.normFactor

        writePNG(result.metalOutput, width: width, height: height,
                 normalize: normFactor, name: "metal_\(sceneName)")
        writePNG(result.brylicOutput, width: width, height: height,
                 normalize: normFactor, name: "brylic_\(sceneName)")

        // Error heatmap amplified 10x for visibility
        let amplified = result.errorMap.map { min(1.0, $0 * 10.0) }
        writePNG(amplified, width: width, height: height,
                 normalize: 1.0, name: "error_\(sceneName)")
    }

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

    // MARK: - Reporting

    private func printSceneReport(_ name: String, _ r: ParityResult) {
        print("")
        print("=== bryLIC Parity: \(name) ===")
        print("  SSIM:         \(fmt4(r.ssim))")
        print("  Hist chi2:    \(fmt4(r.histogramDistance))")
        print("  Mean |error|: \(fmt6(r.meanAbsError))")
        print("  Max  |error|: \(fmt6(r.maxAbsError))")
        print("  P95  |error|: \(fmt6(r.p95AbsError))")
        print("  P99  |error|: \(fmt6(r.p99AbsError))")
        print("  Images:       \(Self.outputDir)")
    }

    private func printSummaryTable(_ results: [(String, ParityResult)]) {
        print("")
        print("=== bryLIC Parity Summary ===")
        print("NOTE: All metrics are advisory. Differences are expected due to:")
        print("  - Integration method (RK2 vs exact pixel-crossing)")
        print("  - Field sampling (bilinear vs nearest-neighbor)")
        print("  - Accumulation precision (float32->float16 vs float32)")
        print("")

        let nameWidth = 22
        let header = "\(pad("Scene", nameWidth)) \(lpad("SSIM", 8)) "
            + "\(lpad("Chi2", 8)) "
            + "\(lpad("Mean|E|", 9)) \(lpad("Max|E|", 9)) "
            + "\(lpad("P95", 9)) \(lpad("P99", 9))"
        print(header)
        print(String(repeating: "-", count: nameWidth + 54))

        for (name, r) in results {
            let line = "\(pad(name, nameWidth)) \(lpad(fmt4(r.ssim), 8)) "
                + "\(lpad(fmt4(r.histogramDistance), 8)) "
                + "\(lpad(fmt6(r.meanAbsError), 9)) \(lpad(fmt6(r.maxAbsError), 9)) "
                + "\(lpad(fmt6(r.p95AbsError), 9)) \(lpad(fmt6(r.p99AbsError), 9))"
            print(line)
        }

        print("")
        print("Images: \(Self.outputDir)")
        print("")
    }

    // MARK: - Formatting helpers

    private func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    private func lpad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    private func fmt4(_ v: Float) -> String { String(format: "%.4f", v) }
    private func fmt6(_ v: Float) -> String { String(format: "%.6f", v) }
}

// MARK: - Types

struct ParityResult {
    let ssim: Float
    let histogramDistance: Float
    let meanAbsError: Float
    let maxAbsError: Float
    let p95AbsError: Float
    let p99AbsError: Float
    let metalOutput: [Float]
    let brylicOutput: [Float]
    let errorMap: [Float]
    let fullSum: Float
    let normFactor: Float
}

struct SceneConfig: Decodable {
    let name: String
    let field: String
    let iterations: Int
    let mask: Bool
    let edge_gain_strength: Float
    let edge_gain_power: Float
    let domain_edge_gain_strength: Float
    let domain_edge_gain_power: Float
}

struct ParityMetadata: Decodable {
    let width: Int
    let height: Int
    let L: Float
    let h: Float
    let steps: Int
    let kernel_len: Int
    let full_sum: Float
    let center_weight: Float
    let scenes: [SceneConfig]
    let noise_seed: Int
    let noise_rng: String
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
