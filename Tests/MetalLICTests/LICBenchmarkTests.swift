import XCTest
import Metal
@testable import MetalLIC

/// GPU benchmark harness for M3 performance tuning.
///
/// Measures pure GPU execution time using `MTLCommandBuffer.gpuStartTime`/`gpuEndTime`.
/// Textures are pre-allocated with `.private` storage to avoid managed-mode overhead.
/// Each scenario runs configurable warm-up + measurement iterations, then reports
/// min/median/mean/p95/max/CV GPU time, wall-clock time, CPU encode overhead, and derived FPS.
///
/// Benchmarks are skipped by default in `swift test`. Enable with:
///   RUN_BENCHMARKS=1 swift test --filter LICBenchmarkTests
///
/// Run a single scenario:
///   RUN_BENCHMARKS=1 swift test --filter LICBenchmarkTests/testBenchmark_2K_vortex
///
/// Note: For consistent results, run benchmarks standalone (with --filter) rather than
/// as part of the full test suite, to avoid thermal state from preceding tests.
final class LICBenchmarkTests: XCTestCase {

    var device: MTLDevice!
    var encoder: LICEncoder!
    var commandQueue: MTLCommandQueue!

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != nil,
            "Set RUN_BENCHMARKS=1 to run benchmarks")
        device = MetalLIC.device
        try XCTSkipIf(device == nil, "No Metal device")
        encoder = try LICEncoder(device: device)
        commandQueue = device.makeCommandQueue()!
    }

    // MARK: - Types

    enum FieldType {
        case uniform
        case vortex
    }

    struct BenchmarkScenario {
        let name: String
        let width: Int
        let height: Int
        let fieldType: FieldType
        let L: Float
        let h: Float
        let iterations: Int
        let warmUpCount: Int
        let measureCount: Int
        let targetFPS: Double?

        init(name: String,
             width: Int, height: Int,
             fieldType: FieldType,
             L: Float = 30, h: Float = 1.0,
             iterations: Int = 1,
             warmUpCount: Int = 5, measureCount: Int = 20,
             targetFPS: Double? = nil) {
            self.name = name
            self.width = width
            self.height = height
            self.fieldType = fieldType
            self.L = L
            self.h = h
            self.iterations = iterations
            self.warmUpCount = warmUpCount
            self.measureCount = measureCount
            self.targetFPS = targetFPS
        }

        var resolutionLabel: String { "\(width)x\(height)" }
    }

    struct FrameTiming {
        let gpuMs: Double
        let wallMs: Double
        let encodeMs: Double
    }

    struct BenchmarkResult {
        let scenario: BenchmarkScenario
        let gpuMin: Double
        let gpuMedian: Double
        let gpuMean: Double
        let gpuP95: Double
        let gpuMax: Double
        let gpuCV: Double          // coefficient of variation (stddev / mean)
        let wallMedian: Double
        let encodeMean: Double
        let encodeMax: Double
        let gpuTimingAvailable: Bool

        var fps: Double {
            let ms = gpuTimingAvailable ? gpuMedian : wallMedian
            return ms > 0 ? 1000.0 / ms : 0
        }

        var meetsTarget: Bool? {
            guard let target = scenario.targetFPS else { return nil }
            return fps >= target
        }
    }

    // MARK: - Individual benchmarks

    func testBenchmark_1080p_uniform() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "1080p uniform",
            width: 1920, height: 1080,
            fieldType: .uniform))
        printReport(result)
    }

    func testBenchmark_1080p_vortex() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "1080p vortex",
            width: 1920, height: 1080,
            fieldType: .vortex))
        printReport(result)
    }

    func testBenchmark_2K_uniform() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "2K uniform",
            width: 2048, height: 2048,
            fieldType: .uniform,
            targetFPS: 60))
        printReport(result)
    }

    func testBenchmark_2K_vortex() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "2K vortex",
            width: 2048, height: 2048,
            fieldType: .vortex,
            targetFPS: 60))
        printReport(result)
    }

    func testBenchmark_4K_uniform() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "4K uniform",
            width: 3840, height: 2160,
            fieldType: .uniform,
            targetFPS: 30))
        printReport(result)
    }

    func testBenchmark_4K_vortex() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "4K vortex",
            width: 3840, height: 2160,
            fieldType: .vortex,
            targetFPS: 30))
        printReport(result)
    }

    func testBenchmark_2K_vortex_3pass() throws {
        let result = try runBenchmark(BenchmarkScenario(
            name: "2K vortex 3-pass",
            width: 2048, height: 2048,
            fieldType: .vortex,
            iterations: 3))
        printReport(result)
    }

    func testBenchmark_summary() throws {
        let scenarios = [
            BenchmarkScenario(name: "1080p uniform", width: 1920, height: 1080,
                              fieldType: .uniform),
            BenchmarkScenario(name: "1080p vortex", width: 1920, height: 1080,
                              fieldType: .vortex),
            BenchmarkScenario(name: "2K uniform", width: 2048, height: 2048,
                              fieldType: .uniform, targetFPS: 60),
            BenchmarkScenario(name: "2K vortex", width: 2048, height: 2048,
                              fieldType: .vortex, targetFPS: 60),
            BenchmarkScenario(name: "4K uniform", width: 3840, height: 2160,
                              fieldType: .uniform, targetFPS: 30),
            BenchmarkScenario(name: "4K vortex", width: 3840, height: 2160,
                              fieldType: .vortex, targetFPS: 30),
            BenchmarkScenario(name: "2K vortex 3-pass", width: 2048, height: 2048,
                              fieldType: .vortex, iterations: 3),
        ]

        var results: [BenchmarkResult] = []
        for scenario in scenarios {
            let result = try autoreleasepool {
                try runBenchmark(scenario)
            }
            results.append(result)
        }
        printSummaryTable(results)
    }

    // MARK: - Core benchmark runner

    private func runBenchmark(_ scenario: BenchmarkScenario) throws -> BenchmarkResult {
        let (params, weights) = try LICKernel.build(L: scenario.L, h: scenario.h)
        let config = LICPipelineConfig()

        let textures = try prepareTextures(
            width: scenario.width, height: scenario.height,
            fieldType: scenario.fieldType)

        // Warm-up: prime caches and driver, discard results.
        for _ in 0..<scenario.warmUpCount {
            let cb = commandQueue.makeCommandBuffer()!
            try encodeScenario(cb, scenario: scenario, params: params,
                               weights: weights, config: config, textures: textures)
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Measurement loop: one command buffer per iteration for clean GPU timestamps.
        var timings: [FrameTiming] = []
        timings.reserveCapacity(scenario.measureCount)

        for _ in 0..<scenario.measureCount {
            let cb = commandQueue.makeCommandBuffer()!

            let encodeStart = CFAbsoluteTimeGetCurrent()
            try encodeScenario(cb, scenario: scenario, params: params,
                               weights: weights, config: config, textures: textures)
            let encodeEnd = CFAbsoluteTimeGetCurrent()

            let wallStart = CFAbsoluteTimeGetCurrent()
            cb.commit()
            cb.waitUntilCompleted()
            let wallEnd = CFAbsoluteTimeGetCurrent()

            guard cb.status == .completed else {
                XCTFail("Command buffer failed: \(cb.error?.localizedDescription ?? "unknown")")
                continue
            }

            let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
            let wallMs = (wallEnd - wallStart) * 1000.0
            let encodeMs = (encodeEnd - encodeStart) * 1000.0

            timings.append(FrameTiming(gpuMs: gpuMs, wallMs: wallMs, encodeMs: encodeMs))
        }

        return computeResult(scenario: scenario, timings: timings)
    }

    private func encodeScenario(
        _ cb: MTLCommandBuffer,
        scenario: BenchmarkScenario,
        params: LicParams,
        weights: [Float],
        config: LICPipelineConfig,
        textures: (input: MTLTexture, vectorField: MTLTexture, output: MTLTexture)
    ) throws {
        if scenario.iterations == 1 {
            try encoder.encode(
                commandBuffer: cb,
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: textures.output,
                config: config)
        } else {
            try encoder.encodeMultiPass(
                commandBuffer: cb,
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: textures.output,
                config: config,
                iterations: scenario.iterations)
        }
    }

    // MARK: - Texture preparation

    /// Creates `.private` GPU-resident textures for the timing loop.
    /// Data is uploaded to `.managed` staging textures, then blit-copied.
    private func prepareTextures(
        width: Int, height: Int,
        fieldType: FieldType
    ) throws -> (input: MTLTexture, vectorField: MTLTexture, output: MTLTexture) {
        // Create staging textures and fill with CPU data.
        let stagingInput = try encoder.makeInputTexture(width: width, height: height)
        let stagingVec = try encoder.makeVectorFieldTexture(width: width, height: height)

        fillInputNoise(stagingInput, width: width, height: height)
        fillVectorField(stagingVec, width: width, height: height, fieldType: fieldType)

        // Create .private textures.
        let privateInput = try makePrivateTexture(
            format: .r32Float, width: width, height: height,
            usage: .shaderRead)
        let privateVec = try makePrivateTexture(
            format: .rg32Float, width: width, height: height,
            usage: .shaderRead)
        let privateOutput = try makePrivateTexture(
            format: .r16Float, width: width, height: height,
            usage: [.shaderRead, .shaderWrite])

        // Blit staging -> private.
        let cb = commandQueue.makeCommandBuffer()!
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw LICError.encoderCreationFailed
        }
        blitCopy(blit, from: stagingInput, to: privateInput)
        blitCopy(blit, from: stagingVec, to: privateVec)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return (privateInput, privateVec, privateOutput)
    }

    private func makePrivateTexture(
        format: MTLPixelFormat,
        width: Int, height: Int,
        usage: MTLTextureUsage
    ) throws -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false)
        desc.usage = usage
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw LICError.textureCreationFailed
        }
        return tex
    }

    private func blitCopy(_ encoder: MTLBlitCommandEncoder,
                          from src: MTLTexture, to dst: MTLTexture) {
        encoder.copy(
            from: src, sourceSlice: 0, sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: src.width, height: src.height, depth: 1),
            to: dst, destinationSlice: 0, destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    }

    // MARK: - Data generators

    private func fillInputNoise(_ texture: MTLTexture, width: Int, height: Int) {
        var data = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                data[y * width + x] = noise(x: x, y: y)
            }
        }
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: &data,
            bytesPerRow: width * MemoryLayout<Float>.stride)
    }

    private func fillVectorField(_ texture: MTLTexture,
                                 width: Int, height: Int,
                                 fieldType: FieldType) {
        let field: [Float]
        switch fieldType {
        case .uniform:
            field = makeUniformField(width: width, height: height)
        case .vortex:
            field = makeVortexField(width: width, height: height)
        }
        var mutableField = field
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: &mutableField,
            bytesPerRow: width * 2 * MemoryLayout<Float>.stride)
    }

    /// Deterministic hash noise matching existing test patterns.
    private func noise(x: Int, y: Int, seed: UInt32 = 0) -> Float {
        var h = seed &+ UInt32(x) &* 374761393 &+ UInt32(y) &* 668265263
        h = (h ^ (h >> 13)) &* 1274126177
        h = h ^ (h >> 16)
        return Float(h & 0xFFFF) / 65535.0
    }

    /// Uniform field (1, 0) — coherent streamlines, best-case texture cache.
    private func makeUniformField(width: Int, height: Int) -> [Float] {
        var data = [Float](repeating: 0, count: width * height * 2)
        for i in stride(from: 0, to: data.count, by: 2) {
            data[i]     = 1.0  // u
            data[i + 1] = 0.0  // v
        }
        return data
    }

    /// Vortex field — divergent streamlines, stresses texture cache.
    private func makeVortexField(width: Int, height: Int) -> [Float] {
        let cx = Float(width) / 2.0
        let cy = Float(height) / 2.0
        var data = [Float](repeating: 0, count: width * height * 2)
        for j in 0..<height {
            for i in 0..<width {
                let x = Float(i) - cx + 0.5
                let y = Float(j) - cy + 0.5
                let len = sqrtf(x * x + y * y)
                let idx = (j * width + i) * 2
                if len < 1e-6 {
                    data[idx]     = 0.0
                    data[idx + 1] = 0.0
                } else {
                    data[idx]     = -y / len
                    data[idx + 1] =  x / len
                }
            }
        }
        return data
    }

    // MARK: - Statistics

    private func computeResult(scenario: BenchmarkScenario,
                                timings: [FrameTiming]) -> BenchmarkResult {
        let gpuTimes = timings.map(\.gpuMs).sorted()
        let wallTimes = timings.map(\.wallMs).sorted()
        let encodeTimes = timings.map(\.encodeMs)

        let gpuAvailable = !gpuTimes.isEmpty && gpuTimes[0] > 0
        let n = Double(max(gpuTimes.count, 1))
        let mean = gpuTimes.reduce(0, +) / n
        let variance = gpuTimes.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / n
        let cv = mean > 0 ? sqrt(variance) / mean : 0

        return BenchmarkResult(
            scenario: scenario,
            gpuMin: gpuTimes.first ?? 0,
            gpuMedian: percentile(gpuTimes, 0.5),
            gpuMean: mean,
            gpuP95: percentile(gpuTimes, 0.95),
            gpuMax: gpuTimes.last ?? 0,
            gpuCV: cv,
            wallMedian: percentile(wallTimes, 0.5),
            encodeMean: encodeTimes.reduce(0, +) / Double(max(encodeTimes.count, 1)),
            encodeMax: encodeTimes.max() ?? 0,
            gpuTimingAvailable: gpuAvailable)
    }

    /// Linear-interpolation percentile on a sorted array.
    private func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = p * Double(sorted.count - 1)
        let lo = Int(idx)
        let hi = min(lo + 1, sorted.count - 1)
        let frac = idx - Double(lo)
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    // MARK: - Reporting

    private func printReport(_ result: BenchmarkResult) {
        let s = result.scenario
        print("")
        print("=== LIC Benchmark: \(s.name) ===")
        print("Device:     \(device.name)")
        print("Resolution: \(s.resolutionLabel)")
        print("Params:     L=\(Int(s.L)) h=\(s.h) iterations=\(s.iterations)")
        print("Runs:       \(s.warmUpCount) warm-up + \(s.measureCount) measured")
        print("")

        if result.gpuTimingAvailable {
            let cvPct = String(format: "%.1f%%", result.gpuCV * 100)
            print("GPU time (ms):")
            print("  min    = \(fmt(result.gpuMin))")
            print("  median = \(fmt(result.gpuMedian))")
            print("  mean   = \(fmt(result.gpuMean))")
            print("  p95    = \(fmt(result.gpuP95))")
            print("  max    = \(fmt(result.gpuMax))")
            print("  CV     = \(cvPct)\(result.gpuCV > 0.10 ? "  (high variance — results may be unreliable)" : "")")
        } else {
            print("GPU timing not available (gpuStartTime returned 0).")
        }

        print("Wall time (ms):")
        print("  median = \(fmt(result.wallMedian))")
        print("Encode overhead (ms):")
        print("  mean   = \(fmt(result.encodeMean))")
        print("  max    = \(fmt(result.encodeMax))")

        let source = result.gpuTimingAvailable ? "GPU" : "wall"
        print("")
        print("FPS (\(source) median): \(String(format: "%.1f", result.fps))")

        if let target = s.targetFPS {
            let pass = result.fps >= target
            print("Target: \(Int(target)) fps -> \(pass ? "PASS" : "FAIL")")
        }
        print("")
    }

    private func printSummaryTable(_ results: [BenchmarkResult]) {
        print("")
        print("=== LIC GPU Benchmark Summary ===")
        print("Device: \(device.name)")
        print("")

        let header = "\(pad("Scenario", 22)) \(pad("Resolution", 12)) "
            + "\(lpad("GPU med", 9)) \(lpad("GPU p95", 9)) \(lpad("GPU max", 9)) "
            + "\(lpad("CV", 6)) "
            + "\(lpad("Wall md", 9)) \(lpad("Encode", 8)) "
            + "\(lpad("FPS", 7))   Target"
        print(header)
        print(String(repeating: "\u{2500}", count: 106))

        for r in results {
            let source = r.gpuTimingAvailable ? r.gpuMedian : r.wallMedian
            let medStr = r.gpuTimingAvailable ? fmtMs(r.gpuMedian) : fmtMs(r.wallMedian)
            let p95Str = r.gpuTimingAvailable ? fmtMs(r.gpuP95) : "--"
            let maxStr = r.gpuTimingAvailable ? fmtMs(r.gpuMax) : "--"
            let cvStr = r.gpuTimingAvailable
                ? String(format: "%.1f%%", r.gpuCV * 100) : "--"
            let wallStr = fmtMs(r.wallMedian)
            let encStr = fmtMs(r.encodeMean)
            let fpsVal = source > 0 ? 1000.0 / source : 0
            let fpsStr = String(format: "%.1f", fpsVal)

            var targetStr = ""
            if let target = r.scenario.targetFPS {
                let pass = r.fps >= target
                targetStr = "\(Int(target))fps \(pass ? "PASS" : "FAIL")"
            }

            let line = "\(pad(r.scenario.name, 22)) \(pad(r.scenario.resolutionLabel, 12)) "
                + "\(lpad(medStr, 9)) \(lpad(p95Str, 9)) \(lpad(maxStr, 9)) "
                + "\(lpad(cvStr, 6)) "
                + "\(lpad(wallStr, 9)) \(lpad(encStr, 8)) "
                + "\(lpad(fpsStr, 7))   \(targetStr)"
            print(line)
        }
        print("")
    }

    private func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    private func lpad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    private func fmt(_ ms: Double) -> String {
        String(format: "%.2f", ms)
    }

    private func fmtMs(_ ms: Double) -> String {
        String(format: "%.2fms", ms)
    }
}
