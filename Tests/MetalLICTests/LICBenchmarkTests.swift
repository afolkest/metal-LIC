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
        let threadgroupSize: MTLSize?

        init(name: String,
             width: Int, height: Int,
             fieldType: FieldType,
             L: Float = 30, h: Float = 1.0,
             iterations: Int = 1,
             warmUpCount: Int = 5, measureCount: Int = 20,
             targetFPS: Double? = nil,
             threadgroupSize: MTLSize? = nil) {
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
            self.threadgroupSize = threadgroupSize
        }

        var resolutionLabel: String { "\(width)x\(height)" }

        var threadgroupLabel: String {
            guard let tg = threadgroupSize else { return "default" }
            return "\(tg.width)x\(tg.height)"
        }
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

    // MARK: - Pipelined throughput

    /// Measures sustained throughput with multiple in-flight command buffers
    /// vs the serial baseline. Reports FPS improvement from pipelining.
    func testBenchmark_pipelined_2K_vortex() throws {
        let width = 2048, height = 2048
        let L: Float = 30, h: Float = 1.0
        let maxInFlight = 3
        let warmUpCount = 5
        let measureCount = 30

        let (params, weights) = try LICKernel.build(L: L, h: h)
        let config = LICPipelineConfig()
        let textures = try prepareTextures(
            width: width, height: height, fieldType: .vortex)

        // --- Serial baseline (existing pattern) ---

        let serialScenario = BenchmarkScenario(
            name: "2K vortex (serial)",
            width: width, height: height,
            fieldType: .vortex, L: L, h: h,
            warmUpCount: warmUpCount, measureCount: measureCount,
            targetFPS: 60)
        let serialResult = try runBenchmark(serialScenario)

        // --- Pipelined measurement ---

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue,
            maxInFlight: maxInFlight)

        // Create per-frame output textures (distinct to avoid write hazards).
        let outputTextures = try (0..<maxInFlight).map { _ in
            try makePrivateTexture(
                format: .r16Float, width: width, height: height,
                usage: [.shaderRead, .shaderWrite])
        }

        // Warm-up.
        try dispatcher.warmUp(
            params: params, kernelWeights: weights,
            inputTexture: textures.input,
            vectorField: textures.vectorField,
            outputTexture: outputTextures[0],
            config: config)

        // Additional warm-up dispatches to stabilize.
        for i in 0..<warmUpCount {
            try dispatcher.dispatch(
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: outputTextures[i % maxInFlight],
                config: config)
        }
        dispatcher.waitForAllFrames()

        // Measurement: dispatch all frames, collect GPU times from callbacks.
        var gpuTimes: [Double] = []
        let lock = NSLock()

        let wallStart = CFAbsoluteTimeGetCurrent()

        for i in 0..<measureCount {
            try dispatcher.dispatch(
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: outputTextures[i % maxInFlight],
                config: config) { cb in
                    if cb.status == .completed {
                        let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
                        lock.lock()
                        gpuTimes.append(gpuMs)
                        lock.unlock()
                    }
                }
        }
        dispatcher.waitForAllFrames()

        let wallEnd = CFAbsoluteTimeGetCurrent()
        let totalWallMs = (wallEnd - wallStart) * 1000.0
        let sustainedFPS = Double(measureCount) / (totalWallMs / 1000.0)

        // --- Report ---

        gpuTimes.sort()
        let gpuMedian = !gpuTimes.isEmpty
            ? gpuTimes[gpuTimes.count / 2] : 0
        let gpuMean = !gpuTimes.isEmpty
            ? gpuTimes.reduce(0, +) / Double(gpuTimes.count) : 0

        let serialWallMs = serialResult.wallMedian
        let serialWallFPS = serialWallMs > 0 ? 1000.0 / serialWallMs : 0
        let improvement = serialWallFPS > 0
            ? (sustainedFPS - serialWallFPS) / serialWallFPS * 100 : 0

        print("")
        print("=== Pipelined Throughput: 2K vortex ===")
        print("Device:       \(device.name)")
        print("Resolution:   \(width)x\(height)")
        print("Params:       L=\(Int(L)) h=\(h)")
        print("In-flight:    \(maxInFlight)")
        print("Frames:       \(warmUpCount) warm-up + \(measureCount) measured")
        print("")
        print("Serial baseline (wall-clock):")
        print("  Wall median: \(fmtMs(serialWallMs))")
        print("  FPS:         \(String(format: "%.1f", serialWallFPS))")
        print("")
        print("Pipelined:")
        print("  GPU median: \(fmtMs(gpuMedian))")
        print("  GPU mean:   \(fmtMs(gpuMean))")
        print("  Wall total: \(fmtMs(totalWallMs))  (\(measureCount) frames)")
        print("  Sustained:  \(String(format: "%.1f", sustainedFPS)) FPS")
        print("")
        print("Improvement:  \(String(format: "%+.1f%%", improvement))")
        print("  (Improvement reflects reduced CPU-GPU sync stalls.")
        print("   Gains are larger when encode time is significant vs GPU time.)")
        print("")
    }

    func testBenchmark_pipelined_2K_vortex_3pass() throws {
        let width = 2048, height = 2048
        let L: Float = 30, h: Float = 1.0
        let iterations = 3
        let maxInFlight = 3
        let warmUpCount = 5
        let measureCount = 30

        let (params, weights) = try LICKernel.build(L: L, h: h)
        let config = LICPipelineConfig()
        let textures = try prepareTextures(
            width: width, height: height, fieldType: .vortex)

        // --- Serial baseline ---

        let serialScenario = BenchmarkScenario(
            name: "2K vortex 3-pass (serial)",
            width: width, height: height,
            fieldType: .vortex, L: L, h: h,
            iterations: iterations,
            warmUpCount: warmUpCount, measureCount: measureCount)
        let serialResult = try runBenchmark(serialScenario)

        // --- Pipelined measurement ---

        let dispatcher = LICDispatcher(
            encoder: encoder, commandQueue: commandQueue,
            maxInFlight: maxInFlight)

        let outputTextures = try (0..<maxInFlight).map { _ in
            try makePrivateTexture(
                format: .r16Float, width: width, height: height,
                usage: [.shaderRead, .shaderWrite])
        }

        try dispatcher.warmUp(
            params: params, kernelWeights: weights,
            inputTexture: textures.input,
            vectorField: textures.vectorField,
            outputTexture: outputTextures[0],
            config: config, iterations: iterations)

        for i in 0..<warmUpCount {
            try dispatcher.dispatch(
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: outputTextures[i % maxInFlight],
                config: config, iterations: iterations)
        }
        dispatcher.waitForAllFrames()

        var gpuTimes: [Double] = []
        let lock = NSLock()

        let wallStart = CFAbsoluteTimeGetCurrent()

        for i in 0..<measureCount {
            try dispatcher.dispatch(
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: outputTextures[i % maxInFlight],
                config: config, iterations: iterations) { cb in
                    if cb.status == .completed {
                        let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
                        lock.lock()
                        gpuTimes.append(gpuMs)
                        lock.unlock()
                    }
                }
        }
        dispatcher.waitForAllFrames()

        let wallEnd = CFAbsoluteTimeGetCurrent()
        let totalWallMs = (wallEnd - wallStart) * 1000.0
        let sustainedFPS = Double(measureCount) / (totalWallMs / 1000.0)

        gpuTimes.sort()
        let gpuMedian = !gpuTimes.isEmpty
            ? gpuTimes[gpuTimes.count / 2] : 0
        let gpuMean = !gpuTimes.isEmpty
            ? gpuTimes.reduce(0, +) / Double(gpuTimes.count) : 0

        let serialWallMs = serialResult.wallMedian
        let serialWallFPS = serialWallMs > 0 ? 1000.0 / serialWallMs : 0
        let improvement = serialWallFPS > 0
            ? (sustainedFPS - serialWallFPS) / serialWallFPS * 100 : 0

        print("")
        print("=== Pipelined Throughput: 2K vortex 3-pass ===")
        print("Device:       \(device.name)")
        print("Resolution:   \(width)x\(height)")
        print("Params:       L=\(Int(L)) h=\(h) iterations=\(iterations)")
        print("In-flight:    \(maxInFlight)")
        print("Frames:       \(warmUpCount) warm-up + \(measureCount) measured")
        print("")
        print("Serial baseline (wall-clock):")
        print("  Wall median: \(fmtMs(serialWallMs))")
        print("  FPS:         \(String(format: "%.1f", serialWallFPS))")
        print("")
        print("Pipelined:")
        print("  GPU median: \(fmtMs(gpuMedian))")
        print("  GPU mean:   \(fmtMs(gpuMean))")
        print("  Wall total: \(fmtMs(totalWallMs))  (\(measureCount) frames)")
        print("  Sustained:  \(String(format: "%.1f", sustainedFPS)) FPS")
        print("")
        print("Improvement:  \(String(format: "%+.1f%%", improvement))")
        print("  (Multi-pass benefits more from pipelining: CPU encodes N passes")
        print("   while GPU executes previous frame's N passes.)")
        print("")
    }

    // MARK: - Threadgroup size sweep

    /// Threadgroup sizes to test. Apple Silicon SIMD width is 32 threads,
    /// so all candidates are multiples of 32 for full SIMD utilization.
    private static let threadgroupCandidates: [(w: Int, h: Int)] = [
        (8, 4),     // 32 — 1 SIMD group, wide
        (4, 8),     // 32 — 1 SIMD group, tall
        (8, 8),     // 64 — 2 SIMD groups (current default)
        (16, 4),    // 64 — 2 SIMD groups, wide
        (16, 8),    // 128 — 4 SIMD groups
        (8, 16),    // 128 — 4 SIMD groups, tall
        (16, 16),   // 256 — 8 SIMD groups
        (32, 4),    // 128 — 4 SIMD groups, very wide
        (32, 8),    // 256 — 8 SIMD groups, wide
        (32, 16),   // 512 — 16 SIMD groups
        (32, 32),   // 1024 — max typical
    ]

    func testThreadgroupSweep_2K_vortex() throws {
        try runThreadgroupSweep(
            label: "2K vortex",
            width: 2048, height: 2048,
            fieldType: .vortex)
    }

    func testThreadgroupSweep_4K_vortex() throws {
        try runThreadgroupSweep(
            label: "4K vortex",
            width: 3840, height: 2160,
            fieldType: .vortex)
    }

    func testThreadgroupSweep_2K_uniform() throws {
        try runThreadgroupSweep(
            label: "2K uniform",
            width: 2048, height: 2048,
            fieldType: .uniform)
    }

    func testThreadgroupSweep_4K_uniform() throws {
        try runThreadgroupSweep(
            label: "4K uniform",
            width: 3840, height: 2160,
            fieldType: .uniform)
    }

    private func runThreadgroupSweep(
        label: String,
        width: Int, height: Int,
        fieldType: FieldType
    ) throws {
        let config = LICPipelineConfig()
        let pipeline = try encoder.buildPipeline(for: config)
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let simdWidth = pipeline.threadExecutionWidth

        print("")
        print("=== Threadgroup Sweep: \(label) ===")
        print("Device:     \(device.name)")
        print("Resolution: \(width)x\(height)")
        print("SIMD width: \(simdWidth)")
        print("Max threads/threadgroup: \(maxThreads)")
        print("")

        // Filter candidates that exceed hardware limit.
        let validCandidates = Self.threadgroupCandidates.filter {
            $0.w * $0.h <= maxThreads
        }

        var results: [BenchmarkResult] = []
        for (tw, th) in validCandidates {
            let tgSize = MTLSize(width: tw, height: th, depth: 1)
            let scenario = BenchmarkScenario(
                name: "\(tw)x\(th) (\(tw * th))",
                width: width, height: height,
                fieldType: fieldType,
                warmUpCount: 5, measureCount: 20,
                threadgroupSize: tgSize)
            let result = try autoreleasepool {
                try runBenchmark(scenario)
            }
            results.append(result)
        }

        printThreadgroupSweepTable(results, simdWidth: simdWidth)
    }

    private func printThreadgroupSweepTable(_ results: [BenchmarkResult],
                                            simdWidth: Int) {
        guard let best = results.min(by: {
            let a = $0.gpuTimingAvailable ? $0.gpuMedian : $0.wallMedian
            let b = $1.gpuTimingAvailable ? $1.gpuMedian : $1.wallMedian
            return a < b
        }) else { return }

        let bestMs = best.gpuTimingAvailable ? best.gpuMedian : best.wallMedian

        let header = "\(pad("Threadgroup", 16)) \(lpad("Threads", 8)) "
            + "\(lpad("SIMDs", 6)) "
            + "\(lpad("GPU med", 9)) \(lpad("GPU p95", 9)) "
            + "\(lpad("CV", 6)) "
            + "\(lpad("FPS", 7))   vs best"
        print(header)
        print(String(repeating: "\u{2500}", count: 80))

        for r in results {
            guard let tg = r.scenario.threadgroupSize else { continue }
            let threads = tg.width * tg.height
            let simds = threads / simdWidth
            let medMs = r.gpuTimingAvailable ? r.gpuMedian : r.wallMedian
            let p95Ms = r.gpuTimingAvailable ? r.gpuP95 : r.wallMedian
            let cvStr = r.gpuTimingAvailable
                ? String(format: "%.1f%%", r.gpuCV * 100) : "--"
            let fpsVal = medMs > 0 ? 1000.0 / medMs : 0
            let delta = bestMs > 0
                ? String(format: "%+.1f%%", (medMs - bestMs) / bestMs * 100)
                : "--"
            let marker = medMs <= bestMs * 1.001 ? " <-- BEST" : ""

            let line = "\(pad(r.scenario.threadgroupLabel, 16)) \(lpad("\(threads)", 8)) "
                + "\(lpad("\(simds)", 6)) "
                + "\(lpad(fmtMs(medMs), 9)) \(lpad(fmtMs(p95Ms), 9)) "
                + "\(lpad(cvStr, 6)) "
                + "\(lpad(String(format: "%.1f", fpsVal), 7))   \(delta)\(marker)"
            print(line)
        }
        print("")
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
                config: config,
                threadgroupSize: scenario.threadgroupSize)
        } else {
            try encoder.encodeMultiPass(
                commandBuffer: cb,
                params: params, kernelWeights: weights,
                inputTexture: textures.input,
                vectorField: textures.vectorField,
                outputTexture: textures.output,
                config: config,
                iterations: scenario.iterations,
                threadgroupSize: scenario.threadgroupSize)
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
        let variance = gpuTimes.count > 1
            ? gpuTimes.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / (n - 1)
            : 0.0
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

    // MARK: - Occupancy & register pressure analysis

    /// Queries Metal pipeline properties across all specialization variants to assess
    /// occupancy and register pressure. Reports maxTotalThreadsPerThreadgroup (the key
    /// indicator — lower values mean higher register pressure), threadExecutionWidth,
    /// and threadgroup memory usage.
    ///
    /// On Apple Silicon, the compiler reduces maxTotalThreadsPerThreadgroup when a kernel
    /// uses too many registers. Typical values:
    ///   1024 = low register pressure (ideal)
    ///    512 = moderate pressure
    ///    256 = high pressure (occupancy-limited)
    func testOccupancyAnalysis() throws {
        print("")
        print("=== Occupancy & Register Pressure Analysis ===")
        print("Device: \(device.name)")
        print("")

        // --- 1. Pipeline property survey across all specialization variants ---

        struct VariantInfo {
            let label: String
            let config: LICPipelineConfig
        }

        let variants: [VariantInfo] = [
            VariantInfo(label: "Default (no mask, no edge gains, no debug)",
                        config: LICPipelineConfig(maskEnabled: false, edgeGainsEnabled: false, debugMode: 0)),
            VariantInfo(label: "Mask only",
                        config: LICPipelineConfig(maskEnabled: true, edgeGainsEnabled: false, debugMode: 0)),
            VariantInfo(label: "Mask + edge gains",
                        config: LICPipelineConfig(maskEnabled: true, edgeGainsEnabled: true, debugMode: 0)),
            VariantInfo(label: "Edge gains only",
                        config: LICPipelineConfig(maskEnabled: false, edgeGainsEnabled: true, debugMode: 0)),
            VariantInfo(label: "Debug: step count heat map",
                        config: LICPipelineConfig(maskEnabled: false, edgeGainsEnabled: false, debugMode: 1)),
            VariantInfo(label: "Debug: boundary hits",
                        config: LICPipelineConfig(maskEnabled: true, edgeGainsEnabled: false, debugMode: 2)),
            VariantInfo(label: "Debug: kernel support ratio",
                        config: LICPipelineConfig(maskEnabled: false, edgeGainsEnabled: false, debugMode: 3)),
            VariantInfo(label: "Full (mask + edge gains + debug 1)",
                        config: LICPipelineConfig(maskEnabled: true, edgeGainsEnabled: true, debugMode: 1)),
        ]

        print("Pipeline Properties by Specialization Variant:")
        print("")
        let hdr = "\(pad("Variant", 48)) \(lpad("MaxTh", 6)) \(lpad("SIMD", 5)) "
            + "\(lpad("SIMDs", 6)) \(lpad("TgMem", 6)) \(lpad("Occupancy", 10))"
        print(hdr)
        print(String(repeating: "\u{2500}", count: 85))

        var defaultMaxThreads: Int = 1024

        for v in variants {
            let pipeline = try encoder.buildPipeline(for: v.config)
            let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
            let simdWidth = pipeline.threadExecutionWidth
            let tgMemBytes = pipeline.staticThreadgroupMemoryLength
            let maxSimds = maxThreads / simdWidth

            if v.config == LICPipelineConfig() {
                defaultMaxThreads = maxThreads
            }

            // Occupancy estimate: maxThreads / 1024 (assuming 1024 is hardware max)
            let occupancyPct = Double(maxThreads) / 1024.0 * 100.0
            let occupancyStr = String(format: "%.0f%%", occupancyPct)

            let line = "\(pad(v.label, 48)) \(lpad("\(maxThreads)", 6)) \(lpad("\(simdWidth)", 5)) "
                + "\(lpad("\(maxSimds)", 6)) \(lpad("\(tgMemBytes)B", 6)) \(lpad(occupancyStr, 10))"
            print(line)
        }
        print("")

        // --- 2. Device capability context ---

        print("Device Capabilities:")
        print("  Max threadgroup memory:  \(device.maxThreadgroupMemoryLength) bytes")
        print("  Max threads/threadgroup: \(defaultMaxThreads) (from default pipeline)")
        print("  Recommended max working set size: \(device.recommendedMaxWorkingSetSize / (1024*1024)) MB")
        if #available(macOS 13.0, *) {
            print("  Architecture: \(device.architecture.name)")
        }
        print("")

        // --- 3. Register pressure assessment ---

        print("Register Pressure Assessment:")
        print("")
        if defaultMaxThreads >= 1024 {
            print("  Status: LOW register pressure")
            print("  maxTotalThreadsPerThreadgroup = \(defaultMaxThreads) (hardware maximum)")
            print("  -> Occupancy is NOT limited by register usage.")
            print("  -> Performance is likely memory-bandwidth or ALU bound, not occupancy bound.")
            print("")
            print("  Recommendation: No register-reduction optimizations needed.")
            print("  Proceed to bandwidth analysis to determine the actual bottleneck.")
        } else if defaultMaxThreads >= 512 {
            print("  Status: MODERATE register pressure")
            print("  maxTotalThreadsPerThreadgroup = \(defaultMaxThreads) (< 1024 hardware max)")
            print("  -> Occupancy reduced to ~\(defaultMaxThreads * 100 / 1024)% of theoretical max.")
            print("  -> May be limiting latency hiding for texture fetches.")
            print("")
            print("  Recommendations:")
            print("  1. Split forward/backward integration into separate loops to reduce live register range")
            print("  2. Reuse float2 temporaries (v, v1 can alias since they don't overlap)")
            print("  3. Move boundary processing to a second pass if edge gains are rare")
            print("  4. Use Xcode GPU Capture -> Shader Profiler for exact register count")
        } else {
            print("  Status: HIGH register pressure")
            print("  maxTotalThreadsPerThreadgroup = \(defaultMaxThreads) (< 512)")
            print("  -> Occupancy severely limited. This is likely the primary bottleneck.")
            print("")
            print("  Recommendations:")
            print("  1. Aggressive register reduction: split kernel into multiple passes")
            print("  2. Reduce float2 live variables in inner loop (currently ~5 float2 + scalars)")
            print("  3. Consider half precision for intermediate positions if quality allows")
            print("  4. Use Xcode GPU Capture -> Shader Profiler for exact register count")
        }
        print("")

        // --- 4. Shader code analysis summary ---

        print("Shader Register Analysis (from source):")
        print("")
        print("  Inner loop live variables (per direction):")
        print("    float2 x        — current position (2 regs)")
        print("    float2 v        — direction at x (2 regs)")
        print("    float2 v1       — direction at midpoint (2 regs)")
        print("    float2 x1       — midpoint position (2 regs)")
        print("    float2 x_next   — next position (2 regs)")
        print("    bool v_valid    — direction validity (1 reg)")
        print("    bool v1_valid   — midpoint validity (1 reg)")
        print("    uint step_count — loop counter (1 reg)")
        print("    float w         — kernel weight (1 reg)")
        print("    float s         — input sample (1 reg)")
        print("  Outer scope (live across loop):")
        print("    float2 x0       — pixel center (2 regs)")
        print("    float W, H      — dimensions (2 regs)")
        print("    float value     — accumulator (1 reg)")
        print("    float used_sum  — weight sum (1 reg)")
        print("    bool hit_domain_edge, hit_mask_edge (2 regs)")
        print("    uint total_steps (1 reg)")
        print("  Estimated total: ~24 scalar registers in inner loop")
        print("  (Actual count may differ — compiler may spill or optimize)")
        print("")

        // --- 5. Comparative benchmark: effect of threadgroup size on occupancy ---

        print("Occupancy Sensitivity Test (2K vortex, varying threadgroup size):")
        print("  Testing whether larger threadgroups (more concurrent threads) improve perf...")
        print("")

        let testSizes: [(w: Int, h: Int, label: String)] = [
            (8, 4, "8x4 (32 = 1 SIMD)"),
            (8, 8, "8x8 (64 = 2 SIMDs)"),
            (16, 16, "16x16 (256 = 8 SIMDs)"),
            (32, 32, "32x32 (1024 = 32 SIMDs)"),
        ]

        let validSizes = testSizes.filter { $0.w * $0.h <= defaultMaxThreads }

        var benchResults: [(label: String, gpuMs: Double)] = []
        for (tw, th, label) in validSizes {
            let scenario = BenchmarkScenario(
                name: label,
                width: 2048, height: 2048,
                fieldType: .vortex,
                warmUpCount: 3, measureCount: 10,
                threadgroupSize: MTLSize(width: tw, height: th, depth: 1))
            let result = try autoreleasepool {
                try runBenchmark(scenario)
            }
            let ms = result.gpuTimingAvailable ? result.gpuMedian : result.wallMedian
            benchResults.append((label: label, gpuMs: ms))
        }

        let bestMs = benchResults.min(by: { $0.gpuMs < $1.gpuMs })?.gpuMs ?? 1.0
        let worstMs = benchResults.max(by: { $0.gpuMs < $1.gpuMs })?.gpuMs ?? 1.0
        let spread = worstMs > 0 ? (worstMs - bestMs) / bestMs * 100 : 0

        let tblHdr = "\(pad("Threadgroup", 32)) \(lpad("GPU med", 9)) \(lpad("FPS", 7)) \(lpad("vs best", 9))"
        print(tblHdr)
        print(String(repeating: "\u{2500}", count: 60))
        for r in benchResults {
            let fps = r.gpuMs > 0 ? 1000.0 / r.gpuMs : 0
            let delta = String(format: "%+.1f%%", (r.gpuMs - bestMs) / bestMs * 100)
            let marker = r.gpuMs <= bestMs * 1.001 ? " <-- BEST" : ""
            let line = "\(pad(r.label, 32)) \(lpad(fmtMs(r.gpuMs), 9)) \(lpad(String(format: "%.1f", fps), 7)) \(lpad(delta, 9))\(marker)"
            print(line)
        }
        print("")

        // Interpret spread in context of register pressure findings.
        let registersLimiting = defaultMaxThreads < 1024
        if spread < 5.0 {
            print("  Interpretation: <5% spread across threadgroup sizes.")
            print("  -> Performance is NOT occupancy-sensitive.")
            print("  -> Kernel is likely ALU-bound. Move to bandwidth analysis.")
        } else if registersLimiting {
            print("  Interpretation: \(String(format: "%.0f%%", spread)) spread AND maxThreads=\(defaultMaxThreads) < 1024.")
            print("  -> Register pressure is limiting threadgroup size AND larger groups help.")
            print("  -> Register reduction should yield measurable gains.")
            print("  Recommendations:")
            print("    1. Split forward/backward integration to reduce live register range")
            print("    2. Reuse float2 temporaries (v, v1 can alias)")
            print("    3. Use Xcode GPU Capture -> Shader Profiler for exact register count")
        } else {
            print("  Interpretation: \(String(format: "%.0f%%", spread)) spread, but maxThreads=\(defaultMaxThreads) (hardware max).")
            print("  -> Kernel is TEXTURE-LATENCY BOUND, not register-limited.")
            print("  -> Large threadgroups (32x32) are optimal for hiding ~120 texture fetches/pixel.")
            print("  -> No register-reduction work needed.")
            print("  -> Proceed to BANDWIDTH ANALYSIS to determine if access patterns can be improved.")
        }
        print("")
    }

    // MARK: - Bandwidth analysis

    /// Determines whether the LIC kernel is ALU-bound, bandwidth-bound, or texture-latency-bound.
    ///
    /// Runs three diagnostic experiments:
    /// 1. Resolution scaling (1080p/2K/4K, fixed L=30) — checks if ns/fetch is constant across pixel counts
    /// 2. L-scaling (4K, L=5..50) — checks if ns/fetch changes with streamline length
    /// 3. Cache efficiency (uniform vs vortex ratio) — measures texture cache impact of divergent streamlines
    ///
    /// Key metric: **nanoseconds per texture fetch** = GPU time / (pixels × fetches_per_pixel).
    /// - Constant across resolutions + L values → texture unit at steady-state throughput
    /// - Increases with L → cache pressure from longer streamlines
    /// - Large uniform/vortex gap → texture-latency-bound (cache misses dominate)
    func testBandwidthAnalysis() throws {
        print("")
        print("=== Bandwidth Analysis ===")
        print("Device: \(device.name)")
        if #available(macOS 13.0, *) {
            print("Architecture: \(device.architecture.name)")
        }
        print("")

        // --- Theoretical model ---

        let defaultL: Float = 30.0
        let defaultH: Float = 1.0
        let defaultSteps = Int(round(defaultL / defaultH))

        // Per step: 2 vector field samples (RK2) + 1 input sample = 3
        // Total: center input (1) + 2 directions × steps × 3
        let fetchesPerPixel = 1 + 2 * defaultSteps * 3  // 181 for L=30, h=1.0
        let vectorFetchesPerPixel = 2 * defaultSteps * 2  // 120
        let inputFetchesPerPixel = 1 + 2 * defaultSteps   // 61

        // Bilinear filtering reads a 2×2 texel footprint.
        // Vector field: rg32Float = 8 bytes/texel × 4 texels = 32 bytes/sample
        // Input: r32Float = 4 bytes/texel × 4 texels = 16 bytes/sample
        let bytesPerVectorFetch: Double = 32.0
        let bytesPerInputFetch: Double = 16.0
        let theoreticalBytesPerPixel =
            Double(vectorFetchesPerPixel) * bytesPerVectorFetch +
            Double(inputFetchesPerPixel) * bytesPerInputFetch +
            2.0  // output write: r16Float

        // Minimum bytes: read entire texture once (no re-reads, perfect cache).
        // Vector field: W×H×8, Input: W×H×4, Output: W×H×2 → 14 bytes/pixel.
        let minimumBytesPerPixel: Double = 14.0

        print("--- Theoretical Model (L=\(Int(defaultL)), h=\(defaultH)) ---")
        print("  Texture fetches per pixel:  \(fetchesPerPixel)  (\(vectorFetchesPerPixel) vector + \(inputFetchesPerPixel) input)")
        print("  Bytes/fetch (no cache):     \(String(format: "%.1f", theoreticalBytesPerPixel / Double(fetchesPerPixel))) avg  (32B vector bilinear, 16B input bilinear)")
        print("  Theoretical bytes/pixel:    \(Int(theoreticalBytesPerPixel)) B  (no-cache upper bound)")
        print("  Minimum bytes/pixel:        \(Int(minimumBytesPerPixel)) B  (perfect cache: each texel read once)")
        print("")

        // --- 1. Resolution scaling ---

        print("--- Resolution Scaling (L=\(Int(defaultL))) ---")
        print("")

        struct ResolutionResult {
            let label: String
            let width: Int
            let height: Int
            let fieldType: FieldType
            let gpuMs: Double
            let pixels: Int
            let fetches: Int
        }

        let resolutions: [(label: String, w: Int, h: Int)] = [
            ("1080p", 1920, 1080),
            ("2K",    2048, 2048),
            ("4K",    3840, 2160),
        ]

        var resResults: [ResolutionResult] = []

        for (label, w, h) in resolutions {
            for fieldType in [FieldType.uniform, FieldType.vortex] {
                let fieldLabel = fieldType == .uniform ? "uniform" : "vortex"
                let scenario = BenchmarkScenario(
                    name: "\(label) \(fieldLabel)",
                    width: w, height: h,
                    fieldType: fieldType,
                    L: defaultL, h: defaultH,
                    warmUpCount: 5, measureCount: 20)
                let result = try autoreleasepool { try runBenchmark(scenario) }
                let ms = result.gpuTimingAvailable ? result.gpuMedian : result.wallMedian
                let pixels = w * h
                let fetches = pixels * fetchesPerPixel
                resResults.append(ResolutionResult(
                    label: "\(label) \(fieldLabel)",
                    width: w, height: h,
                    fieldType: fieldType,
                    gpuMs: ms,
                    pixels: pixels,
                    fetches: fetches))
            }
        }

        let resHeader = "\(pad("Scenario", 18)) \(lpad("Pixels", 10)) "
            + "\(lpad("GPU ms", 9)) \(lpad("Gfetch/s", 9)) "
            + "\(lpad("ns/px", 8)) \(lpad("GB/s*", 8))"
        print(resHeader)
        print(String(repeating: "\u{2500}", count: 66))

        for r in resResults {
            let gfetchPerSec = Double(r.fetches) / (r.gpuMs * 1e6)  // fetches/ns = Gfetch/s
            let nsPerPixel = r.gpuMs * 1e6 / Double(r.pixels)
            let gbps = Double(r.pixels) * theoreticalBytesPerPixel / (r.gpuMs * 1e6)  // bytes / ns = GB/s

            let line = "\(pad(r.label, 18)) \(lpad(String(format: "%.1fM", Double(r.pixels) / 1e6), 10)) "
                + "\(lpad(fmtMs(r.gpuMs), 9)) \(lpad(String(format: "%.1f", gfetchPerSec), 9)) "
                + "\(lpad(String(format: "%.1f", nsPerPixel), 8)) \(lpad(String(format: "%.0f", gbps), 8))"
            print(line)
        }
        print("  * GB/s = no-cache upper bound (assumes every bilinear fetch misses cache)")
        print("    Real device bandwidth is ~200 GB/s (M1 Pro). Values >> peak confirm heavy cache reuse.")
        print("")

        // --- 2. L-scaling ---

        print("--- L-Scaling at 4K (3840x2160) ---")
        print("")

        struct LScaleResult {
            let L: Float
            let steps: Int
            let fetchesPerPixel: Int
            let gpuMsUniform: Double
            let gpuMsVortex: Double
        }

        let lValues: [Float] = [5, 10, 20, 30, 50]
        var lResults: [LScaleResult] = []

        for L in lValues {
            let steps = Int(round(L / defaultH))
            let fetches = 1 + 2 * steps * 3

            var gpuByField: [FieldType: Double] = [:]
            for fieldType in [FieldType.uniform, FieldType.vortex] {
                let scenario = BenchmarkScenario(
                    name: "4K L=\(Int(L))",
                    width: 3840, height: 2160,
                    fieldType: fieldType,
                    L: L, h: defaultH,
                    warmUpCount: 5, measureCount: 20)
                let result = try autoreleasepool { try runBenchmark(scenario) }
                gpuByField[fieldType] = result.gpuTimingAvailable ? result.gpuMedian : result.wallMedian
            }

            lResults.append(LScaleResult(
                L: L,
                steps: steps,
                fetchesPerPixel: fetches,
                gpuMsUniform: gpuByField[.uniform] ?? 0,
                gpuMsVortex: gpuByField[.vortex] ?? 0))
        }

        let pixels4K = 3840 * 2160

        let lHeader = "\(lpad("L", 4)) \(lpad("steps", 6)) \(lpad("fetch/px", 9)) "
            + "\(lpad("GPU uni", 9)) \(lpad("GPU vtx", 9)) "
            + "\(lpad("Gf/s uni", 9)) \(lpad("Gf/s vtx", 9)) "
            + "\(lpad("vtx/uni", 8))"
        print(lHeader)
        print(String(repeating: "\u{2500}", count: 68))

        for r in lResults {
            let totalFetches = Double(pixels4K) * Double(r.fetchesPerPixel)
            let gfsUniform = totalFetches / (r.gpuMsUniform * 1e6)
            let gfsVortex = totalFetches / (r.gpuMsVortex * 1e6)
            let ratio = r.gpuMsUniform > 0 ? r.gpuMsVortex / r.gpuMsUniform : 0

            let line = "\(lpad("\(Int(r.L))", 4)) \(lpad("\(r.steps)", 6)) \(lpad("\(r.fetchesPerPixel)", 9)) "
                + "\(lpad(fmtMs(r.gpuMsUniform), 9)) \(lpad(fmtMs(r.gpuMsVortex), 9)) "
                + "\(lpad(String(format: "%.1f", gfsUniform), 9)) \(lpad(String(format: "%.1f", gfsVortex), 9)) "
                + "\(lpad(String(format: "%.2fx", ratio), 8))"
            print(line)
        }
        print("")

        // --- 3. Cache efficiency summary ---

        print("--- Cache Efficiency (vortex / uniform time ratio) ---")
        print("")

        let cacheHeader = "\(pad("Resolution", 12)) \(lpad("Uniform ms", 11)) \(lpad("Vortex ms", 11)) \(lpad("Ratio", 8))"
        print(cacheHeader)
        print(String(repeating: "\u{2500}", count: 45))

        for (label, _, _) in resolutions {
            let uniResult = resResults.first { $0.label == "\(label) uniform" }
            let vtxResult = resResults.first { $0.label == "\(label) vortex" }
            if let u = uniResult, let v = vtxResult {
                let ratio = u.gpuMs > 0 ? v.gpuMs / u.gpuMs : 0
                let line = "\(pad(label, 12)) \(lpad(fmtMs(u.gpuMs), 11)) \(lpad(fmtMs(v.gpuMs), 11)) \(lpad(String(format: "%.2fx", ratio), 8))"
                print(line)
            }
        }
        print("")

        // --- 4. Automated interpretation ---

        print("--- Interpretation ---")
        print("")

        // Check throughput stability across resolutions (uniform, best-case).
        let uniformRes = resResults.filter { $0.fieldType == .uniform }
        let gfsValues = uniformRes.map { r -> Double in
            Double(r.fetches) / (r.gpuMs * 1e6)  // Gfetch/s
        }
        let gfsMin = gfsValues.min() ?? 1
        let gfsMax = gfsValues.max() ?? 1
        let gfsSpread = gfsMin > 0 ? (gfsMax - gfsMin) / gfsMin * 100 : 0

        // Check L-scaling linearity (vortex, worst-case).
        let baseL = lResults.first { Int($0.L) == 5 }
        let maxL = lResults.first { Int($0.L) == 50 }
        let fetchRatio = (baseL != nil && maxL != nil)
            ? Double(maxL!.fetchesPerPixel) / Double(baseL!.fetchesPerPixel) : 10.0
        let timeRatio = (baseL != nil && maxL != nil && baseL!.gpuMsVortex > 0)
            ? maxL!.gpuMsVortex / baseL!.gpuMsVortex : 0

        // Check cache gap.
        let vtx4K = resResults.first { $0.label == "4K vortex" }?.gpuMs ?? 1
        let uni4K = resResults.first { $0.label == "4K uniform" }?.gpuMs ?? 1
        let cacheGap4K = uni4K > 0 ? vtx4K / uni4K : 0

        // Gfetch/s trend across L (vortex) — dropping throughput = growing cache pressure.
        let gfsL5Vtx = (baseL != nil && baseL!.gpuMsVortex > 0)
            ? Double(pixels4K) * Double(baseL!.fetchesPerPixel) / (baseL!.gpuMsVortex * 1e6) : 0
        let gfsL50Vtx = (maxL != nil && maxL!.gpuMsVortex > 0)
            ? Double(pixels4K) * Double(maxL!.fetchesPerPixel) / (maxL!.gpuMsVortex * 1e6) : 0
        let gfsDrop = gfsL5Vtx > 0 ? (gfsL5Vtx - gfsL50Vtx) / gfsL5Vtx * 100 : 0

        // Resolution scaling: is throughput stable?
        if gfsSpread < 15.0 {
            print("  Resolution scaling: Gfetch/s varies \(String(format: "%.0f%%", gfsSpread)) across 1080p→4K (uniform).")
            print("  -> Texture throughput is resolution-independent. Good.")
        } else {
            print("  Resolution scaling: Gfetch/s varies \(String(format: "%.0f%%", gfsSpread)) across 1080p→4K (uniform).")
            print("  -> Throughput changes with resolution; possible memory bus saturation at higher res.")
        }
        print("")

        // L-scaling.
        let scalingEfficiency = fetchRatio > 0 ? timeRatio / fetchRatio : 0
        print("  L-scaling (vortex): L=5→50, fetch count \(String(format: "%.1f", fetchRatio))x, GPU time \(String(format: "%.1f", timeRatio))x.")
        print("  Scaling efficiency: \(String(format: "%.0f%%", scalingEfficiency * 100)) (100% = time scales perfectly with fetch count).")
        if gfsDrop > 15 {
            print("  Texture throughput drops \(String(format: "%.0f%%", gfsDrop)) from L=5 to L=50 → cache pressure increases with streamline length.")
        } else if gfsDrop < -10 {
            print("  Texture throughput increases \(String(format: "%.0f%%", abs(gfsDrop))) — short kernels underutilize the texture unit.")
        } else {
            print("  Texture throughput is stable across L values → per-fetch cost is constant.")
        }
        print("")

        // Cache gap → bottleneck determination.
        if cacheGap4K < 1.15 {
            print("  Cache efficiency: vortex/uniform ratio = \(String(format: "%.2f", cacheGap4K))x at 4K.")
            print("  -> Minimal cache impact. Kernel is BANDWIDTH-BOUND.")
            print("  -> Streamline divergence does not significantly increase cost.")
            print("  Recommendations:")
            print("    1. Evaluate format tightening: rg16Float vector field (halves vector bandwidth)")
            print("    2. Evaluate r16Float input (halves input bandwidth, quality validation needed)")
            print("    3. Check if texture compression (ASTC) is viable for input")
        } else if cacheGap4K < 1.5 {
            print("  Cache efficiency: vortex/uniform ratio = \(String(format: "%.2f", cacheGap4K))x at 4K.")
            print("  -> Moderate cache impact. Kernel is MIXED bandwidth + texture-latency.")
            print("  Recommendations:")
            print("    1. Format tightening may help (smaller texels → more fit in cache lines)")
            print("    2. Consider spatial reordering of threadgroups to improve coherence")
        } else {
            print("  Cache efficiency: vortex/uniform ratio = \(String(format: "%.2f", cacheGap4K))x at 4K.")
            print("  -> Large cache gap. Kernel is TEXTURE-LATENCY BOUND on divergent fields.")
            print("  -> Divergent streamlines cause cache misses; each dependent fetch stalls.")
            print("  Recommendations:")
            print("    1. Format tightening helps (smaller texels → less bandwidth per cache miss)")
            print("    2. Current 32x32 threadgroups already maximize latency hiding")
            print("    3. On uniform/coherent fields, kernel runs near peak — no action needed")
            print("    4. Divergent-field cost is inherent to the algorithm (dependent texture chain)")
        }
        print("")

        // Final bandwidth number.
        let achievedGbps4KVtx = Double(pixels4K) * theoreticalBytesPerPixel / (vtx4K * 1e6)
        let achievedGbps4KUni = Double(pixels4K) * theoreticalBytesPerPixel / (uni4K * 1e6)
        let minGbps4KVtx = Double(pixels4K) * minimumBytesPerPixel / (vtx4K * 1e6)
        let minGbps4KUni = Double(pixels4K) * minimumBytesPerPixel / (uni4K * 1e6)

        print("  Achieved bandwidth at 4K (no-cache model):")
        print("    Uniform: \(String(format: "%.1f", achievedGbps4KUni)) GB/s    Vortex: \(String(format: "%.1f", achievedGbps4KVtx)) GB/s")
        print("  Achieved bandwidth at 4K (perfect-cache model):")
        print("    Uniform: \(String(format: "%.1f", minGbps4KUni)) GB/s    Vortex: \(String(format: "%.1f", minGbps4KVtx)) GB/s")
        print("  (Device peak bandwidth is typically 100-400 GB/s for Apple Silicon)")
        print("")
    }
}
