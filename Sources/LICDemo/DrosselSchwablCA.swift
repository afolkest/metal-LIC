import MetalKit

// Must match DrosselSchwabl.metal layout exactly
struct DrosselSchwablParams {
    var width: UInt32
    var height: UInt32
    var frameNumber: UInt32
    var growthProb: Float
    var lightningProb: Float
    var heatDecay: Float
}

final class DrosselSchwablCA: CellularAutomaton {

    let name = "Drossel-Schwabl"

    // MARK: - Textures

    private var stateA: MTLTexture   // rg32Float: R = state (0/1/2), G = heat
    private var stateB: MTLTexture
    private(set) var heatTexture: MTLTexture

    var licInputTexture: MTLTexture { heatTexture }

    // MARK: - Pipeline

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let stepPipeline: MTLComputePipelineState
    private let initPipeline: MTLComputePipelineState
    private let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

    // MARK: - State

    private var frame: UInt32 = 0
    private var pingPong = false
    private var vectorFieldTexture: MTLTexture?

    var isPaused = false

    // MARK: - Parameters

    // Growth p and ratio f/p exposed on log10 scale for usable slider ranges.
    // f/p controls fire character (small = large dramatic burns, large = frequent small fires).
    // p controls tempo (how fast trees regrow).
    // Actual probabilities computed in encodeStep: p = 10^growthExp, f = 10^fpRatioExp * p.
    private static let defaultParams: [(String, String, Float, Float, Float)] = [
        ("fpRatioExp",   "f/p Ratio (log₁₀)",  -6.0, -1.0, -3.73),  // f/p ≈ 1.86e-4
        ("growthExp",    "Growth p (log₁₀)",    -4.0, -0.5, -2.75),  // p ≈ 1.78e-3
        ("heatDecay",    "Heat Decay",           0.01, 1.0,  1.0),
        ("substeps",     "Steps / Frame",        1.0,  20.0, 1.0),
    ]

    let parameters: [CAParameter]
    private var values: [String: Float]

    // MARK: - Init

    init(device: MTLDevice, commandQueue: MTLCommandQueue, library: MTLLibrary,
         resolution: Int) throws {
        self.device = device
        self.commandQueue = commandQueue

        guard let stepFunc = library.makeFunction(name: "drosselSchwablKernel") else {
            throw CAError.functionNotFound("drosselSchwablKernel")
        }
        self.stepPipeline = try device.makeComputePipelineState(function: stepFunc)

        guard let initFunc = library.makeFunction(name: "drosselSchwablInitKernel") else {
            throw CAError.functionNotFound("drosselSchwablInitKernel")
        }
        self.initPipeline = try device.makeComputePipelineState(function: initFunc)

        let w = resolution
        let h = resolution

        self.stateA = DrosselSchwablCA.makeTexture(device: device, format: .rg32Float,
                                                    width: w, height: h, label: "DS State A")
        self.stateB = DrosselSchwablCA.makeTexture(device: device, format: .rg32Float,
                                                    width: w, height: h, label: "DS State B")
        self.heatTexture = DrosselSchwablCA.makeTexture(device: device, format: .r32Float,
                                                         width: w, height: h, label: "DS Heat")

        self.parameters = DrosselSchwablCA.defaultParams.map { (id, label, min, max, def) in
            CAParameter(id: id, label: label, min: min, max: max, defaultValue: def)
        }
        self.values = Dictionary(uniqueKeysWithValues:
            DrosselSchwablCA.defaultParams.map { ($0.0, $0.4) }
        )

        runInit()
    }

    // MARK: - Protocol

    func getValue(for id: String) -> Float {
        values[id] ?? 0
    }

    func setValue(_ value: Float, for id: String) {
        values[id] = value
    }

    func setVectorField(_ texture: MTLTexture) {
        vectorFieldTexture = texture
    }

    func encodeStep(commandBuffer: MTLCommandBuffer) {
        guard !isPaused else { return }

        let substeps = max(1, Int(values["substeps"]! + 0.5))
        let growthProb = pow(10.0, values["growthExp"]!)
        let fpRatio = pow(10.0, values["fpRatioExp"]!)
        let lightningProb = fpRatio * growthProb
        let heatDecay = values["heatDecay"]!

        let w = stateA.width
        let h = stateA.height
        let gridSize = MTLSize(width: w, height: h, depth: 1)

        for _ in 0..<substeps {
            let src = pingPong ? stateB : stateA
            let dst = pingPong ? stateA : stateB

            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(stepPipeline)
            enc.setTexture(src, index: 0)
            enc.setTexture(dst, index: 1)
            enc.setTexture(heatTexture, index: 2)

            var params = DrosselSchwablParams(
                width: UInt32(w), height: UInt32(h), frameNumber: frame,
                growthProb: growthProb,
                lightningProb: lightningProb,
                heatDecay: heatDecay
            )
            enc.setBytes(&params, length: MemoryLayout<DrosselSchwablParams>.stride, index: 0)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()

            frame += 1
            pingPong.toggle()
        }
    }

    func reset() {
        runInit()
    }

    // MARK: - Private

    private func runInit() {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(initPipeline)
        enc.setTexture(stateA, index: 0)
        enc.setTexture(heatTexture, index: 1)
        var seed = UInt32.random(in: 0..<UInt32.max)
        enc.setBytes(&seed, length: MemoryLayout<UInt32>.stride, index: 0)

        let gridSize = MTLSize(width: stateA.width, height: stateA.height, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        frame = 0
        pingPong = false
    }

    private static func makeTexture(device: MTLDevice, format: MTLPixelFormat,
                                     width: Int, height: Int, label: String) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else {
            fatalError("Failed to create texture: \(label)")
        }
        tex.label = label
        return tex
    }

    enum CAError: Error {
        case functionNotFound(String)
    }
}
