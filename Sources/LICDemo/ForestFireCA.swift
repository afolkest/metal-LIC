import MetalKit

// Must match ForestFire.metal layout exactly
struct ForestFireParams {
    var width: UInt32
    var height: UInt32
    var frameNumber: UInt32
    var igniteProb: Float
    var spreadProb: Float
    var fuelRegen: Float
    var fuelConsume: Float
    var heatDecay: Float
    var fuelToIgnite: Float
    var fuelToSpread: Float
    var fuelToSustain: Float
    var windStrength: Float
}

final class ForestFireCA: CellularAutomaton {

    let name = "Forest Fire"

    // MARK: - Textures

    private var stateA: MTLTexture
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

    private static let defaultParams: [(String, String, Float, Float, Float)] = [
        ("igniteProb",   "Ignite Prob",    0.0,  0.01,  0.00115),
        ("spreadProb",   "Spread Prob",    0.0,  0.2,   0.02),
        ("fuelRegen",    "Fuel Regen",     0.0,  0.05,  0.008),
        ("fuelConsume",  "Fuel Consume",   0.0,  0.2,   0.0617),
        ("heatDecay",    "Heat Decay",     0.0,  0.2,   0.0677),
        ("fuelToIgnite", "Fuel to Ignite", 0.0,  1.0,   0.3),
        ("fuelToSpread", "Fuel to Spread", 0.0,  1.0,   0.2),
        ("fuelToSustain","Fuel to Sustain",0.0,  1.0,   0.1),
        ("windStrength", "Wind Strength",  0.0,  10.0,  2.49),
    ]

    let parameters: [CAParameter]
    private var values: [String: Float]

    // MARK: - Init

    init(device: MTLDevice, commandQueue: MTLCommandQueue, library: MTLLibrary,
         resolution: Int) throws {
        self.device = device
        self.commandQueue = commandQueue

        // Pipelines
        guard let stepFunc = library.makeFunction(name: "forestFireKernel") else {
            throw CAError.functionNotFound("forestFireKernel")
        }
        self.stepPipeline = try device.makeComputePipelineState(function: stepFunc)

        guard let initFunc = library.makeFunction(name: "forestFireInitKernel") else {
            throw CAError.functionNotFound("forestFireInitKernel")
        }
        self.initPipeline = try device.makeComputePipelineState(function: initFunc)

        // Textures
        let w = resolution
        let h = resolution

        self.stateA = ForestFireCA.makeTexture(device: device, format: .rg32Float,
                                                width: w, height: h, label: "CA State A")
        self.stateB = ForestFireCA.makeTexture(device: device, format: .rg32Float,
                                                width: w, height: h, label: "CA State B")
        self.heatTexture = ForestFireCA.makeTexture(device: device, format: .r32Float,
                                                     width: w, height: h, label: "CA Heat")

        // Parameters
        self.parameters = ForestFireCA.defaultParams.map { (id, label, min, max, def) in
            CAParameter(id: id, label: label, min: min, max: max, defaultValue: def)
        }
        self.values = Dictionary(uniqueKeysWithValues:
            ForestFireCA.defaultParams.map { ($0.0, $0.4) }
        )

        // Initialize state
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
        guard let vectorTex = vectorFieldTexture else { return }

        let src = pingPong ? stateB : stateA
        let dst = pingPong ? stateA : stateB

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(stepPipeline)
        enc.setTexture(src, index: 0)
        enc.setTexture(dst, index: 1)
        enc.setTexture(heatTexture, index: 2)
        enc.setTexture(vectorTex, index: 3)

        let w = stateA.width
        let h = stateA.height
        var params = ForestFireParams(
            width: UInt32(w), height: UInt32(h), frameNumber: frame,
            igniteProb:   values["igniteProb"]!,
            spreadProb:   values["spreadProb"]!,
            fuelRegen:    values["fuelRegen"]!,
            fuelConsume:  values["fuelConsume"]!,
            heatDecay:    values["heatDecay"]!,
            fuelToIgnite: values["fuelToIgnite"]!,
            fuelToSpread: values["fuelToSpread"]!,
            fuelToSustain:values["fuelToSustain"]!,
            windStrength: values["windStrength"]!
        )
        enc.setBytes(&params, length: MemoryLayout<ForestFireParams>.stride, index: 0)

        let gridSize = MTLSize(width: w, height: h, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()

        frame += 1
        pingPong.toggle()
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
