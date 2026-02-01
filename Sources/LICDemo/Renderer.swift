import MetalKit
import MetalLIC

// Must match ForestFire.metal layout
struct ForestFireParams {
    var width: UInt32
    var height: UInt32
    var frameNumber: UInt32
    var growthRate: Float
    var ignitionProb: Float
    var burnRate: Float
    var spreadRate: Float
    var diffusion: Float
}

// Must match Display.metal layout
struct DisplayParams {
    var fullSum: Float
    var gamma: Float
}

final class Renderer: NSObject, MTKViewDelegate {

    // MARK: - Constants

    private let resolution = 1024
    private let maxInFlight = 3

    // MARK: - Metal core

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let semaphore: DispatchSemaphore

    // MARK: - LIC engine

    private let licEncoder: LICEncoder
    private var licParams: LicParams
    private var licWeights: [Float]
    private var currentL: Float = 20.0

    // MARK: - Forest fire CA

    private let caComputePipeline: MTLComputePipelineState
    private let caInitPipeline: MTLComputePipelineState
    private var caStateA: MTLTexture
    private var caStateB: MTLTexture
    private var caFrame: UInt32 = 0
    private var caPingPong: Bool = false // false = A→B, true = B→A
    private var caPaused: Bool = false

    // MARK: - Display

    private let displayPipeline: MTLRenderPipelineState
    private let displaySampler: MTLSamplerState

    // MARK: - Textures

    private let vectorField: MTLTexture
    private let licOutput: MTLTexture

    // MARK: - Vector field state

    private var currentPreset: Int = 1

    // MARK: - CA default params

    private var caGrowthRate: Float = 0.003
    private var caIgnitionProb: Float = 0.0001
    private var caBurnRate: Float = 0.05
    private var caSpreadRate: Float = 0.15
    private var caDiffusion: Float = 0.02

    // MARK: - Threadgroup size

    private let caThreadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

    // MARK: - Init

    init(device: MTLDevice, view: MTKView) {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.commandQueue = queue
        self.semaphore = DispatchSemaphore(value: maxInFlight)

        // --- LIC encoder ---
        do {
            self.licEncoder = try LICEncoder(device: device)
        } catch {
            fatalError("Failed to create LIC encoder: \(error)")
        }

        // Build LIC kernel (L=20, h=1.0)
        let licResult: (params: LicParams, weights: [Float])
        do {
            licResult = try LICKernel.build(L: 20.0, h: 1.0)
        } catch {
            fatalError("Failed to build LIC kernel: \(error)")
        }
        self.licParams = licResult.params
        self.licWeights = licResult.weights

        // --- Compile demo shaders ---
        let demoLibrary: MTLLibrary
        do {
            demoLibrary = try Renderer.makeDemoLibrary(device: device)
        } catch {
            fatalError("Failed to compile demo shaders: \(error)")
        }

        // CA compute pipelines
        do {
            guard let caFunc = demoLibrary.makeFunction(name: "forestFireKernel") else {
                fatalError("forestFireKernel not found")
            }
            self.caComputePipeline = try device.makeComputePipelineState(function: caFunc)

            guard let caInitFunc = demoLibrary.makeFunction(name: "forestFireInitKernel") else {
                fatalError("forestFireInitKernel not found")
            }
            self.caInitPipeline = try device.makeComputePipelineState(function: caInitFunc)
        } catch {
            fatalError("Failed to create CA pipeline: \(error)")
        }

        // Display render pipeline
        do {
            guard let vertFunc = demoLibrary.makeFunction(name: "displayVertex"),
                  let fragFunc = demoLibrary.makeFunction(name: "displayFragment") else {
                fatalError("Display shader functions not found")
            }
            let pipeDesc = MTLRenderPipelineDescriptor()
            pipeDesc.vertexFunction = vertFunc
            pipeDesc.fragmentFunction = fragFunc
            pipeDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat
            self.displayPipeline = try device.makeRenderPipelineState(descriptor: pipeDesc)
        } catch {
            fatalError("Failed to create display pipeline: \(error)")
        }

        // Display sampler (normalized coords, linear filtering)
        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        samplerDesc.normalizedCoordinates = true
        guard let samp = device.makeSamplerState(descriptor: samplerDesc) else {
            fatalError("Failed to create display sampler")
        }
        self.displaySampler = samp

        // --- Create textures ---
        let w = resolution
        let h = resolution

        self.caStateA = Renderer.makePrivateTexture(device: device, format: .r32Float,
                                                     width: w, height: h, label: "CA State A")
        self.caStateB = Renderer.makePrivateTexture(device: device, format: .r32Float,
                                                     width: w, height: h, label: "CA State B")
        self.vectorField = Renderer.makePrivateTexture(device: device, format: .rg32Float,
                                                        width: w, height: h, label: "Vector Field")
        self.licOutput = Renderer.makePrivateTexture(device: device, format: .r16Float,
                                                      width: w, height: h, label: "LIC Output")

        super.init()

        // Initialize CA state on GPU
        initializeCA()

        // Upload initial vector field (preset 1 = vortex)
        uploadVectorField(preset: currentPreset)
    }

    // MARK: - Shader compilation

    private static func makeDemoLibrary(device: MTLDevice) throws -> MTLLibrary {
        // Load both shader files from bundle resources and compile together
        guard let ffURL = Bundle.module.url(forResource: "ForestFire", withExtension: "metal",
                                             subdirectory: "Shaders"),
              let dispURL = Bundle.module.url(forResource: "Display", withExtension: "metal",
                                               subdirectory: "Shaders") else {
            fatalError("Demo shader sources not found in bundle")
        }
        let ffSource = try String(contentsOf: ffURL, encoding: .utf8)
        let dispSource = try String(contentsOf: dispURL, encoding: .utf8)

        // Combine sources (each has its own includes, Metal handles dedup)
        let combined = ffSource + "\n" + dispSource
        return try device.makeLibrary(source: combined, options: nil)
    }

    // MARK: - Texture creation (private storage)

    private static func makePrivateTexture(device: MTLDevice, format: MTLPixelFormat,
                                            width: Int, height: Int,
                                            label: String) -> MTLTexture {
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

    // MARK: - CA initialization

    private func initializeCA() {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(caInitPipeline)
        enc.setTexture(caStateA, index: 0)
        var seed: UInt32 = UInt32.random(in: 0..<UInt32.max)
        enc.setBytes(&seed, length: MemoryLayout<UInt32>.stride, index: 0)

        let gridSize = MTLSize(width: resolution, height: resolution, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: caThreadgroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        caFrame = 0
        caPingPong = false
    }

    // MARK: - Vector field upload

    private func uploadVectorField(preset: Int) {
        let w = resolution
        let h = resolution
        let data = VectorFieldGenerator.generate(preset: preset, width: w, height: h)

        // Upload via staging buffer + blit
        let bytesPerRow = w * MemoryLayout<SIMD2<Float>>.stride
        let totalBytes = bytesPerRow * h

        guard let staging = device.makeBuffer(bytes: data,
                                               length: totalBytes,
                                               options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else { return }

        blit.copy(from: staging, sourceOffset: 0,
                  sourceBytesPerRow: bytesPerRow,
                  sourceBytesPerImage: totalBytes,
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: vectorField, destinationSlice: 0,
                  destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Frame rendering

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        semaphore.wait()

        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            semaphore.signal()
            return
        }

        let sem = semaphore
        cmdBuf.addCompletedHandler { _ in sem.signal() }

        // --- Pass 1: Forest fire CA ---
        if !caPaused {
            let srcTex = caPingPong ? caStateB : caStateA
            let dstTex = caPingPong ? caStateA : caStateB

            guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(caComputePipeline)
            enc.setTexture(srcTex, index: 0)
            enc.setTexture(dstTex, index: 1)

            var params = ForestFireParams(
                width: UInt32(resolution), height: UInt32(resolution),
                frameNumber: caFrame,
                growthRate: caGrowthRate, ignitionProb: caIgnitionProb,
                burnRate: caBurnRate, spreadRate: caSpreadRate,
                diffusion: caDiffusion
            )
            enc.setBytes(&params, length: MemoryLayout<ForestFireParams>.stride, index: 0)

            let gridSize = MTLSize(width: resolution, height: resolution, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: caThreadgroupSize)
            enc.endEncoding()

            caFrame += 1
            caPingPong.toggle()
        }

        // Current CA state (the one we just wrote, or last written if paused)
        let currentCA = caPingPong ? caStateB : caStateA

        // --- Pass 2: LIC ---
        do {
            try licEncoder.encode(
                commandBuffer: cmdBuf,
                params: licParams,
                kernelWeights: licWeights,
                inputTexture: currentCA,
                vectorField: vectorField,
                outputTexture: licOutput
            )
        } catch {
            print("LIC encode error: \(error)")
            semaphore.signal()
            return
        }

        // --- Pass 3: Display ---
        guard let renderEnc = cmdBuf.makeRenderCommandEncoder(descriptor: passDesc) else {
            semaphore.signal()
            return
        }
        renderEnc.setRenderPipelineState(displayPipeline)
        renderEnc.setFragmentTexture(licOutput, index: 0)
        renderEnc.setFragmentSamplerState(displaySampler, index: 0)
        var displayParams = DisplayParams(fullSum: licParams.fullSum, gamma: 2.2)
        renderEnc.setFragmentBytes(&displayParams,
                                    length: MemoryLayout<DisplayParams>.stride,
                                    index: 0)
        renderEnc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renderEnc.endEncoding()

        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - Keyboard handling

    func handleKeyDown(_ event: NSEvent) {
        guard let chars = event.charactersIgnoringModifiers else { return }

        switch chars {
        case " ":
            caPaused.toggle()
        case "1": switchPreset(1)
        case "2": switchPreset(2)
        case "3": switchPreset(3)
        case "4": switchPreset(4)
        case "5": switchPreset(5)
        case "r", "R":
            initializeCA()
        case "+", "=":
            adjustKernelLength(delta: 2)
        case "-", "_":
            adjustKernelLength(delta: -2)
        default:
            break
        }
    }

    private func switchPreset(_ preset: Int) {
        guard preset != currentPreset else { return }
        currentPreset = preset
        uploadVectorField(preset: preset)
    }

    private func adjustKernelLength(delta: Float) {
        currentL = max(2.0, min(60.0, currentL + delta))
        do {
            let result = try LICKernel.build(L: currentL, h: 1.0)
            licParams = result.params
            licWeights = result.weights
        } catch {
            print("Failed to rebuild kernel: \(error)")
        }
    }
}
