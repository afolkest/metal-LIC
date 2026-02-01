import MetalKit
import MetalLIC

// Must match Display.metal layout
struct DisplayParams {
    var fullSum: Float
    var exposure: Float
    var contrast: Float
    var brightness: Float
    var gamma: Float
}

// Must match NoiseBlend.metal layout
struct NoiseBlendParams {
    var fireWeight: Float
}

final class Renderer: NSObject, MTKViewDelegate {

    // MARK: - Constants

    private var resolution: Int = 1024
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

    // MARK: - CA (protocol-based)

    private var ca: CellularAutomaton

    // MARK: - Noise blend

    private let noiseBlendPipeline: MTLComputePipelineState
    private var noiseTex: MTLTexture
    private var blendedTex: MTLTexture

    // MARK: - Display

    private let displayPipeline: MTLRenderPipelineState
    private let displaySampler: MTLSamplerState

    // MARK: - Textures

    private var vectorField: MTLTexture
    private var licOutput: MTLTexture

    // MARK: - Shader library (kept for CA recreation on resolution change)

    private let demoLibrary: MTLLibrary

    // MARK: - Vector field state

    private var currentPreset: Int = 6

    // MARK: - Recording

    private let videoRecorder = VideoRecorder()
    private var stagingBuffers: [MTLBuffer] = []
    private var stagingIndex = 0
    private var wasRecording = false
    private var recordingWidth = 0
    private var recordingHeight = 0

    // MARK: - FPS tracking

    private var lastFrameTime: CFAbsoluteTime = 0
    private var fpsFrameCount: Int = 0
    private var fpsTimeAccum: CFAbsoluteTime = 0

    // MARK: - Settings

    let settings: DemoSettings

    // MARK: - Threadgroup size

    private let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

    // MARK: - Init

    init(device: MTLDevice, view: MTKView, settings: DemoSettings) {
        self.device = device
        self.settings = settings
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

        // Build LIC kernel
        let licResult: (params: LicParams, weights: [Float])
        do {
            licResult = try LICKernel.build(L: 20.0, h: 1.0)
        } catch {
            fatalError("Failed to build LIC kernel: \(error)")
        }
        self.licParams = licResult.params
        self.licWeights = licResult.weights

        // --- Compile demo shaders ---
        do {
            self.demoLibrary = try Renderer.makeDemoLibrary(device: device)
        } catch {
            fatalError("Failed to compile demo shaders: \(error)")
        }

        // Noise blend pipeline
        do {
            guard let blendFunc = self.demoLibrary.makeFunction(name: "noiseBlendKernel") else {
                fatalError("noiseBlendKernel not found")
            }
            self.noiseBlendPipeline = try device.makeComputePipelineState(function: blendFunc)
        } catch {
            fatalError("Failed to create noise blend pipeline: \(error)")
        }

        // Display render pipeline
        do {
            guard let vertFunc = self.demoLibrary.makeFunction(name: "displayVertex"),
                  let fragFunc = self.demoLibrary.makeFunction(name: "displayFragment") else {
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

        self.vectorField = Renderer.makePrivateTexture(device: device, format: .rg32Float,
                                                        width: w, height: h, label: "Vector Field")
        self.licOutput = Renderer.makePrivateTexture(device: device, format: .r16Float,
                                                      width: w, height: h, label: "LIC Output")
        self.blendedTex = Renderer.makePrivateTexture(device: device, format: .r32Float,
                                                       width: w, height: h, label: "Blended")
        self.noiseTex = Renderer.makeStaticNoise(device: device, commandQueue: queue,
                                                   width: w, height: h)

        // --- Create CA ---
        do {
            let fireCA = try ForestFireCA(device: device, commandQueue: queue,
                                           library: self.demoLibrary, resolution: resolution)
            self.ca = fireCA
        } catch {
            fatalError("Failed to create CA: \(error)")
        }

        super.init()

        // Initialize settings from CA
        settings.populateFromCA(ca)
        settings.vectorPreset = currentPreset

        // Upload initial vector field
        uploadVectorField(preset: currentPreset)
        ca.setVectorField(vectorField)
    }

    // MARK: - Shader compilation

    private static func makeDemoLibrary(device: MTLDevice) throws -> MTLLibrary {
        guard let ffURL = Bundle.module.url(forResource: "ForestFire", withExtension: "metal",
                                             subdirectory: "Shaders"),
              let dispURL = Bundle.module.url(forResource: "Display", withExtension: "metal",
                                               subdirectory: "Shaders"),
              let blendURL = Bundle.module.url(forResource: "NoiseBlend", withExtension: "metal",
                                                subdirectory: "Shaders") else {
            fatalError("Demo shader sources not found in bundle")
        }
        let ffSource = try String(contentsOf: ffURL, encoding: .utf8)
        let dispSource = try String(contentsOf: dispURL, encoding: .utf8)
        let blendSource = try String(contentsOf: blendURL, encoding: .utf8)

        let combined = ffSource + "\n" + dispSource + "\n" + blendSource
        return try device.makeLibrary(source: combined, options: nil)
    }

    // MARK: - Texture creation

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

    private static func makeStaticNoise(device: MTLDevice, commandQueue: MTLCommandQueue,
                                          width: Int, height: Int) -> MTLTexture {
        // Generate white noise on CPU, upload to GPU
        var pixels = [Float](repeating: 0, count: width * height)
        for i in 0..<pixels.count {
            pixels[i] = Float.random(in: 0...1)
        }

        let tex = makePrivateTexture(device: device, format: .r32Float,
                                      width: width, height: height, label: "Static Noise")

        let bytesPerRow = width * MemoryLayout<Float>.stride
        let totalBytes = bytesPerRow * height

        guard let staging = device.makeBuffer(bytes: pixels, length: totalBytes,
                                               options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else {
            fatalError("Failed to upload noise texture")
        }

        blit.copy(from: staging, sourceOffset: 0,
                  sourceBytesPerRow: bytesPerRow,
                  sourceBytesPerImage: totalBytes,
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: tex, destinationSlice: 0,
                  destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return tex
    }

    // MARK: - Vector field upload

    private func uploadVectorField(preset: Int) {
        let w = resolution
        let h = resolution
        let data = VectorFieldGenerator.generate(preset: preset, width: w, height: h)

        let bytesPerRow = w * MemoryLayout<SIMD2<Float>>.stride
        let totalBytes = bytesPerRow * h

        guard let staging = device.makeBuffer(bytes: data, length: totalBytes,
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

    // MARK: - Settings sync

    private func syncSettings() {
        // Sync CA params from UI
        for param in ca.parameters {
            if let val = settings.caValues[param.id] {
                ca.setValue(val, for: param.id)
            }
        }

        // Handle CA reset
        if settings.resetRequested {
            ca.reset()
            settings.resetRequested = false
        }

        // Kernel length change
        let newL = settings.kernelLength
        if abs(newL - currentL) > 0.5 {
            adjustKernelLength(to: newL)
        }

        // Preset change
        if settings.vectorPreset != currentPreset {
            switchPreset(settings.vectorPreset)
        }

        // Resolution change
        if settings.resolution != resolution {
            rebuildForResolution(settings.resolution)
        }

        // Recording state transitions
        let wantRecord = settings.isRecording
        if !wantRecord && wasRecording {
            wasRecording = false
            videoRecorder.stopRecording { url in
                if let url = url {
                    print("Video saved to: \(url.path)")
                }
            }
            stagingBuffers = []
        } else if wantRecord && !wasRecording {
            wasRecording = true
        }
    }

    // MARK: - Frame rendering

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        // FPS tracking
        let now = CFAbsoluteTimeGetCurrent()
        if lastFrameTime > 0 {
            fpsTimeAccum += now - lastFrameTime
            fpsFrameCount += 1
            if fpsTimeAccum >= 0.5 {
                let measuredFps = Double(fpsFrameCount) / fpsTimeAccum
                DispatchQueue.main.async { [weak self] in
                    self?.settings.fps = measuredFps
                }
                fpsFrameCount = 0
                fpsTimeAccum = 0
            }
        }
        lastFrameTime = now

        semaphore.wait()

        syncSettings()

        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            semaphore.signal()
            return
        }

        // --- Pass 1: CA step ---
        ca.encodeStep(commandBuffer: cmdBuf)

        // --- Pass 2: Noise blend ---
        do {
            guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(noiseBlendPipeline)
            enc.setTexture(ca.licInputTexture, index: 0)
            enc.setTexture(noiseTex, index: 1)
            enc.setTexture(blendedTex, index: 2)

            var blendParams = NoiseBlendParams(fireWeight: settings.fireWeight)
            enc.setBytes(&blendParams, length: MemoryLayout<NoiseBlendParams>.stride, index: 0)

            let gridSize = MTLSize(width: resolution, height: resolution, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        // --- Pass 3: LIC (1 or 2 passes) ---
        do {
            try licEncoder.encodeMultiPass(
                commandBuffer: cmdBuf,
                params: licParams,
                kernelWeights: licWeights,
                inputTexture: blendedTex,
                vectorField: vectorField,
                outputTexture: licOutput,
                iterations: settings.licPasses
            )
        } catch {
            print("LIC encode error: \(error)")
            semaphore.signal()
            return
        }

        // --- Pass 4: Display ---
        guard let renderEnc = cmdBuf.makeRenderCommandEncoder(descriptor: passDesc) else {
            semaphore.signal()
            return
        }
        renderEnc.setRenderPipelineState(displayPipeline)
        renderEnc.setFragmentTexture(licOutput, index: 0)
        renderEnc.setFragmentSamplerState(displaySampler, index: 0)
        var displayParams = DisplayParams(
            fullSum: licParams.fullSum,
            exposure: settings.exposure,
            contrast: settings.contrast,
            brightness: settings.brightness,
            gamma: settings.gamma
        )
        renderEnc.setFragmentBytes(&displayParams,
                                    length: MemoryLayout<DisplayParams>.stride,
                                    index: 0)
        renderEnc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renderEnc.endEncoding()

        // --- Recording: blit drawable to staging buffer ---
        var capturedStaging: MTLBuffer? = nil
        var capturedWidth = 0
        var capturedHeight = 0

        if wasRecording {
            let drawableTex = drawable.texture
            let w = drawableTex.width
            let h = drawableTex.height

            // Lazily start the writer on the first recorded frame
            if !videoRecorder.isRecording {
                do {
                    let url = try videoRecorder.startRecording(width: w, height: h)
                    recordingWidth = w
                    recordingHeight = h
                    stagingBuffers = (0..<maxInFlight).map {_ in
                        device.makeBuffer(length: w * h * 4, options: .storageModeShared)!
                    }
                    stagingIndex = 0
                    print("Recording to: \(url.path)")
                } catch {
                    print("Failed to start recording: \(error)")
                    wasRecording = false
                    DispatchQueue.main.async { [weak self] in
                        self?.settings.isRecording = false
                    }
                }
            }

            if videoRecorder.isRecording && w == recordingWidth && h == recordingHeight {
                let staging = stagingBuffers[stagingIndex % maxInFlight]
                stagingIndex += 1

                if let blit = cmdBuf.makeBlitCommandEncoder() {
                    blit.copy(from: drawableTex,
                              sourceSlice: 0, sourceLevel: 0,
                              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                              sourceSize: MTLSize(width: w, height: h, depth: 1),
                              to: staging,
                              destinationOffset: 0,
                              destinationBytesPerRow: w * 4,
                              destinationBytesPerImage: w * h * 4)
                    blit.endEncoding()

                    capturedStaging = staging
                    capturedWidth = w
                    capturedHeight = h
                }
            }
        }

        let sem = semaphore
        let recorder = videoRecorder
        cmdBuf.addCompletedHandler { _ in
            if let staging = capturedStaging {
                recorder.appendFrame(from: staging, width: capturedWidth, height: capturedHeight)
            }
            sem.signal()
        }

        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - Keyboard handling

    func handleKeyDown(_ event: NSEvent) {
        guard let chars = event.charactersIgnoringModifiers else { return }

        switch chars {
        case " ":
            ca.isPaused.toggle()
        case "1": settings.vectorPreset = 1
        case "2": settings.vectorPreset = 2
        case "3": settings.vectorPreset = 3
        case "4": settings.vectorPreset = 4
        case "5": settings.vectorPreset = 5
        case "6": settings.vectorPreset = 6
        case "r", "R":
            settings.resetRequested = true
        case "+", "=":
            settings.kernelLength = min(120, settings.kernelLength + 2)
        case "-", "_":
            settings.kernelLength = max(2, settings.kernelLength - 2)
        default:
            break
        }
    }

    private func switchPreset(_ preset: Int) {
        guard preset != currentPreset, preset >= 1, preset <= 6 else { return }
        currentPreset = preset
        uploadVectorField(preset: preset)
        ca.setVectorField(vectorField)
    }

    private func rebuildForResolution(_ newRes: Int) {
        // Called from draw() which already holds one semaphore slot,
        // so drain only the remaining in-flight work.
        for _ in 0..<(maxInFlight - 1) {
            semaphore.wait()
        }

        resolution = newRes
        let w = newRes
        let h = newRes

        vectorField = Renderer.makePrivateTexture(device: device, format: .rg32Float,
                                                    width: w, height: h, label: "Vector Field")
        licOutput = Renderer.makePrivateTexture(device: device, format: .r16Float,
                                                  width: w, height: h, label: "LIC Output")
        blendedTex = Renderer.makePrivateTexture(device: device, format: .r32Float,
                                                   width: w, height: h, label: "Blended")
        noiseTex = Renderer.makeStaticNoise(device: device, commandQueue: commandQueue,
                                             width: w, height: h)

        do {
            let fireCA = try ForestFireCA(device: device, commandQueue: commandQueue,
                                           library: demoLibrary, resolution: newRes)
            ca = fireCA
            settings.populateFromCA(ca)
        } catch {
            print("Failed to recreate CA: \(error)")
        }

        uploadVectorField(preset: currentPreset)
        ca.setVectorField(vectorField)

        // Restore the slots we drained (not the one draw() holds —
        // draw() will return without submitting work this frame,
        // and its completed handler will signal its own slot).
        for _ in 0..<(maxInFlight - 1) {
            semaphore.signal()
        }
    }

    private func adjustKernelLength(to newL: Float) {
        let clamped = max(2.0, min(120.0, newL))
        currentL = clamped
        do {
            let result = try LICKernel.build(L: clamped, h: 1.0)
            licParams = result.params
            licWeights = result.weights
        } catch {
            print("Failed to rebuild kernel: \(error)")
        }
    }
}
