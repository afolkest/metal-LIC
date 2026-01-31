import Metal
import Foundation

/// Manages pipelined LIC dispatch with multiple in-flight command buffers.
///
/// Uses a semaphore to limit concurrent GPU work, allowing the CPU to encode
/// frame N+1 while the GPU executes frame N. Per-frame ping-pong textures
/// prevent aliasing between in-flight multi-pass dispatches.
///
/// - Note: The caller must ensure that `inputTexture`, `vectorField`, and
///   `maskTexture` are not modified while any in-flight frame references them.
///   The `outputTexture` for each dispatch must either be unique per frame or
///   the caller must use the completion callback to know when it is safe to read.
public final class LICDispatcher {

    /// Maximum number of command buffers in flight simultaneously.
    public let maxInFlight: Int

    /// The underlying encoder (shared across frames).
    public let encoder: LICEncoder

    /// The command queue used for submission.
    public let commandQueue: MTLCommandQueue

    private let semaphore: DispatchSemaphore
    private var frameSlots: [FrameSlot]
    private var frameIndex: Int = 0

    /// Creates a pipelined dispatcher.
    ///
    /// - Parameters:
    ///   - encoder: A configured `LICEncoder` with pre-built pipelines.
    ///   - commandQueue: The command queue for buffer submission.
    ///   - maxInFlight: Maximum concurrent in-flight command buffers (default 3).
    public init(encoder: LICEncoder, commandQueue: MTLCommandQueue, maxInFlight: Int = 3) {
        precondition(maxInFlight >= 1, "maxInFlight must be >= 1")
        self.encoder = encoder
        self.commandQueue = commandQueue
        self.maxInFlight = maxInFlight
        self.semaphore = DispatchSemaphore(value: maxInFlight)
        self.frameSlots = (0..<maxInFlight).map { _ in FrameSlot() }
    }

    /// Dispatches a single LIC frame.
    ///
    /// Blocks the calling thread if `maxInFlight` frames are already in flight.
    /// The completion handler is called on an unspecified thread when the GPU
    /// finishes this frame; inspect `commandBuffer.status` and timing properties
    /// (`gpuStartTime`, `gpuEndTime`) in the callback.
    ///
    /// - Parameters:
    ///   - params: LIC parameters.
    ///   - kernelWeights: Hann kernel weights.
    ///   - inputTexture: `r32Float` input (must remain valid until completion).
    ///   - vectorField: `rg32Float` vector field.
    ///   - outputTexture: `r16Float` output (written by GPU).
    ///   - maskTexture: Optional `r8Uint` mask.
    ///   - config: Pipeline configuration.
    ///   - iterations: Number of convolution passes (>= 1).
    ///   - threadgroupSize: Optional threadgroup override.
    ///   - completion: Called when the GPU finishes. Receives the command buffer.
    public func dispatch(
        params: LicParams,
        kernelWeights: [Float],
        inputTexture: MTLTexture,
        vectorField: MTLTexture,
        outputTexture: MTLTexture,
        maskTexture: MTLTexture? = nil,
        config: LICPipelineConfig = LICPipelineConfig(),
        iterations: Int = 1,
        threadgroupSize: MTLSize? = nil,
        completion: ((MTLCommandBuffer) -> Void)? = nil
    ) throws {
        semaphore.wait()

        let slot = frameSlots[frameIndex % maxInFlight]
        frameIndex += 1

        guard let cb = commandQueue.makeCommandBuffer() else {
            semaphore.signal()
            throw LICError.encoderCreationFailed
        }

        do {
            if iterations > 1 {
                let pingPong = try slot.ensurePingPongTextures(
                    encoder: encoder,
                    width: outputTexture.width,
                    height: outputTexture.height)
                try encoder.encodeMultiPass(
                    commandBuffer: cb,
                    params: params, kernelWeights: kernelWeights,
                    inputTexture: inputTexture, vectorField: vectorField,
                    outputTexture: outputTexture, maskTexture: maskTexture,
                    config: config, iterations: iterations,
                    threadgroupSize: threadgroupSize,
                    pingPongTextures: pingPong)
            } else {
                try encoder.encode(
                    commandBuffer: cb,
                    params: params, kernelWeights: kernelWeights,
                    inputTexture: inputTexture, vectorField: vectorField,
                    outputTexture: outputTexture, maskTexture: maskTexture,
                    config: config, threadgroupSize: threadgroupSize)
            }
        } catch {
            semaphore.signal()
            throw error
        }

        cb.addCompletedHandler { [semaphore] buffer in
            semaphore.signal()
            completion?(buffer)
        }

        cb.commit()
    }

    /// Blocks until all in-flight frames have completed.
    ///
    /// Call before reading the final output texture or shutting down.
    public func waitForAllFrames() {
        // Acquire all permits — guarantees all prior frames have signaled.
        for _ in 0..<maxInFlight {
            semaphore.wait()
        }
        // Release them so the dispatcher is reusable.
        for _ in 0..<maxInFlight {
            semaphore.signal()
        }
    }

    /// Performs a warm-up dispatch to prime GPU caches and driver state.
    ///
    /// Dispatches a single frame synchronously and waits for completion.
    /// Call once after initialization, before the real-time loop begins.
    public func warmUp(
        params: LicParams,
        kernelWeights: [Float],
        inputTexture: MTLTexture,
        vectorField: MTLTexture,
        outputTexture: MTLTexture,
        maskTexture: MTLTexture? = nil,
        config: LICPipelineConfig = LICPipelineConfig(),
        iterations: Int = 1
    ) throws {
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw LICError.encoderCreationFailed
        }

        if iterations > 1 {
            let slot = frameSlots[0]
            let pingPong = try slot.ensurePingPongTextures(
                encoder: encoder,
                width: outputTexture.width,
                height: outputTexture.height)
            try encoder.encodeMultiPass(
                commandBuffer: cb,
                params: params, kernelWeights: kernelWeights,
                inputTexture: inputTexture, vectorField: vectorField,
                outputTexture: outputTexture, maskTexture: maskTexture,
                config: config, iterations: iterations,
                pingPongTextures: pingPong)
        } else {
            try encoder.encode(
                commandBuffer: cb,
                params: params, kernelWeights: kernelWeights,
                inputTexture: inputTexture, vectorField: vectorField,
                outputTexture: outputTexture, maskTexture: maskTexture,
                config: config)
        }

        cb.commit()
        cb.waitUntilCompleted()
    }
}

// MARK: - Per-frame resource slot

/// Holds per-frame ping-pong textures for multi-pass dispatch.
/// One slot per in-flight frame prevents aliasing between concurrent GPU work.
private final class FrameSlot {
    var pingTexture: MTLTexture?
    var pongTexture: MTLTexture?
    var width: Int = 0
    var height: Int = 0

    /// Returns the ping-pong texture pair, allocating or resizing as needed.
    func ensurePingPongTextures(
        encoder: LICEncoder,
        width: Int,
        height: Int
    ) throws -> (MTLTexture, MTLTexture) {
        if self.width == width && self.height == height,
           let ping = pingTexture, let pong = pongTexture {
            return (ping, pong)
        }
        let ping = try encoder.makeOutputTexture(width: width, height: height)
        ping.label = "LIC dispatcher ping"
        let pong = try encoder.makeOutputTexture(width: width, height: height)
        pong.label = "LIC dispatcher pong"
        self.pingTexture = ping
        self.pongTexture = pong
        self.width = width
        self.height = height
        return (ping, pong)
    }
}
