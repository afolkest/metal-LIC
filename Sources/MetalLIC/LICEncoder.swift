import Metal
import Foundation

/// Configuration for building a specialized LIC pipeline variant.
public struct LICPipelineConfig: Hashable {
    public var maskEnabled: Bool
    public var edgeGainsEnabled: Bool
    public var debugMode: UInt32

    public init(maskEnabled: Bool = false,
                edgeGainsEnabled: Bool = false,
                debugMode: UInt32 = 0) {
        self.maskEnabled = maskEnabled
        self.edgeGainsEnabled = edgeGainsEnabled
        self.debugMode = debugMode
    }
}

/// Encodes LIC compute dispatches to a Metal command buffer.
///
/// Owns the compute pipeline(s), samplers, and parameter buffer.
/// Textures are caller-owned and passed to `encode(...)`.
public final class LICEncoder {

    public let device: MTLDevice

    // Cached pipeline states keyed by configuration.
    private var pipelines: [LICPipelineConfig: MTLComputePipelineState] = [:]
    private let library: MTLLibrary

    // Samplers (Section 4.2)
    private let inputSampler: MTLSamplerState   // linear clamp, pixel coords
    private let vectorSampler: MTLSamplerState   // linear clamp, pixel coords

    // 1x1 zero mask for when masking is disabled (keeps signature uniform).
    private let dummyMask: MTLTexture

    public init(device: MTLDevice) throws {
        self.device = device
        self.library = try ShaderLibrary.makeLibrary(device: device)

        // --- Samplers (Section 10: linear clamp, pixel coordinates) ---
        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        samplerDesc.normalizedCoordinates = false  // pixel coordinates (Section 5.1)

        guard let inputSamp = device.makeSamplerState(descriptor: samplerDesc) else {
            throw LICError.samplerCreationFailed
        }
        self.inputSampler = inputSamp

        guard let vecSamp = device.makeSamplerState(descriptor: samplerDesc) else {
            throw LICError.samplerCreationFailed
        }
        self.vectorSampler = vecSamp

        // --- 1x1 zero mask (Section 4.2: bind even when masking disabled) ---
        let maskDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Uint,
            width: 1, height: 1,
            mipmapped: false)
        maskDesc.usage = .shaderRead
        maskDesc.storageMode = .managed
        guard let mask = device.makeTexture(descriptor: maskDesc) else {
            throw LICError.textureCreationFailed
        }
        var zero: UInt8 = 0
        mask.replace(region: MTLRegionMake2D(0, 0, 1, 1),
                     mipmapLevel: 0,
                     withBytes: &zero,
                     bytesPerRow: 1)
        self.dummyMask = mask

        // --- Build default pipeline (no mask, no edge gains, no debug) ---
        try buildPipeline(for: LICPipelineConfig())
    }

    // MARK: - Pipeline management

    /// Ensures a pipeline variant exists for the given configuration.
    @discardableResult
    public func buildPipeline(for config: LICPipelineConfig) throws -> MTLComputePipelineState {
        if let existing = pipelines[config] { return existing }

        let constants = MTLFunctionConstantValues()
        var maskEnabled = config.maskEnabled
        var edgeGainsEnabled = config.edgeGainsEnabled
        var debugMode = config.debugMode
        constants.setConstantValue(&maskEnabled, type: .bool, index: 0)
        constants.setConstantValue(&edgeGainsEnabled, type: .bool, index: 1)
        constants.setConstantValue(&debugMode, type: .uint, index: 2)

        let function = try library.makeFunction(name: "licKernel",
                                                constantValues: constants)
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[config] = pipeline
        return pipeline
    }

    // MARK: - Encode

    /// Encodes a single LIC pass into a command buffer.
    ///
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode into.
    ///   - params: LIC parameters (from `LICKernel.build`).
    ///   - kernelWeights: Hann kernel weights array.
    ///   - inputTexture: `r32Float` input texture.
    ///   - vectorField: `rg32Float` vector field texture.
    ///   - outputTexture: `r16Float` output texture.
    ///   - maskTexture: Optional `r8Uint` mask. Pass nil when masking is disabled.
    ///   - config: Pipeline configuration (determines which specialized variant to use).
    public func encode(
        commandBuffer: MTLCommandBuffer,
        params: LicParams,
        kernelWeights: [Float],
        inputTexture: MTLTexture,
        vectorField: MTLTexture,
        outputTexture: MTLTexture,
        maskTexture: MTLTexture? = nil,
        config: LICPipelineConfig = LICPipelineConfig()
    ) throws {
        guard let pipeline = pipelines[config] else {
            throw LICError.pipelineNotBuilt(config)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LICError.encoderCreationFailed
        }

        encoder.setComputePipelineState(pipeline)

        // --- Textures (Section 4.2) ---
        encoder.setTexture(inputTexture, index: 0)
        encoder.setTexture(vectorField, index: 1)
        encoder.setTexture(maskTexture ?? dummyMask, index: 2)
        encoder.setTexture(outputTexture, index: 3)

        // --- Samplers (Section 4.2) ---
        encoder.setSamplerState(inputSampler, index: 0)
        encoder.setSamplerState(vectorSampler, index: 1)

        // --- Buffers (Section 4.2) ---
        var mutableParams = params
        encoder.setBytes(&mutableParams,
                         length: MemoryLayout<LicParams>.stride,
                         index: 0)
        guard !kernelWeights.isEmpty else {
            throw LICError.emptyKernelWeights
        }
        kernelWeights.withUnsafeBufferPointer { ptr in
            encoder.setBytes(ptr.baseAddress!,
                             length: ptr.count * MemoryLayout<Float>.stride,
                             index: 1)
        }

        // --- Dispatch ---
        let w = outputTexture.width
        let h = outputTexture.height
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let gridSize = MTLSize(width: w, height: h, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
    }
}

// MARK: - Texture helpers

extension LICEncoder {

    /// Creates an `r32Float` texture (for input).
    public func makeInputTexture(width: Int, height: Int) throws -> MTLTexture {
        try makeTexture(format: .r32Float, width: width, height: height,
                        usage: [.shaderRead, .shaderWrite])
    }

    /// Creates an `rg32Float` texture (for vector field).
    public func makeVectorFieldTexture(width: Int, height: Int) throws -> MTLTexture {
        try makeTexture(format: .rg32Float, width: width, height: height,
                        usage: .shaderRead)
    }

    /// Creates an `r16Float` texture (for output).
    public func makeOutputTexture(width: Int, height: Int) throws -> MTLTexture {
        try makeTexture(format: .r16Float, width: width, height: height,
                        usage: [.shaderRead, .shaderWrite])
    }

    /// Creates an `r8Uint` texture (for mask).
    public func makeMaskTexture(width: Int, height: Int) throws -> MTLTexture {
        try makeTexture(format: .r8Uint, width: width, height: height,
                        usage: .shaderRead)
    }

    private func makeTexture(format: MTLPixelFormat,
                             width: Int, height: Int,
                             usage: MTLTextureUsage) throws -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format,
            width: width, height: height,
            mipmapped: false)
        desc.usage = usage
        desc.storageMode = .managed
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw LICError.textureCreationFailed
        }
        return tex
    }
}

// MARK: - Errors

public enum LICError: Error {
    case samplerCreationFailed
    case textureCreationFailed
    case encoderCreationFailed
    case pipelineNotBuilt(LICPipelineConfig)
    case emptyKernelWeights
}
