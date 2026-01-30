/// Parameters for the LIC compute kernel.
///
/// Memory layout must match the MSL `LicParams` in `Shaders/LIC.metal` exactly:
/// 11 fields × 4 bytes = 44 bytes, no padding, 4-byte alignment.
public struct LicParams {
    /// Integration step size in pixels (default 1.0).
    public var h: Float
    /// Zero-vector guard threshold (1e-12). Vectors with squared length below this are treated as zero.
    public var eps2: Float
    /// Number of integration steps per direction: `round(L / h)`.
    public var steps: UInt32
    /// Kernel center index: `kernel_len / 2`.
    public var kmid: UInt32
    /// Kernel array length: `2 * steps + 1` (always odd).
    public var kernelLen: UInt32
    /// Sum of all kernel weights.
    public var fullSum: Float
    /// Weight at the kernel center: `kernel[kmid]`.
    public var centerWeight: Float
    /// Mask edge gain strength (0 = disabled).
    public var edgeGainStrength: Float
    /// Mask edge gain exponent.
    public var edgeGainPower: Float
    /// Domain edge gain strength (0 = disabled).
    public var domainEdgeGainStrength: Float
    /// Domain edge gain exponent.
    public var domainEdgeGainPower: Float

    public init(
        h: Float,
        eps2: Float = 1e-12,
        steps: UInt32,
        kmid: UInt32,
        kernelLen: UInt32,
        fullSum: Float,
        centerWeight: Float,
        edgeGainStrength: Float = 0,
        edgeGainPower: Float = 2,
        domainEdgeGainStrength: Float = 0,
        domainEdgeGainPower: Float = 2
    ) {
        self.h = h
        self.eps2 = eps2
        self.steps = steps
        self.kmid = kmid
        self.kernelLen = kernelLen
        self.fullSum = fullSum
        self.centerWeight = centerWeight
        self.edgeGainStrength = edgeGainStrength
        self.edgeGainPower = edgeGainPower
        self.domainEdgeGainStrength = domainEdgeGainStrength
        self.domainEdgeGainPower = domainEdgeGainPower
    }
}
