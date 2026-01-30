import Metal

/// MetalLIC — Real-time Line Integral Convolution engine for Apple Silicon.
public enum MetalLIC {

    /// Returns the default Metal device, or nil if Metal is unavailable.
    public static var device: MTLDevice? {
        MTLCreateSystemDefaultDevice()
    }
}
