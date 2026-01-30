import Metal
import Foundation

/// Loads the Metal shader library from bundled source.
enum ShaderLibrary {

    enum Error: Swift.Error {
        case shaderSourceNotFound
        case deviceUnavailable
    }

    /// Compiles and returns the Metal library from the bundled LIC.metal source.
    ///
    /// SPM bundles the .metal file as a resource (not compiled at build time).
    /// We compile from source at runtime, which works for both `swift build` and Xcode.
    static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let url = Bundle.module.url(
            forResource: "LIC",
            withExtension: "metal",
            subdirectory: "Shaders"
        )
        guard let url else { throw Error.shaderSourceNotFound }
        let source = try String(contentsOf: url, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }
}
