import XCTest
import Metal
@testable import MetalLIC

final class MetalLICTests: XCTestCase {

    func testMetalDeviceAvailable() throws {
        let device = MetalLIC.device
        XCTAssertNotNil(device, "Metal device must be available on Apple Silicon")
    }

    func testMetalDeviceSupportsFamily() throws {
        let device = try XCTUnwrap(MetalLIC.device)
        XCTAssertTrue(
            device.supportsFamily(.apple7),
            "Device must support Apple family 7+ (M1 or later)"
        )
    }

    func testShaderLibraryCompiles() throws {
        let device = try XCTUnwrap(MetalLIC.device)
        let library = try ShaderLibrary.makeLibrary(device: device)
        XCTAssertNotNil(
            library.makeFunction(name: "licKernel"),
            "licKernel function must exist in compiled shader library"
        )
    }

    // MARK: - LicParams layout

    func testLicParamsSize() {
        // 11 fields × 4 bytes = 44 bytes, matching MSL struct layout
        XCTAssertEqual(MemoryLayout<LicParams>.size, 44)
        XCTAssertEqual(MemoryLayout<LicParams>.stride, 44)
        XCTAssertEqual(MemoryLayout<LicParams>.alignment, 4)
    }

    func testLicParamsFieldOffsets() {
        // Verify each field sits at the expected byte offset to match Metal's layout.
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.h), 0)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.eps2), 4)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.steps), 8)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.kmid), 12)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.kernelLen), 16)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.fullSum), 20)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.centerWeight), 24)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.edgeGainStrength), 28)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.edgeGainPower), 32)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.domainEdgeGainStrength), 36)
        XCTAssertEqual(MemoryLayout<LicParams>.offset(of: \.domainEdgeGainPower), 40)
    }
}
