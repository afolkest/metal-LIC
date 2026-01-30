import XCTest
@testable import MetalLIC

final class LICKernelTests: XCTestCase {

    // MARK: - Parameter validation

    func testInvalidL_throws() {
        XCTAssertThrowsError(try LICKernel.build(L: 0)) { error in
            XCTAssertTrue(error is LICKernel.Error)
        }
        XCTAssertThrowsError(try LICKernel.build(L: -1))
    }

    func testInvalidH_throws() {
        XCTAssertThrowsError(try LICKernel.build(L: 30, h: 0))
        XCTAssertThrowsError(try LICKernel.build(L: 30, h: -1))
    }

    // MARK: - steps == 0 edge case (Section 6.1)

    func testStepsZero_degenerateKernel() throws {
        // L=0.3, h=1.0 → round(0.3) = 0
        let (params, weights) = try LICKernel.build(L: 0.3, h: 1.0)
        XCTAssertEqual(params.steps, 0)
        XCTAssertEqual(params.kernelLen, 1)
        XCTAssertEqual(params.kmid, 0)
        XCTAssertEqual(params.fullSum, 1.0)
        XCTAssertEqual(params.centerWeight, 1.0)
        XCTAssertEqual(weights, [1.0])
    }

    // MARK: - Kernel shape properties

    func testOddLength() throws {
        for L: Float in [1, 5, 10, 30, 100] {
            let (params, weights) = try LICKernel.build(L: L)
            XCTAssertEqual(params.kernelLen % 2, 1, "Kernel length must be odd (L=\(L))")
            XCTAssertEqual(UInt32(weights.count), params.kernelLen)
        }
    }

    func testSymmetry() throws {
        let (_, weights) = try LICKernel.build(L: 30)
        let n = weights.count
        for i in 0..<n / 2 {
            XCTAssertEqual(weights[i], weights[n - 1 - i], accuracy: 1e-7,
                           "Kernel must be symmetric: index \(i) vs \(n - 1 - i)")
        }
    }

    func testCenterWeightIsOne() throws {
        // Hann at center: 0.5 * (1 + cos(0)) = 1.0
        let (params, weights) = try LICKernel.build(L: 30)
        XCTAssertEqual(weights[Int(params.kmid)], 1.0, accuracy: 1e-7)
        XCTAssertEqual(params.centerWeight, 1.0, accuracy: 1e-7)
    }

    func testAllWeightsNonNegative() throws {
        let (_, weights) = try LICKernel.build(L: 30)
        for (i, w) in weights.enumerated() {
            XCTAssertGreaterThanOrEqual(w, 0, "Weight at index \(i) must be >= 0")
        }
    }

    // MARK: - Computed fields

    func testDefaultParams() throws {
        // L=30, h=1.0 → steps=30, kernel_len=61, kmid=30
        let (params, weights) = try LICKernel.build(L: 30)
        XCTAssertEqual(params.steps, 30)
        XCTAssertEqual(params.kernelLen, 61)
        XCTAssertEqual(params.kmid, 30)
        XCTAssertEqual(params.h, 1.0)
        XCTAssertEqual(params.eps2, 1e-12)
        XCTAssertEqual(weights.count, 61)
    }

    func testFullSumMatchesWeights() throws {
        let (params, weights) = try LICKernel.build(L: 30)
        let computed = weights.reduce(0, +)
        XCTAssertEqual(params.fullSum, computed, accuracy: 1e-5)
    }

    func testStepsRounding() throws {
        // L=10, h=3 → round(10/3) = round(3.333) = 3
        let (params, _) = try LICKernel.build(L: 10, h: 3)
        XCTAssertEqual(params.steps, 3)
        XCTAssertEqual(params.kernelLen, 7)
        XCTAssertEqual(params.kmid, 3)
    }

    func testStepsRoundingTiesAwayFromZero() throws {
        // L=5, h=2 → L/h = 2.5 → round ties away from zero → 3
        let (params, _) = try LICKernel.build(L: 5, h: 2)
        XCTAssertEqual(params.steps, 3)
    }

    func testHalfStepSize() throws {
        // L=30, h=0.5 → steps = round(60) = 60, kernel_len = 121
        let (params, weights) = try LICKernel.build(L: 30, h: 0.5)
        XCTAssertEqual(params.steps, 60)
        XCTAssertEqual(params.kernelLen, 121)
        XCTAssertEqual(weights.count, 121)
    }

    // MARK: - Edge gain defaults

    func testEdgeGainDefaults() throws {
        let (params, _) = try LICKernel.build(L: 30)
        XCTAssertEqual(params.edgeGainStrength, 0)
        XCTAssertEqual(params.edgeGainPower, 2)
        XCTAssertEqual(params.domainEdgeGainStrength, 0)
        XCTAssertEqual(params.domainEdgeGainPower, 2)
    }

    func testCustomEdgeGains() throws {
        let (params, _) = try LICKernel.build(
            L: 30, edgeGainStrength: 0.5, edgeGainPower: 3,
            domainEdgeGainStrength: 0.8, domainEdgeGainPower: 1.5
        )
        XCTAssertEqual(params.edgeGainStrength, 0.5)
        XCTAssertEqual(params.edgeGainPower, 3)
        XCTAssertEqual(params.domainEdgeGainStrength, 0.8)
        XCTAssertEqual(params.domainEdgeGainPower, 1.5)
    }
}
