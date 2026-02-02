import SwiftUI

final class DemoSettings: ObservableObject {

    // MARK: - Info (set by renderer)

    @Published var fps: Double = 0

    // MARK: - Resolution

    @Published var resolution: Int = 1024

    // MARK: - Display

    @Published var exposure: Float = 1.0
    @Published var contrast: Float = 1.0
    @Published var brightness: Float = 0.0
    @Published var gamma: Float = 2.2

    // MARK: - Blend

    @Published var fireWeight: Float = 0.781

    // MARK: - LIC

    @Published var kernelLength: Float = 20.0
    @Published var licPasses: Int = 1

    // MARK: - Vector field

    @Published var vectorPreset: Int = 6

    // MARK: - CA

    @Published var caName: String = ""
    @Published var availableCAs: [String] = []
    @Published var selectedCAIndex: Int = 0
    @Published var caValues: [String: Float] = [:]
    @Published var caParameters: [CAParameter] = []

    // MARK: - Recording

    @Published var isRecording = false

    // MARK: - Actions

    @Published var resetRequested = false

    func populateFromCA(_ ca: CellularAutomaton) {
        caName = ca.name
        caParameters = ca.parameters
        caValues = Dictionary(uniqueKeysWithValues:
            ca.parameters.map { ($0.id, ca.getValue(for: $0.id)) }
        )
    }
}
