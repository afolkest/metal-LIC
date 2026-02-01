import SwiftUI

struct SliderOverlay: View {
    @ObservedObject var settings: DemoSettings
    @State private var isExpanded = true

    var body: some View {
        VStack(alignment: .trailing, spacing: 0) {
            // Toggle button
            Button(action: { withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() } }) {
                Image(systemName: isExpanded ? "chevron.right" : "chevron.left")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .padding(.bottom, 4)

            if isExpanded {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        displaySection
                        Divider()
                        blendSection
                        Divider()
                        licSection
                        Divider()
                        caSection
                    }
                    .padding(12)
                }
                .frame(minWidth: 260, idealWidth: 260, maxWidth: 260,
                       maxHeight: 600)
                .background(.ultraThinMaterial)
                .cornerRadius(8)
            }
        }
        .padding(8)
    }

    // MARK: - Sections

    private var displaySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Display").font(.headline).foregroundColor(.white)
            paramSlider(label: "Exposure", value: $settings.exposure, range: 0.1...5.0)
            paramSlider(label: "Contrast", value: $settings.contrast, range: 0.1...3.0)
            paramSlider(label: "Brightness", value: $settings.brightness, range: -0.5...0.5)
            paramSlider(label: "Gamma", value: $settings.gamma, range: 0.5...4.0)
        }
    }

    private var blendSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Blend").font(.headline).foregroundColor(.white)
            paramSlider(label: "Fire Weight", value: $settings.fireWeight, range: 0...1)
        }
    }

    private var licSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("LIC").font(.headline).foregroundColor(.white)
            paramSlider(label: "Kernel Length", value: $settings.kernelLength, range: 2...60)
        }
    }

    private var caSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Fire CA").font(.headline).foregroundColor(.white)
                Spacer()
                Button("Reset") { settings.resetRequested = true }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            ForEach(settings.caParameters) { param in
                let binding = Binding<Float>(
                    get: { settings.caValues[param.id] ?? param.defaultValue },
                    set: { settings.caValues[param.id] = $0 }
                )
                paramSlider(label: param.label, value: binding,
                            range: param.min...param.max)
            }
        }
    }

    // MARK: - Helpers

    private func paramSlider(label: String, value: Binding<Float>,
                              range: ClosedRange<Float>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.8))
                Spacer()
                Text(formatValue(value.wrappedValue, range: range))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.6))
            }
            Slider(value: value, in: range)
                .controlSize(.small)
        }
    }

    private func formatValue(_ v: Float, range: ClosedRange<Float>) -> String {
        if range.upperBound - range.lowerBound > 10 {
            return String(format: "%.1f", v)
        } else if range.upperBound - range.lowerBound > 1 {
            return String(format: "%.2f", v)
        } else {
            return String(format: "%.4f", v)
        }
    }
}
