import SwiftUI
import AppKit

@main
struct LICDemoApp: App {
    @StateObject private var settings = DemoSettings()

    init() {
        // SPM executables don't get a proper activation policy by default,
        // so key events never reach the app. Force regular app mode.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup {
            HStack(spacing: 0) {
                MetalView(settings: settings)
                    .aspectRatio(1, contentMode: .fit)
                SliderOverlay(settings: settings)
            }
            .frame(minWidth: 780, minHeight: 512)
        }
        .defaultSize(width: 1300, height: 1024)
    }
}
