import SwiftUI

@main
struct LICDemoApp: App {
    var body: some Scene {
        WindowGroup {
            MetalView()
                .frame(minWidth: 512, minHeight: 512)
        }
        .defaultSize(width: 1024, height: 1024)
    }
}
