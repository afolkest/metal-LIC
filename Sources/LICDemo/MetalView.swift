import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    @ObservedObject var settings: DemoSettings

    func makeNSView(context: Context) -> KeyableMTKView {
        let view = KeyableMTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        view.preferredFramesPerSecond = 60
        view.isPaused = false
        view.enableSetNeedsDisplay = false

        let renderer = Renderer(device: device, view: view, settings: settings)
        view.delegate = renderer
        context.coordinator.renderer = renderer

        // Use a local event monitor so keyboard shortcuts work even when
        // SwiftUI sliders have focus.
        context.coordinator.keyMonitor = NSEvent.addLocalMonitorForEvents(
            matching: .keyDown
        ) { event in
            renderer.handleKeyDown(event)
            return event
        }

        return view
    }

    func updateNSView(_ nsView: KeyableMTKView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var renderer: Renderer?
        var keyMonitor: Any?

        deinit {
            if let monitor = keyMonitor {
                NSEvent.removeMonitor(monitor)
            }
        }
    }
}

class KeyableMTKView: MTKView {
    override var acceptsFirstResponder: Bool { true }
}
