import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
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

        let renderer = Renderer(device: device, view: view)
        view.delegate = renderer
        view.keyHandler = renderer
        context.coordinator.renderer = renderer

        DispatchQueue.main.async {
            view.window?.makeFirstResponder(view)
        }

        return view
    }

    func updateNSView(_ nsView: KeyableMTKView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var renderer: Renderer?
    }
}

class KeyableMTKView: MTKView {
    weak var keyHandler: Renderer?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        keyHandler?.handleKeyDown(event)
    }
}
