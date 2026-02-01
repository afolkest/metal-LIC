import MetalKit

struct CAParameter: Identifiable {
    let id: String
    let label: String
    let min: Float
    let max: Float
    let defaultValue: Float
}

protocol CellularAutomaton: AnyObject {
    var name: String { get }
    var licInputTexture: MTLTexture { get }
    var parameters: [CAParameter] { get }
    func getValue(for id: String) -> Float
    func setValue(_ value: Float, for id: String)
    func setVectorField(_ texture: MTLTexture)
    func encodeStep(commandBuffer: MTLCommandBuffer)
    func reset()
    var isPaused: Bool { get set }
}
