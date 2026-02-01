#include <metal_stdlib>
using namespace metal;

struct DisplayParams {
    float fullSum;   // kernel weight sum for normalization
    float gamma;     // display gamma (2.2)
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Fullscreen triangle from vertex_id (no vertex buffer needed).
// Three vertices that cover the entire clip space:
//   id=0: (-1, -1)  id=1: (3, -1)  id=2: (-1, 3)
vertex VertexOut displayVertex(uint vid [[vertex_id]]) {
    VertexOut out;
    float2 pos = float2((vid << 1) & 2, vid & 2);
    out.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    out.texCoord = float2(pos.x, 1.0 - pos.y); // flip Y for Metal texture coords
    return out;
}

// Sample r16Float LIC output, normalize by fullSum, apply gamma, output grayscale.
fragment float4 displayFragment(
    VertexOut                          in      [[stage_in]],
    texture2d<float, access::sample>   licTex  [[texture(0)]],
    sampler                            samp    [[sampler(0)]],
    constant DisplayParams&            params  [[buffer(0)]])
{
    float raw = licTex.sample(samp, in.texCoord).r;
    float normalized = raw / max(params.fullSum, 1e-6);
    normalized = saturate(normalized);
    float display = pow(normalized, 1.0 / params.gamma);
    return float4(display, display, display, 1.0);
}
