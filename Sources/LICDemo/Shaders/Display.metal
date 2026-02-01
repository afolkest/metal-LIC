#include <metal_stdlib>
using namespace metal;

struct DisplayParams {
    float fullSum;     // kernel weight sum for normalization
    float exposure;    // multiplicative (default 1.0)
    float contrast;    // pivot around 0.5 (default 1.0)
    float brightness;  // additive offset (default 0.0)
    float gamma;       // display gamma (default 2.2)
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Fullscreen triangle from vertex_id (no vertex buffer needed).
vertex VertexOut displayVertex(uint vid [[vertex_id]]) {
    VertexOut out;
    float2 pos = float2((vid << 1) & 2, vid & 2);
    out.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    out.texCoord = float2(pos.x, 1.0 - pos.y);
    return out;
}

// Sample r16Float LIC output, apply exposure/contrast/brightness/gamma, output grayscale.
fragment float4 displayFragment(
    VertexOut                          in      [[stage_in]],
    texture2d<float, access::sample>   licTex  [[texture(0)]],
    sampler                            samp    [[sampler(0)]],
    constant DisplayParams&            params  [[buffer(0)]])
{
    float raw = licTex.sample(samp, in.texCoord).r;

    // Normalize by kernel weight sum
    float v = raw / max(params.fullSum, 1e-6);

    // Exposure
    v *= params.exposure;

    // Contrast (pivot around 0.5)
    v = (v - 0.5) * params.contrast + 0.5;

    // Brightness offset
    v += params.brightness;

    // Clamp before gamma
    v = saturate(v);

    // Gamma correction
    v = pow(v, 1.0 / params.gamma);

    return float4(v, v, v, 1.0);
}
