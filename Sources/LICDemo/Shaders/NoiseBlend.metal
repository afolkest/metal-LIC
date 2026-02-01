#include <metal_stdlib>
using namespace metal;

struct NoiseBlendParams {
    float fireWeight; // blend factor: 1.0 = all fire, 0.0 = all noise
};

kernel void noiseBlendKernel(
    texture2d<float, access::read>   heatTex    [[texture(0)]],
    texture2d<float, access::read>   noiseTex   [[texture(1)]],
    texture2d<float, access::write>  blendedOut [[texture(2)]],
    constant NoiseBlendParams&       params     [[buffer(0)]],
    uint2                            gid        [[thread_position_in_grid]])
{
    uint w = blendedOut.get_width();
    uint h = blendedOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float heat  = heatTex.read(gid).r;
    float noise = noiseTex.read(gid).r;
    float blended = params.fireWeight * heat + (1.0 - params.fireWeight) * noise;

    blendedOut.write(float4(blended, 0.0, 0.0, 1.0), gid);
}
