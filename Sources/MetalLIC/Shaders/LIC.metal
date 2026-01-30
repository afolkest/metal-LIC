#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Function constants (compile-time specialization, Section 4.2)
// ---------------------------------------------------------------------------
constant bool kMaskEnabled      [[function_constant(0)]];
constant bool kEdgeGainsEnabled [[function_constant(1)]];
constant uint kDebugMode        [[function_constant(2)]];

// ---------------------------------------------------------------------------
// LicParams — shared with Swift host (must match layout exactly)
// ---------------------------------------------------------------------------
struct LicParams {
    float h;                        // integration step size (px)
    float eps2;                     // zero-vector guard (1e-12)
    uint  steps;                    // round(L / h)
    uint  kmid;                     // kernel center index
    uint  kernel_len;               // kernel array length (2*steps + 1)
    float full_sum;                 // sum of all kernel weights
    float center_weight;            // kernel[kmid]
    float edge_gain_strength;       // mask edge gain strength
    float edge_gain_power;          // mask edge gain exponent
    float domain_edge_gain_strength; // domain edge gain strength
    float domain_edge_gain_power;   // domain edge gain exponent
};

// ---------------------------------------------------------------------------
// LIC compute kernel (placeholder — full implementation in M1)
// ---------------------------------------------------------------------------
kernel void licKernel(
    texture2d<float, access::sample>  inputTex   [[texture(0)]],
    texture2d<float, access::sample>  vectorTex  [[texture(1)]],
    texture2d<uint,  access::read>    maskTex    [[texture(2)]],
    texture2d<half,  access::write>   outputTex  [[texture(3)]],
    constant LicParams&               params     [[buffer(0)]],
    constant float*                   kernelW    [[buffer(1)]],
    sampler                           inputSamp  [[sampler(0)]],
    sampler                           vectorSamp [[sampler(1)]],
    uint2                             gid        [[thread_position_in_grid]])
{
    if (gid.x >= outputTex.get_width() || gid.y >= outputTex.get_height()) {
        return;
    }

    // Placeholder: write zero. Real integration logic comes next.
    outputTex.write(half4(0.0h), gid);
}
