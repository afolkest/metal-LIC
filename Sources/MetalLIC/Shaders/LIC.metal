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
// Helpers
// ---------------------------------------------------------------------------

/// Sample and normalize a vector field direction.
/// Returns (normalized_direction, is_valid).
/// Invalid means NaN/inf in the raw sample — caller must stop integration.
/// Zero/near-zero returns (0,0) with is_valid = true (still sample, no advance).
inline float2 sampleDirection(texture2d<float, access::sample> vectorTex,
                              sampler vectorSamp,
                              float2 pos,
                              float eps2,
                              thread bool& is_valid)
{
    float2 raw = vectorTex.sample(vectorSamp, pos).rg;
    float len2 = dot(raw, raw);
    if (!isfinite(len2)) {
        is_valid = false;
        return float2(0.0);
    }
    is_valid = true;
    if (len2 < eps2) {
        return float2(0.0);    // zero vector: no advance, still sample
    }
    return raw * rsqrt(len2);  // direction-only
}

/// Domain boundary test: valid domain is [0.5, W-0.5] x [0.5, H-0.5].
inline bool isOutsideDomain(float2 pos, float W, float H) {
    return pos.x < 0.5f || pos.x > W - 0.5f ||
           pos.y < 0.5f || pos.y > H - 0.5f;
}

// ---------------------------------------------------------------------------
// LIC compute kernel (Sections 7–9, 14)
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
    // Bounds check
    if (gid.x >= outputTex.get_width() || gid.y >= outputTex.get_height()) {
        return;
    }

    // Pixel center in pixel coordinates (Section 5.1)
    float2 x0 = float2(gid) + 0.5f;
    float W = float(outputTex.get_width());
    float H = float(outputTex.get_height());

    // ------------------------------------------------------------------
    // Starting-pixel mask check (Section 9)
    // ------------------------------------------------------------------
    bool starting_masked = false;
    if (kMaskEnabled) {
        if (maskTex.read(gid).r != 0) {
            starting_masked = true;
            float center_sample = inputTex.sample(inputSamp, x0).r;
            float result = (params.full_sum != 0.0f)
                ? params.full_sum * center_sample
                : center_sample;

            if (kDebugMode == 0) {
                outputTex.write(half4(half(result), 0.0h, 0.0h, 1.0h), gid);
            } else if (kDebugMode == 1) {
                outputTex.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);  // no steps
            } else if (kDebugMode == 2) {
                outputTex.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);  // no boundary hit
            } else if (kDebugMode == 3) {
                half r = half(params.center_weight / max(params.full_sum, 1e-12f));
                outputTex.write(half4(r, 0.0h, 0.0h, 1.0h), gid);
            }
            return;
        }
    }

    // ------------------------------------------------------------------
    // Initialize center sample (Section 7, step 2)
    // ------------------------------------------------------------------
    float center_sample = inputTex.sample(inputSamp, x0).r;
    float value    = params.center_weight * center_sample;
    float used_sum = params.center_weight;

    // Integration tracking
    bool hit_domain_edge = false;
    bool hit_mask_edge   = false;
    uint total_steps     = 0;

    // ------------------------------------------------------------------
    // Forward (dir=0, sign=+1) and backward (dir=1, sign=-1) integration
    // Section 7: "For backward integration, use -v(x)."
    // ------------------------------------------------------------------
    for (int dir = 0; dir < 2; dir++) {
        float dir_sign = (dir == 0) ? 1.0f : -1.0f;
        float2 x = x0;

        for (uint step_count = 1; step_count <= params.steps; step_count++) {

            // --- Sample direction at current position ---
            bool v_valid;
            float2 v = sampleDirection(vectorTex, vectorSamp, x,
                                       params.eps2, v_valid);
            // NaN/inf: stop before sampling, no boundary hit (Section 7)
            if (!v_valid) break;

            v *= dir_sign;  // negate direction for backward pass

            // --- RK2 step 1: midpoint (Section 7.1, step 1) ---
            float2 x1 = x + 0.5f * params.h * v;
            if (isOutsideDomain(x1, W, H)) {
                hit_domain_edge = true;
                break;
            }

            // --- Sample direction at midpoint ---
            bool v1_valid;
            float2 v1 = sampleDirection(vectorTex, vectorSamp, x1,
                                        params.eps2, v1_valid);
            if (!v1_valid) break;

            v1 *= dir_sign;

            // --- RK2 step 2: full step (Section 7.1, step 2) ---
            float2 x_next = x + params.h * v1;
            if (isOutsideDomain(x_next, W, H)) {
                hit_domain_edge = true;
                break;
            }

            // --- Mask check (Section 7.1, step 3) ---
            if (kMaskEnabled) {
                uint2 mask_idx = uint2(floor(x_next));
                if (maskTex.read(mask_idx).r != 0) {
                    hit_mask_edge = true;
                    break;
                }
            }

            // --- Sample & accumulate (Section 7.1, step 4 / Section 8) ---
            uint k = (dir == 0)
                ? (params.kmid + step_count)    // forward
                : (params.kmid - step_count);   // backward
            float w = kernelW[k];
            float s = inputTex.sample(inputSamp, x_next).r;
            value    += w * s;
            used_sum += w;
            total_steps++;

            // --- Advance (Section 7.1, step 5) ---
            x = x_next;
        }
    }

    // ------------------------------------------------------------------
    // Boundary processing: renormalization + edge gains (Section 9)
    // Gated by kEdgeGainsEnabled function constant.
    // ------------------------------------------------------------------
    if (kEdgeGainsEnabled) {
        bool needs_boundary = (used_sum > params.center_weight)
                           && (used_sum < params.full_sum);
        bool apply_mask_edge = hit_mask_edge && !starting_masked;

        if (needs_boundary && (apply_mask_edge || hit_domain_edge)) {
            float support_factor = clamp(
                (used_sum - params.center_weight) /
                (params.full_sum - params.center_weight),
                0.0f, 1.0f);

            // Renormalize once
            value *= params.full_sum / used_sum;

            // Mask edge gain
            if (apply_mask_edge && params.edge_gain_strength > 0.0f) {
                float t = clamp(
                    (params.full_sum - used_sum) / params.full_sum,
                    0.0f, 1.0f);
                float gain = 1.0f + params.edge_gain_strength
                    * pow(t, params.edge_gain_power) * support_factor;
                value *= gain;
            }

            // Domain edge gain
            if (hit_domain_edge && params.domain_edge_gain_strength > 0.0f) {
                float t = clamp(
                    (params.full_sum - used_sum) / params.full_sum,
                    0.0f, 1.0f);
                float gain = 1.0f + params.domain_edge_gain_strength
                    * pow(t, params.domain_edge_gain_power) * support_factor;
                value *= gain;
            }
        }
    }

    // ------------------------------------------------------------------
    // Output (Section 14: debug visualization)
    // ------------------------------------------------------------------
    if (kDebugMode == 0) {
        outputTex.write(half4(half(value), 0.0h, 0.0h, 1.0h), gid);
    } else if (kDebugMode == 1) {
        // Step count heat map: (fwd + bwd steps) / (2 * steps)
        float ratio = (params.steps > 0)
            ? float(total_steps) / float(2 * params.steps)
            : 0.0f;
        outputTex.write(half4(half(saturate(ratio)), 0.0h, 0.0h, 1.0h), gid);
    } else if (kDebugMode == 2) {
        // Boundary hit: 0.0=none, 0.5=mask only, 0.75=domain only, 1.0=both
        half val = 0.0h;
        if (hit_mask_edge && !hit_domain_edge)  val = 0.5h;
        if (!hit_mask_edge && hit_domain_edge)  val = 0.75h;
        if (hit_mask_edge && hit_domain_edge)   val = 1.0h;
        outputTex.write(half4(val, 0.0h, 0.0h, 1.0h), gid);
    } else if (kDebugMode == 3) {
        // Kernel support ratio: used_sum / full_sum
        float ratio = (params.full_sum > 0.0f)
            ? used_sum / params.full_sum
            : 0.0f;
        outputTex.write(half4(half(saturate(ratio)), 0.0h, 0.0h, 1.0h), gid);
    }
}
