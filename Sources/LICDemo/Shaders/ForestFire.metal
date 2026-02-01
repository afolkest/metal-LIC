#include <metal_stdlib>
using namespace metal;

struct ForestFireParams {
    uint  width;
    uint  height;
    uint  frameNumber;
    float growthRate;     // regrowth speed
    float ignitionProb;   // spontaneous ignition probability
    float burnRate;       // how fast burning cells decay
    float spreadRate;     // fire spread probability per burning neighbor
    float diffusion;      // spatial smoothing toward neighbor average
};

// Simple hash-based RNG: deterministic per (x, y, frame)
static float pcgHash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / float(0xFFFFFFFFu);
}

static float rng(uint x, uint y, uint frame, uint salt) {
    uint seed = x * 1973u + y * 9277u + frame * 26699u + salt * 12893u;
    return pcgHash(seed);
}

kernel void forestFireKernel(
    texture2d<float, access::read>   stateIn  [[texture(0)]],
    texture2d<float, access::write>  stateOut [[texture(1)]],
    constant ForestFireParams&       params   [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float cell = stateIn.read(gid).r;
    float r0 = rng(gid.x, gid.y, params.frameNumber, 0u);
    float r1 = rng(gid.x, gid.y, params.frameNumber, 1u);

    // Count burning neighbors (Moore neighborhood, toroidal wrapping)
    float burningCount = 0.0;
    float neighborSum = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            uint nx = (gid.x + uint(dx + int(params.width))) % params.width;
            uint ny = (gid.y + uint(dy + int(params.height))) % params.height;
            float n = stateIn.read(uint2(nx, ny)).r;
            neighborSum += n;
            if (n > 0.85) burningCount += 1.0;
        }
    }
    float neighborAvg = neighborSum / 8.0;

    float result;

    if (cell > 0.85) {
        // Burning: decay
        result = cell - params.burnRate;
        if (result <= 0.85) {
            result = 0.0; // snap to ash
        }
    } else if (cell > 0.15) {
        // Vegetation: grow, diffuse, maybe catch fire
        result = cell + params.growthRate;
        // Spatial diffusion
        result = mix(result, neighborAvg, params.diffusion);
        // Ignition from burning neighbors
        if (burningCount > 0.0 && r0 < params.spreadRate * burningCount) {
            result = 0.95; // ignite
        }
        // Rare spontaneous ignition
        if (r1 < params.ignitionProb) {
            result = 0.95;
        }
        result = clamp(result, 0.0, 0.84); // stay vegetation unless ignited
        if (burningCount > 0.0 && r0 < params.spreadRate * burningCount) {
            result = 0.95;
        }
        if (r1 < params.ignitionProb) {
            result = 0.95;
        }
    } else {
        // Empty/ash: regrow slowly with random variation
        result = cell + params.growthRate * (0.5 + 0.5 * r0);
        result = clamp(result, 0.0, 0.84);
    }

    stateOut.write(float4(result, 0.0, 0.0, 1.0), gid);
}

// Initialize CA state: ~80% mid-vegetation, ~15% mature, ~5% burning seeds
kernel void forestFireInitKernel(
    texture2d<float, access::write>  stateOut [[texture(0)]],
    constant uint&                   seed     [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]])
{
    uint w = stateOut.get_width();
    uint h = stateOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float r = rng(gid.x, gid.y, seed, 42u);

    float value;
    if (r < 0.05) {
        // 5% burning seeds
        value = 0.90 + rng(gid.x, gid.y, seed, 99u) * 0.10;
    } else if (r < 0.20) {
        // 15% mature vegetation (close to burning threshold)
        value = 0.65 + rng(gid.x, gid.y, seed, 77u) * 0.19;
    } else {
        // 80% mid-vegetation
        value = 0.20 + rng(gid.x, gid.y, seed, 55u) * 0.45;
    }

    stateOut.write(float4(value, 0.0, 0.0, 1.0), gid);
}
