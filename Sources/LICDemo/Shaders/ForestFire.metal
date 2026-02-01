#include <metal_stdlib>
using namespace metal;

struct ForestFireParams {
    uint  width;
    uint  height;
    uint  frameNumber;
    float igniteProb;    // spontaneous ignition probability
    float spreadProb;    // base spread probability per burning neighbor
    float fuelRegen;     // fuel regrowth rate per step
    float fuelConsume;   // fuel consumed per burning step
    float heatDecay;     // heat decay rate when not burning
    float fuelToIgnite;  // minimum fuel for spontaneous ignition
    float fuelToSpread;  // minimum fuel for fire spread
    float fuelToSustain; // minimum fuel to keep burning
    float windStrength;  // wind influence on spread direction
};

// PCG hash-based RNG
static float pcgHash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / float(0xFFFFFFFFu);
}

static float rng(uint x, uint y, uint frame, uint salt) {
    uint seed = x * 1973u + y * 9277u + frame * 26699u + salt * 12893u;
    return pcgHash(seed);
}

// Initialize CA: fuel=0.8–1.0, 5% burning seeds (heat=1.0), rest heat=0
kernel void forestFireInitKernel(
    texture2d<float, access::write>  stateOut [[texture(0)]],
    texture2d<float, access::write>  heatOut  [[texture(1)]],
    constant uint&                   seed     [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]])
{
    uint w = stateOut.get_width();
    uint h = stateOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float r = rng(gid.x, gid.y, seed, 42u);
    float fuelR = rng(gid.x, gid.y, seed, 55u);

    float fuel = 0.8 + fuelR * 0.2; // start with high fuel
    float heat = 0.0;

    if (r < 0.05) {
        // 5% burning seeds
        heat = 1.0;
    }

    stateOut.write(float4(fuel, heat, 0.0, 1.0), gid);
    heatOut.write(float4(heat, 0.0, 0.0, 1.0), gid);
}

// 3-field forest fire CA: fuel + heat in rg32Float, 4-connectivity, anisotropic wind spread
kernel void forestFireKernel(
    texture2d<float, access::read>   stateIn    [[texture(0)]],
    texture2d<float, access::write>  stateOut   [[texture(1)]],
    texture2d<float, access::write>  heatOut    [[texture(2)]],
    texture2d<float, access::read>   vectorTex  [[texture(3)]],
    constant ForestFireParams&       params     [[buffer(0)]],
    uint2                            gid        [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float2 state = stateIn.read(gid).rg;
    float fuel = state.r;
    float heat = state.g;
    bool wasBurning = heat >= 0.999;

    float newFuel = fuel;
    float newHeat = heat;

    if (wasBurning) {
        // Burning: consume fuel
        newFuel = fuel - params.fuelConsume;
        if (newFuel < params.fuelToSustain) {
            // Extinguish — not enough fuel
            newHeat = 0.0;
        } else {
            newHeat = 1.0; // still burning
        }
    } else {
        // Not burning: regenerate fuel (only if cooled enough, matching Python's heat < 0.3)
        if (heat < 0.3) {
            newFuel = min(fuel + params.fuelRegen, 1.0);
        }

        // Check 4-connected neighbors (von Neumann) for fire spread
        bool ignited = false;
        float2 windDir = vectorTex.read(gid).rg;

        // Cardinal offsets: right, up, left, down
        const int2 offsets[4] = { int2(1,0), int2(0,1), int2(-1,0), int2(0,-1) };

        for (int i = 0; i < 4; i++) {
            int2 off = offsets[i];
            uint nx = (gid.x + uint(off.x + int(params.width))) % params.width;
            uint ny = (gid.y + uint(off.y + int(params.height))) % params.height;

            float2 nState = stateIn.read(uint2(nx, ny)).rg;
            bool neighborBurning = nState.g >= 0.999;

            if (neighborBurning && newFuel >= params.fuelToSpread) {
                // spreadDir = direction fire travels FROM neighbor TO us = -offset
                float2 spreadDir = float2(-float(off.x), -float(off.y));
                float windMod = clamp(1.0 + params.windStrength * dot(spreadDir, windDir), 0.0, 20.0);
                float prob = params.spreadProb * newFuel * windMod;

                float r = rng(gid.x, gid.y, params.frameNumber, uint(i + 10));
                if (r < prob) {
                    ignited = true;
                }
            }
        }

        // Spontaneous ignition
        if (!ignited && newFuel >= params.fuelToIgnite) {
            float r = rng(gid.x, gid.y, params.frameNumber, 100u);
            if (r < params.igniteProb) {
                ignited = true;
            }
        }

        // Decay residual heat
        if (!ignited) {
            newHeat = max(0.0, heat - params.heatDecay);
        } else {
            newHeat = 1.0;
        }
    }

    stateOut.write(float4(newFuel, newHeat, 0.0, 1.0), gid);
    heatOut.write(float4(newHeat, 0.0, 0.0, 1.0), gid);
}
