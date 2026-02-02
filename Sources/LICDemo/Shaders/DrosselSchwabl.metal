#include <metal_stdlib>
using namespace metal;

// Namespaced to avoid collision with ForestFire.metal when files are concatenated.
namespace ds {

static float pcgHash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / float(0xFFFFFFFFu);
}

static float rng(uint x, uint y, uint frame, uint salt) {
    uint seed = x * 1973u + y * 9277u + frame * 26699u + salt * 12893u;
    return pcgHash(seed);
}

} // namespace ds

struct DrosselSchwablParams {
    uint  width;
    uint  height;
    uint  frameNumber;
    float growthProb;     // p: empty → tree probability
    float lightningProb;  // f: tree spontaneous ignition probability
    float heatDecay;      // visualization: heat decay rate per step
};

// Initialize: ~60% trees, sparse burning seeds, heat = 1.0 where burning
kernel void drosselSchwablInitKernel(
    texture2d<float, access::write>  stateOut [[texture(0)]],
    texture2d<float, access::write>  heatOut  [[texture(1)]],
    constant uint&                   seed     [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]])
{
    uint w = stateOut.get_width();
    uint h = stateOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float r1 = ds::rng(gid.x, gid.y, seed, 42u);
    float r2 = ds::rng(gid.x, gid.y, seed, 77u);

    float state = 0.0;  // empty
    float heat = 0.0;

    if (r1 < 0.6) {
        state = 1.0;  // tree
        heat = 1.0;
        if (r2 < 0.002) {
            state = 2.0;  // burning seed
        }
    }

    stateOut.write(float4(state, heat, 0.0, 1.0), gid);
    heatOut.write(float4(heat, 0.0, 0.0, 1.0), gid);
}

// Discrete 3-state Drossel-Schwabl forest fire:
//   burning → empty
//   tree + burning neighbor → burning
//   tree + lightning (prob f) → burning
//   empty + growth (prob p) → tree
// Heat layer for LIC visualization decays independently.
kernel void drosselSchwablKernel(
    texture2d<float, access::read>   stateIn  [[texture(0)]],
    texture2d<float, access::write>  stateOut [[texture(1)]],
    texture2d<float, access::write>  heatOut  [[texture(2)]],
    constant DrosselSchwablParams&   params   [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float2 old = stateIn.read(gid).rg;
    int c = int(old.r + 0.5);
    float oldHeat = old.g;

    float newState;
    float newHeat;

    if (c == 2) {
        // Burning → empty
        newState = 0.0;
        newHeat = max(0.0f, oldHeat - params.heatDecay);
    } else if (c == 1) {
        // Tree: check 4-connected neighbors for fire
        uint w = params.width;
        uint h = params.height;

        const int2 offsets[4] = { int2(1,0), int2(0,1), int2(-1,0), int2(0,-1) };
        bool hasFire = false;

        for (int i = 0; i < 4; i++) {
            uint nx = (gid.x + uint(offsets[i].x + int(w))) % w;
            uint ny = (gid.y + uint(offsets[i].y + int(h))) % h;
            int nc = int(stateIn.read(uint2(nx, ny)).r + 0.5);
            if (nc == 2) {
                hasFire = true;
                break;
            }
        }

        if (hasFire) {
            newState = 2.0;
            newHeat = 1.0;
        } else {
            float r = ds::rng(gid.x, gid.y, params.frameNumber, 100u);
            if (r < params.lightningProb) {
                newState = 2.0;
                newHeat = 1.0;
            } else {
                newState = 1.0;
                newHeat = 1.0;
            }
        }
    } else {
        // Empty: growth check
        float r = ds::rng(gid.x, gid.y, params.frameNumber, 200u);
        if (r < params.growthProb) {
            newState = 1.0;
            newHeat = 1.0;
        } else {
            newState = 0.0;
            newHeat = max(0.0f, oldHeat - params.heatDecay);
        }
    }

    stateOut.write(float4(newState, newHeat, 0.0, 1.0), gid);
    heatOut.write(float4(newHeat, 0.0, 0.0, 1.0), gid);
}
