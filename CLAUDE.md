# CLAUDE.md

## What this is

A real-time Line Integral Convolution (LIC) engine for macOS Apple Silicon, written in Swift + Metal. LIC convolves an input texture along a 2D vector field to produce streakline visualizations — used here for fine-art output, not scientific visualization.

## Key facts

- **Platform**: macOS, Apple Silicon (M1+), Metal compute shaders
- **Algorithm**: RK2 streamline integration, Hann kernel, bilinear vector field sampling, float32 accumulation, float16 output
- **Input**: arbitrary non-negative texture (GPU-generated or CPU-uploaded) + 2D direction-only vector field
- **Output**: grayscale float16 LIC image
- **Performance target**: 4K @ ~30 fps, 2K @ 60 fps on M1 Pro with default params (L=30, h=1.0, 1 iteration)
- **Quality bar**: extremely high — no visible artifacts (aliasing, banding, directional bias, edge darkening)

## What counts as success

1. GPU output matches a spec-exact CPU reference within float16 tolerance
2. No visible artifacts on standard test scenes (uniform, vortex, saddle, shear, radial, random fields)
3. Deterministic: identical inputs always produce identical output
4. Meets performance targets on M1 Pro

## Implementation status

Work follows three milestones in `docs/04-implementation-plan.md`:
- **M1**: Core shader + host encoder + smoke tests
- **M2**: Full spec compliance (masks, edge gains, multi-pass) + CPU reference + validation tests
- **M3**: Performance tuning + bryLIC parity checks

## Where to find details

The `docs/` folder is the source of truth. Read these when you need specifics:

- `docs/00-requirements-and-goals.md` — scope, constraints, performance/quality targets
- `docs/01-algorithm-decisions.md` — locked decisions, paper takeaways, deferred items
- `docs/02-algorithm-spec.md` — **the spec** (authoritative for all algorithm details: kernel construction, integration steps, boundary handling, resource bindings, MSL signature)
- `docs/03-validation-plan.md` — test strategy (CPU reference, GPU comparison, bryLIC parity)
- `docs/04-implementation-plan.md` — milestone checklist with exit criteria

When in doubt, `02-algorithm-spec.md` wins — it is the single source of truth for how the algorithm works.

## Important design choices to know upfront

- **Raw weighted sum**: output is not globally normalized. Renormalization only happens on boundary/mask truncation.
- **Kernel is built on the host**: the wrapper constructs the discrete Hann kernel and passes it to the shader as a buffer. Odd-length only.
- **1:1 coordinate mapping**: vector field, input, mask, and output all share the same resolution. Scaling is the caller's job.
- **Resource bindings are fixed**: texture/sampler/buffer slots are specified exactly in the spec (Section 4.2). Don't improvise.
- **Function constants for specialization**: use Metal `function_constant` to eliminate mask, edge gain, and debug branches at compile time. Cache specialized pipeline variants at init — never branch at runtime for dispatch-uniform state. See spec Section 4.2.
- **Debug visualization built in**: the shader has a `kDebugMode` function constant (step count heat map, boundary hits, kernel support ratio). Use these during development instead of printf debugging.
- **Boundary handling has precise rules**: closed domain truncation, mask-stop semantics, conditional renormalization + edge gains. Get the details from spec Section 9.
