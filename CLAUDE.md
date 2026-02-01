# CLAUDE.md

## What this is

A real-time Line Integral Convolution (LIC) engine for macOS Apple Silicon, written in Swift + Metal. LIC convolves an input texture along a 2D vector field to produce streakline visualizations — used here for fine-art output, not scientific visualization.

The goal is a fast, artifact-free GPU implementation suitable for real-time creative applications. The code is authoritative for current behavior; design-phase documents are preserved in `docs/design-history/` for rationale and context.

## Key facts

- **Platform**: macOS, Apple Silicon (M1+), Metal compute shaders, SPM package
- **Algorithm**: RK2 streamline integration, Hann kernel, bilinear vector field sampling, float32 accumulation, float16 output
- **Input**: arbitrary non-negative `r32Float` texture (GPU-generated or CPU-uploaded) + 2D direction-only `rg32Float` vector field
- **Output**: grayscale `r16Float` LIC image (raw weighted sum, not globally normalized)
- **Performance target**: 4K @ ~30 fps, 2K @ 60 fps on M1 Pro with default params (L=30, h=1.0, 1 iteration)
- **Quality bar**: no visible artifacts (aliasing, banding, directional bias, edge darkening)

## Architecture

### Source files (`Sources/MetalLIC/`)

| File | Purpose |
|------|---------|
| `Shaders/LIC.metal` | Compute kernel — RK2 integration, Hann convolution, boundary handling, debug viz |
| `LICEncoder.swift` | Command buffer encoding, pipeline management (4 cached variants), texture helpers |
| `LICDispatcher.swift` | Pipelined multi-buffered dispatch with semaphore flow control (maxInFlight=3) |
| `LICKernel.swift` | Host-side Hann kernel construction, populates `LicParams` + weight array |
| `LicParams.swift` | Shared parameter struct — must match Metal layout exactly (44 bytes, no padding) |
| `LICReferenceCPU.swift` | Spec-exact CPU reference for validation (Float64 and Float32 variants) |

### Data flow

```
LICKernel.build(L, h, ...)
    → (LicParams, [Float] weights)
        → LICEncoder.encode(commandBuffer, params, weights, textures...)
            → GPU: licKernel per-thread
                → r16Float output
```

For multi-pass: `LICEncoder.encodeMultiPass` chains passes with ping-pong `r16Float` textures. `LICDispatcher` manages multiple in-flight command buffers to avoid CPU/GPU sync stalls.

## Design invariants

These are the non-obvious rules. Violating any of them will introduce bugs.

- **Raw weighted sum**: output = `sum(kernel[k] * sample[k])`. No global normalization. Renormalization only on boundary/mask truncation.
- **Kernel built on host**: discrete Hann kernel passed as a buffer. Odd-length only. `steps = round(L / h)`, `kernel_len = 2 * steps + 1`.
- **1:1 coordinate mapping**: vector field, input, mask, and output share the same resolution. Scaling is the caller's job.
- **Fixed resource bindings**: texture 0–3 (input, vector, mask, output), sampler 0–1 (input, vector), buffer 0–1 (params, weights). Don't change these.
- **Function constants for specialization**: `kMaskEnabled` (bool, index 0), `kEdgeGainsEnabled` (bool, index 1), `kDebugMode` (uint, index 2). Pipeline variants cached at init. Never branch at runtime for dispatch-uniform state.
- **Pixel coordinates**: centers at `(x + 0.5, y + 0.5)`. Valid domain is `[0.5, W-0.5] x [0.5, H-0.5]`. Samplers use `normalizedCoordinates = false`.
- **Boundary handling**: closed domain truncation. Renormalization applies when `used_sum < full_sum` AND a domain/mask edge was hit. Edge gain multipliers are gated by `kEdgeGainsEnabled`. NaN/Inf termination does NOT record a boundary hit and does NOT trigger renormalization.
- **Mask semantics**: starting-pixel masked → return `full_sum * center_sample` (no integration). Entering a masked pixel during integration → stop, truncate, renormalize.
- **Multi-pass**: ping-pong with `r16Float` intermediates. Mask semantics apply per pass. Metal auto-promotes float16→float32 on sample.
- **Determinism**: identical inputs must always produce identical output.

## Debug visualization

The shader has a `kDebugMode` function constant. Use these instead of printf debugging:

| Mode | Output | Use for |
|------|--------|---------|
| 0 | Normal LIC | Production |
| 1 | Step count / (2 * steps) | Finding early-termination patterns |
| 2 | Boundary hit flags (0/0.5/0.75/1.0) | Seeing where domain/mask boundaries are hit |
| 3 | used_sum / full_sum | Visualizing kernel truncation extent |

## Testing

Three tiers, gated by environment variables:

```bash
# Core tests (always run) — invariants, smoke, kernel, diagnostics,
# dispatcher, GPU/CPU parity on 16x16 grids
swift test

# bryLIC parity (advisory, no hard assertions) — 15 scenes comparing
# against Python bryLIC reference. Writes PNGs to output/parity/.
# Requires: python3 scripts/generate_brylic_reference.py
RUN_PARITY=1 swift test --filter LICParityTests

# Performance benchmarks — throughput at 1080p/2K/4K, threadgroup
# sweeps, pipelined dispatch, occupancy analysis
RUN_BENCHMARKS=1 swift test --filter LICBenchmarkTests
```

## What counts as success

1. GPU output matches CPU reference within float16 tolerance
2. No visible artifacts on standard test scenes (uniform, vortex, saddle, shear, radial, random)
3. Deterministic
4. Meets performance targets on M1 Pro

## Design history

`docs/design-history/` contains the original design specs from the v1 build-out:

- `00-requirements-and-goals.md` — scope, constraints, performance/quality targets
- `01-algorithm-decisions.md` — locked decisions, paper takeaways (Cabral & Leedom 1993, Hero & Pang 2007, Qin et al. 2010), deferred items
- `02-algorithm-spec.md` — original algorithm spec (boundary handling, resource bindings, MSL signature)
- `03-validation-plan.md` — test strategy rationale (CPU reference, bryLIC parity approach)
- `04-implementation-plan.md` — milestone checklist, performance analysis data (threadgroup tuning, occupancy, bandwidth)

These are useful for rationale and context but may drift from current code. When in doubt, the code wins.
