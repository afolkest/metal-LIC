# Requirements and Goals
**Consistency rule**: requirements must match `docs/02-algorithm-spec.md` for all v1 decisions. The spec is the source of truth.

## Problem statement
- Deliver a fast, high-quality GPU LIC for real-time interactive visual art on macOS (Apple Silicon).
- Support arbitrary non-negative input textures (not just white noise) convolved along static 2D vector fields.

## Scope
- In-scope features:
  - 2D LIC on static vector fields.
  - Arbitrary non-negative input texture (noise or artist-provided) as the seed signal.
  - Input textures may be GPU-generated (e.g., CA) or CPU-provided for simple/offline use.
  - Grayscale LIC output, computed on GPU (Metal).
  - User-specifiable kernel length (baseline default provided).
  - Direction-only (normalized) vector field integration.
- Out-of-scope features:
  - 3D LIC or time-varying vector fields.
  - Temporal coherence / flicker handling for animated sequences.
  - Vector-field resampling or preprocessing beyond minimal normalization in the core algorithm (wrapper-level resampling for scaling/zoom is allowed).
  - UI/UX tooling (controls, authoring UI) beyond minimal parameter hooks.

## Target platforms
- macOS on Apple Silicon (M1+).
- Primary dev target: M1 Pro.
- Metal compute pipeline.

## Performance targets
- Target resolution: single 4K output (3840x2160).
- Stretch goal: 4K at 60 fps (quality-dependent).
- Expected baseline: 4K at ~30 fps for high-quality settings; 2K at 60 fps likely for high-quality.
- Memory budget: keep GPU memory footprint modest; avoid large multi-pass buffers where possible.
 
## Real-time defaults (v1)
- Kernel half-length L: ~30 px at 1024-scale (full length ~60 px).
- Step size h: 1.0 px (0.5 px is "ultra" and not default for real-time).
- Iterations: 1 (multi-pass is allowed but not default for real-time).
- Edge gains: off by default (strength = 0).
- Precision: float32 integration/accumulation; float16 output.

## Real-time implementation guidance
- Precreate and reuse pipelines, samplers, buffers, and textures; no per-frame allocation.
- Keep textures GPU-resident and avoid per-frame CPU readbacks.
- Use multiple in-flight command buffers to avoid CPU-GPU sync stalls.
- Prefer GPU noise animation (procedural or small seed updates) over full texture uploads.
- CPU-provided textures are acceptable for simple use, but avoid per-frame full-size uploads for real-time.
- Use a warm-up dispatch at startup to avoid first-frame timing spikes.
- Tune threadgroup size with profiling; start with 8x8 or 16x16.

## Quality targets
- Visual fidelity: extremely high; suitable for fine-art output.
- Acceptable artifacts: none visible (avoid aliasing, banding, directional bias, edge darkening).
- Determinism: deterministic output given the same inputs and parameters.

## Input data
- Vector field dimensionality: 2D.
- Temporal behavior: static.
- Data format / source: numerical grid (resolution often 2K-4K); no analytic field required.
- Coordinate system / scaling: vector field normalized to direction only.

## Output
- Grayscale LIC.
- Output format: float16 (compute in float16/float32 as needed; defer 8-bit conversion to downstream tooling).
- Export/compositing outside the core implementation.

## User controls
- Primary: kernel half-length `L` (default ~30 px at 1024-scale; full length ~60 px; adjustable).
- Secondary (optional): step size (1.0 px default; 0.5 px “ultra”), integration method (RK2 only; RK4 deferred), noise/texture input selection.
- Kernel type: fixed for v1 (Hann/cosine window).

## Constraints and dependencies
- Core implementation should stay minimal and focused; helper scripts/tools allowed for development only.
- No mandatory external dependencies beyond Metal and standard macOS frameworks.
- Host/shader resource bindings are fixed in the algorithm spec (Section 4.2) to avoid ambiguity across implementations.

## Success criteria
- High-quality LIC at 2K/4K with no visible artifacts.
- Meets or approaches real-time performance on M1 Pro.
- Clear parameterization that enables artistic control without complicating core logic.
- Validation includes CPU reference checks and bryLIC parity as a warning signal (see `docs/03-validation-plan.md`).

## Risks
- 4K60 may be infeasible for long kernels or high sample counts.
- Performance vs quality tradeoffs may require multiple modes.
- Integration accuracy vs speed (step size, RK order) may affect artifact risk.
