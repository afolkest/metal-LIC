# Requirements and Goals

## Problem statement
- Deliver a fast, high-quality GPU LIC for real-time interactive visual art on macOS (Apple Silicon).
- Support arbitrary input textures (not just white noise) convolved along static 2D vector fields.

## Scope
- In-scope features:
  - 2D LIC on static vector fields.
  - Arbitrary input texture (noise or artist-provided) as the seed signal.
  - Grayscale LIC output, computed on GPU (Metal).
  - User-specifiable kernel length (baseline default provided).
  - Direction-only (normalized) vector field integration.
- Out-of-scope features:
  - 3D LIC or time-varying vector fields.
  - Temporal coherence / flicker handling for animated sequences.
  - Vector-field resampling or preprocessing beyond minimal normalization.
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
- Primary: kernel length (default ~30 pixels at 1024-scale; adjustable).
- Secondary (optional): step size, integration method, noise/texture input selection.
- Kernel type: fixed for v1 (exact type TBD; e.g., Gaussian or box).

## Constraints and dependencies
- Core implementation should stay minimal and focused; helper scripts/tools allowed for development only.
- No mandatory external dependencies beyond Metal and standard macOS frameworks.

## Success criteria
- High-quality LIC at 2K/4K with no visible artifacts.
- Meets or approaches real-time performance on M1 Pro.
- Clear parameterization that enables artistic control without complicating core logic.

## Risks
- 4K60 may be infeasible for long kernels or high sample counts.
- Performance vs quality tradeoffs may require multiple modes.
- Integration accuracy vs speed (step size, RK order) may affect artifact risk.

