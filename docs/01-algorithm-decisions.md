# Algorithm Decisions (Draft)

This document captures what is already decided, what is unknown, and what needs research.
It will be updated as we read the papers in `LIC_papers/`.

## What we know (decisions locked for v1)
- **Domain**: 2D static vector fields on a grid.
- **Output**: grayscale LIC only.
- **Seed signal**: arbitrary input texture (not limited to white noise).
- **Vector handling**: direction-only (normalized vectors).
- **Kernel length**: user-specifiable; baseline default ~30 px at a 1024-scale.
- **Kernel shape**: Hann/cosine window (symmetric).
- **Precision**: float16 output (compute in float16 or float32 as needed).
- **Platform**: macOS, Apple Silicon, Metal compute.
- **Quality bar**: extremely high; visible artifacts are unacceptable.
- **Integration**: RK2 (midpoint) with bilinear sampling of the vector field.
- **Boundary policy (domain)**: closed boundaries truncate streamlines; renormalize by kernel weight sum.
- **Boundary policy (mask)**: optional mask boundaries; stop when entering a masked pixel.
- **Edge gain / halo**: optional post-gain when truncation occurs (default off).

## Baseline algorithm (assumed until proven otherwise)
- For each output pixel, integrate a streamline forward and backward.
- Sample the input texture along the streamline and convolve with a kernel.
- Combine forward/backward samples into a single scalar output.

## Paper takeaways (first pass)
### Cabral & Leedom 1993 (SIGGRAPH LIC)
- Use **direction-only** vectors during advection; magnitude is handled separately (optional post-processing).
- **Symmetric forward/backward integration** is critical; bias creates false spirals.
- **Normalization**: divide by the integral of the kernel to keep brightness stable. Fixed normalization can highlight singularities but may darken edges.
- **Aliasing**: thin 1D sampling over white noise can alias; **low-pass filter the input texture** (cheaper than widening the kernel).
- **Integration**: their method is DDA-style with per-cell boundary stepping; they note RK4 may improve quality but did not study it.
- **Edge/zero-vector handling**: terminate on singularities or zero vectors; can return constant or input pixel.
- **Parameter L**: too large washes out texture; too small under-filters. Smallest effective L is preferred for both quality and speed.

### Hero & Pang 2007 (Hardware LIC/HyperLIC)
- GPU implementation is straightforward: store vector field as texture, compute streamline per pixel on GPU.
- Approximate neighbor-based LIC variants are faster but **noticeably lower quality** (angular aliasing, isotropy).
- **Precision matters**: low-precision textures can degrade quality; use float textures where possible.
- **Boundary**: clamped noise sampling can smear edges; **wrap** for noise can reduce edge artifacts (works best when input is random).

### Qin et al. 2010 (CUDA LIC)
- RK4 integration is a common choice in LIC for stability/accuracy.
- GPU parallelism scales well because each pixel is independent.

## Reference implementation notes (bryLIC)
- **Streamline stepping**: DDA-style grid traversal using time-to-next-pixel; vectors are treated as piecewise-constant per cell (no bilinear interpolation).
- **Boundary policy**: `closed` boundaries **truncate streamlines** (stop on edge hit); `periodic` boundaries wrap. Truncation is intentional to avoid edge smearing and enable edge-aware effects.
- **Mask boundaries**: optional boolean mask; streamlines stop when entering a blocked pixel. If the starting pixel is masked, output is the center sample scaled by kernel sum.
- **Kernel normalization on truncation**: if the kernel is truncated, the output is renormalized by `full_sum / used_sum` to avoid darkening.
- **Edge gain (halo)**: optional gain based on how much of the kernel was truncated, applied only when a streamline actually hits a boundary (mask or domain). This specifically addresses edge darkening and can create aesthetic halos.
- **Kernel shape used in scripts/tests**: cosine/Hann window (`0.5 * (1 + cos(pi * x / L))`), symmetric; default streamlength ~30 px at 1024-scale.
- **Polarization mode**: optional vector “sign continuity” (flip when dot product with last step is negative). Useful for sign-ambiguous fields (e.g., eigenvectors), not required for velocity fields.

## Known unknowns (to decide after reading)
### Integration
- Step size: fixed in texture space vs adaptive; relation to kernel length L.
- Handling of zero vectors / stagnation points (terminate, clamp, or fallback).
- Symmetry enforcement across forward/backward integration.
- Optional RK4 “ultra quality” mode vs single RK2 path.

### Kernel / convolution
- Kernel normalization strategy (per-pixel vs fixed).
- Forward/backward weighting (must be symmetric).

### Sampling & filtering
- Texture sampling mode (nearest vs linear).
- **Prefiltering** for input textures to avoid aliasing (especially for white noise).
- Input texture size/tiling strategy to avoid visible periodicity.

### Boundary behavior
- Noise sampling: wrap vs clamp (wrap reduces edge smears for random inputs).
- Avoiding edge darkening / bias from truncated streamlines.

### Performance strategy
- Single-pass vs multi-pass LIC.
- Use of precomputed streamline buffers vs on-the-fly integration.
- Threadgroup layout and memory access patterns.
- Precision tradeoffs (float16 vs float32 in key steps).

### Data layout / coordination
- Vector field resolution vs output resolution mapping.
- Coordinate transforms between vector grid and output grid.
- Handling of non-square or non-uniform fields.

## Unknown unknowns (to capture as we read)
- Any algorithmic pitfalls or artifacts specific to GPU LIC.
- Known optimizations that preserve quality at high resolution.
- Stable “best practice” choices for kernel length and step size.
- Recommended strategies for minimizing directional bias.

## Papers to read (first pass)
- `LIC_papers/Cab93.md` (Cabral & Leedom): baseline LIC definition, kernel/integration choices.
- `LIC_papers/Her00.md`: GPU or fast LIC insights (performance + artifacts).
- `LIC_papers/Qin10.md`: performance or accuracy tradeoffs (verify scope).
- `LIC_papers/uflic-a-line-integral-convolution-algorithm-for-visualizing-unst.md`: improvements/variants.

## Next steps
- Read `Cab93.md` first to anchor baseline method.
- Update “Known unknowns” with concrete decisions + rationale.
- Create `docs/02-algorithm-spec.md` once decisions are locked.
