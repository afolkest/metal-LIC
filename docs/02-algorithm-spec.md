# Algorithm Spec (Draft)

This document specifies the v1 Line Integral Convolution (LIC) algorithm for Metal.
Items marked **TBD** are explicit decisions that must be finalized before implementation.

## 1) Purpose
Compute a high-quality grayscale LIC on the GPU for static 2D vector fields, using an arbitrary input texture as the seed signal.

## 2) Definitions & notation
- **V(x)**: 2D vector field sampled on a grid.
- **v(x)**: normalized direction field, `v(x) = normalize(V(x))`.
- **x**: continuous 2D position in texture space.
- **h**: integration step size in texture-space units (default 1.0 px; optional 0.5 px for higher detail).
- **L**: kernel half-length in texture-space units (user parameter).
- **kernel[k]**: discrete symmetric kernel weights, indexed over the streamline.
- **kmid**: center index of the kernel (`kernel.len() / 2`).
- **full_sum**: sum of all kernel weights.
- **used_sum**: sum of kernel weights actually sampled for this pixel.

## 3) Inputs
- **Vector field texture**: 2-channel float texture representing V(x).
- **Input texture**: arbitrary grayscale signal (noise or artist-provided).
- **Optional mask texture**: 1-channel boolean or float mask indicating blocked pixels.
- **Parameters**:
  - `L`: kernel half-length in pixels (default ~30 px at 1024-scale).
  - `h`: step size (default 1.0 px; optional 0.5 px “ultra”).
  - `kernel`: fixed Hann/cosine window (v1).
  - `boundary_mode`: closed for domain (truncate streamlines).
  - `edge_gain_strength`: mask edge gain strength (default 0).
  - `edge_gain_power`: mask edge gain exponent (default 2).
  - `domain_edge_gain_strength`: domain edge gain strength (default 0).
  - `domain_edge_gain_power`: domain edge gain exponent (default 2).
  - `noise_sample_mode`: wrap for generated noise; clamp for artist textures.
  - `input_sample_mode`: linear default (nearest optional).
  - `precision_mode`: float16 output; float32 accumulation/integration.

## 4) Output
- **Grayscale LIC** image as float16 texture.

## 5) Coordinate mapping (**TBD**)
Define mapping from output pixel coordinates to vector-field texture coordinates.
- Option A: vector field resolution == output resolution (1:1).
- Option B: vector field sampled at a different resolution; scale accordingly.

## 6) Kernel
- **Shape**: Hann/cosine window (symmetric).
- **Definition** (for |s| <= L):
  - `w(s) = 0.5 * (1 + cos(pi * s / L))`
  - `w(s) = 0` for |s| > L.
- **Discrete form**:
  - Kernel is represented as a 1D array `kernel[k]` with center at `kmid`.
  - `full_sum = sum(kernel)` (used for boundary processing).
  - The convolution output is the weighted sum of samples; it is **not** globally normalized.
  - Renormalization is applied **only** when a boundary or mask truncates the kernel (see Section 9).

## 7) Streamline integration
For each output pixel:
1. Let `x0` be the output pixel center in texture coordinates.
2. Initialize accumulation with the center sample at `x0`.
3. Integrate **forward** and **backward** along `v(x)` using RK2 (midpoint):
   - `x1 = x + 0.5 * h * v(x)`
   - `x_next = x + h * v(x1)`
   - For backward integration, use `-v(x)`.
4. Use **bilinear sampling** of the vector field for RK2.

**Termination conditions**:
- Hit domain boundary (closed): stop and truncate.
- Hit masked pixel: stop and truncate.
- Reached kernel length `L`.
- If a sampled vector is NaN, stop integration in that direction (no boundary hit is recorded).
- If the vector is zero, the streamline does not advance; sampling continues at the same pixel.

## 8) Sampling & convolution
For each step along the streamline (forward and backward):
1. Compute signed distance `s` from the center (accumulate `h`).
2. Select the corresponding discrete kernel weight `kernel[k]`.
3. Sample input texture at `x`.
4. Accumulate `value += kernel[k] * sample` and `used_sum += kernel[k]`.
5. Stop when `|s| >= L` or termination condition occurs.

Final output:
- `output = value` (raw weighted sum).
- Boundary/mask truncation may trigger renormalization and edge gain (Section 9).

**Input prefiltering (optional)**:
- The core algorithm does not require prefiltering.
- For generated white noise, a low-pass prefilter is recommended to reduce aliasing artifacts.
- Prefiltering is the caller’s responsibility (or an optional preprocessing pass).

## 9) Boundary & mask handling
- **Domain boundary**: closed. Streamline truncates at boundary.
- **Mask**: when entering a masked pixel, terminate integration.
  - If the starting pixel is masked, return `full_sum * center_sample` immediately (or `center_sample` if `full_sum == 0`).

**Boundary processing (matches bryLIC behavior)**:
- `center_weight = kernel[kmid]`
- `needs_boundary_processing = (used_sum > center_weight) && (used_sum < full_sum)`
- `hit_domain_edge`: true if either forward or backward integration hit a closed boundary.
- `hit_mask_edge`: true if either direction entered a masked pixel.
- `apply_mask_edge = hit_mask_edge && starting_pixel_not_masked`

If `needs_boundary_processing && (apply_mask_edge || hit_domain_edge)` then:
1. **Support factor**  
   `support_factor = clamp((used_sum - center_weight) / (full_sum - center_weight), 0..1)`
2. **Renormalize ONCE**  
   `value *= full_sum / used_sum`
3. **Mask edge gain** (if `apply_mask_edge` and `edge_gain_strength > 0`)  
   `t = clamp((full_sum - used_sum) / full_sum, 0..1)`  
   `gain = 1 + edge_gain_strength * t^edge_gain_power * support_factor`  
   `value *= gain`
4. **Domain edge gain** (if `hit_domain_edge` and `domain_edge_gain_strength > 0`)  
   `t = clamp((full_sum - used_sum) / full_sum, 0..1)`  
   `gain = 1 + domain_edge_gain_strength * t^domain_edge_gain_power * support_factor`  
   `value *= gain`

Notes:
- Renormalization and gains are **only** applied when a boundary/mask truncates the kernel.
- If truncation happens for other reasons (e.g., NaNs in the vector field), no renormalization is performed.
- Periodic boundaries wrap and do **not** set `hit_domain_edge`.

## 10) Sampling mode
- Vector field sampling: bilinear (for RK2).
- Input texture sampling: linear (default); nearest optional.
- Noise sampling: wrap for generated noise; clamp for artist textures.

## 11) Precision
- Output stored in float16.
- Integrate and accumulate in float32 to avoid artifacts.

## 12) Determinism
Given identical inputs and parameters, output must be deterministic.

## 13) Open decisions (TBD)
- Vector field resolution mapping to output (1:1 vs scaled).
- Optional RK4 “ultra” mode vs RK2 only.
- Input texture tiling strategy to avoid visible periodicity (especially for noise).

## Appendix A) Edge gain formula (reference)
Let:
- `full_sum = sum(kernel)`
- `center_weight = kernel[kmid]`
- `used_sum = sum of sampled kernel weights`
- `t = clamp((full_sum - used_sum) / full_sum, 0..1)`
- `support_factor = clamp((used_sum - center_weight) / (full_sum - center_weight), 0..1)`

Apply **only if** `needs_boundary_processing && (apply_mask_edge || hit_domain_edge)`:
1. Renormalize once: `value *= full_sum / used_sum`
2. Mask edge gain (if `apply_mask_edge`):  
   `value *= 1 + edge_gain_strength * t^edge_gain_power * support_factor`
3. Domain edge gain (if `hit_domain_edge`):  
   `value *= 1 + domain_edge_gain_strength * t^domain_edge_gain_power * support_factor`
