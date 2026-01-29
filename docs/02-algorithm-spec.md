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
- **L**: kernel half-length in texture-space units (wrapper-level parameter used to build the kernel).
- **kernel[k]**: discrete symmetric kernel weights, indexed over the streamline.
- **kmid**: center index of the kernel (`kernel.len() / 2`).
- **full_sum**: sum of all kernel weights.
- **used_sum**: sum of kernel weights actually sampled for this pixel.

## 3) Inputs
 - **Vector field texture**: 2-channel float texture representing V(x) (direction field).
 - **Input texture**: arbitrary non-negative grayscale signal (noise or artist-provided).
 - **Optional mask texture**: 1-channel **uint** mask where `0 = unblocked`, `>0 = blocked`.
- **Parameters**:
  - `L`: kernel half-length in pixels (default ~30 px at 1024-scale), **wrapper-level**.
  - `h`: step size (default 1.0 px; optional 0.5 px “ultra”).
  - `kernel`: **precomputed** fixed Hann/cosine window (v1).
  - `boundary_mode`: closed for domain (truncate streamlines). Periodic is reserved for later.
  - `edge_gain_strength`: mask edge gain strength (default 0).
  - `edge_gain_power`: mask edge gain exponent (default 2).
  - `domain_edge_gain_strength`: domain edge gain strength (default 0).
  - `domain_edge_gain_power`: domain edge gain exponent (default 2).
  - Mask IDs > 0 are treated identically in v1 (single gain); per‑ID gains are reserved for later.
  - `uv_mode`: **velocity only** (polarization deferred).
  - `noise_sample_mode`: wrap for generated noise; clamp for artist textures.
  - `input_sample_mode`: **linear (fixed for v1)**.
  - `precision_mode`: float16 output; float32 accumulation/integration.
  - `iterations`: number of convolution passes (>= 1).

## 4) Output
- **Grayscale LIC** image as float16 texture.
- Output values are the **raw weighted sum** (not globally normalized).
- Input texture values are expected to be **non-negative**.
 - For `iterations > 1`, the output of each pass is used as the input texture for the next pass.

## 4.1) Texture formats (v1)
- Input texture: `r32Float`
- Vector field: `rg32Float`
- Output: `r16Float`
- Mask: `r8Uint`

## 5) Coordinate mapping
**v1 mapping is 1:1**: vector field, input texture, mask, and output share the same resolution.
- Output pixel centers are at `(x + 0.5, y + 0.5)` in texture space.
- Image space uses the standard image convention (x right, y down).
- Vector components are interpreted in **pixel units** (direction-only); `h` is measured in pixels.

**Scaling/transforming fields is out of core scope**:
- If a different resolution or zoom is desired, resample the vector field (and mask) to the output resolution in a wrapper before invoking the core LIC.

## 6) Kernel
- **Shape**: Hann/cosine window (symmetric).
- **Discrete form (RK2 steps)**:
  - `steps = round(L / h)`
  - `N = 2 * steps + 1`, `kmid = N // 2`
  - `s_i = (i - kmid) * h` (signed distance per step)
  - `kernel[i] = 0.5 * (1 + cos(pi * s_i / L))` for `|s_i| <= L`, else `0`
  - **No weight scaling by `h`** (matches bryLIC raw weighted sum behavior).
- **Normalization**:
  - The convolution output is the raw weighted sum (not globally normalized).
  - Renormalization is applied **only** when a boundary or mask truncates the kernel (see Section 9).
 - The kernel is **constructed by the wrapper** and passed into the core GPU kernel.
 - The wrapper must provide an **odd-length** kernel with a well-defined center (`kmid`).
 - Even-length kernels are invalid in v1 (treat as input error in wrapper).

## 7) Streamline integration
For each output pixel:
1. Let `x0` be the output pixel center in texture coordinates.
2. Initialize accumulation with the center sample at `x0`.
3. Integrate **forward** and **backward** along `v(x)` using RK2 (midpoint):
   - `x1 = x + 0.5 * h * v(x)`
   - `x_next = x + h * v(x1)`
   - For backward integration, use `-v(x)`.
4. Use **bilinear sampling** of the vector field for RK2.
5. **Normalize the sampled vector** after each bilinear lookup:
   - `len2 = u*u + v*v`
   - If `len2 < eps2` or not finite: treat as zero vector.
   - Else `v = (u, v) * rsqrt(len2)` (direction-only).
   - Use `eps2 = 1e-12` (float32).
6. Each RK2 step corresponds to one kernel index in the forward/backward ranges.

**Termination conditions**:
- Hit domain boundary (closed): stop and truncate.
- Hit masked pixel: stop and truncate.
- Reached kernel length `L` (i.e., step count exceeds `steps`).
- If a sampled vector is NaN/inf, stop integration in that direction **before sampling** (no boundary hit is recorded).
- If the vector is zero (or near-zero), the streamline does not advance; sampling continues at the same pixel.

## 8) Sampling & convolution
For each step along the streamline (forward and backward):
1. Each step advances by `h`; `step_count` determines `k` and `s_i`:
   - Forward: `k = kmid + step_count`
   - Backward: `k = kmid - step_count`
2. Select `kernel[k]` (equivalently `kernel(s_i)`).
3. Sample input texture at `x`.
4. Accumulate `value += kernel[k] * sample` and `used_sum += kernel[k]`.
5. Stop when `step_count > steps` or termination condition occurs.

Final output:
- `output = value` (raw weighted sum).
- Boundary/mask truncation may trigger renormalization and edge gain (Section 9).

## 8.1) Multi-pass convolution
- If `iterations > 1`, run the LIC pass repeatedly.
- Each pass uses the **previous pass output** as the new input texture.
- Use ping‑pong textures to avoid read/write hazards.
 - Mask semantics apply **per pass** (starting masked pixels return `full_sum * center_sample` each pass).

**Input prefiltering (optional)**:
- The core algorithm does not require prefiltering.
- For generated white noise, a low-pass prefilter is recommended to reduce aliasing artifacts.
- Prefiltering is the caller’s responsibility (or an optional preprocessing pass).

## 9) Boundary & mask handling
- **Domain boundary**: closed. Streamline truncates at boundary.
 - **Mask**: when entering a masked pixel (`mask != 0`), terminate integration.
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
- Periodic boundaries are **not implemented in v1** (reserved for later).

## 10) Sampling mode
- Vector field sampling: bilinear (for RK2).
- Input texture sampling: **linear (fixed for v1)**.
- Mask sampling: nearest (integer read).
- Noise sampling: wrap for generated noise; clamp for artist textures.

## 11) Precision
**Quality mode (v1)**:
- Vector field texture: **float32** (required).
- Input texture: **float32 recommended** (float16 allowed later as a fast mode).
- Output stored in float16.
- Integrate and accumulate in float32 to avoid artifacts.

## 12) Determinism
Given identical inputs and parameters, output must be deterministic.

## 13) Open decisions (TBD)
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
