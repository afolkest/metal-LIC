# Algorithm Spec (v1)

This document specifies the v1 Line Integral Convolution (LIC) algorithm for Metal.

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
 - **Input source**: the input texture may be GPU-generated or CPU-provided (see Section 3.1).
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
  - `noise_sample_mode`: clamp (fixed for v1; no tiling).
  - `input_sample_mode`: **linear (fixed for v1)**.
  - `precision_mode`: float16 output; float32 accumulation/integration.
  - `iterations`: number of convolution passes (>= 1).

## 3.1) Input texture sources
The LIC kernel consumes a GPU `r32Float` input texture with **non-negative** values.

**Allowed sources**:
- **GPU producer (recommended for real-time)**: any compute pass (e.g., cellular automata) that writes to the input texture. The producer should encode into the same command buffer and output a GPU-resident `r32Float` texture.
- **CPU-provided texture (simple/offline)**: upload to a GPU `r32Float` texture before running LIC. This is acceptable for occasional updates; avoid per-frame full-size uploads for real-time use.

**Producer contract (reference)**:
The producer writes into a provided GPU texture (no CPU readback) and can be chained directly before LIC.
```swift
protocol TextureProducer {
    var outputTexture: MTLTexture { get }
    func encode(commandBuffer: MTLCommandBuffer, time: Float)
}

// Example usage:
producer.encode(commandBuffer: cb, time: t)
licEncoder.encode(commandBuffer: cb, inputTexture: producer.outputTexture)
```

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

## 4.2) Resource bindings (v1)
These bindings are fixed for v1 to avoid ambiguity across host/shader code.

**Textures**:
- `texture(0)`: input texture (`r32Float`, sample access).
- `texture(1)`: vector field (`rg32Float`, sample access).
- `texture(2)`: mask (`r8Uint`, read access). If masking is disabled, bind a 1x1 zero mask texture and use the same code path.
- `texture(3)`: output (`r16Float`, write access).

**Samplers**:
- `sampler(0)`: input sampler (linear clamp; no tiling).
- `sampler(1)`: vector sampler (linear clamp).

**Buffers**:
- `buffer(0)`: params struct `LicParams` (read-only).
- `buffer(1)`: kernel weights array `float kernel[kernel_len]` (read-only).

**Function constants**:
Use Metal function constants for compile-time specialization of code paths that are uniform across a dispatch. The compiler eliminates dead branches entirely, avoiding per-pixel runtime cost.
- `function_constant(0)`: `kMaskEnabled` (`bool`) — controls mask texture reads and mask boundary logic.
- `function_constant(1)`: `kEdgeGainsEnabled` (`bool`) — controls edge gain multipliers in boundary processing (Section 9). Renormalization itself is unconditional when a boundary/mask truncates the kernel.
- `function_constant(2)`: `kDebugMode` (`uint`) — selects debug visualization output (0 = off / normal LIC, 1 = step count heat map, 2 = boundary hit visualization, 3 = used_sum / full_sum ratio).

The host builds specialized `MTLComputePipelineState` variants for each active combination and caches them at init time. When masking is disabled, a 1x1 zero mask texture is still bound (keeps the signature uniform) but the shader skips mask reads entirely.

**MSL signature (reference)**:
```metal
// Function constants (compile-time specialization)
constant bool kMaskEnabled      [[function_constant(0)]];
constant bool kEdgeGainsEnabled [[function_constant(1)]];
constant uint kDebugMode        [[function_constant(2)]];

struct LicParams {
    float h;
    float eps2;
    uint  steps;
    uint  kmid;
    uint  kernel_len;
    float full_sum;
    float center_weight;
    float edge_gain_strength;
    float edge_gain_power;
    float domain_edge_gain_strength;
    float domain_edge_gain_power;
};

kernel void licKernel(
    texture2d<float, access::sample>  inputTex   [[texture(0)]],
    texture2d<float, access::sample>  vectorTex  [[texture(1)]],
    texture2d<uint, access::read>     maskTex    [[texture(2)]],
    texture2d<half, access::write>    outputTex  [[texture(3)]],
    constant LicParams&               params     [[buffer(0)]],
    constant float*                   kernel     [[buffer(1)]],
    sampler                           inputSamp  [[sampler(0)]],
    sampler                           vectorSamp [[sampler(1)]],
    uint2                             gid        [[thread_position_in_grid]]
);
```

## 5) Coordinate mapping
**v1 mapping is 1:1**: vector field, input texture, mask, and output share the same resolution.
- Output pixel centers are at `(x + 0.5, y + 0.5)` in texture space.
- Image space uses the standard image convention (x right, y down).
- Vector components are interpreted in **pixel units** (direction-only); `h` is measured in pixels.

**Scaling/transforming fields is out of core scope**:
- If a different resolution or zoom is desired, resample the vector field (and mask) to the output resolution in a wrapper before invoking the core LIC.

## 5.1) Sampling coordinates (canonical)
- All positions `x` are in **pixel coordinates**; pixel `(i, j)` center is `(i + 0.5, j + 0.5)`.
- Samplers must use **pixel coordinates** (`normalizedCoordinates = false` / `coord::pixel`).
  - If normalized sampling is used, convert with `uv = x / float2(width, height)` before sampling.
- Valid domain for sampling/integration is `x in [0.5, width - 0.5]` and `y in [0.5, height - 0.5]`.
  - Leaving this domain is a **domain boundary hit** (closed boundary behavior).

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

## 6.1) Parameter validity (canonical)
- Require `L > 0` and `h > 0` (wrapper error otherwise).
- `steps = round(L / h)` with ties **away from zero** (equivalently `floor(L / h + 0.5)` for positive inputs).
- If `steps == 0`, set `kernel_len = 1`, `kernel[0] = 1`, and `full_sum = center_weight = 1`.

## 7) Streamline integration
For each output pixel:
1. Let `x0` be the output pixel center in texture coordinates.
2. Initialize accumulation with the center sample at `x0`:
   - `value = kernel[kmid] * sample(inputTex, x0)`
   - `used_sum = kernel[kmid]`
3. Integrate **forward** and **backward** along `v(x)` using RK2 (midpoint). RK4 is deferred to post‑v1:
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

## 7.1) Per-step boundary & mask checks (canonical order)
For each step `step_count = 1..steps` in a direction:
1. Compute `x1 = x + 0.5 * h * v(x)`. If `x1` is outside the domain, set `hit_domain_edge = true` and stop (no sample).
2. Compute `x_next = x + h * v(x1)`. If `x_next` is outside the domain, set `hit_domain_edge = true` and stop (no sample).
3. Mask lookup uses pixel mapping: `mask_idx = uint2(floor(x_next))`.
   - If `mask[mask_idx] != 0`, set `hit_mask_edge = true` and stop (no sample).
4. Sample input at `x_next`, accumulate, and update `used_sum`.
5. Set `x = x_next`.
   - Even if `x_next == x` (zero/near-zero vector), still count the step and sample at the current pixel to ensure termination.

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
- Multi-pass is supported in v1. If `iterations > 1`, run the LIC pass repeatedly.
- Each pass uses the **previous pass output** as the new input texture.
- Use ping‑pong textures to avoid read/write hazards.
- Ping‑pong textures use `r16Float` (same as output). Metal promotes `r16Float` to float32 automatically when sampled through `texture2d<float, access::sample>`, so no special handling is needed. This precision is acceptable for v1 iteration counts.
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
- `needs_boundary_processing = (used_sum < full_sum)`
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
- Renormalization (step 2) is **always** applied when a boundary/mask truncates the kernel, regardless of `kEdgeGainsEnabled`. Edge gain multipliers (steps 3–4) are gated by `kEdgeGainsEnabled`.
- If truncation happens for other reasons (e.g., NaNs in the vector field), no renormalization is performed.
- Periodic boundaries are **not implemented in v1** (reserved for later).

## 10) Sampling mode
- Vector field sampling: bilinear (for RK2).
- Input texture sampling: **linear clamp (fixed for v1; no tiling)**.
- Mask sampling: nearest (integer read).

## 11) Precision
**Quality mode (v1)**:
- Vector field texture: **float32** (required).
- Input texture: **float32 recommended** (float16 allowed later as a fast mode).
- Output stored in float16.
- Integrate and accumulate in float32 to avoid artifacts.

## 12) Determinism
Given identical inputs and parameters, output must be deterministic.

## 13) Real-time defaults & performance notes
**Defaults for real-time (v1)**:
- `L`: ~30 px at 1024-scale (kernel half-length; full length ~60 px).
- `h`: 1.0 px (0.5 px reserved for "ultra").
- `iterations`: 1.
- `edge_gain_strength` / `domain_edge_gain_strength`: 0 (off).
- Precision: float32 integration/accumulation; float16 output.

**Performance guidance (implementation)**:
- Precreate and reuse compute pipelines, samplers, buffers, and textures.
- Keep textures GPU-resident; avoid per-frame CPU readbacks.
- Use multiple in-flight command buffers to avoid CPU-GPU sync stalls.
- Prefer GPU noise animation (procedural or small seed updates) over full texture uploads.
- CPU-provided textures are acceptable for simple use, but avoid per-frame full-size uploads for real-time.
- Run a warm-up dispatch at startup to avoid first-frame timing spikes.
- Threadgroup size: use the largest square that fits `maxTotalThreadsPerThreadgroup` (32×32 = 1024 on M1 Pro). Larger threadgroups hide texture-fetch latency from divergent streamlines; measured 39–65% faster than 8×8 on vortex fields. Uniform fields are bandwidth-limited and insensitive to threadgroup size.
- Use function constants to eliminate dead code paths (mask, edge gains, debug) at compile time; cache specialized pipelines.
- Profile occupancy and register pressure via GPU capture; reduce live registers if occupancy is low.
- Check ALU-bound vs memory-bound limiter at 4K; streamline walks diverge and may stress texture cache.

## 14) Debug visualization
The shader supports a compile-time debug mode via the `kDebugMode` function constant (Section 4.2). When `kDebugMode != 0`, the shader writes a debug value to the output instead of the LIC result:

| `kDebugMode` | Output | Purpose |
|---|---|---|
| 0 | Normal LIC value | Production |
| 1 | Step count (forward + backward) / (2 * steps) | Integration length heat map — reveals early termination |
| 2 | Boundary hit flags (0.0 = none, 0.5 = mask only, 0.75 = domain only, 1.0 = both) | Boundary hit visualization |
| 3 | used_sum / full_sum | Kernel support ratio — reveals truncation extent |

Debug pipelines are built at init alongside production pipelines using the same function constant mechanism. Debug output is float16 in [0, 1] for direct visualization.

## 15) Validation
See `docs/03-validation-plan.md` for the v1 validation plan (CPU reference, GPU vs CPU checks, and bryLIC parity warnings).

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
