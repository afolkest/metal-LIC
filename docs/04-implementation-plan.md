# Implementation Plan

## M1: Core LIC (shader + host + basic test harness)

- [ ] Project scaffolding — Swift package with Metal support, build config for macOS/Apple Silicon
- [ ] Shared `LicParams` struct definition (Swift + MSL), matching Section 4.2 exactly
- [ ] Kernel weight construction — Hann window, `steps = round(L/h)`, odd-length validation, `full_sum`/`center_weight` computation, `steps == 0` edge case
- [ ] Metal shader — RK2 integration (forward + backward), bilinear vector field sampling, normalize-after-sample with `eps2` guard, center sample init, kernel indexing (`kmid ± step_count`)
- [ ] Host encoder — compute pipeline creation, texture allocation (r32Float input, rg32Float vector field, r8Uint mask, r16Float output), sampler setup (linear clamp × 2, pixel coordinates), buffer binding per Section 4.2
- [ ] Domain boundary checks — closed boundaries; check `x1` then `x_next` against `[0.5, W-0.5]`/`[0.5, H-0.5]` before sampling, set `hit_domain_edge` on first violation
- [ ] Zero-vector handling (no advance, still count step + sample) and NaN/inf termination (stop before sample, no boundary hit recorded)
- [ ] Simple test harness — generate uniform vector field + white noise input, run LIC, write output image, visual check
- [ ] Smoke test with vortex field to catch directional bias / spiral artifacts

**Exit criterion**: shader produces visually plausible LIC on uniform + vortex fields, no crashes.

## M2: Full spec compliance + validation

- [ ] Mask texture support (r8Uint nearest read, starting-pixel masked path, stop-on-enter; check `mask[floor(x_next)]` before sampling)
- [ ] Boundary processing — renormalization + edge gains (Section 9: support_factor, mask edge gain, domain edge gain)
- [ ] Multi-pass convolution (ping-pong r16Float textures, per-pass mask semantics)
- [ ] CPU reference implementation (spec-exact: RK2 + bilinear, same kernel, same boundary/mask rules, same `round(L/h)` and domain bounds)
- [ ] GPU vs CPU comparison tests (MAE, max-abs error, f16 tolerances)
- [ ] Deterministic unit tests on small grids (16×16, 32×32) for each invariant in validation plan Section 1

**Exit criterion**: GPU matches CPU reference within f16 tolerance on all validation plan test scenes.

## M3: Performance + polish

- [ ] Profile on M1 Pro at 2K and 4K
- [ ] Threadgroup size tuning (start 8×8 / 16×16, measure)
- [ ] Resource reuse — precreated pipelines, samplers, buffers; no per-frame allocation
- [ ] Multiple in-flight command buffers
- [ ] Warm-up dispatch
- [ ] bryLIC parity checks (SSIM, histogram distance, error heatmaps — advisory only)

**Exit criterion**: 2K@60fps or 4K@~30fps on M1 Pro with default params (L=30, h=1.0, 1 iteration).
