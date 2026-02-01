# Implementation Plan

## M1: Core LIC (shader + host + basic test harness)

- [x] Project scaffolding — Swift package with Metal support, build config for macOS/Apple Silicon
- [x] Shared `LicParams` struct definition (Swift + MSL), matching Section 4.2 exactly
- [x] Kernel weight construction — Hann window, `steps = round(L/h)`, odd-length validation, `full_sum`/`center_weight` computation, `steps == 0` edge case
- [x] Metal shader — RK2 integration (forward + backward), bilinear vector field sampling, normalize-after-sample with `eps2` guard, center sample init, kernel indexing (`kmid ± step_count`)
- [x] Host encoder — compute pipeline creation, texture allocation (r32Float input, rg32Float vector field, r8Uint mask, r16Float output), sampler setup (linear clamp × 2, pixel coordinates), buffer binding per Section 4.2
- [x] Domain boundary checks — closed boundaries; check `x1` then `x_next` against `[0.5, W-0.5]`/`[0.5, H-0.5]` before sampling, set `hit_domain_edge` on first violation
- [x] Zero-vector handling (no advance, still count step + sample) and NaN/inf termination (stop before sample, no boundary hit recorded)
- [x] Function constants — declare `kMaskEnabled`, `kEdgeGainsEnabled`, `kDebugMode` as MSL function constants; build specialized pipeline variants at init; cache by configuration key
- [x] Debug visualization — `kDebugMode` function constant (0=off, 1=step count heat map, 2=boundary hit, 3=used_sum ratio); when active, output debug value instead of LIC result
- [x] Simple test harness — generate uniform vector field + white noise input, run LIC, write output image, visual check
- [x] Smoke test with vortex field to catch directional bias / spiral artifacts

**Exit criterion**: shader produces visually plausible LIC on uniform + vortex fields, no crashes. Function constants compile and select correct code paths.

## M2: Full spec compliance + validation

- [x] Mask texture support (r8Uint nearest read, starting-pixel masked path, stop-on-enter; check `mask[floor(x_next)]` before sampling)
- [x] Boundary processing — renormalization + edge gains (Section 9: support_factor, mask edge gain, domain edge gain)
- [x] Multi-pass convolution (ping-pong r16Float textures, per-pass mask semantics)
- [x] CPU reference implementation (spec-exact: RK2 + bilinear, same kernel, same boundary/mask rules, same `round(L/h)` and domain bounds)
- [x] GPU vs CPU comparison tests (MAE, max-abs error, f16 tolerances)
- [x] Deterministic unit tests on small grids (16×16) for: determinism, kernel indexing, zero-vector, NaN/inf, mask semantics (starting-pixel + stop-on-enter)
- [x] Deterministic unit tests for multi-pass (single-iteration equivalence, GPU/CPU match, differs-from-single, mask per-pass, determinism)
- [x] Deterministic unit tests for remaining invariants: boundary truncation (edgeGainsEnabled gating, GPU/CPU match with boundary processing disabled vs enabled)

**Exit criterion**: GPU matches CPU reference within f16 tolerance on all validation plan test scenes.

## M3: Performance + polish

- [x] Profile on M1 Pro at 2K and 4K
- [x] Threadgroup size tuning (start 8×8 / 16×16, measure)
- [x] Occupancy & register pressure analysis — all 8 pipeline variants at maxThreads=1024 (100% occupancy); no register pressure. 82% threadgroup-size spread confirms kernel is texture-latency bound, not register-limited. 32×32 threadgroups optimal. No register reduction needed.
- [x] Bandwidth analysis — Resolution scaling (1080p/2K/4K) + L-scaling (L=5..50) + cache efficiency (uniform vs vortex). 93 Gfetch/s uniform throughput is constant across resolutions (not bus-saturated). Vortex/uniform ratio 1.25–1.33× with throughput dropping 18% from L=5→50 on vortex fields. No-cache model shows 2478 GB/s (12× M1 Pro peak) confirming >90% cache reuse. Verdict: MIXED texture-latency + moderate cache pressure on divergent fields; NOT bandwidth-bound. Format tightening would improve cache efficiency; divergent-field penalty is inherent to dependent texture fetch chains.
- [x] Resource reuse — all 4 production pipeline variants (mask × edgeGains) pre-built at init; threadgroup sizes cached per-pipeline; samplers and dummy mask already at init; params/weights use setBytes (correct for <4KB data). No per-frame allocation in the dispatch path.
- [x] Multiple in-flight command buffers — `LICDispatcher` with semaphore-based pipelining (maxInFlight=3 default), per-frame ping-pong texture pools for multi-pass isolation, completion callbacks for GPU timing. `LICEncoder.encodeMultiPass` extended with optional `pingPongTextures` parameter for safe concurrent dispatch.
- [x] Warm-up dispatch — `LICDispatcher.warmUp()` primes GPU caches, TLB entries, and driver state with a synchronous single-frame dispatch before the real-time loop.
- [x] bryLIC parity checks (SSIM, histogram distance, error heatmaps — advisory only). Python generator (`scripts/generate_brylic_reference.py`) produces reference fixtures; Swift test (`LICParityTests`) computes SSIM (0.88–0.99), histogram chi-squared (<0.10), and per-pixel error stats across 14 scenes (6 baseline: uniform, vortex, saddle, radial, vortex_3pass, vortex_masked; 8 stress: vortex_2pass, vortex_edge_gain, radial_domain_gain, vortex_both_gains, radial_unnorm, shear, zero_patch, nan_patch). Divergence is expected due to RK2 vs pixel-crossing integration and bilinear vs nearest-neighbor field sampling. Run with `RUN_PARITY=1 swift test --filter LICParityTests`.

**Exit criterion**: 2K@60fps or 4K@~30fps on M1 Pro with default params (L=30, h=1.0, 1 iteration).

## Maybe / deferred

- [ ] r32Float ping-pong intermediates for multi-pass — avoids float16 quantization between passes, but the implicit float32→float16 texture-write conversion differs from MSL's `half()` cast (doubles single-pass error), and removing the float16 sync point causes GPU-vs-CPU arithmetic differences to compound ~fullSum× per pass. Requires either a per-pass comparison test methodology (GPU readback between passes) or a function-constant-controlled output conversion. Only worth pursuing if float16 intermediates cause visible banding artifacts in practice.
