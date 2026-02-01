# Validation Plan (v1)

This plan verifies that the Metal LIC implementation matches the v1 spec, is deterministic, and does not deviate visually from a trusted reference (bryLIC) beyond expected algorithmic differences.

## 1) Deterministic unit tests (spec invariants)
Small grids (e.g., 16x16, 32x32), fixed seeds, CPU reference values.

**Core invariants**:
- **Determinism**: identical inputs/params produce identical outputs (byte‑for‑byte).
- **Zero vector handling**: zero/near‑zero vectors do not advance; sampling remains at the current pixel.
- **NaN/inf handling**: terminate integration in that direction before sampling; no boundary hit recorded; no renorm/gain.
- **Boundary truncation**: closed boundaries stop integration and trigger renormalization/gains only when truncation is due to an actual boundary/mask hit.
- **Mask semantics**:
  - Starting masked pixel returns `full_sum * center_sample`.
  - Entering masked pixel stops integration and may trigger mask edge gain.
- **Kernel indexing**: `steps = round(L / h)`, `N = 2*steps + 1`, center at `kmid`.
- **Multi‑pass**: output of pass N becomes input to pass N+1; starting‑mask behavior applies per pass.

## 2) GPU vs CPU reference (spec‑aligned)
Implement a CPU reference that exactly matches the v1 spec (RK2 + bilinear sampling + kernel/edge rules).

**Comparison**:
- Use the same inputs and params as GPU.
- Compare outputs with tolerances appropriate for float16 output:
  - Report MAE and max‑abs error.
  - Also compare normalized outputs (divide by `full_sum`) to remove global scale.
- Fail only if errors exceed agreed thresholds.

## 3) bryLIC parity checks (warning system)
Use bryLIC as a *visual regression alarm*, not a strict correctness oracle, since it uses DDA stepping and cell‑constant vectors.

**Test scenes** (same inputs, kernels, lengths):
- Constant uniform field.
- Vortex.
- Saddle.
- Shear.
- Radial inflow/outflow.
- Random field.
- With and without masks.
- Domain boundary hits.

**Metrics**:
- SSIM / correlation on normalized outputs.
- Histogram distance (e.g., L1 or EMD).
- Error heatmaps for visual inspection.

**Policy**:
- Thresholds are advisory; large deviations trigger investigation.
- Accept that small differences are expected due to integration and sampling differences.

## 4) Artifacts checklist (qualitative)
- No directional bias or spiraling in uniform fields.
- No edge darkening unless edge gains are enabled.
- No visible banding at 2K/4K.
- No shimmer/flicker beyond intended noise animation.

