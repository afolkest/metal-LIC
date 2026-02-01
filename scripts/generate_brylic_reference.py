#!/usr/bin/env python3
"""Generate bryLIC reference data for metal-LIC parity checks.

Produces raw binary fixture files that the Swift parity test loads.
All inputs are generated deterministically so metal-LIC receives
identical data.

Usage:
    python3 scripts/generate_brylic_reference.py

Prerequisites:
    pip install -e /path/to/brylic   (or wherever brylic is installed)
"""

import json
import sys
from pathlib import Path

import numpy as np


def build_hann_kernel(L: float, h: float = 1.0) -> np.ndarray:
    """Build Hann kernel matching LICKernel.build(L:h:) exactly.

    steps = round(L / h)
    kernel_len = 2 * steps + 1
    kernel[i] = 0.5 * (1 + cos(pi * s_i / L))
    where s_i = (i - kmid) * h
    """
    steps = int(round(L / h))
    if steps == 0:
        return np.array([1.0], dtype=np.float32)

    kernel_len = 2 * steps + 1
    kmid = kernel_len // 2

    kernel = np.zeros(kernel_len, dtype=np.float32)
    for i in range(kernel_len):
        si = float(i - kmid) * h
        if abs(si) <= L:
            kernel[i] = np.float32(0.5 * (1.0 + np.cos(np.pi * si / L)))

    return kernel


def make_vector_fields(width: int, height: int) -> dict:
    """Generate test vector fields.

    Coordinate convention matches LICImageTests.swift:280-317:
    center = (width/2, height/2), offsets use (i - cx + 0.5).
    """
    cx = width / 2.0
    cy = height / 2.0

    fields = {}

    # Coordinate grids (row-major: j=row/y, i=col/x)
    j_coords = np.arange(height, dtype=np.float32)
    i_coords = np.arange(width, dtype=np.float32)
    ii, jj = np.meshgrid(i_coords, j_coords)

    x = ii - np.float32(cx) + np.float32(0.5)
    y = jj - np.float32(cy) + np.float32(0.5)
    r = np.sqrt(x * x + y * y).astype(np.float32)
    near_zero = r < 1e-6
    r_safe = np.where(near_zero, np.float32(1.0), r)

    # --- Unit-length fields ---

    # Uniform: (1, 0)
    fields["uniform"] = (
        np.ones((height, width), dtype=np.float32),
        np.zeros((height, width), dtype=np.float32),
    )

    # Vortex: (-y/r, x/r)
    fields["vortex"] = (
        np.where(near_zero, np.float32(0), -y / r_safe).astype(np.float32),
        np.where(near_zero, np.float32(0), x / r_safe).astype(np.float32),
    )

    # Saddle: (x/r, -y/r)
    fields["saddle"] = (
        np.where(near_zero, np.float32(0), x / r_safe).astype(np.float32),
        np.where(near_zero, np.float32(0), -y / r_safe).astype(np.float32),
    )

    # Radial: (x/r, y/r)
    fields["radial"] = (
        np.where(near_zero, np.float32(0), x / r_safe).astype(np.float32),
        np.where(near_zero, np.float32(0), y / r_safe).astype(np.float32),
    )

    # --- Non-unit / stress fields ---

    # Radial unnormalized: (x, y) raw — magnitude varies with distance from center
    fields["radial_unnorm"] = (
        np.where(near_zero, np.float32(0), x).astype(np.float32),
        np.where(near_zero, np.float32(0), y).astype(np.float32),
    )

    # Shear: sharp direction discontinuity at y = midpoint
    # Top half: (1, 0), bottom half: (-1, 0)
    shear_u = np.ones((height, width), dtype=np.float32)
    shear_u[height // 2:, :] = np.float32(-1.0)
    shear_v = np.zeros((height, width), dtype=np.float32)
    fields["shear"] = (shear_u, shear_v)

    # Zero patch: vortex field with 64x64 block of zeros at center
    zp_u, zp_v = fields["vortex"]
    zp_u = zp_u.copy()
    zp_v = zp_v.copy()
    cy_i, cx_i = height // 2, width // 2
    zp_u[cy_i - 32:cy_i + 32, cx_i - 32:cx_i + 32] = 0.0
    zp_v[cy_i - 32:cy_i + 32, cx_i - 32:cx_i + 32] = 0.0
    fields["zero_patch"] = (zp_u, zp_v)

    # NaN patch: vortex field with 64x64 block of NaN at center
    np_u, np_v = fields["vortex"]
    np_u = np_u.copy()
    np_v = np_v.copy()
    np_u[cy_i - 32:cy_i + 32, cx_i - 32:cx_i + 32] = np.nan
    np_v[cy_i - 32:cy_i + 32, cx_i - 32:cx_i + 32] = np.nan
    fields["nan_patch"] = (np_u, np_v)

    return fields


def make_circular_mask(width: int, height: int, radius_frac: float = 0.15) -> np.ndarray:
    """Circular mask (True = blocked) centered on image."""
    cx = width / 2.0
    cy = height / 2.0
    j_coords = np.arange(height, dtype=np.float32)
    i_coords = np.arange(width, dtype=np.float32)
    ii, jj = np.meshgrid(i_coords, j_coords)
    x = ii - np.float32(cx) + np.float32(0.5)
    y = jj - np.float32(cy) + np.float32(0.5)
    dist = np.sqrt(x * x + y * y)
    return dist < (radius_frac * min(width, height))


# Scene definitions — each scene carries its full config so the Swift test
# can be data-driven instead of parsing scene names.
SCENES = [
    # --- Original baseline scenes ---
    {"name": "uniform",       "field": "uniform", "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "vortex",        "field": "vortex",  "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "saddle",        "field": "saddle",  "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "radial",        "field": "radial",  "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "vortex_3pass",  "field": "vortex",  "iterations": 3, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "vortex_masked", "field": "vortex",  "iterations": 1, "mask": True,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},

    # --- Stress-test scenes ---
    {"name": "vortex_2pass",  "field": "vortex",  "iterations": 2, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "vortex_edge_gain", "field": "vortex", "iterations": 1, "mask": True,
     "edge_gain_strength": 0.5, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "radial_domain_gain", "field": "radial", "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.5, "domain_edge_gain_power": 2.0},
    {"name": "vortex_both_gains", "field": "vortex", "iterations": 1, "mask": True,
     "edge_gain_strength": 0.5, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.3, "domain_edge_gain_power": 2.0},
    {"name": "radial_unnorm", "field": "radial_unnorm", "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "shear",         "field": "shear",   "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "zero_patch",    "field": "zero_patch", "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
    {"name": "nan_patch",     "field": "nan_patch",  "iterations": 1, "mask": False,
     "edge_gain_strength": 0.0, "edge_gain_power": 2.0,
     "domain_edge_gain_strength": 0.0, "domain_edge_gain_power": 2.0},
]


def main():
    width, height = 512, 512
    L, h = 30.0, 1.0

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "Tests" / "MetalLICTests" / "Fixtures" / "brylic_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import brylic
    try:
        import brylic
    except ImportError:
        print("ERROR: brylic not installed. Install with:")
        print("  pip install -e ../brylic")
        sys.exit(1)

    # Generate deterministic noise (numpy PCG64 seed)
    rng = np.random.Generator(np.random.PCG64(12345))
    noise = rng.uniform(0.0, 1.0, size=(height, width)).astype(np.float32)
    noise.tofile(out_dir / "noise.bin")
    print(f"Noise: shape={noise.shape}, range=[{noise.min():.4f}, {noise.max():.4f}]")

    # Build Hann kernel
    kernel = build_hann_kernel(L, h)
    full_sum = float(kernel.sum())
    center_weight = float(kernel[len(kernel) // 2])
    kernel.tofile(out_dir / "kernel.bin")
    print(f"Kernel: len={len(kernel)}, full_sum={full_sum:.4f}, center={center_weight:.4f}")

    # Generate vector fields
    fields = make_vector_fields(width, height)
    for name, (u, v) in fields.items():
        u.tofile(out_dir / f"field_{name}_u.bin")
        v.tofile(out_dir / f"field_{name}_v.bin")
        has_nan = "NaN" if np.any(np.isnan(u)) or np.any(np.isnan(v)) else ""
        u_finite = u[np.isfinite(u)]
        v_finite = v[np.isfinite(v)]
        print(f"Field '{name}': u=[{u_finite.min():.3f},{u_finite.max():.3f}], "
              f"v=[{v_finite.min():.3f},{v_finite.max():.3f}] {has_nan}")

    # Generate mask
    mask = make_circular_mask(width, height)
    mask.astype(np.uint8).tofile(out_dir / "mask_circular.bin")
    print(f"Mask: {mask.sum()} blocked pixels ({100*mask.mean():.1f}%)")

    # Run bryLIC on each scene
    print()
    for scene in SCENES:
        name = scene["name"]
        u, v = fields[scene["field"]]
        iterations = scene["iterations"]
        mask_arr = mask if scene["mask"] else None

        egs = scene["edge_gain_strength"]
        egp = scene["edge_gain_power"]
        degs = scene["domain_edge_gain_strength"]
        degp = scene["domain_edge_gain_power"]

        gains_str = ""
        if egs > 0:
            gains_str += f" egs={egs}"
        if degs > 0:
            gains_str += f" degs={degs}"

        print(f"Running bryLIC: {name} ({width}x{height}, L={L}, "
              f"iters={iterations}{gains_str})...", end=" ")

        result = brylic.convolve(
            noise,
            u, v,
            kernel=kernel,
            uv_mode="velocity",
            boundaries="closed",
            iterations=iterations,
            mask=mask_arr,
            edge_gain_strength=egs,
            edge_gain_power=egp,
            domain_edge_gain_strength=degs,
            domain_edge_gain_power=degp,
        )

        result.astype(np.float32).tofile(out_dir / f"brylic_{name}.bin")
        print(f"range=[{result.min():.4f}, {result.max():.4f}], mean={result.mean():.4f}")

    # Save metadata with per-scene config
    metadata = {
        "width": width,
        "height": height,
        "L": L,
        "h": h,
        "steps": int(round(L / h)),
        "kernel_len": len(kernel),
        "full_sum": full_sum,
        "center_weight": center_weight,
        "noise_seed": 12345,
        "noise_rng": "numpy.random.PCG64",
        "scenes": [
            {
                "name": s["name"],
                "field": s["field"],
                "iterations": s["iterations"],
                "mask": s["mask"],
                "edge_gain_strength": s["edge_gain_strength"],
                "edge_gain_power": s["edge_gain_power"],
                "domain_edge_gain_strength": s["domain_edge_gain_strength"],
                "domain_edge_gain_power": s["domain_edge_gain_power"],
            }
            for s in SCENES
        ],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFixtures written to: {out_dir}")
    print(f"Scenes ({len(SCENES)}): {[s['name'] for s in SCENES]}")


if __name__ == "__main__":
    main()
