"""18S-conductance anisotropic diffusion (vs isotropic Gaussian KDE) for the
tinted-cytoplasm game.

The KDE step is replaced by N iterations of 4-neighbor weighted diffusion
where edge conductance c(p,q) = s18(p) · s18(q). The downstream Dirichlet
posterior + grey-abstain stays identical, so the only thing that changes
is *how* the per-lineage densities are smoothed:

    isotropic Gaussian → leaks across 18S valleys between adjacent cells
    anisotropic 18S-diffusion → color floods within a bright cytoplasm region,
                                 stops at dim valleys

Output: 2×3 grid comparison at σ≈1 µm and σ≈2 µm equivalents, plus the
reference 18S+anchor scatter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
ROIS_DIR    = DATA / "benchmark_doublet_rois"
FIGS_DIR    = ROOT / "figs" / "tinted_cytoplasm_diffusion"

PIXEL_SIZE_UM = 0.2125
LINEAGES = ["Melanoma", "Myeloid", "Tcell", "Plasma", "Fibroblast", "Endothelial", "Keratinocyte"]
LINEAGE_COLOURS = {
    "Melanoma":     "#E41A1C",
    "Myeloid":      "#FF7F00",
    "Tcell":        "#377EB8",
    "Plasma":       "#984EA3",
    "Fibroblast":   "#4DAF4A",
    "Endothelial":  "#00CED1",
    "Keratinocyte": "#F781BF",
}

# Posterior + grey-abstain hyperparameters (same as Gaussian version)
ALPHA          = 0.5
N_MIN_EVIDENCE = 3.0
GREY = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def norm01(im, lo=1.0, hi=99.0):
    l, h = np.percentile(im, [lo, hi])
    return np.clip((im.astype(np.float32) - l) / max(h - l, 1e-6), 0, 1)


def hex_to_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)


def make_conductance(s18_raw, mode="linear"):
    """Convert raw 18S intensity → per-pixel conductance ∈ [0, 1].

    mode='linear': 1-99 percentile clip → linear rescale to [0, 1].
                   Edge weight c(p)·c(q) is then product of linear values.
    mode='log'   : log1p(s18) → 1-99 percentile clip → linear rescale to [0, 1].
                   Captures log-normal structure of fluorescence intensities,
                   compressing the bright tail and stretching the dim end.
                   Approximately equivalent to using geometric-mean edge weights
                   on linearly-rescaled values.
    """
    if mode == "linear":
        arr = s18_raw.astype(np.float32)
    elif mode == "log":
        arr = np.log1p(s18_raw.astype(np.float32))
    else:
        raise ValueError(f"unknown conductance mode: {mode}")
    lo, hi = np.percentile(arr, [1, 99])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def diffuse_anisotropic(rho, conductance, n_iter, dt=0.2):
    """4-neighbor anisotropic diffusion with per-edge conductance c(p)·c(q).

    rho: (K, H, W) float32 — per-lineage density to smooth
    conductance: (H, W) float32 in [0, 1]
    n_iter: number of diffusion steps; effective σ_px = sqrt(n_iter · dt) in
            homogeneous-conductance regions. In low-conductance regions
            (dim 18S valleys), the local σ_eff is much smaller — exactly the
            desired anisotropy.
    dt: step size, must be < 1/4 for 4-connected stability. 0.2 is safe.

    Boundary handling: zero-flux Neumann (no transport across image edge).
    Implemented by zeroing edge conductances at the boundary, which prevents
    the np.roll wrap-around from leaking mass.

    Conservation of mass: each edge contributes equal-and-opposite to its two
    endpoints, so global Σ rho is exactly preserved.
    """
    c = conductance
    # Edge conductances (between p and each of its 4 neighbours):
    # c_right[i, j] is the conductance of the edge from (i, j) → (i, j+1)
    c_right = c * np.roll(c, -1, axis=-1)
    c_left  = c * np.roll(c,  1, axis=-1)
    c_down  = c * np.roll(c, -1, axis=-2)
    c_up    = c * np.roll(c,  1, axis=-2)
    # Zero-flux boundary: kill the edges that wrap around the image extent
    c_right[..., :, -1] = 0.0   # last column has no right neighbour
    c_left[...,  :,  0] = 0.0   # first column has no left neighbour
    c_down[...,  -1, :] = 0.0   # last row has no down neighbour
    c_up[...,    0,  :] = 0.0   # first row has no up neighbour

    for _ in range(n_iter):
        flux  = (np.roll(rho, -1, axis=-1) - rho) * c_right
        flux += (np.roll(rho,  1, axis=-1) - rho) * c_left
        flux += (np.roll(rho, -1, axis=-2) - rho) * c_down
        flux += (np.roll(rho,  1, axis=-2) - rho) * c_up
        rho = rho + dt * flux
    return rho


def make_tint_from_rho(rho_smoothed, sigma_eff_px, colors,
                       alpha=ALPHA, n_min=N_MIN_EVIDENCE):
    """Apply Dirichlet posterior + grey-abstain to a smoothed lineage density.

    rho_smoothed: (K, H, W) — per-lineage density after any smoothing
    sigma_eff_px: effective σ in pixels (for converting ρ → effective N)
    Returns (H, W, 3) tint and (H, W) effective N_total.
    """
    K = rho_smoothed.shape[0]
    gauss_norm = 2.0 * np.pi * sigma_eff_px ** 2
    N = rho_smoothed * gauss_norm
    N_total = N.sum(axis=0)

    pi_post = (N + alpha) / (N_total + K * alpha)[None]
    tint_post = np.einsum("khw,kc->hwc", pi_post, colors)
    confidence = N_total / (N_total + n_min)
    tint_final = (confidence[..., None] * tint_post
                  + (1.0 - confidence[..., None]) * GREY[None, None, :])
    return tint_final, N_total


def main(benchmark_id, sigma_um_list=(1.0, 2.0)):
    roi = ROIS_DIR / benchmark_id
    assert roi.exists(), f"missing ROI: {roi}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    dapi = tifffile.imread(roi / "morphology_DAPI.tif")
    s18  = tifffile.imread(roi / "morphology_18S.tif")
    cpsam = tifffile.imread(roi / "cells_cpsam_masks.tif")
    tx   = pd.read_parquet(roi / "transcripts.parquet")
    meta = json.loads((roi / "metadata.json").read_text())
    H, W = s18.shape
    cx0 = meta["bbox_crop_um"]["x0"]; cy0 = meta["bbox_crop_um"]["y0"]
    cps_focal = meta["cps_id_at_bookmark"]

    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    tx["gene"] = [gene_names[g] for g in tx["gene_id"].values]
    tx["lineage"] = tx["gene"].map(gl)
    anch = tx[tx["lineage"].isin(LINEAGES)].copy()
    anch["px"] = np.rint((anch["x_um"] - cx0) / PIXEL_SIZE_UM).astype(int)
    anch["py"] = np.rint((anch["y_um"] - cy0) / PIXEL_SIZE_UM).astype(int)
    anch = anch[(anch["px"] >= 0) & (anch["px"] < W) &
                (anch["py"] >= 0) & (anch["py"] < H)].copy()
    print(f"[{benchmark_id}] {len(anch)} anchored tx ({meta['lineage_pair']})")

    # Point-mass per-lineage raster (K, H, W)
    K = len(LINEAGES)
    lin_to_idx = {L: i for i, L in enumerate(LINEAGES)}
    li_arr = anch["lineage"].map(lin_to_idx).to_numpy()
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)

    colors = np.stack([hex_to_rgb(LINEAGE_COLOURS[L]) for L in LINEAGES])
    s18_n     = norm01(s18); dapi_n = norm01(dapi)
    c_linear  = make_conductance(s18, mode="linear")
    c_log     = make_conductance(s18, mode="log")
    focal_mask = cpsam == cps_focal
    ext = [cx0, cx0 + W * PIXEL_SIZE_UM,
           cy0 + H * PIXEL_SIZE_UM, cy0]

    # Quick check of intensity distribution (printed, not plotted in main fig)
    pos = s18[s18 > 0].astype(np.float32)
    if pos.size:
        import scipy.stats as ss
        log_skew  = float(ss.skew(np.log1p(pos)))
        lin_skew  = float(ss.skew(pos))
        print(f"  s18 skew  (linear vs log1p): {lin_skew:+.2f}  vs  {log_skew:+.2f}")
        print("  → log scaling brings 18S much closer to Gaussian if |log_skew| < |lin_skew|")

    # Build per-σ × per-mode rendering
    DT = 0.2
    results = {}   # (sigma_um, mode) → out_rgb
    for sigma_um in sigma_um_list:
        sigma_px = sigma_um / PIXEL_SIZE_UM
        n_iter = int(round(sigma_px ** 2 / DT))

        # Gaussian KDE baseline (single, shared across mode columns)
        rho_g = rho_pts.copy()
        for k in range(K):
            rho_g[k] = gaussian_filter(rho_g[k], sigma=sigma_px)
        tint_g, _ = make_tint_from_rho(rho_g, sigma_px, colors)
        results[(sigma_um, "gaussian")] = s18_n[..., None] * tint_g

        # Anisotropic diffusion — linear conductance
        rho_d = diffuse_anisotropic(rho_pts.copy(), c_linear, n_iter=n_iter, dt=DT)
        tint_d, _ = make_tint_from_rho(rho_d, sigma_px, colors)
        results[(sigma_um, "linear")] = s18_n[..., None] * tint_d

        # Anisotropic diffusion — log conductance
        rho_d = diffuse_anisotropic(rho_pts.copy(), c_log, n_iter=n_iter, dt=DT)
        tint_d, _ = make_tint_from_rho(rho_d, sigma_px, colors)
        results[(sigma_um, "log")] = s18_n[..., None] * tint_d

        print(f"  σ={sigma_um}µm  ↔  N_iter={n_iter}")

    # Figure: 3 rows × (1 reference + n_sigmas) cols
    n_sigmas = len(sigma_um_list)
    fig, axes = plt.subplots(3, 1 + n_sigmas, figsize=(5 + 5 * n_sigmas, 15))

    # Column 0 (reference): row 0 = 18S + scatter; row 1 = c_linear; row 2 = c_log
    axes[0, 0].imshow(s18_n, cmap="gray", extent=ext, aspect="equal")
    axes[0, 0].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                       extent=ext, origin="upper")
    in_cell = cpsam[anch["py"].to_numpy(), anch["px"].to_numpy()] == cps_focal
    for li, L in enumerate(LINEAGES):
        sel = (~in_cell) & (li_arr == li)
        if sel.any():
            sub = anch.iloc[np.where(sel)[0]]
            axes[0, 0].scatter(sub["x_um"], sub["y_um"], s=6,
                               c=LINEAGE_COLOURS[L], alpha=0.3, edgecolor="none")
    for li, L in enumerate(LINEAGES):
        sel = in_cell & (li_arr == li)
        if sel.any():
            sub = anch.iloc[np.where(sel)[0]]
            axes[0, 0].scatter(sub["x_um"], sub["y_um"], s=22,
                               c=LINEAGE_COLOURS[L], edgecolor="black", linewidth=0.4,
                               label=f"{L} ({len(sub)})", alpha=0.95)
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.85)
    axes[0, 0].set_title(f"reference: 18S + anchors\n{meta['lineage_pair']}  "
                          f"top1={meta['top1_n']} top2={meta['top2_n']}")
    axes[1, 0].imshow(c_linear, cmap="gray", extent=ext, aspect="equal", vmin=0, vmax=1)
    axes[1, 0].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                       extent=ext, origin="upper")
    axes[1, 0].set_title("c_linear  (1-99% clip → linear)")
    axes[2, 0].imshow(c_log, cmap="gray", extent=ext, aspect="equal", vmin=0, vmax=1)
    axes[2, 0].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                       extent=ext, origin="upper")
    axes[2, 0].set_title("c_log  (log1p → 1-99% clip)")

    # σ columns: row 0 = Gaussian KDE, row 1 = diffusion-linear, row 2 = diffusion-log
    def flag(s):
        if s <= 0.5:   return "  (under)"
        if s >= 4.0:   return "  (over)"
        if 0.75 <= s <= 1.5: return "  ← sweet"
        return ""
    for ci, sigma_um in enumerate(sigma_um_list):
        tag = flag(sigma_um)
        axes[0, 1 + ci].imshow(results[(sigma_um, "gaussian")], extent=ext, aspect="equal")
        axes[0, 1 + ci].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                                 extent=ext, origin="upper")
        axes[0, 1 + ci].set_title(f"Gaussian KDE  σ={sigma_um} µm{tag}")
        axes[1, 1 + ci].imshow(results[(sigma_um, "linear")], extent=ext, aspect="equal")
        axes[1, 1 + ci].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                                 extent=ext, origin="upper")
        axes[1, 1 + ci].set_title(f"diffusion · c_linear  σ_eff={sigma_um} µm{tag}")
        axes[2, 1 + ci].imshow(results[(sigma_um, "log")], extent=ext, aspect="equal")
        axes[2, 1 + ci].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                                 extent=ext, origin="upper")
        axes[2, 1 + ci].set_title(f"diffusion · c_log  σ_eff={sigma_um} µm{tag}")

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"{benchmark_id} — Gaussian KDE vs 18S-anisotropic diffusion "
                 f"(both with Dirichlet posterior α={ALPHA}, "
                 f"grey ≤ N_eff={N_MIN_EVIDENCE})", y=1.001)
    plt.tight_layout()
    out_png = FIGS_DIR / f"{benchmark_id}_diffusion_vs_kde.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0])
    args = ap.parse_args()
    main(args.benchmark_id, args.sigmas)
