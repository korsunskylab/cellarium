"""Sigma sweep for the cytoplasm-tinting game on one benchmark mini-ROI.

Renders a 2×4 grid:
    [0,0] reference (18S + anchor scatter, focal cell outlined)
    [0,1..3], [1,0..3]   tinted-cytoplasm at 6 σ values from undersmoothed to oversmoothed

Output: figs/tinted_cytoplasm_sigma_sweep/<benchmark_id>.png

Usage:
    python tint_cytoplasm_sigma_sweep.py [benchmark_id]
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
FIGS_DIR    = ROOT / "figs" / "tinted_cytoplasm_sigma_sweep"

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
SIGMA_SWEEP_UM = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Posterior coloring parameters (Dirichlet-multinomial w/ uniform prior + grey
# abstain when local effective-evidence is below threshold). Tuned 2026-05-14
# against the visual feedback that σ=1µm + 1 transcript shouldn't tint a region.
ALPHA       = 0.5   # per-lineage Dirichlet pseudocount (smooths posterior)
N_MIN_EVIDENCE = 3.0  # local effective-transcript count below this → fade to grey
GREY = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def norm01(im, lo=1.0, hi=99.0):
    l, h = np.percentile(im, [lo, hi])
    return np.clip((im.astype(np.float32) - l) / max(h - l, 1e-6), 0, 1)


def hex_to_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)


def make_tint(rho_counts, sigma_px, colors):
    """Posterior tint with uniform Dirichlet prior + grey fade for low-evidence.

    rho_counts: (K, H, W) raw count rasters (point masses at transcript pixels).
    sigma_px:   Gaussian-filter scale.
    colors:     (K, 3) per-lineage RGB.

    Returns (H, W, 3) tinted image and (H, W) effective-N (interpretable
    'transcripts contributing to this pixel's estimate').
    """
    K = rho_counts.shape[0]
    rho = np.empty_like(rho_counts)
    for k in range(K):
        rho[k] = gaussian_filter(rho_counts[k], sigma=sigma_px)

    # Convert ρ (kernel density) to effective transcript count N at each pixel.
    # gaussian_filter preserves the integral, so multiplying by 2πσ² recovers
    # 'how many transcripts effectively contribute' as a peak-equivalent count.
    gauss_norm = 2.0 * np.pi * sigma_px ** 2
    N = rho * gauss_norm
    N_total = N.sum(axis=0)

    # Dirichlet-multinomial posterior mean with uniform prior:
    #   π_t = (N_t + α) / (N_total + K·α)
    pi_post = (N + ALPHA) / (N_total + K * ALPHA)[None]
    tint_post = np.einsum("khw,kc->hwc", pi_post, colors)

    # Confidence gate: fade to grey when effective evidence is below threshold.
    confidence = N_total / (N_total + N_MIN_EVIDENCE)
    tint_final = (confidence[..., None] * tint_post
                  + (1.0 - confidence[..., None]) * GREY[None, None, :])
    return tint_final, N_total


def main(benchmark_id):
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
    print(f"[{benchmark_id}] {len(anch)} anchored tx in crop  ({meta['lineage_pair']})")

    # Rasterized counts per lineage (reused across all σ)
    K = len(LINEAGES)
    lin_to_idx = {L: i for i, L in enumerate(LINEAGES)}
    li_arr = anch["lineage"].map(lin_to_idx).to_numpy()
    rho_counts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_counts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)

    colors = np.stack([hex_to_rgb(LINEAGE_COLOURS[L]) for L in LINEAGES])
    s18_n = norm01(s18); dapi_n = norm01(dapi)
    focal_mask = cpsam == cps_focal
    ext = [cx0, cx0 + W * PIXEL_SIZE_UM,
           cy0 + H * PIXEL_SIZE_UM, cy0]

    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    axes = axes.flatten()

    # Panel 0: 18S + anchors reference
    axes[0].imshow(s18_n, cmap="gray", extent=ext, aspect="equal")
    axes[0].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                    extent=ext, origin="upper")
    in_cell = cpsam[anch["py"].to_numpy(), anch["px"].to_numpy()] == cps_focal
    for li, L in enumerate(LINEAGES):
        # outside-cell faint
        sel = (~in_cell) & (li_arr == li)
        if sel.any():
            sub = anch.iloc[np.where(sel)[0]]
            axes[0].scatter(sub["x_um"], sub["y_um"], s=6,
                            c=LINEAGE_COLOURS[L], alpha=0.3, edgecolor="none")
    for li, L in enumerate(LINEAGES):
        sel = in_cell & (li_arr == li)
        if sel.any():
            sub = anch.iloc[np.where(sel)[0]]
            axes[0].scatter(sub["x_um"], sub["y_um"], s=22,
                            c=LINEAGE_COLOURS[L], edgecolor="black", linewidth=0.4,
                            label=f"{L} ({len(sub)})", alpha=0.95)
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.85)
    axes[0].set_title(f"reference: 18S + anchors\n{meta['lineage_pair']}  "
                       f"top1={meta['top1_n']} top2={meta['top2_n']}")

    # Panels 1..6: tinted at each σ (posterior with grey-abstain prior)
    for i, sigma_um in enumerate(SIGMA_SWEEP_UM):
        ax = axes[i + 1]
        sigma_px = sigma_um / PIXEL_SIZE_UM
        tint, _ = make_tint(rho_counts, sigma_px, colors)
        out_rgb = s18_n[..., None] * tint
        ax.imshow(out_rgb, extent=ext, aspect="equal")
        ax.contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                   extent=ext, origin="upper")
        flag = ""
        if sigma_um <= 0.5:   flag = "  (undersmoothed)"
        elif sigma_um >= 4.0: flag = "  (oversmoothed)"
        elif 0.75 <= sigma_um <= 1.5: flag = "  ← sweet spot"
        ax.set_title(f"σ = {sigma_um} µm{flag}")

    # Panel 7: lineage colour legend (one panel reserved)
    axes[7].axis("off")
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=LINEAGE_COLOURS[L],
                          markersize=14, label=L) for L in LINEAGES]
    axes[7].legend(handles=handles, loc="center", fontsize=12, frameon=False,
                   title="lineage colours", title_fontsize=13)

    for ax in axes[:7]:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"{benchmark_id}  —  σ sweep on tinted cytoplasm "
                 f"(rank {meta['rank']}, {meta['lineage_pair']})  ·  "
                 f"posterior with Dirichlet prior α={ALPHA}, "
                 f"grey ≤ N_eff={N_MIN_EVIDENCE} transcripts",
                 y=1.005)
    plt.tight_layout()
    out_png = FIGS_DIR / f"{benchmark_id}_sigma_sweep.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    args = ap.parse_args()
    main(args.benchmark_id)
