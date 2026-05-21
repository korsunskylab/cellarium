"""Inspect 3D Z-structure of anchored transcripts inside one benchmark mini-ROI.

The local data snapshot only has the 2D EDF morphology projection (no per-Z
DAPI/18S stacks), so the morphology background here is fixed across Z bins.
But the transcripts are Z-resolved (z_um), so we can bin them by Z plane to
see whether the lineage pattern shifts through depth.

Renders a 2-row figure:
  Top:    standard 2-panel reference  (DAPI+18S overlay | 18S + ALL anchors)
  Bottom: 3 panels (one per Z plane)  (18S + anchors in that Z bin only)

Z planes are defined from the acquisition step (3 µm). Default bin edges span
the empirical Z range of transcripts inside the focal cell.

Usage:
    python inspect_doublet_z.py [benchmark_id]
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
ROIS_DIR    = DATA / "benchmark_doublet_rois"
FIGS_DIR    = ROOT / "figs" / "zstack_inspection"

PIXEL_SIZE_UM = 0.2125
N_ZBINS = 9   # Xenium Prime acquires ~9 imaging Z planes per cycle
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


def norm01(im, lo=1.0, hi=99.0):
    l, h = np.percentile(im, [lo, hi])
    return np.clip((im.astype(np.float32) - l) / max(h - l, 1e-6), 0, 1)


def scatter_anchors(ax, anch_sub, in_cell_mask, lineages_list, colours,
                    in_cell_size=24, out_cell_size=6):
    li_arr = anch_sub["lineage"].map({L: i for i, L in enumerate(lineages_list)}).to_numpy()
    for li, L in enumerate(lineages_list):
        sel = (~in_cell_mask) & (li_arr == li)
        if sel.any():
            sub = anch_sub.iloc[np.where(sel)[0]]
            ax.scatter(sub["x_um"], sub["y_um"], s=out_cell_size,
                       c=colours[L], alpha=0.3, edgecolor="none")
    for li, L in enumerate(lineages_list):
        sel = in_cell_mask & (li_arr == li)
        if sel.any():
            sub = anch_sub.iloc[np.where(sel)[0]]
            ax.scatter(sub["x_um"], sub["y_um"], s=in_cell_size,
                       c=colours[L], edgecolor="black", linewidth=0.4,
                       label=f"{L} ({len(sub)})", alpha=0.95)


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
    anch["in_focal"] = cpsam[anch["py"].to_numpy(), anch["px"].to_numpy()] == cps_focal

    # Z bins from acquisition step (3 µm). Span the data with N_ZBINS equal-width bins.
    z_focal = anch.loc[anch["in_focal"], "z_um"]
    z_min, z_max = float(z_focal.min()), float(z_focal.max())
    edges = np.linspace(z_min, z_max, N_ZBINS + 1)
    print(f"[{benchmark_id}] focal-cell anchored tx Z range: [{z_min:.2f}, {z_max:.2f}] µm")
    print(f"  bin edges: {[round(float(e), 2) for e in edges]}")

    s18_n  = norm01(s18); dapi_n = norm01(dapi)
    focal_mask = cpsam == cps_focal
    ext = [cx0, cx0 + W * PIXEL_SIZE_UM,
           cy0 + H * PIXEL_SIZE_UM, cy0]

    # Layout: row 0 = 2 reference panels (DAPI+18S | 18S+all-anchors);
    # rows 1..ceil(N_ZBINS/3) = grid of Z bins (3 per row).
    ncols = 3
    nrows_z = (N_ZBINS + ncols - 1) // ncols   # ceil division
    nrows_total = 1 + nrows_z
    fig = plt.figure(figsize=(ncols * 5, nrows_total * 5))
    gs = fig.add_gridspec(nrows_total, ncols, hspace=0.28, wspace=0.10)

    # Top-left: DAPI (blue) + 18S (yellow)
    ax_topleft = fig.add_subplot(gs[0, 0])
    rgb_overlay = np.stack([s18_n, s18_n, dapi_n], axis=-1)
    ax_topleft.imshow(rgb_overlay, extent=ext, aspect="equal")
    ax_topleft.contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                       extent=ext, origin="upper")
    ax_topleft.set_title(f"DAPI (blue) + 18S (yellow)\n"
                          f"cps_id={cps_focal}, {meta['lineage_pair']}")

    # Top-middle: 18S + ALL anchors (reference)
    ax_topright = fig.add_subplot(gs[0, 1])
    ax_topright.imshow(s18_n, cmap="gray", extent=ext, aspect="equal")
    ax_topright.contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                        extent=ext, origin="upper")
    scatter_anchors(ax_topright, anch, anch["in_focal"].to_numpy(),
                    LINEAGES, LINEAGE_COLOURS)
    ax_topright.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_topright.set_title(f"18S + ALL anchors (any z)\n"
                           f"top1={meta['top1_n']} top2={meta['top2_n']}  "
                           f"ratio={meta['top2_n']/max(meta['top1_n'],1):.2f}")

    # Hide the extra cell in row 0
    for col in range(2, ncols):
        ax_empty = fig.add_subplot(gs[0, col])
        ax_empty.axis("off")

    # Per-Z-bin scatter (rows 1..nrows_total-1)
    z_bins_summary = []
    for bi in range(N_ZBINS):
        z_lo = edges[bi]; z_hi = edges[bi + 1]
        if bi == N_ZBINS - 1:
            in_bin = (anch["z_um"] >= z_lo) & (anch["z_um"] <= z_hi)
        else:
            in_bin = (anch["z_um"] >= z_lo) & (anch["z_um"] < z_hi)
        sub = anch[in_bin].copy()

        r = 1 + bi // ncols
        c = bi % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(s18_n, cmap="gray", extent=ext, aspect="equal")
        ax.contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                   extent=ext, origin="upper")
        scatter_anchors(ax, sub, sub["in_focal"].to_numpy(),
                        LINEAGES, LINEAGE_COLOURS,
                        in_cell_size=22, out_cell_size=5)
        in_focal_sub = sub[sub["in_focal"]]
        n_focal = len(in_focal_sub)
        lin_focal = in_focal_sub["lineage"].value_counts().to_dict()
        # Compact label: "{lineage}:n, {lineage}:n"
        lin_str = ", ".join(f"{L[:4]}:{n}" for L, n in
                            sorted(lin_focal.items(), key=lambda kv: -kv[1])[:3])
        ax.set_title(f"z ∈ [{z_lo:.2f}, {z_hi:.2f}] µm\n"
                     f"in-focal: {n_focal} anchors  ({lin_str})", fontsize=9)
        z_bins_summary.append((z_lo, z_hi, n_focal, lin_focal))

    # Hide unused cells in the last Z-bin row
    for slot in range(N_ZBINS, nrows_z * ncols):
        r = 1 + slot // ncols
        c = slot % ncols
        ax_empty = fig.add_subplot(gs[r, c])
        ax_empty.axis("off")

    for ax in fig.axes:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"{benchmark_id}  —  Z stratification of anchored transcripts "
                 f"(rank {meta['rank']}, {meta['lineage_pair']})",
                 y=0.995)
    out_png = FIGS_DIR / f"{benchmark_id}_zstack.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")

    print("\nPer-Z-bin focal-cell anchor counts:")
    for z_lo, z_hi, n, lin in z_bins_summary:
        print(f"  [{z_lo:.2f}, {z_hi:.2f}] µm  n={n:>3d}  {lin}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_009")
    args = ap.parse_args()
    main(args.benchmark_id)
