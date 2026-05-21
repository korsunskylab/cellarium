"""Prototype the 'color 18S by mRNA evidence' game on a benchmark mini-ROI.

Pipeline (per-pixel soft type field → tinted cytoplasm image):
  1. KDE per lineage on anchored transcripts (Gaussian, σ in microns).
  2. Per-pixel posterior π_t(p) = ρ_t(p) / Σ_t ρ_t(p)   (uniform prior).
  3. Mix lineage colours: tint(p) = Σ_t π_t(p) · c_t.
  4. Final image: out_rgb(p) = 18S(p) · tint(p).
     • 18S brightness gates visibility (background → black for free).
     • Single-type cytoplasm → saturated lineage colour.
     • Heterotypic doublet cytoplasm → two visibly different-coloured halves.

Renders a 4-panel comparison + saves to figs/tinted_cytoplasm_proto/.

Usage:
    python tint_cytoplasm_proto.py [benchmark_id]   # default: DB_top100_006
    python tint_cytoplasm_proto.py DB_top100_006 --sigma-um 4
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
FIGS_DIR    = ROOT / "figs" / "tinted_cytoplasm_proto"

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


def norm01(im: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    l, h = np.percentile(im, [lo, hi])
    return np.clip((im.astype(np.float32) - l) / max(h - l, 1e-6), 0, 1)


def hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)


def main(benchmark_id: str, sigma_um: float) -> None:
    roi = ROIS_DIR / benchmark_id
    assert roi.exists(), f"missing ROI dir: {roi}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load mini-ROI ─────────────────────────────────────────────────────
    dapi = tifffile.imread(roi / "morphology_DAPI.tif")
    s18  = tifffile.imread(roi / "morphology_18S.tif")
    cpsam = tifffile.imread(roi / "cells_cpsam_masks.tif")
    tx   = pd.read_parquet(roi / "transcripts.parquet")
    meta = json.loads((roi / "metadata.json").read_text())
    H, W = s18.shape
    crop_x0 = meta["bbox_crop_um"]["x0"]
    crop_y0 = meta["bbox_crop_um"]["y0"]
    cps_id_focal = meta["cps_id_at_bookmark"]
    print(f"[{benchmark_id}] shape={H}×{W} px  transcripts={len(tx)}  "
          f"lineage_pair={meta['lineage_pair']}")

    # ── Map gene_id → lineage (anchored only) ─────────────────────────────
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    tx["gene"] = [gene_names[g] for g in tx["gene_id"].values]
    tx["lineage"] = tx["gene"].map(gl)
    anch = tx[tx["lineage"].isin(LINEAGES)].copy()
    print(f"  anchored: {len(anch)} / {len(tx)} ({100*len(anch)/max(len(tx),1):.1f}%)")
    print(f"  per-lineage counts:")
    for L in LINEAGES:
        print(f"    {L:<14s} {int((anch['lineage']==L).sum())}")

    # ── Pixel coords inside crop ──────────────────────────────────────────
    anch["px"] = np.rint((anch["x_um"] - crop_x0) / PIXEL_SIZE_UM).astype(int)
    anch["py"] = np.rint((anch["y_um"] - crop_y0) / PIXEL_SIZE_UM).astype(int)
    anch = anch[(anch["px"] >= 0) & (anch["px"] < W) &
                (anch["py"] >= 0) & (anch["py"] < H)].copy()

    # ── KDE per lineage ───────────────────────────────────────────────────
    sigma_px = sigma_um / PIXEL_SIZE_UM
    K = len(LINEAGES)
    lin_to_idx = {L: i for i, L in enumerate(LINEAGES)}
    rho = np.zeros((K, H, W), dtype=np.float32)
    # Rasterize counts then convolve
    li_arr = anch["lineage"].map(lin_to_idx).to_numpy()
    np.add.at(rho, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    for k in range(K):
        rho[k] = gaussian_filter(rho[k], sigma=sigma_px)

    # ── Per-pixel posterior π_t ───────────────────────────────────────────
    total = rho.sum(axis=0)
    pi = np.divide(rho, np.maximum(total[None], 1e-8),
                   out=np.zeros_like(rho), where=total[None] > 1e-8)

    # ── Mix lineage colours ───────────────────────────────────────────────
    colors = np.stack([hex_to_rgb(LINEAGE_COLOURS[L]) for L in LINEAGES])  # (K, 3)
    tint = np.einsum("khw,kc->hwc", pi, colors)                            # (H, W, 3)

    # ── Final tinted-cytoplasm image: 18S brightness × tint ───────────────
    s18_n  = norm01(s18)
    dapi_n = norm01(dapi)
    out_rgb = s18_n[..., None] * tint

    # ── Plot 4 panels ─────────────────────────────────────────────────────
    ext = [crop_x0, crop_x0 + W * PIXEL_SIZE_UM,
           crop_y0 + H * PIXEL_SIZE_UM, crop_y0]   # left, right, bottom, top (origin upper)
    focal_mask = cpsam == cps_id_focal

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))

    # 1. DAPI (blue) + 18S (yellow)
    rgb_dapi_s18 = np.stack([s18_n, s18_n, dapi_n], axis=-1)
    axes[0].imshow(rgb_dapi_s18, extent=ext, aspect="equal")
    axes[0].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                    extent=ext, origin="upper")
    axes[0].set_title(f"DAPI (blue) + 18S (yellow)\ncps_id={cps_id_focal}")

    # 2. 18S + anchor scatter (same as benchmark plots)
    axes[1].imshow(s18_n, cmap="gray", extent=ext, aspect="equal")
    axes[1].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                    extent=ext, origin="upper")
    in_cell = cpsam[anch["py"].to_numpy(), anch["px"].to_numpy()] == cps_id_focal
    out_cell = ~in_cell
    for li, L in enumerate(LINEAGES):
        sel = out_cell & (li_arr == li)
        if sel.any():
            sub = anch.iloc[np.where(sel)[0]]
            axes[1].scatter(sub["x_um"], sub["y_um"], s=8,
                            c=LINEAGE_COLOURS[L], alpha=0.35, edgecolor="none")
    for li, L in enumerate(LINEAGES):
        sel = in_cell & (li_arr == li)
        if sel.any():
            sub = anch.iloc[np.where(sel)[0]]
            axes[1].scatter(sub["x_um"], sub["y_um"], s=28,
                            c=LINEAGE_COLOURS[L], edgecolor="black", linewidth=0.4,
                            label=f"{L} (n={len(sub)})", alpha=0.95)
    axes[1].legend(loc="upper right", fontsize=7, framealpha=0.85)
    axes[1].set_title(f"18S + anchor tx\n{meta['lineage_pair']}  "
                       f"top1={meta['top1_n']} top2={meta['top2_n']}")

    # 3. Tinted cytoplasm
    axes[2].imshow(out_rgb, extent=ext, aspect="equal")
    axes[2].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                    extent=ext, origin="upper")
    axes[2].set_title(f"Tinted cytoplasm  (σ={sigma_um:.1f} µm)\n"
                       f"out = 18S × Σ_t π_t(p) · c_t")
    # Lineage colour legend (compact)
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=LINEAGE_COLOURS[L],
                          markersize=8, label=L) for L in LINEAGES]
    axes[2].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    # 4. Confidence map: max(π_t) (saturated where one type clearly dominates)
    confidence = pi.max(axis=0)
    im4 = axes[3].imshow(confidence, extent=ext, aspect="equal",
                          cmap="magma", vmin=1.0/K, vmax=1.0)
    axes[3].contour(focal_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                    extent=ext, origin="upper")
    axes[3].set_title(f"max(π_t)  (confidence of dominant type)\n"
                       f"yellow = unambiguous, dark = mixed/empty")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.02)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"{benchmark_id}  —  rank {meta['rank']}  ({meta['lineage_pair']})",
                 y=1.005)
    plt.tight_layout()
    out_png = FIGS_DIR / f"{benchmark_id}_sigma{sigma_um:.1f}um.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    ap.add_argument("--sigma-um", type=float, default=4.0)
    args = ap.parse_args()
    main(args.benchmark_id, args.sigma_um)
