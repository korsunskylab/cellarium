"""Test the pair-specific gradient edge detector on DB_top100_003.

Original eval_cuts_2x2 uses DiZenzo λ_max on the full multichannel π, then
global-percentile hysteresis. That throws away intracellular heterotypic
boundaries because the intercellular gradients (where π switches between
cells) are much stronger and dominate the percentile.

This script replaces the detector with:
    edge_field = |∇(π_a − π_b)|       (single scalar per pixel)
    NMS along ∇ direction
    hysteresis at PER-CELL percentiles (inside the focal cell only)

For DB_top100_003: a=Fibroblast, b=Melanoma.

Outputs a multi-panel diagnostic + CP-SAM intervention test.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import (
    make_pi, nms, SIGMA_PRE_PX, SIGMA_POST_PX, T_HIGH_PERC, T_LOW_PERC,
)
from tint_cytoplasm_diffusion import LINEAGES, LINEAGE_COLOURS, hex_to_rgb, norm01

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
BID = "DB_top100_003"
OUT_DIR = ROOT / "figs" / "eval_cut_pair_gradient"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125
SIGMA_CUT_PX = int(round(1.0 / PIXEL_SIZE_UM))   # 1 µm
DEPTH_MAX = 0.9

CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=True, normalize=False)


def label_overlay(mask, alpha=0.55, seed=3):
    n = int(mask.max())
    if n == 0: return np.zeros((*mask.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n+1))])
    shuf = perm[mask]
    rgba = plt.get_cmap("tab20")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (mask > 0)
    edges = find_boundaries(mask, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def hysteresis_inside_region(magnitude, region_mask, t_low_perc, t_high_perc):
    """Compute hysteresis percentile thresholds ONLY over pixels inside region_mask.
    Then apply hysteresis to the whole image (so edges adjacent to the region get
    propagated naturally), but the thresholds reflect the in-region distribution."""
    vals = magnitude[region_mask & (magnitude > 0)]
    if vals.size == 0:
        return np.zeros_like(magnitude, dtype=bool)
    t_lo = np.percentile(vals, t_low_perc)
    t_hi = np.percentile(vals, t_high_perc)
    # Standard hysteresis: seed at high, grow into low
    strong = magnitude >= t_hi
    weak   = (magnitude >= t_lo) & (magnitude < t_hi)
    # BFS / connected components from strong seeds through weak
    from scipy.ndimage import label as cc_label, binary_dilation
    out = strong.copy()
    while True:
        grown = binary_dilation(out, iterations=1) & (weak | strong)
        if grown.sum() == out.sum(): break
        out = grown
    return out


def main():
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair_lins = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair_lins[0], pair_lins[1]
    ka = LINEAGES.index(lin_a); kb = LINEAGES.index(lin_b)
    print(f"loading {BID} (cps={cps_focal}, pair: {lin_a} × {lin_b})…")
    rdir = ROIS / BID
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
    d = make_pi(rdir, gene_names, gl)
    s18, dapi, pi = d["s18"], d["dapi"], d["pi"]
    s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    anch = d["anch"]
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape
    print(f"  shape {s18.shape}, focal area {focal_region.sum()} px")

    # ----- pair-specific signed lineage map -----
    pi_diff = (pi[ka] - pi[kb]).astype(np.float32)
    pi_diff_smoothed = gaussian_filter(pi_diff, sigma=SIGMA_PRE_PX)

    # ----- gradient magnitude -----
    from scipy.ndimage import sobel
    gy = sobel(pi_diff_smoothed, axis=0); gx = sobel(pi_diff_smoothed, axis=1)
    grad_mag = np.hypot(gx, gy)
    grad_theta = np.arctan2(gy, gx)
    grad_post = gaussian_filter(grad_mag, sigma=SIGMA_POST_PX)

    # ----- NMS -----
    nms_map = nms(grad_post, grad_theta)

    # ----- hysteresis at PER-CELL percentiles (inside focal cell) -----
    bin_edges_in_focal = hysteresis_inside_region(
        nms_map, focal_region, T_LOW_PERC, T_HIGH_PERC,
    )
    n_inside_focal = int((bin_edges_in_focal & focal_region).sum())
    print(f"  per-focal-cell hysteresis: {bin_edges_in_focal.sum()} binary edge px total, "
          f"{n_inside_focal} inside focal")

    # Also compute the OLD (global-percentile) edges for comparison
    t_lo_global = np.percentile(nms_map[nms_map > 0], T_LOW_PERC)
    t_hi_global = np.percentile(nms_map[nms_map > 0], T_HIGH_PERC)
    from skimage.filters import apply_hysteresis_threshold
    bin_edges_global = apply_hysteresis_threshold(nms_map, t_lo_global, t_hi_global)
    n_inside_focal_global = int((bin_edges_global & focal_region).sum())
    print(f"  OLD (global-percentile) hysteresis: {bin_edges_global.sum()} binary px total, "
          f"{n_inside_focal_global} inside focal")

    # ----- cut 18S using new edges -----
    cut_field = gaussian_filter(bin_edges_in_focal.astype(np.float32), sigma=SIGMA_CUT_PX)
    cut_field = cut_field / max(cut_field.max(), 1e-6) * DEPTH_MAX
    s18_cut = s18.astype(np.float32) * (1.0 - cut_field)

    # ----- run CP-SAM control + cut -----
    print("\nloading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def normalize_to_wsi(arr, ch_key):
        a = arr.astype(np.float32)
        q_lo = wsi_pct[ch_key]["q_lo"]; q_hi = wsi_pct[ch_key]["q_hi"]
        return ((a - q_lo) / max(q_hi - q_lo, 1e-6)).astype(np.float32)
    dapi_norm = normalize_to_wsi(dapi, "DAPI")
    s18_norm  = normalize_to_wsi(s18, "18S")
    s18_cut_norm = normalize_to_wsi(s18_cut, "18S")

    def run_cp(d, s):
        img = np.stack([d, s], axis=-1).astype(np.float32)
        masks_out, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return masks_out.astype(np.int32)

    print("  control (raw 18S)…")
    masks_ctrl = run_cp(dapi_norm, s18_norm)
    print("  cut (pair-gradient edges)…")
    masks_cut  = run_cp(dapi_norm, s18_cut_norm)

    def count_focal(masks_out):
        focal_area = focal_region.sum()
        if focal_area == 0: return 0, []
        lbls = np.unique(masks_out[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted(
            [(int(L), int(((masks_out == L) & focal_region).sum()) / focal_area) for L in lbls],
            key=lambda kv: -kv[1])
        n_at = sum(1 for _, c in covers if c >= 0.05)
        return n_at, covers[:5]

    n_ctrl, cov_ctrl = count_focal(masks_ctrl)
    n_cut, cov_cut = count_focal(masks_cut)
    print(f"\nControl n_focal = {n_ctrl}, covers = {[f'{c*100:.0f}%' for _, c in cov_ctrl]}")
    print(f"Cut     n_focal = {n_cut}, covers = {[f'{c*100:.0f}%' for _, c in cov_cut]}")

    # ----- render multi-panel diagnostic -----
    ys, xs = np.where(focal_region)
    pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    def add_focal(ax, c="cyan", lw=1.5):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors=c, linewidths=lw)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 0: π_a, π_b, π_diff, |∇π_diff|
    im = axes[0,0].imshow(pi[ka][crop], cmap="Greens", vmin=0, vmax=1)
    add_focal(axes[0,0]); axes[0,0].set_title(f"(0,0) π_{lin_a}")
    fig.colorbar(im, ax=axes[0,0], fraction=0.04, pad=0.02)
    im = axes[0,1].imshow(pi[kb][crop], cmap="Reds", vmin=0, vmax=1)
    add_focal(axes[0,1]); axes[0,1].set_title(f"(0,1) π_{lin_b}")
    fig.colorbar(im, ax=axes[0,1], fraction=0.04, pad=0.02)
    im = axes[0,2].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1)
    add_focal(axes[0,2]); axes[0,2].set_title(f"(0,2) π_{lin_a} − π_{lin_b}  (signed)")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04, pad=0.02)
    im = axes[0,3].imshow(grad_post[crop], cmap="magma")
    add_focal(axes[0,3]); axes[0,3].set_title(f"(0,3) |∇(π_{lin_a}−π_{lin_b})|  (post-smoothed)")
    fig.colorbar(im, ax=axes[0,3], fraction=0.04, pad=0.02)

    # Row 1: NMS + global-percentile edges + per-cell-percentile edges + edge comparison
    im = axes[1,0].imshow(nms_map[crop], cmap="magma")
    add_focal(axes[1,0]); axes[1,0].set_title("(1,0) NMS-thinned edge magnitude")
    fig.colorbar(im, ax=axes[1,0], fraction=0.04, pad=0.02)
    axes[1,1].imshow(bin_edges_global[crop], cmap="gray")
    add_focal(axes[1,1])
    axes[1,1].set_title(f"(1,1) OLD: global-percentile edges\n{n_inside_focal_global} inside focal")
    axes[1,2].imshow(bin_edges_in_focal[crop], cmap="gray")
    add_focal(axes[1,2])
    axes[1,2].set_title(f"(1,2) NEW: per-focal-cell percentile edges\n{n_inside_focal} inside focal")
    # overlay
    axes[1,3].imshow(s18_n[crop], cmap="gray")
    # red dots = NEW edges, cyan dots = OLD edges
    new_only = bin_edges_in_focal & ~bin_edges_global
    shared = bin_edges_in_focal & bin_edges_global
    old_only = bin_edges_global & ~bin_edges_in_focal
    overlay = np.zeros((*new_only[crop].shape, 4), dtype=np.float32)
    overlay[new_only[crop]] = (1, 0.2, 0.2, 0.9)        # red = new only
    overlay[shared[crop]]   = (1, 1, 0.2, 0.9)          # yellow = both
    overlay[old_only[crop]] = (0.2, 0.6, 1.0, 0.9)      # blue = old only
    axes[1,3].imshow(overlay)
    add_focal(axes[1,3])
    axes[1,3].set_title("(1,3) edges overlay\nred=NEW only, yellow=both, blue=OLD only")

    # Row 2: cut 18S + control CP-SAM + cut CP-SAM + summary
    axes[2,0].imshow(norm01(s18_cut[crop]), cmap="gray")
    add_focal(axes[2,0])
    axes[2,0].set_title(f"(2,0) cut 18S (σ_cut=1µm, depth={DEPTH_MAX})")
    axes[2,1].imshow(s18_n[crop], cmap="gray")
    axes[2,1].imshow(label_overlay(masks_ctrl[crop]))
    add_focal(axes[2,1])
    axes[2,1].set_title(f"(2,1) CONTROL CP-SAM  (n_focal={n_ctrl})")
    axes[2,2].imshow(norm01(s18_cut[crop]), cmap="gray")
    axes[2,2].imshow(label_overlay(masks_cut[crop]))
    add_focal(axes[2,2])
    axes[2,2].set_title(f"(2,2) CUT CP-SAM  (n_focal={n_cut}, Δ={n_cut-n_ctrl})")
    axes[2,3].axis("off")
    summary = (
        f"DB_top100_003 (Fib×Mel, cps={cps_focal})\n"
        f"  focal area: {focal_region.sum()} px\n"
        f"\n"
        f"EDGE DETECTOR COMPARISON:\n"
        f"  OLD (global-percentile DiZenzo on full π):\n"
        f"    edges inside focal: {n_inside_focal_global} px\n"
        f"  NEW (per-focal-cell percentile on |∇(π_a−π_b)|):\n"
        f"    edges inside focal: {n_inside_focal} px  ({n_inside_focal/max(n_inside_focal_global,1):.1f}× more)\n"
        f"\n"
        f"CP-SAM RESULT:\n"
        f"  control (raw 18S):  {n_ctrl} masks at focal\n"
        f"    covers: {[f'{c*100:.0f}%' for _, c in cov_ctrl]}\n"
        f"  cut (new edges):    {n_cut} masks at focal\n"
        f"    covers: {[f'{c*100:.0f}%' for _, c in cov_cut]}\n"
        f"    Δ = {n_cut - n_ctrl} (positive = more splits)\n"
        f"\n"
        f"PIPELINE PARAMS:\n"
        f"  σ_pre={SIGMA_PRE_PX}, σ_post={SIGMA_POST_PX}\n"
        f"  hysteresis: T_LO={T_LOW_PERC}%, T_HI={T_HIGH_PERC}% (in-focal)\n"
        f"  cut: σ_cut={SIGMA_CUT_PX}px ({SIGMA_CUT_PX*PIXEL_SIZE_UM:.1f}µm), depth_max={DEPTH_MAX}\n"
    )
    axes[2,3].text(0.02, 0.98, summary, transform=axes[2,3].transAxes,
                    fontsize=9, family="monospace", va="top")

    plt.suptitle(
        f"Pair-specific gradient ∇(π_{lin_a}−π_{lin_b}) — heterotypic edge detector test on {BID}\n"
        f"(intracellular boundary detection; per-focal-cell percentile hysteresis)",
        fontsize=12, y=1.0)
    plt.tight_layout()
    out_png = OUT_DIR / f"{BID}_pair_gradient.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
