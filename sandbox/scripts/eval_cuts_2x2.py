"""2x2 CP-SAM intervention test on one mini-ROI:

  axis 1: edge method   A = ∇(π·18S)              B = ∇π then × 18S
  axis 2: edge format   NMS  = post-NMS magnitude (continuous, varying along curve)
                        BIN  = hysteresis binary edges

For each of the 4 cut profiles: Gaussian-smooth the edge field with σ_cut, scale
to depth ∈ [0, depth_max], multiply 18S by (1 − cut_depth). Then run CP-SAM on
(DAPI + cut_18S) with the validated max-recall settings.

Plus a control: CP-SAM on (DAPI + raw 18S).

For the focal benchmark cell, count how many CP-SAM masks cover ≥5% of its
WSI-merged region. If treatment count > control count → intervention split it.
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
from cellpose import models
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries

# Reuse the edge-detection plumbing from edges_three_methods
from edges_three_methods import (
    SIGMA_PRE_PX, SIGMA_POST_PX, T_HIGH_PERC, T_LOW_PERC,
    dizenzo_lam_max, nms, hysteresis, method_A, method_B, make_pi, edges_pipeline,
)
from tint_cytoplasm_diffusion import LINEAGE_COLOURS, LINEAGES, hex_to_rgb, norm01

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
ROIS_DIR    = DATA / "benchmark_doublet_rois"
FIGS_DIR    = ROOT / "figs" / "eval_cuts_2x2"
WSI_PCT_JSON       = DATA / "wsi_morphology_percentiles.json"
WSI_DAPI_FULL_PATH = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S_FULL_PATH  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"

PIXEL_SIZE_UM = 0.2125
SIGMA_CUT_UM  = 1.0                                # cut profile width (validated 2D)
SIGMA_CUT_PX  = SIGMA_CUT_UM / PIXEL_SIZE_UM       # ≈ 4.7 px
DEPTH_MAX     = 0.9                                # max fraction to dim 18S
MIN_COVER     = 0.05                               # threshold to count as "covering focal"

CPSAM_KW = dict(diameter=None, niter=200,
                cellprob_threshold=-5.0, flow_threshold=0.0, augment=True)


def get_wsi_percentiles():
    """Cache whole-slide 1-99 percentiles for DAPI + 18S so ROI inputs to
    CP-SAM are normalized on the same scale as the WSI run. Matches
    `norm01_uint16` from scripts/run_cpsam_whole_slide.py (seed=0, 10M sample).
    """
    if WSI_PCT_JSON.exists():
        return json.loads(WSI_PCT_JSON.read_text())
    print(f"computing WSI percentiles (one-time, ~10 s)...")
    out = {}
    for name, path in [("DAPI", WSI_DAPI_FULL_PATH), ("18S", WSI_18S_FULL_PATH)]:
        arr = tifffile.imread(path)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, arr.size, size=10_000_000)
        sample = arr.ravel()[idx]
        q_lo, q_hi = np.percentile(sample, [1, 99])
        out[name] = {"q_lo": float(q_lo), "q_hi": float(q_hi)}
        print(f"  {name}: q_lo={q_lo:.0f}  q_hi={q_hi:.0f}")
    WSI_PCT_JSON.write_text(json.dumps(out, indent=2))
    return out


def norm_with_wsi(arr, q_lo, q_hi):
    """Clip + scale to [0,1] using pre-computed whole-slide percentiles."""
    a = arr.astype(np.float32)
    rng = max(q_hi - q_lo, 1e-6)
    return np.clip((a - q_lo) / rng, 0, 1).astype(np.float32)


def make_cut_profile(field, sigma_cut_px, depth_max):
    """Gaussian-smooth field, normalize by its max, scale to [0, depth_max]."""
    smoothed = gaussian_filter(field.astype(np.float32), sigma=sigma_cut_px)
    m = smoothed.max()
    if m < 1e-12:
        return np.zeros_like(smoothed, dtype=np.float32)
    return (smoothed / m * depth_max).astype(np.float32)


def apply_cut(s18_n, cut_depth):
    return (s18_n * (1.0 - cut_depth)).astype(np.float32)


def run_cpsam(model, dapi_n, s18_n):
    img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
    masks, _, _ = model.eval(img, channel_axis=-1, **CPSAM_KW)
    return masks.astype(np.int32)


def count_focal_covers(masks, focal_region, min_cover=MIN_COVER):
    focal_area = int(focal_region.sum())
    if focal_area == 0:
        return 0, []
    labels_in = np.unique(masks[focal_region])
    labels_in = labels_in[labels_in != 0]
    out = []
    for L in labels_in:
        cov = int(((masks == L) & focal_region).sum()) / focal_area
        if cov >= min_cover:
            out.append((int(L), float(cov)))
    out.sort(key=lambda kv: -kv[1])
    return len(out), out


def count_collateral_splits(masks_control, masks_treat, focal_region, min_area=50):
    """How many non-focal cells in the control got split by the treatment?

    For each label in masks_control that doesn't overlap the focal region:
      - find which treatment labels overlap it
      - if ≥2 treatment labels cover ≥10% of the control label, count as split
    """
    # Non-focal control labels = control labels that don't overlap focal_region
    focal_labels = set(int(L) for L in np.unique(masks_control[focal_region]) if L > 0)
    splits = 0
    for L in np.unique(masks_control):
        if L == 0 or L in focal_labels:
            continue
        cell = (masks_control == L)
        if cell.sum() < min_area:
            continue
        t_labels = np.unique(masks_treat[cell])
        t_labels = t_labels[t_labels != 0]
        # count how many treatment labels cover ≥10% of this control cell
        n_overlap = sum(
            1 for tl in t_labels
            if ((masks_treat == tl) & cell).sum() / cell.sum() >= 0.10
        )
        if n_overlap >= 2:
            splits += 1
    return splits


def label_overlay(masks, alpha=0.5):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    rgba = np.zeros((*masks.shape, 4), dtype=np.float32)
    if n > 0:
        perm = np.concatenate([[0], np.random.default_rng(0).permutation(np.arange(1, n + 1))])
        shuf = perm[masks]
        rgba = plt.get_cmap("nipy_spectral")(shuf / max(n, 1))
        rgba[..., 3] = alpha * (masks > 0)
        edges = find_boundaries(masks, mode="outer")
        rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def render(bid, data, conditions, results, focal_region):
    s18_n  = data["s18_n"]
    dapi_n = data["dapi_n"]
    anch   = data["anch"]; li_arr = data["li_arr"]; meta = data["meta"]
    colors = np.stack([hex_to_rgb(LINEAGE_COLOURS[L]) for L in LINEAGES])
    tint_rgb = np.einsum("khw,kc->hwc", data["pi"], colors)
    rgb_tint = (s18_n[..., None] * tint_rgb).astype(np.float32)

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    # Row 0: reference + edge inputs that produced each cut
    axes[0, 0].imshow(s18_n, cmap="gray")
    for li, L in enumerate(LINEAGES):
        sub = anch[li_arr == li]
        if len(sub):
            axes[0, 0].scatter(sub["px"], sub["py"], s=8,
                                c=LINEAGE_COLOURS[L], edgecolor="none", alpha=0.5)
    axes[0, 0].contour(focal_region, levels=[0.5], colors="cyan", linewidths=1.0)
    axes[0, 0].set_title(f"18S + anchors\n{meta['lineage_pair']} (rank {meta['rank']})")

    for ci, name in enumerate(["A_NMS", "A_BIN", "B_NMS", "B_BIN"]):
        edge_field = conditions[name]["edge_field_display"]
        axes[0, 1 + ci].imshow(edge_field, cmap="magma")
        axes[0, 1 + ci].contour(focal_region, levels=[0.5], colors="cyan", linewidths=0.7)
        axes[0, 1 + ci].set_title(f"{name}\n{conditions[name]['short_descr']}")

    # Row 1: 18S input fed to CP-SAM (control + 4 cut variants)
    axes[1, 0].imshow(s18_n, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].contour(focal_region, levels=[0.5], colors="cyan", linewidths=1.0)
    axes[1, 0].set_title("control: raw 18S")
    for ci, name in enumerate(["A_NMS", "A_BIN", "B_NMS", "B_BIN"]):
        cut_s18 = conditions[name]["s18_cut"]
        axes[1, 1 + ci].imshow(cut_s18, cmap="gray", vmin=0, vmax=1)
        axes[1, 1 + ci].contour(focal_region, levels=[0.5], colors="cyan", linewidths=1.0)
        axes[1, 1 + ci].set_title(f"{name}: 18S after cut")

    # Row 2: CP-SAM masks (control + 4)
    axes[2, 0].imshow(s18_n, cmap="gray")
    axes[2, 0].imshow(label_overlay(results["control"]["masks"]))
    axes[2, 0].contour(focal_region, levels=[0.5], colors="cyan", linewidths=1.0)
    n_c = results["control"]["n_focal"]
    axes[2, 0].set_title(f"control CP-SAM\n{n_c} mask(s) cover focal")

    for ci, name in enumerate(["A_NMS", "A_BIN", "B_NMS", "B_BIN"]):
        r = results[name]
        axes[2, 1 + ci].imshow(s18_n, cmap="gray")
        axes[2, 1 + ci].imshow(label_overlay(r["masks"]))
        axes[2, 1 + ci].contour(focal_region, levels=[0.5], colors="cyan", linewidths=1.0)
        delta = r["n_focal"] - n_c
        flag = ("  SPLIT" if delta > 0 else "  EXTRA-MERGE" if delta < 0 else "")
        axes[2, 1 + ci].set_title(f"{name} CP-SAM\n{r['n_focal']} mask(s) cover focal "
                                    f"(Δ={delta:+d}){flag}\n"
                                    f"collateral splits: {r['collateral']}")

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"{bid} — 2×2 cut intervention (A vs B × NMS vs binary)  "
                 f"σ_cut={SIGMA_CUT_UM} µm, depth={DEPTH_MAX}",
                 y=1.005)
    plt.tight_layout()
    out_png = FIGS_DIR / f"{bid}.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")


def main(bid):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"σ_cut = {SIGMA_CUT_UM} µm = {SIGMA_CUT_PX:.1f} px,  depth_max = {DEPTH_MAX}")

    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]

    data = make_pi(ROIS_DIR / bid, gene_names, gl)
    s18_n  = data["s18_n"]
    dapi_n = data["dapi_n"]
    print(f"[{bid}] π {data['pi'].shape}  ({data['meta']['lineage_pair']})")

    # Focal region from cached WSI CP-SAM mask
    cpsam_roi = tifffile.imread(ROIS_DIR / bid / "cells_cpsam_masks.tif")
    cps_focal = int(data["meta"]["cps_id_at_bookmark"])
    focal_region = (cpsam_roi == cps_focal)
    print(f"  focal region: {int(focal_region.sum())} px")

    # Compute methods A and B → NMS magnitude + binary edges
    lam_A, theta_A = method_A(data["pi"], s18_n)
    lam_B, theta_B = method_B(data["pi"], s18_n)
    nms_A = nms(lam_A, theta_A)
    nms_B = nms(lam_B, theta_B)
    bin_A, _ = edges_pipeline(lam_A, theta_A)
    bin_B, _ = edges_pipeline(lam_B, theta_B)
    print(f"  Method A: {int(bin_A.sum())} binary edge px")
    print(f"  Method B: {int(bin_B.sum())} binary edge px")

    # Build the four cut profiles
    conditions = {
        "A_NMS": {
            "edge_field_display": nms_A,
            "short_descr": "NMS magnitude (cut depth ∝ strength)",
            "cut_depth": make_cut_profile(nms_A, SIGMA_CUT_PX, DEPTH_MAX),
        },
        "A_BIN": {
            "edge_field_display": bin_A.astype(np.float32),
            "short_descr": "binary edges (uniform depth)",
            "cut_depth": make_cut_profile(bin_A.astype(np.float32), SIGMA_CUT_PX, DEPTH_MAX),
        },
        "B_NMS": {
            "edge_field_display": nms_B,
            "short_descr": "NMS magnitude (cut depth ∝ strength)",
            "cut_depth": make_cut_profile(nms_B, SIGMA_CUT_PX, DEPTH_MAX),
        },
        "B_BIN": {
            "edge_field_display": bin_B.astype(np.float32),
            "short_descr": "binary edges (uniform depth)",
            "cut_depth": make_cut_profile(bin_B.astype(np.float32), SIGMA_CUT_PX, DEPTH_MAX),
        },
    }
    for name, c in conditions.items():
        c["s18_cut"] = apply_cut(s18_n, c["cut_depth"])

    # Load cellpose
    print("loading cellpose-SAM model on MPS...")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # CONTROL run
    masks_control = run_cpsam(m_sam, dapi_n, s18_n)
    n_focal_c, cov_c = count_focal_covers(masks_control, focal_region)
    print(f"  CONTROL: {n_focal_c} mask(s) cover focal  "
          f"{[f'{cov:.0%}' for _, cov in cov_c[:4]]}")
    results = {"control": {"masks": masks_control, "n_focal": n_focal_c, "cov": cov_c}}

    # 4 TREATMENT runs
    for name, c in conditions.items():
        masks_t = run_cpsam(m_sam, dapi_n, c["s18_cut"])
        n_focal_t, cov_t = count_focal_covers(masks_t, focal_region)
        collateral = count_collateral_splits(masks_control, masks_t, focal_region)
        results[name] = {"masks": masks_t, "n_focal": n_focal_t,
                         "cov": cov_t, "collateral": collateral}
        print(f"  {name:6s}: {n_focal_t} mask(s) cover focal (Δ={n_focal_t-n_focal_c:+d}), "
              f"collateral splits: {collateral}")

    render(bid, data, conditions, results, focal_region)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    args = ap.parse_args()
    main(args.benchmark_id)
