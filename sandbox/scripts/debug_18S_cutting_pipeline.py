"""Intermediate-step diagnostic for the 18S cutting pipeline on DB_top100_003.

The user observed: in the eval_cuts_2x2 figure, the focal doublet has clearly
distinct Fibroblast (green anchors) and Melanoma (red anchors) territories, yet
the edge map shows NO edges between them. This script renders every intermediate
field of the canonical pipeline so we can see where the signal is lost.

Uses the canonical `make_pi` from edges_three_methods (which itself uses
`diffuse_anisotropic` + Dirichlet posterior from tint_cytoplasm_diffusion).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
import tifffile

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))

from edges_three_methods import (
    make_pi, dizenzo_lam_max, nms, hysteresis,
    SIGMA_PRE_PX, SIGMA_POST_PX, T_HIGH_PERC, T_LOW_PERC,
    method_A, method_B,
)
from tint_cytoplasm_diffusion import LINEAGES, LINEAGE_COLOURS, ALPHA, hex_to_rgb, norm01

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
BID = "DB_top100_003"
OUT_DIR = ROOT / "figs" / "debug_18S_cutting"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125


def label_overlay(mask, alpha=0.5, seed=3):
    n = int(mask.max())
    if n == 0: return np.zeros((*mask.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n+1))])
    shuf = perm[mask]
    rgba = plt.get_cmap("tab20")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (mask > 0)
    return rgba


def main():
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    print(f"loading {BID} (cps={cps_focal}, {row['lineage_pair']})…")
    rdir = ROISDIR = ROIS / BID
    # gene_names + gene→lineage map needed by make_pi
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()

    d = make_pi(ROISDIR, gene_names, gl)
    s18, dapi, pi = d["s18"], d["dapi"], d["pi"]    # pi: (K, H, W)
    s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    anch = d["anch"]; li_arr = d["li_arr"]
    masks = tifffile.imread(ROISDIR / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape
    K = len(LINEAGES)
    print(f"  shape {s18.shape}, focal area {focal_region.sum()} px")
    print(f"  {len(anch)} anchored transcripts. lineage counts:")
    counts = anch["lineage"].value_counts()
    print(f"    {dict(counts)}")

    # Method-B edges: edges on π (DiZenzo) → NMS → hysteresis, then weight by 18S
    pi_sm = np.stack([gaussian_filter(pi[k], sigma=SIGMA_PRE_PX) for k in range(K)], axis=0)
    lam, theta = dizenzo_lam_max(pi_sm, sigma_pre_px=0, sigma_post_px=0)  # already smoothed
    lam_post = gaussian_filter(lam, sigma=SIGMA_POST_PX)
    nms_map = nms(lam_post, theta)
    # Hysteresis: use NMS percentiles as thresholds
    t_lo = np.percentile(nms_map[nms_map > 0], T_LOW_PERC) if (nms_map > 0).any() else 0
    t_hi = np.percentile(nms_map[nms_map > 0], T_HIGH_PERC) if (nms_map > 0).any() else 0
    bin_edges = hysteresis(nms_map, t_lo, t_hi)
    edges_inside_focal = int((bin_edges & focal_region).sum())
    print(f"  binary edges: {bin_edges.sum()} total, {edges_inside_focal} INSIDE focal cell")

    # ------------------------------------------------------------------
    # Render: per-lineage π + RGB composite + edges
    # ------------------------------------------------------------------
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

    # Build π RGB composite using lineage colours
    pi_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for k, L in enumerate(LINEAGES):
        rgb = np.array(hex_to_rgb(LINEAGE_COLOURS[L]), dtype=np.float32) / 255.0
        pi_rgb += pi[k][..., None] * rgb[None, None, :]
    pi_rgb = np.clip(pi_rgb, 0, 1)

    # Identify the relevant lineages for THIS doublet
    pair_lins = row["lineage_pair"].split(" × ")
    pair_idx = [LINEAGES.index(L) for L in pair_lins if L in LINEAGES]
    lin_a, lin_b = pair_lins[0], pair_lins[1]
    ka = LINEAGES.index(lin_a); kb = LINEAGES.index(lin_b)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    # Row 0: raw + anchors
    axes[0,0].imshow(s18_n[crop], cmap="gray"); add_focal(axes[0,0])
    axes[0,0].set_title("(0,0) 18S (normalized)")
    axes[0,1].imshow(dapi_n[crop], cmap="gray"); add_focal(axes[0,1])
    axes[0,1].set_title("(0,1) DAPI (normalized)")
    axes[0,2].imshow(s18_n[crop], cmap="gray"); add_focal(axes[0,2])
    legend_handles = []
    for L in LINEAGES:
        sub = anch[anch["lineage"] == L]
        sub_in = sub[(sub["px"] >= x0) & (sub["px"] < x1) &
                      (sub["py"] >= y0) & (sub["py"] < y1)]
        if len(sub_in) == 0: continue
        color = LINEAGE_COLOURS[L]
        axes[0,2].scatter(sub_in["px"] - x0, sub_in["py"] - y0,
                           s=6, c=color, alpha=0.85, edgecolors="none")
        legend_handles.append(Patch(color=color, label=f"{L} ({len(sub_in)})"))
    axes[0,2].set_title(f"(0,2) anchors by lineage  [{lin_a}×{lin_b} doublet]")
    if legend_handles:
        axes[0,2].legend(handles=legend_handles, fontsize=6, loc="upper right")
    axes[0,3].imshow(s18_n[crop], cmap="gray")
    axes[0,3].imshow(label_overlay(masks[crop])); add_focal(axes[0,3])
    axes[0,3].set_title("(0,3) WSI CP-SAM masks (cyan=focal)")

    # Row 1: π_a and π_b — the two relevant lineages
    im = axes[1,0].imshow(pi[ka][crop], cmap="Greens", vmin=0, vmax=1); add_focal(axes[1,0])
    axes[1,0].set_title(f"(1,0) π_{lin_a}  (Dirichlet posterior, 0..1)")
    fig.colorbar(im, ax=axes[1,0], fraction=0.04, pad=0.02)
    im = axes[1,1].imshow(pi[kb][crop], cmap="Reds", vmin=0, vmax=1); add_focal(axes[1,1])
    axes[1,1].set_title(f"(1,1) π_{lin_b}  (Dirichlet posterior, 0..1)")
    fig.colorbar(im, ax=axes[1,1], fraction=0.04, pad=0.02)
    # difference panel — most informative
    pi_diff = pi[ka] - pi[kb]
    im = axes[1,2].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[1,2])
    axes[1,2].set_title(f"(1,2) π_{lin_a} − π_{lin_b}  (red={lin_b}, blue={lin_a})")
    fig.colorbar(im, ax=axes[1,2], fraction=0.04, pad=0.02)
    axes[1,3].imshow(pi_rgb[crop]); add_focal(axes[1,3], c="white")
    axes[1,3].set_title("(1,3) π RGB composite (all 7 lineages)")

    # Row 2: max-π, argmax-π, mass map, "max π inside focal" histogram
    pi_max = pi.max(axis=0)
    pi_argmax = pi.argmax(axis=0)
    im = axes[2,0].imshow(pi_max[crop], cmap="viridis", vmin=1.0/K, vmax=1.0); add_focal(axes[2,0])
    axes[2,0].set_title(f"(2,0) max π  (1/K={1/K:.2f}=abstain, 1=certain)")
    fig.colorbar(im, ax=axes[2,0], fraction=0.04, pad=0.02)
    # argmax with discrete colormap
    cmap = plt.cm.get_cmap("tab10", K)
    axes[2,1].imshow(pi_argmax[crop], cmap=cmap, vmin=0, vmax=K-1); add_focal(axes[2,1])
    axes[2,1].set_title("(2,1) argmax π (lineage with highest π per pixel)")
    legend_handles = [Patch(color=cmap(i), label=L) for i, L in enumerate(LINEAGES)]
    axes[2,1].legend(handles=legend_handles, fontsize=6, loc="upper right")
    # Where is each lineage "won"?  Restrict to inside focal
    inside_argmax = pi_argmax[focal_region]
    print(f"\nInside focal cell ({focal_region.sum()} px):")
    for k, L in enumerate(LINEAGES):
        n = int((inside_argmax == k).sum())
        if n: print(f"    argmax={L}: {n} px ({100*n/focal_region.sum():.0f}%)")
    # Mass map — sum of all evidence in this region after diffusion (just plot π_max for now)
    # Anchor density inside focal: histogram of lineages
    anch_inside = anch[focal_region[anch["py"].clip(0,H-1), anch["px"].clip(0,W-1)]]
    counts_inside = anch_inside["lineage"].value_counts()
    axes[2,2].axis("off")
    inside_summary = f"Anchors INSIDE focal (n={len(anch_inside)}):\n"
    for L, c in counts_inside.items():
        inside_summary += f"   {L}: {c}\n"
    inside_summary += f"\nargmax-π pixel counts in focal:\n"
    for k, L in enumerate(LINEAGES):
        n = int((inside_argmax == k).sum())
        if n: inside_summary += f"   {L}: {n} px ({100*n/focal_region.sum():.0f}%)\n"
    axes[2,2].text(0.05, 0.95, inside_summary, transform=axes[2,2].transAxes,
                    fontsize=11, family="monospace", va="top")
    axes[2,2].set_title("(2,2) anchor & argmax breakdown inside focal")
    # Gradient of (π_a - π_b) — should be strong at lineage boundary
    grad = np.hypot(*np.gradient(pi_diff))
    im = axes[2,3].imshow(grad[crop], cmap="magma")
    add_focal(axes[2,3])
    axes[2,3].set_title(f"(2,3) |∇(π_{lin_a} − π_{lin_b})|")
    fig.colorbar(im, ax=axes[2,3], fraction=0.04, pad=0.02)

    # Row 3: edges
    im = axes[3,0].imshow(lam_post[crop], cmap="magma")
    add_focal(axes[3,0])
    axes[3,0].set_title("(3,0) DiZenzo λ_max(π)")
    fig.colorbar(im, ax=axes[3,0], fraction=0.04, pad=0.02)
    axes[3,1].imshow(nms_map[crop], cmap="magma")
    add_focal(axes[3,1])
    axes[3,1].set_title("(3,1) NMS-thinned edge magnitude")
    axes[3,2].imshow(bin_edges[crop], cmap="gray")
    add_focal(axes[3,2])
    axes[3,2].set_title(f"(3,2) hysteresis binary edges ({bin_edges[crop].sum()} px in crop, "
                        f"{edges_inside_focal} inside focal)")
    # Cut 18S
    SIGMA_CUT_PX = int(round(1.0 / PIXEL_SIZE_UM))
    DEPTH_MAX = 0.9
    cut_field = gaussian_filter(bin_edges.astype(np.float32), sigma=SIGMA_CUT_PX)
    cut_field = cut_field / max(cut_field.max(), 1e-6) * DEPTH_MAX
    s18_cut = s18.astype(np.float32) * (1.0 - cut_field)
    axes[3,3].imshow(norm01(s18_cut[crop]), cmap="gray")
    add_focal(axes[3,3])
    axes[3,3].set_title(f"(3,3) cut 18S  (σ_cut=1µm, depth={DEPTH_MAX})")

    plt.suptitle(
        f"18S cutting pipeline diagnostic — {BID} (cps={cps_focal}, {row['lineage_pair']})\n"
        f"User concern: anchors show clear {lin_a} vs {lin_b} territories inside focal doublet. "
        f"Does π show this gradient? Where do the edges land?",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    out_png = OUT_DIR / f"{BID}_pipeline_debug.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_png}")
    print(f"  binary edges inside focal: {edges_inside_focal} px / {focal_region.sum()} focal px = "
          f"{100*edges_inside_focal/focal_region.sum():.2f}%")


if __name__ == "__main__":
    main()
