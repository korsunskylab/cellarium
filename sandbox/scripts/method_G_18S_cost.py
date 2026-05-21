"""Method G — 18S-cost-weighted cut placement.

User insight on cell 098: when the π-gradient has two possible paths between
the two lineages, prefer the one through dim 18S. "Hannibal over the Alps"
analogy: cutting through bright 18S is expensive; cutting through dim 18S
(natural cell-cell boundary) is cheap. Choose path 2.

Implementation:
  Same as Method F2 but the final edge field gets an extra `(1 - s18_n)`
  multiplier so that:
    bright 18S  → (1 - s18_n) small  → edge magnitude reduced  → less cut
    dim 18S      → (1 - s18_n) large  → edge magnitude preserved → cut applied

  We keep:
    - ALPHA = 10  (strong Dirichlet prior — 10 pseudo-counts per class)
    - confidence²  (extra blip suppression)
    - cell_evidence = max(DAPI_n, 18S_n) factor (kill background)

  We add:
    - (1 - s18_smooth)  factor where s18_smooth is a slightly Gaussian-smoothed
      normalized 18S. Smoothed slightly so isolated bright pixels don't mask
      a wide neighbourhood.
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
from scipy.ndimage import gaussian_filter, sobel
from skimage.segmentation import find_boundaries
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import make_pi, SIGMA_PX as SIGMA_DIFFUSE_PX
from tint_cytoplasm_diffusion import LINEAGES, LINEAGE_COLOURS, norm01

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = ROOT / "figs" / "method_G_18S_cost"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

ALPHA = 10.0
N_MIN = 3.0
K_SHARP = 8.0
DEPTH_MAX = 0.9
SIGMA_CUT_PX = 2.0
S18_SMOOTH_SIGMA = 2.0   # smoothing for the 18S cost surface

CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=True, normalize=False)


def posterior_with_alpha(anch, li_arr, sigma_diffuse_px, H, W, K, alpha, n_min):
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    rho_smooth = np.stack([gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    pi_post = (N_eff + alpha) / (N_total + K * alpha + 1e-9)[None]
    confidence = N_total / (N_total + n_min)
    pi_abst = confidence[None] * pi_post + (1.0 - confidence[None]) * (1.0 / K)
    return pi_post, pi_abst, N_total, confidence


def label_overlay(mask, alpha=0.5):
    u = np.unique(mask); u = u[u != 0]
    cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], alpha)
    return out


def main(BID="DB_top100_098"):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
    print(f"{BID}  ({lin_a} × {lin_b})")
    rdir = ROIS / BID
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
    d = make_pi(rdir, gene_names, gl)
    s18, dapi = d["s18"], d["dapi"]; s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    anch = d["anch"]; li_arr = d["li_arr"]
    masks_cpsam = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks_cpsam == cps_focal)
    H, W = s18.shape; K = d["pi"].shape[0]

    # Build π with ALPHA=10
    pi_post, pi_abst, N_total, confidence = posterior_with_alpha(
        anch, li_arr, SIGMA_DIFFUSE_PX, H, W, K, alpha=ALPHA, n_min=N_MIN)
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)

    # tanh-sharpened gradient
    t = np.tanh(K_SHARP * pi_diff)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g = np.hypot(gx, gy).astype(np.float32)

    # Weights
    conf_sq = (confidence ** 2).astype(np.float32)
    cell_ev = np.maximum(dapi_n, s18_n).astype(np.float32)
    s18_cost = 1.0 - gaussian_filter(s18_n, sigma=S18_SMOOTH_SIGMA)   # high where 18S dim
    s18_cost = np.clip(s18_cost, 0, 1).astype(np.float32)

    # Method F2 (no cost)
    edge_F2 = g * conf_sq * cell_ev
    # Method G (adds 18S cost weighting)
    edge_G  = edge_F2 * s18_cost

    def smooth_to_depth(edge):
        sm = gaussian_filter(edge, sigma=SIGMA_CUT_PX)
        if sm.max() > 0: sm = sm / sm.max() * DEPTH_MAX
        return sm
    cut_F2 = smooth_to_depth(edge_F2)
    cut_G  = smooth_to_depth(edge_G)
    s18_F2_cut = s18.astype(np.float32) * (1.0 - cut_F2)
    s18_G_cut  = s18.astype(np.float32) * (1.0 - cut_G)

    # CP-SAM
    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_wsi(arr, key):
        return ((arr.astype(np.float32) - wsi_pct[key]["q_lo"]) /
                max(wsi_pct[key]["q_hi"] - wsi_pct[key]["q_lo"], 1e-6)).astype(np.float32)
    dapi_norm = norm_wsi(dapi, "DAPI")
    def run_cp(s_img):
        img = np.stack([dapi_norm, norm_wsi(s_img, "18S")], axis=-1).astype(np.float32)
        m, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m.astype(np.int32)
    def count_focal(m):
        a = focal_region.sum()
        if a == 0: return 0, []
        lbls = np.unique(m[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted([(int(L), int(((m==L)&focal_region).sum())/a) for L in lbls], key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]
    m_ctrl = run_cp(s18); n_c, cov_c = count_focal(m_ctrl)
    m_F2   = run_cp(s18_F2_cut); n_F2, cov_F2 = count_focal(m_F2)
    m_G    = run_cp(s18_G_cut);  n_G, cov_G   = count_focal(m_G)
    print(f"  Control: n={n_c} {[f'{c*100:.0f}%' for _, c in cov_c]}")
    print(f"  F2:      n={n_F2} {[f'{c*100:.0f}%' for _, c in cov_F2]}")
    print(f"  G:       n={n_G} {[f'{c*100:.0f}%' for _, c in cov_G]}")

    # render
    ys, xs = np.where(focal_region); pad = 25
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    def add_focal(ax, c="yellow", lw=1.5):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors=c, linewidths=lw)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    # Row 0
    axes[0,0].imshow(s18_n[crop], cmap="gray"); add_focal(axes[0,0])
    sub = anch[(anch["px"]>=x0)&(anch["px"]<x1)&(anch["py"]>=y0)&(anch["py"]<y1)]
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"]==L]
        if len(s): axes[0,0].scatter(s["px"]-x0, s["py"]-y0, s=4, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
    axes[0,0].set_title("REF: 18S + anchors")
    im = axes[0,1].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,1])
    axes[0,1].set_title(f"π_{lin_a}−π_{lin_b}  (ALPHA={ALPHA})")
    fig.colorbar(im, ax=axes[0,1], fraction=0.04)
    im = axes[0,2].imshow(s18_n[crop], cmap="gray", vmin=0, vmax=1); add_focal(axes[0,2])
    axes[0,2].set_title("18S (normalized)\nHIGH = hard to cut")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04)
    im = axes[0,3].imshow(s18_cost[crop], cmap="viridis", vmin=0, vmax=1); add_focal(axes[0,3])
    axes[0,3].set_title("1 − 18S_smooth  (cut affordability)\nbright = cheap")
    fig.colorbar(im, ax=axes[0,3], fraction=0.04)

    # Row 1: edge fields + cut fields
    vmax = max(edge_F2[focal_region].max(), edge_G[focal_region].max(), 1e-3)
    im = axes[1,0].imshow(edge_F2[crop], cmap="magma", vmax=vmax); add_focal(axes[1,0])
    axes[1,0].set_title("F2 edge field\n(conf² × cell_evidence × |∇tanh|)")
    fig.colorbar(im, ax=axes[1,0], fraction=0.04)
    im = axes[1,1].imshow(edge_G[crop], cmap="magma", vmax=vmax); add_focal(axes[1,1])
    axes[1,1].set_title("G edge field = F2 × (1 − 18S_smooth)\n(bright 18S regions suppressed)")
    fig.colorbar(im, ax=axes[1,1], fraction=0.04)
    im = axes[1,2].imshow(cut_F2[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX); add_focal(axes[1,2])
    axes[1,2].set_title("F2 cut depth field")
    fig.colorbar(im, ax=axes[1,2], fraction=0.04)
    im = axes[1,3].imshow(cut_G[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX); add_focal(axes[1,3])
    axes[1,3].set_title("G cut depth field (18S-cost weighted)")
    fig.colorbar(im, ax=axes[1,3], fraction=0.04)

    # Row 2: CP-SAM
    axes[2,0].imshow(s18_n[crop], cmap="gray")
    axes[2,0].imshow(label_overlay(m_ctrl[crop]))
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"]==L]
        if len(s): axes[2,0].scatter(s["px"]-x0, s["py"]-y0, s=4, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
    add_focal(axes[2,0]); axes[2,0].set_title(f"CONTROL n={n_c}\n{[f'{c*100:.0f}%' for _,c in cov_c[:3]]}")
    axes[2,1].imshow(norm01(s18_F2_cut[crop]), cmap="gray")
    axes[2,1].imshow(label_overlay(m_F2[crop]))
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"]==L]
        if len(s): axes[2,1].scatter(s["px"]-x0, s["py"]-y0, s=4, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
    add_focal(axes[2,1]); axes[2,1].set_title(f"F2 (no cost) n={n_F2}\n{[f'{c*100:.0f}%' for _,c in cov_F2[:3]]}")
    axes[2,2].imshow(norm01(s18_G_cut[crop]), cmap="gray")
    axes[2,2].imshow(label_overlay(m_G[crop]))
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"]==L]
        if len(s): axes[2,2].scatter(s["px"]-x0, s["py"]-y0, s=4, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
    add_focal(axes[2,2]); axes[2,2].set_title(f"G (18S cost) n={n_G}\n{[f'{c*100:.0f}%' for _,c in cov_G[:3]]}")
    axes[2,3].axis("off")
    summary = (
        f"{BID}  ({lin_a} × {lin_b})\n\n"
        f"Method G — 18S-cost weighted cut placement\n"
        f"  • ALPHA={ALPHA} Dirichlet prior\n"
        f"  • conf² × cell_evidence weights\n"
        f"  • × (1 − smoothed_18S_n) — Hannibal's-Alps cost\n\n"
        f"CP-SAM:\n"
        f"  Control:  n_focal={n_c}\n"
        f"  F2:       n_focal={n_F2}\n"
        f"  G:        n_focal={n_G}  (Δ vs F2 = {n_G-n_F2})\n\n"
        f"Edge magnitude inside focal:\n"
        f"  F2 max: {edge_F2[focal_region].max():.3f}\n"
        f"  G  max: {edge_G[focal_region].max():.3f}\n"
        f"\nCut depth inside focal:\n"
        f"  F2 max: {cut_F2[focal_region].max():.2f}\n"
        f"  G  max: {cut_G[focal_region].max():.2f}\n"
    )
    axes[2,3].text(0.02, 0.98, summary, transform=axes[2,3].transAxes,
                    fontsize=10, family="monospace", va="top")

    plt.suptitle(f"Method G — 18S-cost weighting — {BID}", fontsize=12, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_method_G.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("benchmark_id", nargs="?", default="DB_top100_098")
    a = p.parse_args()
    main(a.benchmark_id)
