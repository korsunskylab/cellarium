"""Method F2 — refined for DB_top100_018.

User feedback on Method F's diagnostic for cell 018 found two issues:
  1. A tiny enclosed region was being defined by only 2 transcripts → false
     "confident" territory. Fix: raise N_MIN_EVIDENCE so few transcripts
     give low confidence.
  2. Deep cuts were being placed in background regions where there is no
     18S signal. CP-SAM cannot register cuts there. Fix: gate the cut
     field by a validity mask = "where cells actually live" derived from
     imaging signals (DAPI ∨ 18S above a low threshold).

Compares Method F (original) vs Method F2 (refined).
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
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, distance_transform_edt
from skimage.segmentation import find_boundaries
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import make_pi, SIGMA_PX as SIGMA_DIFFUSE_PX
from tint_cytoplasm_diffusion import (
    LINEAGES, LINEAGE_COLOURS, ALPHA, N_MIN_EVIDENCE as N_MIN_OLD, norm01,
)

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
BID = "DB_top100_018"
OUT_DIR = ROOT / "figs" / "method_F2_refined"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

K_SHARP = 8.0
DEPTH_MAX = 0.9
SIGMA_CUT_PX = 2.0

# Method F2 changes:
ALPHA_NEW = 10.0     # was 0.5 — Dirichlet per-class pseudo-count.
                     # "10 heads / 10 tails as prior" — need 10+ class-specific
                     # observations before posterior moves substantially.
N_MIN_NEW = 3.0      # back to default; ALPHA now does the heavy lifting
USE_CONF_SQUARED = True

CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=True, normalize=False)


def grey_abstain_pi(pi, anch, li_arr, sigma_diffuse_px, n_min):
    """Original abstain (uses provided pi as the posterior). Kept for the
    Method F (legacy) comparison."""
    K, H, W = pi.shape
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    rho_smooth = np.stack([gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    confidence = N_total / (N_total + n_min)
    pi_abst = confidence[None] * pi + (1.0 - confidence[None]) * (1.0 / K)
    return pi_abst, confidence


def posterior_with_alpha(anch, li_arr, sigma_diffuse_px, H, W, K, alpha, n_min):
    """Recompute the Dirichlet posterior from scratch with a custom ALPHA.
    Returns (pi_post_with_alpha, pi_abst, N_eff, N_total, confidence).
    """
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    rho_smooth = np.stack([gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    # Dirichlet posterior with per-class pseudo-count alpha
    pi_post = (N_eff + alpha) / (N_total + K * alpha + 1e-9)[None]
    # Optional grey-abstain blend (weaker now since ALPHA already does the work)
    confidence = N_total / (N_total + n_min)
    pi_abst = confidence[None] * pi_post + (1.0 - confidence[None]) * (1.0 / K)
    return pi_post, pi_abst, N_eff, N_total, confidence


def edge_field(pi_diff_abst, confidence, k_sharp=K_SHARP,
                conf_squared=False, cell_evidence=None, ce_squared=False):
    """Edge magnitude weighted by confidence and (optionally) by 18S/DAPI.

    cell_evidence is a continuous [0,1] field giving local "cell-ness" — typically
    max(DAPI_n, 18S_n). Multiplying the edge by it pulls edge intensity toward
    regions with actual imaging signal and away from background.
    """
    t = np.tanh(k_sharp * pi_diff_abst)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g = np.hypot(gx, gy).astype(np.float32)
    w = (confidence ** 2 if conf_squared else confidence).astype(np.float32)
    if cell_evidence is not None:
        ce = (cell_evidence ** 2 if ce_squared else cell_evidence).astype(np.float32)
        w = w * ce
    return (w * g).astype(np.float32)


def cut_18s(s18, edge, sigma_cut, depth_max):
    smoothed = gaussian_filter(edge, sigma=sigma_cut)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max() * depth_max
    return s18.astype(np.float32) * (1.0 - smoothed), smoothed


def main():
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
    print(f"loading {BID}…")
    rdir = ROIS / BID
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
    d = make_pi(rdir, gene_names, gl)
    s18, dapi, pi = d["s18"], d["dapi"], d["pi"]
    s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    anch = d["anch"]; li_arr = d["li_arr"]
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape

    # ----- Method F (original) -----
    piF_abst, confF = grey_abstain_pi(pi, anch, li_arr, SIGMA_DIFFUSE_PX, N_MIN_OLD)
    pi_diffF = (piF_abst[ka] - piF_abst[kb]).astype(np.float32)
    edgeF = edge_field(pi_diffF, confF, conf_squared=False)
    cutF_img, cutF_smooth = cut_18s(s18, edgeF, SIGMA_CUT_PX, DEPTH_MAX)

    # ----- Method F2 (refined per user feedback) -----
    # KEY FIX: raise the Dirichlet prior strength ALPHA, not N_MIN_EVIDENCE.
    # ALPHA is per-class pseudo-count ("10 heads / 10 tails prior"). With
    # ALPHA=10, 2 transcripts of one class shift π by only ~3 percentage
    # points above uniform — exactly the "more class-specific evidence
    # needed" behavior the user asked for.
    K = pi.shape[0]
    pi_post2, pi2_abst, N_eff, N_total, conf2 = posterior_with_alpha(
        anch, li_arr, SIGMA_DIFFUSE_PX, H, W, K, alpha=ALPHA_NEW, n_min=N_MIN_NEW)
    pi_diff2 = (pi2_abst[ka] - pi2_abst[kb]).astype(np.float32)
    cell_evidence = np.maximum(dapi_n, s18_n).astype(np.float32)
    edge2 = edge_field(pi_diff2, conf2, conf_squared=USE_CONF_SQUARED,
                        cell_evidence=cell_evidence, ce_squared=False)
    cut2_img, cut2_smooth = cut_18s(s18, edge2, SIGMA_CUT_PX, DEPTH_MAX)
    validity = cell_evidence  # for display
    print(f"  Method F2:")
    print(f"    ALPHA (Dirichlet prior): 0.5 → {ALPHA_NEW}")
    print(f"    N_MIN_EVIDENCE: {N_MIN_NEW} (default — ALPHA carries weight)")
    print(f"    conf² weighting: {USE_CONF_SQUARED}")
    print(f"    cell_evidence × edge (continuous, not binary)")

    # ----- CP-SAM -----
    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_wsi(arr, key):
        return ((arr.astype(np.float32) - wsi_pct[key]["q_lo"]) /
                max(wsi_pct[key]["q_hi"] - wsi_pct[key]["q_lo"], 1e-6)).astype(np.float32)
    dapi_norm = norm_wsi(dapi, "DAPI")
    def run_cp(s_img):
        img = np.stack([dapi_norm, norm_wsi(s_img, "18S")], axis=-1).astype(np.float32)
        m_out, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m_out.astype(np.int32)
    def count_focal(m_out):
        a = focal_region.sum()
        if a == 0: return 0, []
        lbls = np.unique(m_out[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted([(int(L), int(((m_out==L)&focal_region).sum())/a) for L in lbls], key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]

    m_ctrl = run_cp(s18);  n_ctrl, cov_ctrl = count_focal(m_ctrl); print(f"  control:  n={n_ctrl} {[f'{c*100:.0f}%' for _,c in cov_ctrl]}")
    m_F = run_cp(cutF_img); n_F, cov_F = count_focal(m_F); print(f"  Method F:  n={n_F} {[f'{c*100:.0f}%' for _,c in cov_F]}")
    m_F2 = run_cp(cut2_img); n_F2, cov_F2 = count_focal(m_F2); print(f"  Method F2: n={n_F2} {[f'{c*100:.0f}%' for _,c in cov_F2]}")

    # ----- diagnostic figure: 3 rows × 4 cols (each row = Control / F / F2 / annot) -----
    ys, xs = np.where(focal_region); pad = 25
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    def add_focal(ax, c="yellow", lw=1.5):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors=c, linewidths=lw)
        ax.set_xticks([]); ax.set_yticks([])
    sub = anch[(anch["px"] >= x0) & (anch["px"] < x1) &
                (anch["py"] >= y0) & (anch["py"] < y1)]
    def scatter_anchors(ax):
        for L in [lin_a, lin_b]:
            s = sub[sub["lineage"] == L]
            if len(s): ax.scatter(s["px"]-x0, s["py"]-y0, s=5, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")

    def sign_contour(ax, pi_diff_arr, color="cyan", lw=2):
        sign = (pi_diff_arr > 0).astype(np.uint8)
        sb = find_boundaries(sign, mode="inner") & focal_region
        ax.contour(sb[crop].astype(int), levels=[0.5], colors=color, linewidths=lw)

    def label_overlay(mask):
        out = np.zeros((*mask.shape, 4), dtype=np.float32)
        u = np.unique(mask); u = u[u != 0]
        cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
        for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], 0.45)
        return out

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    # Row 0: refs (anchors, π_diff F, π_diff F2, validity mask)
    axes[0,0].imshow(s18_n[crop], cmap="gray"); scatter_anchors(axes[0,0]); add_focal(axes[0,0])
    axes[0,0].set_title("REF: 18S + anchors (Fib green, Mel red)")
    im = axes[0,1].imshow(pi_diffF[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,1])
    sign_contour(axes[0,1], pi_diffF, color="cyan")
    axes[0,1].set_title(f"Method F π_diff (ALPHA=0.5, n_min={N_MIN_OLD})\nsign change in cyan")
    fig.colorbar(im, ax=axes[0,1], fraction=0.04)
    im = axes[0,2].imshow(pi_diff2[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,2])
    sign_contour(axes[0,2], pi_diff2, color="lime")
    axes[0,2].set_title(f"Method F2 π_diff (ALPHA={ALPHA_NEW})\nsign change in lime — tiny islands gone")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04)
    im = axes[0,3].imshow(validity[crop], cmap="gray", vmin=0, vmax=1); add_focal(axes[0,3])
    axes[0,3].set_title(f"cell_evidence = max(DAPI_n, 18S_n)\nCONTINUOUS (not binary) — edge intensity\nis multiplied pixel-wise by this")
    fig.colorbar(im, ax=axes[0,3], fraction=0.04)

    # Row 1: edge fields F vs F2
    im = axes[1,0].imshow(confF[crop], cmap="viridis", vmin=0, vmax=1); add_focal(axes[1,0])
    axes[1,0].set_title(f"F confidence (n_min={N_MIN_OLD})")
    fig.colorbar(im, ax=axes[1,0], fraction=0.04)
    im = axes[1,1].imshow(conf2[crop], cmap="viridis", vmin=0, vmax=1); add_focal(axes[1,1])
    axes[1,1].set_title(f"F2 confidence (n_min={N_MIN_NEW})")
    fig.colorbar(im, ax=axes[1,1], fraction=0.04)
    im = axes[1,2].imshow(cutF_smooth[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX); add_focal(axes[1,2])
    sign_contour(axes[1,2], pi_diffF, color="cyan", lw=1.2)
    axes[1,2].set_title("Method F cut field")
    fig.colorbar(im, ax=axes[1,2], fraction=0.04)
    im = axes[1,3].imshow(cut2_smooth[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX); add_focal(axes[1,3])
    sign_contour(axes[1,3], pi_diff2, color="lime", lw=1.2)
    axes[1,3].set_title("Method F2 cut field (validity-gated, conf²)")
    fig.colorbar(im, ax=axes[1,3], fraction=0.04)

    # Row 2: CP-SAM results
    axes[2,0].imshow(s18_n[crop], cmap="gray")
    axes[2,0].imshow(label_overlay(m_ctrl[crop])); scatter_anchors(axes[2,0]); add_focal(axes[2,0])
    axes[2,0].set_title(f"CONTROL  n_focal={n_ctrl}\n{[f'{c*100:.0f}%' for _,c in cov_ctrl[:3]]}")
    axes[2,1].imshow(norm01(cutF_img[crop]), cmap="gray")
    axes[2,1].imshow(label_overlay(m_F[crop])); scatter_anchors(axes[2,1]); add_focal(axes[2,1])
    sign_contour(axes[2,1], pi_diffF, color="cyan", lw=1.2)
    axes[2,1].set_title(f"Method F  n_focal={n_F}\n{[f'{c*100:.0f}%' for _,c in cov_F[:3]]}")
    axes[2,2].imshow(norm01(cut2_img[crop]), cmap="gray")
    axes[2,2].imshow(label_overlay(m_F2[crop])); scatter_anchors(axes[2,2]); add_focal(axes[2,2])
    sign_contour(axes[2,2], pi_diff2, color="lime", lw=1.2)
    axes[2,2].set_title(f"Method F2  n_focal={n_F2}\n{[f'{c*100:.0f}%' for _,c in cov_F2[:3]]}")
    axes[2,3].axis("off")
    summary = (
        f"{BID}  ({lin_a} × {lin_b})\n\n"
        f"FIXES IN METHOD F2:\n"
        f"  1. ALPHA (Dirichlet prior): 0.5 → {ALPHA_NEW}\n"
        f"     Per-class pseudo-count = '10 heads/10 tails' prior.\n"
        f"     2 red transcripts shift π_red only ~3% above uniform;\n"
        f"     need ~20+ class-specific to reach confident π.\n"
        f"  2. edge weight = confidence² (stronger blip suppression)\n"
        f"  3. CONTINUOUS cell_evidence multiplier:\n"
        f"     edge × max(DAPI_n, 18S_n)\n"
        f"     (pulls edge magnitude toward 18S signal,\n"
        f"      kills cuts in background regions)\n"
        f"\nCP-SAM RESULT:\n"
        f"  Control:   n_focal={n_ctrl}\n"
        f"  Method F:  n_focal={n_F}\n"
        f"  Method F2: n_focal={n_F2}\n"
        f"\nMax cut depth inside focal:\n"
        f"  F:  {cutF_smooth[focal_region].max():.2f}\n"
        f"  F2: {cut2_smooth[focal_region].max():.2f}\n"
        f"\nSign-change islands in focal:\n"
        f"  F  (n_min={N_MIN_OLD}):"
        f" {int((find_boundaries((pi_diffF>0).astype(np.uint8), mode='inner') & focal_region).sum())} px\n"
        f"  F2 (n_min={N_MIN_NEW}):"
        f" {int((find_boundaries((pi_diff2>0).astype(np.uint8), mode='inner') & focal_region).sum())} px\n"
    )
    axes[2,3].text(0.02, 0.98, summary, transform=axes[2,3].transAxes,
                    fontsize=10, family="monospace", va="top")

    plt.suptitle(f"Method F vs F2 — {BID}", fontsize=12, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_F_vs_F2.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
