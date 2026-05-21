"""Method F — confidence-weighted tanh gradient, σ_cut from synthetic operating window.

Constraints (from user 2026-05-20):
  • Don't penalize by size (no area threshold).
  • The COLORING / EDGE CONTRAST scheme should not give confidence to small blips —
    use grey-abstain on π AND multiply the edge field by per-pixel confidence.
  • σ_cut should be in the synthetic operating window (2–3 px), not the 5 px we used.
  • Cut profile is a smooth 1D Gaussian (already implicitly the case when we smooth
    a continuous edge field).
  • Plot anchored transcripts on the final CP-SAM panels.

Pipeline:
  1. π_abst from grey-abstained Dirichlet posterior + soft abstain blend
  2. π_diff = π_a − π_b (top-2 lineage pair)
  3. tanh sharpening: t = tanh(K · π_diff)
  4. gradient: g = |∇ t|
  5. CONFIDENCE weighting: edge = confidence × g
  6. cut field: Gaussian-smooth `edge` with σ_cut and scale to depth_max
  7. cut 18S = 18S · (1 − cut)
  8. run CP-SAM control + cut

We test σ_cut ∈ {2, 3} px and a "no confidence weighting" control for sanity.
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
from tint_cytoplasm_diffusion import (
    LINEAGES, LINEAGE_COLOURS, ALPHA, N_MIN_EVIDENCE, norm01,
)

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = ROOT / "figs" / "method_F_confidence_weighted"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

K_SHARP = 8.0
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


def grey_abstain_pi(pi, anch, li_arr, sigma_diffuse_px, n_min):
    K, H, W = pi.shape
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    rho_smooth = np.stack([gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    confidence = N_total / (N_total + n_min)
    pi_abst = confidence[None] * pi + (1.0 - confidence[None]) * (1.0 / K)
    return pi_abst, confidence


def make_edge_field(pi_diff_abst, confidence, K_sharp=K_SHARP):
    """Return continuous edge field = confidence × |∇ tanh(K · π_diff_abst)|."""
    t = np.tanh(K_sharp * pi_diff_abst)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g = np.hypot(gx, gy).astype(np.float32)
    return (confidence * g).astype(np.float32), g  # weighted, raw


def cut_18s(s18, edge_field, sigma_cut_px, depth_max=DEPTH_MAX):
    """Smooth the edge field with σ_cut, normalize to depth_max, multiply 18S."""
    smoothed = gaussian_filter(edge_field, sigma=sigma_cut_px)
    if smoothed.max() > 0: smoothed = smoothed / smoothed.max() * depth_max
    return s18.astype(np.float32) * (1.0 - smoothed), smoothed


def main(BID="DB_top100_003"):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
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
    H, W = s18.shape; K = pi.shape[0]
    print(f"  shape {s18.shape}, focal area {focal_region.sum()} px")

    pi_abst, confidence = grey_abstain_pi(pi, anch, d["li_arr"], SIGMA_DIFFUSE_PX, N_MIN_EVIDENCE)
    pi_diff_abst = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    print(f"  median confidence inside focal: {float(np.median(confidence[focal_region])):.2f}")

    edge_weighted, edge_raw = make_edge_field(pi_diff_abst, confidence)

    variants = []
    # F1: confidence-weighted, σ_cut=2
    cut_F1, smoothed_F1 = cut_18s(s18, edge_weighted, sigma_cut_px=2.0)
    variants.append(("F1: conf-weighted, σ_cut=2px", edge_weighted, cut_F1, smoothed_F1))
    # F2: confidence-weighted, σ_cut=3
    cut_F2, smoothed_F2 = cut_18s(s18, edge_weighted, sigma_cut_px=3.0)
    variants.append(("F2: conf-weighted, σ_cut=3px", edge_weighted, cut_F2, smoothed_F2))
    # F3 control: NO confidence weighting (raw gradient), σ_cut=2 — to see how much blips amp
    cut_F3, smoothed_F3 = cut_18s(s18, edge_raw, sigma_cut_px=2.0)
    variants.append(("F3 control: NO conf weight, σ_cut=2px", edge_raw, cut_F3, smoothed_F3))

    print("\nloading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_wsi(arr, key):
        return ((arr.astype(np.float32) - wsi_pct[key]["q_lo"]) /
                max(wsi_pct[key]["q_hi"] - wsi_pct[key]["q_lo"], 1e-6)).astype(np.float32)
    dapi_norm = norm_wsi(dapi, "DAPI")
    def run_cp(s18_image):
        img = np.stack([dapi_norm, norm_wsi(s18_image, "18S")], axis=-1).astype(np.float32)
        m_out, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m_out.astype(np.int32)
    def count_focal(m_out):
        a = focal_region.sum()
        if a == 0: return 0, []
        lbls = np.unique(m_out[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted(
            [(int(L), int(((m_out == L) & focal_region).sum()) / a) for L in lbls],
            key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]

    print("  CP-SAM control + 3 variants…")
    m_ctrl = run_cp(s18); n_ctrl, cov_ctrl = count_focal(m_ctrl)
    results = [("Control (no cut)", m_ctrl, n_ctrl, cov_ctrl, s18)]
    for name, _, cut_img, _ in variants:
        m = run_cp(cut_img); n, cov = count_focal(m)
        results.append((name, m, n, cov, cut_img))
        print(f"  {name}: n_focal={n} covers={[f'{c*100:.0f}%' for _, c in cov]}")

    # render
    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]; focal_bound = find_boundaries(focal_local, mode="outer")
    def add_focal(ax, c="cyan", lw=1.5):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors=c, linewidths=lw)
        ax.set_xticks([]); ax.set_yticks([])
    def overlay_anchors(ax):
        sub_in = anch[(anch["px"] >= x0) & (anch["px"] < x1) &
                       (anch["py"] >= y0) & (anch["py"] < y1)]
        for L in LINEAGES:
            s = sub_in[sub_in["lineage"] == L]
            if len(s) == 0: continue
            ax.scatter(s["px"] - x0, s["py"] - y0, s=5, c=LINEAGE_COLOURS[L],
                        alpha=0.85, edgecolors="none")

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    # Row 0: references
    axes[0,0].imshow(s18_n[crop], cmap="gray"); add_focal(axes[0,0])
    axes[0,0].set_title("REF: 18S")
    im = axes[0,1].imshow(pi_diff_abst[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,1])
    axes[0,1].set_title(f"π_{lin_a}−π_{lin_b}  (grey-abstained)")
    fig.colorbar(im, ax=axes[0,1], fraction=0.04, pad=0.02)
    im = axes[0,2].imshow(confidence[crop], cmap="viridis", vmin=0, vmax=1); add_focal(axes[0,2])
    axes[0,2].set_title("confidence  (low=blip-like)")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04, pad=0.02)
    axes[0,3].imshow(s18_n[crop], cmap="gray")
    overlay_anchors(axes[0,3]); add_focal(axes[0,3])
    legend_handles = [Patch(color=LINEAGE_COLOURS[L], label=L) for L in [lin_a, lin_b]]
    axes[0,3].legend(handles=legend_handles, fontsize=8, loc="upper right")
    axes[0,3].set_title("anchors (only pair lineages legend-shown)")

    # Row 1: tanh sharpening + edge fields
    t = np.tanh(K_SHARP * pi_diff_abst)
    im = axes[1,0].imshow(t[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[1,0])
    axes[1,0].set_title(f"tanh({K_SHARP}·π_diff_abst)  (sharpened)")
    fig.colorbar(im, ax=axes[1,0], fraction=0.04, pad=0.02)
    im = axes[1,1].imshow(edge_raw[crop], cmap="magma")
    add_focal(axes[1,1])
    axes[1,1].set_title("|∇ tanh|  (no confidence weighting)")
    fig.colorbar(im, ax=axes[1,1], fraction=0.04, pad=0.02)
    im = axes[1,2].imshow(edge_weighted[crop], cmap="magma")
    add_focal(axes[1,2])
    axes[1,2].set_title("conf × |∇ tanh|  — Method F edge field")
    fig.colorbar(im, ax=axes[1,2], fraction=0.04, pad=0.02)
    # Difference panel: how much does conf weighting suppress blips?
    diff = edge_raw[crop] - edge_weighted[crop]
    im = axes[1,3].imshow(diff, cmap="hot")
    add_focal(axes[1,3])
    axes[1,3].set_title("|edge_raw − edge_weighted|\n(blips that got suppressed)")
    fig.colorbar(im, ax=axes[1,3], fraction=0.04, pad=0.02)

    # Row 2: per-variant cut field
    axes[2,0].imshow(s18_n[crop], cmap="gray"); add_focal(axes[2,0])
    axes[2,0].set_title("REF: 18S (uncut)")
    for j, (name, _, _, smoothed) in enumerate(variants):
        im = axes[2, j+1].imshow(smoothed[crop], cmap="magma")
        add_focal(axes[2, j+1])
        axes[2, j+1].set_title(f"{name}\n(cut depth field)")
        fig.colorbar(im, ax=axes[2, j+1], fraction=0.04, pad=0.02)

    # Row 3: CP-SAM with anchors on the result
    for j, (name, m, n, cov, cut_img) in enumerate(results):
        ax = axes[3, j]
        if "Control" in name:
            base = s18_n[crop]
        else:
            base = norm01(cut_img[crop])
        ax.imshow(base, cmap="gray")
        ax.imshow(label_overlay(m[crop]))
        overlay_anchors(ax)
        add_focal(ax)
        cov_str = " ".join(f"{c*100:.0f}%" for _, c in cov[:3])
        ax.set_title(f"{name}\nn_focal={n}, covers={cov_str}")

    plt.suptitle(
        f"Method F — confidence-weighted tanh gradient on {BID} (Fib × Mel)\n"
        f"σ_cut from synthetic operating window. Blips suppressed by ×confidence, not by area filter.",
        fontsize=11, y=1.005)
    plt.tight_layout()
    out_png = OUT_DIR / f"{BID}_method_F.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("benchmark_id", nargs="?", default="DB_top100_003")
    a = p.parse_args()
    main(a.benchmark_id)
