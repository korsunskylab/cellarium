"""Cell 015 diagnostic — why doesn't CP-SAM follow the cut?

Hypothesis: the cut field is not actually located on the green sign-change line.
The edge field = |∇ tanh(K · π_diff)| × confidence² × cell_evidence.

|∇ tanh| peaks at the sign-change line itself (good — this is the green line).
But confidence² and cell_evidence can zero-out portions of that line, so the
*remaining* cut runs somewhere else. CP-SAM then follows the surviving cut
faithfully — just not where we'd hoped.

This script renders each multiplicative factor separately, overlaid on the
green sign-change line, so we can see exactly which factor breaks the chain.

Final panel: where is the cut field actually concentrated vs where is the
green line, and where does CP-SAM end up splitting?
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib.pyplot as plt
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
OUT_DIR = ROOT / "figs" / "method_F_diagnose_015"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
SIGMA_CUT_PX = 10.0; DEPTH_MAX = 1.0  # strongest cut from the sweep
ANCHOR_DOT_SIZE = 22
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
    return pi_abst, confidence


def label_overlay(mask, alpha=0.45):
    u = np.unique(mask); u = u[u != 0]
    cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], alpha)
    return out


def main():
    BID = "DB_top100_015"
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
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape; K = d["pi"].shape[0]

    pi_abst, confidence = posterior_with_alpha(
        anch, li_arr, SIGMA_DIFFUSE_PX, H, W, K, alpha=ALPHA, n_min=N_MIN)
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    t = np.tanh(K_SHARP * pi_diff)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g_raw = np.hypot(gx, gy).astype(np.float32)  # |∇ tanh|
    conf_sq = (confidence ** 2).astype(np.float32)
    cell_ev = np.maximum(dapi_n, s18_n).astype(np.float32)
    edge_F = g_raw * conf_sq * cell_ev

    smoothed = gaussian_filter(edge_F, sigma=SIGMA_CUT_PX)
    if smoothed.max() > 0:
        cut_field = (smoothed / smoothed.max()) * DEPTH_MAX
    else:
        cut_field = smoothed
    s_cut = s18.astype(np.float32) * (1.0 - cut_field)

    sign_field = (pi_diff > 0).astype(np.uint8)
    sign_boundary_focal = find_boundaries(sign_field, mode="inner") & focal_region

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
    m_ctrl = run_cp(s18)
    m_cut = run_cp(s_cut)

    # crop centred on focal
    ys, xs = np.where(focal_region); pad = 35
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]
    in_focal_ctrl = find_boundaries(m_ctrl, mode="inner") & focal_region
    in_focal_cut  = find_boundaries(m_cut,  mode="inner") & focal_region

    # anchor dots restricted to crop
    in_crop_mask = ((anch["py"].to_numpy() >= y0) & (anch["py"].to_numpy() < y1) &
                    (anch["px"].to_numpy() >= x0) & (anch["px"].to_numpy() < x1))
    a_in_py = anch["py"].to_numpy()[in_crop_mask] - y0
    a_in_px = anch["px"].to_numpy()[in_crop_mask] - x0
    a_in_li = li_arr[in_crop_mask]

    def base_overlays(ax, show_anchors=False):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.4)
        ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=2.0)
        if show_anchors:
            for k, lin in enumerate(LINEAGES):
                m_lin = a_in_li == k
                if not m_lin.any(): continue
                ax.scatter(a_in_px[m_lin], a_in_py[m_lin],
                            s=ANCHOR_DOT_SIZE, color=LINEAGE_COLOURS[lin],
                            edgecolors="black", linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])

    # 3 rows x 4 cols
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    # Row 1: raw inputs
    ax = axes[0, 0]
    ax.imshow(norm01(s18[crop]), cmap="gray")
    base_overlays(ax, show_anchors=True)
    ax.set_title("18S + anchors\n(yellow=focal, lime=π sign-change)")

    ax = axes[0, 1]
    pdf_c = pi_diff[crop]
    vm = max(abs(pdf_c.min()), abs(pdf_c.max()), 1e-3)
    im = ax.imshow(pdf_c, cmap="RdBu_r", vmin=-vm, vmax=vm)
    plt.colorbar(im, ax=ax, fraction=0.04)
    base_overlays(ax)
    ax.set_title(f"π_diff = π[{lin_a}] − π[{lin_b}]\n(scalar field; sign-change = lime)")

    ax = axes[0, 2]
    ax.imshow(t[crop], cmap="RdBu_r", vmin=-1, vmax=1)
    base_overlays(ax)
    ax.set_title("tanh(K · π_diff)  (sharpened)")

    ax = axes[0, 3]
    ax.imshow(g_raw[crop], cmap="hot",
              vmax=np.percentile(g_raw[focal_region], 99) if focal_region.any() else 1)
    base_overlays(ax)
    ax.set_title("|∇ tanh|  (this should peak ON the green line)")

    # Row 2: multiplicative factors
    ax = axes[1, 0]
    ax.imshow(conf_sq[crop], cmap="viridis", vmin=0, vmax=1)
    base_overlays(ax)
    ax.set_title("confidence²  (anchor density × dispersion)")

    ax = axes[1, 1]
    ax.imshow(cell_ev[crop], cmap="gray", vmin=0, vmax=1)
    base_overlays(ax)
    ax.set_title("cell_evidence = max(DAPI_n, 18S_n)")

    ax = axes[1, 2]
    ev_pct = np.percentile(edge_F[focal_region], 99) if focal_region.any() else 1.0
    ax.imshow(edge_F[crop], cmap="hot", vmin=0, vmax=ev_pct)
    base_overlays(ax)
    ax.set_title("edge_F = |∇ tanh| × conf² × cell_ev\n(THE cut field — does it match green?)")

    ax = axes[1, 3]
    cf_c = cut_field[crop]
    ax.imshow(cf_c, cmap="hot", vmin=0, vmax=1)
    base_overlays(ax)
    ax.set_title(f"cut_field (σ={SIGMA_CUT_PX}px, depth={DEPTH_MAX})\n"
                  f"(smoothed + normalized)")

    # Row 3: result
    ax = axes[2, 0]
    ax.imshow(norm01(s18[crop]), cmap="gray")
    ax.imshow(label_overlay(m_ctrl[crop]))
    ax.contour(in_focal_ctrl[crop].astype(int), levels=[0.5], colors="white", linewidths=1.4)
    base_overlays(ax)
    a_ctrl = focal_region.sum()
    lbls = np.unique(m_ctrl[focal_region]); lbls = lbls[lbls != 0]
    covers_c = sorted([(int(L), int(((m_ctrl==L)&focal_region).sum())/a_ctrl) for L in lbls], key=lambda kv: -kv[1])[:3]
    cov_c_str = " ".join(f"{c*100:.0f}%" for _, c in covers_c)
    nc = sum(1 for _, c in covers_c if c >= 0.05)
    ax.set_title(f"Control 18S (no cut)\nn_focal={nc}  {cov_c_str}")

    ax = axes[2, 1]
    ax.imshow(norm01(s_cut[crop]), cmap="gray")
    base_overlays(ax)
    ax.set_title("s_cut = 18S × (1 − cut_field)\nwhat CP-SAM actually sees")

    ax = axes[2, 2]
    ax.imshow(norm01(s_cut[crop]), cmap="gray")
    ax.imshow(label_overlay(m_cut[crop]))
    ax.contour(in_focal_cut[crop].astype(int), levels=[0.5], colors="white", linewidths=1.6)
    base_overlays(ax)
    a_cut = focal_region.sum()
    lbls = np.unique(m_cut[focal_region]); lbls = lbls[lbls != 0]
    covers_x = sorted([(int(L), int(((m_cut==L)&focal_region).sum())/a_cut) for L in lbls], key=lambda kv: -kv[1])[:3]
    cov_x_str = " ".join(f"{c*100:.0f}%" for _, c in covers_x)
    nx = sum(1 for _, c in covers_x if c >= 0.05)
    ax.set_title(f"CP-SAM on s_cut\nn_focal={nx}  {cov_x_str}\nwhite=CP split, lime=target")

    ax = axes[2, 3]
    # Overlay: cut field (red) vs CP-SAM split (white) vs green target — same axes
    ax.imshow(norm01(s18[crop]), cmap="gray")
    cut_alpha = np.zeros((*cf_c.shape, 4), dtype=np.float32)
    cut_alpha[..., 0] = 1.0
    cut_alpha[..., 3] = (cf_c / max(cf_c.max(), 1e-6)).clip(0, 1) * 0.75
    ax.imshow(cut_alpha)
    ax.contour(in_focal_cut[crop].astype(int), levels=[0.5], colors="white", linewidths=1.8)
    base_overlays(ax)
    ax.set_title("OVERLAY: red=cut field, lime=target,\nwhite=CP-SAM split, yellow=focal\n(Does white = red?  Does red = lime?)")

    plt.suptitle(
        f"{BID} — diagnose why CP-SAM split ≠ green target.  "
        f"Factor decomposition of edge_F = |∇tanh| × conf² × cell_ev.",
        fontsize=13, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_diagnose.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
