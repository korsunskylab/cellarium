"""Contact sheet for the Method F σ-sweep batch.

For each of the 68 doublets, render a 1×4 row: control | σ=2 | σ=5 | σ=10
(focal-crop 18S with CP-SAM mask overlay + yellow focal contour + lime sign-
change line + white internal split line). Title carries n_focal values.

Output: figs/method_F_batch_all_doublets/_contact_sheet.png  (one big image)
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
OUT_DIR = ROOT / "figs" / "method_F_batch_all_doublets"
SUMMARY = OUT_DIR / "_summary.csv"

ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
SIGMAS_CUT_PX = [2.0, 5.0, 10.0]
DEPTH_MAX = 0.99
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


def render_doublet_row(axes_row, BID, m_sam, norm_dapi, norm_18s, summary_row):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
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
    g_raw = np.hypot(gx, gy).astype(np.float32)
    cell_ev = np.maximum(dapi_n, s18_n).astype(np.float32)
    edge_F = g_raw * (confidence ** 2) * cell_ev
    sign_field = (pi_diff > 0).astype(np.uint8)
    sign_boundary_focal = find_boundaries(sign_field, mode="inner") & focal_region

    def run_cp(s_img):
        img = np.stack([norm_dapi(dapi), norm_18s(s_img)], axis=-1).astype(np.float32)
        m, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m.astype(np.int32)

    m_ctrl = run_cp(s18)
    masks_for_sigma = {}
    for sigma in SIGMAS_CUT_PX:
        smoothed = gaussian_filter(edge_F, sigma=sigma)
        if smoothed.max() > 0:
            cut_field = (smoothed / smoothed.max()) * DEPTH_MAX
        else:
            cut_field = smoothed
        s_cut = s18.astype(np.float32) * (1.0 - cut_field)
        masks_for_sigma[sigma] = run_cp(s_cut)

    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]

    def draw(ax, m_disp, label, n_val, ctrl_n):
        ax.imshow(norm01(s18[crop]), cmap="gray")
        ax.imshow(label_overlay(m_disp[crop]))
        in_focal = find_boundaries(m_disp, mode="inner") & focal_region
        ax.contour(in_focal[crop].astype(int), levels=[0.5], colors="white", linewidths=0.9)
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=0.8)
        ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        if ctrl_n is None:
            colour = "black"
        else:
            colour = ("green" if n_val > ctrl_n else
                      ("red" if n_val < ctrl_n else "black"))
        ax.set_title(f"{label}  n={n_val}", fontsize=7, color=colour, pad=1)

    ctrl_n = int(summary_row["n_control"])
    draw(axes_row[0], m_ctrl, f"{BID[-3:]} ctrl", ctrl_n, None)
    for i, sigma in enumerate(SIGMAS_CUT_PX, start=1):
        n_s = int(summary_row[f"n_sigma{int(sigma)}"])
        draw(axes_row[i], masks_for_sigma[sigma], f"σ={int(sigma)}", n_s, ctrl_n)


def main():
    df = pd.read_csv(SUMMARY).sort_values("benchmark_id").reset_index(drop=True)
    n = len(df)
    # Recompute CP-SAM masks for the contact sheet. (We didn't pickle them
    # from the batch; redoing keeps the contact sheet decoupled.)
    print(f"Building contact sheet for {n} doublets ({n}×4 CP-SAM calls — fast on MPS)…")

    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_dapi(arr):
        return ((arr.astype(np.float32) - wsi_pct["DAPI"]["q_lo"]) /
                max(wsi_pct["DAPI"]["q_hi"] - wsi_pct["DAPI"]["q_lo"], 1e-6)).astype(np.float32)
    def norm_18s(arr):
        return ((arr.astype(np.float32) - wsi_pct["18S"]["q_lo"]) /
                max(wsi_pct["18S"]["q_hi"] - wsi_pct["18S"]["q_lo"], 1e-6)).astype(np.float32)

    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # Layout: 4 columns (ctrl + 3 sigmas); n rows.
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.2 * n))
    for i, r in df.iterrows():
        BID = r["benchmark_id"]
        try:
            render_doublet_row(axes[i, :], BID, m_sam, norm_dapi, norm_18s, r)
        except Exception as e:
            print(f"  {BID}: FAILED ({e})")
            for ax in axes[i, :]: ax.text(0.5, 0.5, f"{BID}: ERR", ha='center'); ax.set_xticks([]); ax.set_yticks([])
        if (i + 1) % 10 == 0: print(f"  rendered {i+1}/{n}")

    plt.suptitle(
        f"Method F σ-sweep contact sheet (depth={DEPTH_MAX}, α={ALPHA}, K_sharp={K_SHARP})\n"
        f"Columns: control / σ=2 / σ=5 / σ=10.   Green title = improved; red = regressed.\n"
        f"Yellow = focal CP-SAM contour;  lime = π sign-change boundary;  white = CP-SAM split.",
        fontsize=11, y=1.0)
    plt.tight_layout()
    out = OUT_DIR / "_contact_sheet.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
