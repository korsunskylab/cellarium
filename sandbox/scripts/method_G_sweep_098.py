"""Cell 098 — sweep σ_cut × depth_max for Method G (18S-cost weighted).

098 didn't split with default F2 or G settings. Try a broader sweep:
  σ_cut ∈ {2, 4, 6, 10}  (going up to ~2 µm wide)
  depth_max ∈ {0.7, 0.9, 0.99, 1.0}

If no combination splits, the limitation is structural (cell 098 has weak/
absent 18S-boundary signal between the Tcell island and surrounding Fib) and
no cut strength via 18S can flip CP-SAM. Useful negative result.
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
OUT_DIR = ROOT / "figs" / "method_G_sweep_098"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125
ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
S18_SMOOTH_SIGMA = 2.0
WIDTHS_PX = [2.0, 4.0, 6.0, 10.0]
DEPTHS    = [0.7, 0.9, 0.99, 1.0]
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
    BID = "DB_top100_098"
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
    g = np.hypot(gx, gy).astype(np.float32)
    s18_cost = (1.0 - gaussian_filter(s18_n, sigma=S18_SMOOTH_SIGMA)).clip(0, 1)
    cell_ev  = np.maximum(dapi_n, s18_n).astype(np.float32)
    edge_G   = g * (confidence ** 2) * cell_ev * s18_cost

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
    def count_focal(m):
        a = focal_region.sum()
        if a == 0: return 0, []
        lbls = np.unique(m[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted([(int(L), int(((m==L)&focal_region).sum())/a) for L in lbls], key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]

    m_ctrl = run_cp(s18); n_c, cov_c = count_focal(m_ctrl)
    print(f"  Control: n={n_c} {[f'{c*100:.0f}%' for _,c in cov_c]}")

    grid = {}
    for sigma_cut in WIDTHS_PX:
        for d_max in DEPTHS:
            smoothed = gaussian_filter(edge_G, sigma=sigma_cut)
            if smoothed.max() > 0:
                cut_field = smoothed / smoothed.max() * d_max
            else:
                cut_field = smoothed
            s_cut = s18.astype(np.float32) * (1.0 - cut_field)
            m_out = run_cp(s_cut); n, cov = count_focal(m_out)
            grid[(sigma_cut, d_max)] = (m_out, n, cov, cut_field, s_cut)
            print(f"    σ={sigma_cut}px depth={d_max}: n={n} {[f'{c*100:.0f}%' for _,c in cov[:3]]}")

    # Render — focal-cell-centered grid
    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]
    def overlays(ax):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.2)
        ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=1.6)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(len(WIDTHS_PX), len(DEPTHS), figsize=(16, 16))
    for i, sigma_cut in enumerate(WIDTHS_PX):
        for j, d_max in enumerate(DEPTHS):
            ax = axes[i, j]
            m_out, n, cov, cut_field, s_cut = grid[(sigma_cut, d_max)]
            ax.imshow(norm01(s_cut[crop]), cmap="gray")
            ax.imshow(label_overlay(m_out[crop]))
            in_focal_bound = find_boundaries(m_out, mode="inner") & focal_region
            ax.contour(in_focal_bound[crop].astype(int), levels=[0.5], colors="white", linewidths=1.0)
            overlays(ax)
            cov_str = " ".join(f"{c*100:.0f}%" for _, c in cov[:3])
            color = "green" if n >= 2 else "red"
            ax.set_title(f"σ={sigma_cut}px, d={d_max}\nn={n}  {cov_str}",
                          fontsize=9, color=color)
    plt.suptitle(
        f"{BID} ({lin_a} × {lin_b}) — Method G sweep (18S-cost weighted)\n"
        f"Control: n={n_c}.  Lime=π sign-change boundary, White=CP-SAM split.",
        fontsize=11, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_G_sweep.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
