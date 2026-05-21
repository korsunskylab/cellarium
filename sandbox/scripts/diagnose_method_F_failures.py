"""Per-cell deep diagnostic for the cells where Method F failed or behaved oddly.

For each cell, render a single big panel that overlays in one image:
  - 18S (gray background)
  - anchors (colored by lineage)
  - π_diff sign change line (red contour: where π_a − π_b = 0)
  - cut field magnitude (heat-colored overlay, only where > 0.05 of max)
  - CP-SAM split line (white contour: boundaries between masks at the focal location)

Plus a small text panel with quantitative diagnostics:
  - max cut depth inside focal
  - fraction of focal pixels with cut depth > 0.3 (significant cut region)
  - max cut depth ON the π_diff sign-change line (does the cut land on the lineage boundary?)
  - alignment metric (% of significant cut pixels that fall on the sign-change line within tolerance)

Run on the three diagnostic cells: 015, 018, 098.
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
    LINEAGES, LINEAGE_COLOURS, ALPHA, N_MIN_EVIDENCE, norm01,
)

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = ROOT / "figs" / "method_F_failures_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

K_SHARP = 8.0
DEPTH_MAX = 0.9
SIGMA_CUT_PX = 2.0

CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=True, normalize=False)


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


def diagnose_cell(BID, m_sam, wsi_pct, gene_names, gl):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
    rdir = ROIS / BID

    d = make_pi(rdir, gene_names, gl)
    s18, dapi, pi = d["s18"], d["dapi"], d["pi"]
    anch = d["anch"]
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape

    pi_abst, confidence = grey_abstain_pi(pi, anch, d["li_arr"], SIGMA_DIFFUSE_PX, N_MIN_EVIDENCE)
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    t = np.tanh(K_SHARP * pi_diff)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g = np.hypot(gx, gy).astype(np.float32)
    edge = confidence * g
    smoothed = gaussian_filter(edge, sigma=SIGMA_CUT_PX)
    if smoothed.max() > 0: smoothed = smoothed / smoothed.max() * DEPTH_MAX
    s18_cut = s18.astype(np.float32) * (1.0 - smoothed)

    # CP-SAM
    def norm_wsi(arr, key):
        return ((arr.astype(np.float32) - wsi_pct[key]["q_lo"]) /
                max(wsi_pct[key]["q_hi"] - wsi_pct[key]["q_lo"], 1e-6)).astype(np.float32)
    dapi_norm = norm_wsi(dapi, "DAPI")
    def run_cp(s_img):
        img = np.stack([dapi_norm, norm_wsi(s_img, "18S")], axis=-1).astype(np.float32)
        m_out, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m_out.astype(np.int32)
    m_ctrl = run_cp(s18)
    m_cut = run_cp(s18_cut)

    # Sign-change curve of π_diff (a 1-px contour)
    sign_pi = (pi_diff > 0).astype(np.uint8)
    sign_boundary = find_boundaries(sign_pi, mode="inner")
    # Restrict to focal cell
    sign_boundary_focal = sign_boundary & focal_region

    # Cut metrics inside focal
    focal_area = int(focal_region.sum())
    max_cut_focal = float(smoothed[focal_region].max())
    sig_cut_mask = smoothed > 0.3
    frac_significant_focal = float((sig_cut_mask & focal_region).sum() / focal_area)
    # Distance from sign-change line
    if sign_boundary_focal.any():
        dt_to_sign = distance_transform_edt(~sign_boundary_focal)
        # Alignment: % of significant-cut focal pixels within K=3 px of sign line
        sig_in_focal = sig_cut_mask & focal_region
        if sig_in_focal.any():
            dists = dt_to_sign[sig_in_focal]
            frac_aligned = float((dists <= 3).sum() / sig_in_focal.sum())
            median_align = float(np.median(dists))
        else:
            frac_aligned = float("nan"); median_align = float("nan")
    else:
        frac_aligned = float("nan"); median_align = float("nan")

    # CP-SAM split line inside the focal region
    # Get the unique labels inside focal_region and find boundaries between them
    cut_labels_in_focal = np.unique(m_cut[focal_region])
    cut_labels_in_focal = cut_labels_in_focal[cut_labels_in_focal != 0]
    # Build a boolean map of CP-SAM split lines inside focal
    cp_split = np.zeros_like(focal_region, dtype=bool)
    if len(cut_labels_in_focal) > 1:
        # boundaries between labels in m_cut, restricted to focal area
        b = find_boundaries(m_cut, mode="inner")
        cp_split = b & focal_region

    def count_focal(m_out):
        if focal_area == 0: return 0, []
        lbls = np.unique(m_out[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted(
            [(int(L), int(((m_out == L) & focal_region).sum()) / focal_area) for L in lbls],
            key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]
    n_ctrl, cov_ctrl = count_focal(m_ctrl)
    n_cut, cov_cut = count_focal(m_cut)

    # Render
    ys, xs = np.where(focal_region); pad = 25
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) main overlay
    ax = axes[0, 0]
    ax.imshow(norm01(s18[crop]), cmap="gray")
    # anchors
    sub = anch[(anch["px"] >= x0) & (anch["px"] < x1) &
                (anch["py"] >= y0) & (anch["py"] < y1)]
    handles = []
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"] == L]
        if len(s): ax.scatter(s["px"]-x0, s["py"]-y0, s=5, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
        handles.append(Patch(color=LINEAGE_COLOURS[L], label=f"{L} ({len(s)})"))
    # π_diff sign-change in cyan
    ax.contour((sign_boundary_focal[crop]).astype(int), levels=[0.5], colors="cyan", linewidths=2)
    # focal outline
    ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
    # CP-SAM split inside focal in white
    ax.contour((cp_split[crop]).astype(int), levels=[0.5], colors="white", linewidths=1.2)
    ax.set_title(f"{BID}  ({lin_a}×{lin_b})\n"
                 f"yellow=focal outline, cyan=π_diff sign-change (anchored boundary), "
                 f"white=CP-SAM split line under cut", fontsize=9)
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])

    # (0,1) π_diff
    im = axes[0,1].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0,1].contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
    axes[0,1].contour((sign_boundary_focal[crop]).astype(int), levels=[0.5], colors="cyan", linewidths=2)
    axes[0,1].set_title(f"π_{lin_a} − π_{lin_b}  (sign change = cyan)")
    axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
    fig.colorbar(im, ax=axes[0,1], fraction=0.04)

    # (0,2) cut depth field
    im = axes[0,2].imshow(smoothed[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX)
    axes[0,2].contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
    axes[0,2].contour((sign_boundary_focal[crop]).astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0,2].set_title(f"cut depth field (σ_cut={SIGMA_CUT_PX}px, depth_max={DEPTH_MAX})")
    axes[0,2].set_xticks([]); axes[0,2].set_yticks([])
    fig.colorbar(im, ax=axes[0,2], fraction=0.04)

    # (1,0) CP-SAM control with anchors
    axes[1,0].imshow(norm01(s18[crop]), cmap="gray")
    # mask overlay
    cnt = np.unique(m_ctrl[crop]); cnt = cnt[cnt != 0]
    rng = np.random.default_rng(7)
    cmap_arr = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(cnt), 1)))
    overlay = np.zeros((*m_ctrl[crop].shape, 4), dtype=np.float32)
    for i, L in enumerate(cnt):
        overlay[m_ctrl[crop] == L] = (*cmap_arr[i % len(cmap_arr)][:3], 0.45)
    axes[1,0].imshow(overlay)
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"] == L]
        if len(s): axes[1,0].scatter(s["px"]-x0, s["py"]-y0, s=5, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
    axes[1,0].contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
    axes[1,0].set_title(f"CONTROL (no cut)  n_focal={n_ctrl}\n"
                        f"covers={[f'{c*100:.0f}%' for _, c in cov_ctrl[:3]]}")
    axes[1,0].set_xticks([]); axes[1,0].set_yticks([])

    # (1,1) CP-SAM with cut
    axes[1,1].imshow(norm01(s18_cut[crop]), cmap="gray")
    cnt = np.unique(m_cut[crop]); cnt = cnt[cnt != 0]
    cmap_arr = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(cnt), 1)))
    overlay = np.zeros((*m_cut[crop].shape, 4), dtype=np.float32)
    for i, L in enumerate(cnt):
        overlay[m_cut[crop] == L] = (*cmap_arr[i % len(cmap_arr)][:3], 0.45)
    axes[1,1].imshow(overlay)
    for L in [lin_a, lin_b]:
        s = sub[sub["lineage"] == L]
        if len(s): axes[1,1].scatter(s["px"]-x0, s["py"]-y0, s=5, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")
    axes[1,1].contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
    axes[1,1].contour((sign_boundary_focal[crop]).astype(int), levels=[0.5], colors="cyan", linewidths=2)
    axes[1,1].set_title(f"CUT (Method F)  n_focal={n_cut}\n"
                        f"covers={[f'{c*100:.0f}%' for _, c in cov_cut[:3]]}")
    axes[1,1].set_xticks([]); axes[1,1].set_yticks([])

    # (1,2) text summary
    axes[1,2].axis("off")
    summary = (
        f"{BID}\n  pair: {lin_a} × {lin_b}, cps={cps_focal}\n"
        f"  focal area: {focal_area} px\n"
        f"\nANCHORS INSIDE FOCAL:\n"
        f"  {lin_a}: {int((anch.loc[(anch['lineage']==lin_a), :].apply(lambda r: focal_region[int(r['py']),int(r['px'])] if 0<=r['py']<H and 0<=r['px']<W else False, axis=1)).sum())}\n"
        f"  {lin_b}: {int((anch.loc[(anch['lineage']==lin_b), :].apply(lambda r: focal_region[int(r['py']),int(r['px'])] if 0<=r['py']<H and 0<=r['px']<W else False, axis=1)).sum())}\n"
        f"\nCUT FIELD INSIDE FOCAL:\n"
        f"  max cut depth: {max_cut_focal:.2f} / {DEPTH_MAX:.1f} max\n"
        f"  % focal pixels w/ cut > 0.3: {100*frac_significant_focal:.0f}%\n"
        f"\nALIGNMENT WITH π_diff SIGN-CHANGE LINE:\n"
        f"  sign-change pixels in focal: {int(sign_boundary_focal.sum())}\n"
        f"  median dist sig-cut → sign line: {median_align:.1f} px\n"
        f"  % sig-cut pixels within 3px of sign line: "
        f"{100*frac_aligned:.0f}%\n" if not np.isnan(frac_aligned) else "  N/A (no sign line)\n"
        f"\nCP-SAM RESULT:\n"
        f"  control n_focal={n_ctrl}, cut n_focal={n_cut}, Δ={n_cut-n_ctrl}\n"
    )
    axes[1,2].text(0.02, 0.98, summary, transform=axes[1,2].transAxes,
                    fontsize=9, family="monospace", va="top")

    plt.suptitle(
        f"Method F failure diagnostic — {BID}",
        fontsize=12, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_diagnostic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")
    return dict(BID=BID, n_ctrl=n_ctrl, n_cut=n_cut,
                max_cut_focal=max_cut_focal,
                frac_significant_focal=frac_significant_focal,
                median_align=median_align, frac_aligned=frac_aligned)


def main():
    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()

    results = []
    for bid in ["DB_top100_015", "DB_top100_018", "DB_top100_098"]:
        print(f"\n=== {bid} ===")
        r = diagnose_cell(bid, m_sam, wsi_pct, gene_names, gl)
        results.append(r)
        print(f"  max_cut_focal={r['max_cut_focal']:.2f}, "
              f"%>0.3={100*r['frac_significant_focal']:.0f}%, "
              f"%aligned_w/_sign_line={100*r['frac_aligned']:.0f}%")

    print(f"\nresults: {results}")


if __name__ == "__main__":
    main()
