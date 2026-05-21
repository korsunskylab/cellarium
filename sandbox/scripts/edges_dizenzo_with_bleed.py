"""Step 2: DiZenzo edges with normalized-convolution color-bleed across a
validity mask derived from imaging signals only — NO CP-SAM outputs.

Pipeline:
  1. π from σ=2 µm linear-conductance diffusion + Dirichlet posterior (unchanged).
  2. Validity mask from DAPI ∨ 18S (purely imaging, pre-segmentation):
        soft weight  w(p) = max(DAPI_n(p), 18S_n(p))   ∈ [0, 1]
        binary mask  M(p) = w(p) > τ_cell             (τ_cell ≈ 0.15)
     Inside nucleus → DAPI bright; inside cytoplasm → 18S bright;
     anywhere inside a cell → at least one is bright; background → both dim.
  3. Normalized convolution to bleed π out of the mask, per channel:
        π_ext_k(p) = G(π_k · w)(p) / G(w)(p)     with σ_bleed = 2 px (small)
  4. DiZenzo structure tensor on π_ext (with pre+post smoothing) → λ_max, θ.
  5. NMS along ∇ direction; hysteresis on percentiles.
  6. Final edges = (edges_bin) AND M.

Output: figs/edges_with_bleed/<bid>.png
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
from scipy.ndimage import gaussian_filter, label, sobel

from tint_cytoplasm_diffusion import (
    ALPHA, GREY, LINEAGE_COLOURS, LINEAGES, N_MIN_EVIDENCE,
    diffuse_anisotropic, hex_to_rgb, make_conductance, norm01,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
ROIS_DIR    = DATA / "benchmark_doublet_rois"
FIGS_DIR    = ROOT / "figs" / "edges_with_bleed"

PIXEL_SIZE_UM   = 0.2125
SIGMA_UM        = 2.0
DT              = 0.2
SIGMA_PX        = SIGMA_UM / PIXEL_SIZE_UM
N_ITER          = int(round(SIGMA_PX ** 2 / DT))

SIGMA_PRE_PX    = 1.0
SIGMA_POST_PX   = 3.0
SIGMA_BLEED_PX  = 2.0                 # bleed kernel ~ 2 px (≈0.4 µm); just enough
                                       # to make π smooth across the mask boundary,
                                       # but not large enough to wash out the
                                       # inside-mask gradients we want to detect

CELL_MASK_TAU   = 0.15                # threshold on max(DAPI_n, 18S_n) to define
                                       # "inside any cell" — fully pre-segmentation

T_HIGH_PERC = 95.0
T_LOW_PERC  = 80.0


def make_pi_and_meta(roi_dir, gene_names, gl):
    s18  = tifffile.imread(roi_dir / "morphology_18S.tif")
    dapi = tifffile.imread(roi_dir / "morphology_DAPI.tif")
    tx   = pd.read_parquet(roi_dir / "transcripts.parquet")
    meta = json.loads((roi_dir / "metadata.json").read_text())
    H, W = s18.shape
    cx0 = meta["bbox_crop_um"]["x0"]; cy0 = meta["bbox_crop_um"]["y0"]

    tx["gene"]    = [gene_names[g] for g in tx["gene_id"].values]
    tx["lineage"] = tx["gene"].map(gl)
    anch = tx[tx["lineage"].isin(LINEAGES)].copy()
    anch["px"] = np.rint((anch["x_um"] - cx0) / PIXEL_SIZE_UM).astype(int)
    anch["py"] = np.rint((anch["y_um"] - cy0) / PIXEL_SIZE_UM).astype(int)
    anch = anch[(anch["px"] >= 0) & (anch["px"] < W) &
                (anch["py"] >= 0) & (anch["py"] < H)].copy()

    K = len(LINEAGES)
    lin_to_idx = {L: i for i, L in enumerate(LINEAGES)}
    li_arr = anch["lineage"].map(lin_to_idx).to_numpy()
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)

    c_linear = make_conductance(s18, mode="linear")
    rho_d    = diffuse_anisotropic(rho_pts, c_linear, n_iter=N_ITER, dt=DT)
    N = rho_d * 2.0 * np.pi * SIGMA_PX ** 2
    N_total    = N.sum(axis=0)
    pi_post    = (N + ALPHA) / (N_total + K * ALPHA)[None]
    confidence = N_total / (N_total + N_MIN_EVIDENCE)

    s18_n  = norm01(s18)
    dapi_n = norm01(dapi)
    return {
        "s18": s18, "s18_n": s18_n,
        "dapi": dapi, "dapi_n": dapi_n,
        "pi": pi_post.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "anch": anch, "li_arr": li_arr, "meta": meta,
    }


def cell_mask_from_dapi_18s(dapi_n, s18_n, tau=CELL_MASK_TAU):
    """Pre-segmentation 'inside-any-cell' mask from imaging alone.

    soft_weight = max(DAPI_n, 18S_n)  ∈ [0, 1]
        — high at nuclei (DAPI bright), high at cytoplasm (18S bright),
          low at extracellular background where both are dim
    binary_mask = soft_weight > tau

    Returns (soft_weight, binary_mask).
    """
    soft = np.maximum(dapi_n, s18_n).astype(np.float32)
    return soft, soft > tau


def normalized_convolution(field, weight, sigma_px):
    """f_smooth(p) = G(f·w)(p) / G(w)(p), per channel.

    Knutsson (1993). Extends valid-region values into low-weight regions via
    a weighted local average, with the same kernel shape used for both
    numerator and denominator so the result is smooth across the mask edge.
    """
    K = field.shape[0]
    den = gaussian_filter(weight, sigma=sigma_px)
    den = np.maximum(den, 1e-6)
    out = np.empty_like(field)
    for k in range(K):
        num = gaussian_filter(field[k] * weight, sigma=sigma_px)
        out[k] = num / den
    return out


def dizenzo_structure_tensor(pi, sigma_pre_px=SIGMA_PRE_PX, sigma_post_px=SIGMA_POST_PX):
    K, H, W = pi.shape
    Jx = np.empty((K, H, W), dtype=np.float32)
    Jy = np.empty((K, H, W), dtype=np.float32)
    for k in range(K):
        p = gaussian_filter(pi[k], sigma=sigma_pre_px)
        Jx[k] = sobel(p, axis=1, mode="reflect") / 8.0
        Jy[k] = sobel(p, axis=0, mode="reflect") / 8.0
    G_xx = (Jx ** 2).sum(axis=0)
    G_yy = (Jy ** 2).sum(axis=0)
    G_xy = (Jx * Jy).sum(axis=0)
    G_xx = gaussian_filter(G_xx, sigma=sigma_post_px)
    G_yy = gaussian_filter(G_yy, sigma=sigma_post_px)
    G_xy = gaussian_filter(G_xy, sigma=sigma_post_px)
    trace = G_xx + G_yy
    det   = G_xx * G_yy - G_xy ** 2
    disc  = np.sqrt(np.maximum((trace / 2) ** 2 - det, 0.0))
    lam_max = trace / 2 + disc
    theta   = 0.5 * np.arctan2(2 * G_xy, G_xx - G_yy + 1e-12)
    return lam_max.astype(np.float32), theta.astype(np.float32)


def nms(magnitude, theta_grad):
    angle = (np.degrees(theta_grad) + 180) % 180
    out = magnitude.copy()
    bins = [
        ((angle <  22.5) | (angle >= 157.5),  ( 0,  1)),
        ((angle >=  22.5) & (angle <  67.5),  (-1,  1)),
        ((angle >=  67.5) & (angle < 112.5),  (-1,  0)),
        ((angle >= 112.5) & (angle < 157.5),  (-1, -1)),
    ]
    for mask, (dy, dx) in bins:
        n1 = np.roll(np.roll(magnitude,  dy, 0),  dx, 1)
        n2 = np.roll(np.roll(magnitude, -dy, 0), -dx, 1)
        suppress = mask & ((magnitude < n1) | (magnitude < n2))
        out[suppress] = 0
    return out


def hysteresis(magnitude, t_low, t_high):
    strong = magnitude >= t_high
    weak   = magnitude >= t_low
    labels, _ = label(weak, structure=np.ones((3, 3), dtype=np.uint8))
    keep_ids = set(np.unique(labels[strong]).tolist())
    keep_ids.discard(0)
    return np.isin(labels, list(keep_ids))


def tint_for_display(pi, confidence, colors):
    tint_post = np.einsum("khw,kc->hwc", pi, colors)
    return (confidence[..., None] * tint_post +
            (1.0 - confidence[..., None]) * GREY[None, None, :])


def render(bid, data, soft_weight, binary_mask, pi_ext, lam_max,
           lam_max_nms, edges_final, colors):
    s18_n      = data["s18_n"]
    pi         = data["pi"]
    confidence = data["confidence"]
    anch       = data["anch"]
    li_arr     = data["li_arr"]
    meta       = data["meta"]
    H, W = s18_n.shape

    tint     = tint_for_display(pi,     confidence, colors)
    tint_ext = tint_for_display(pi_ext, np.ones_like(confidence), colors)
    rgb_18s  = (s18_n[..., None] * tint).astype(np.float32)
    rgb_ext  = tint_ext.astype(np.float32)   # show bleed-extended π raw

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(s18_n, cmap="gray")
    for li, L in enumerate(LINEAGES):
        sub = anch[li_arr == li]
        if len(sub):
            axes[0, 0].scatter(sub["px"], sub["py"], s=10,
                                c=LINEAGE_COLOURS[L], edgecolor="none", alpha=0.6)
    axes[0, 0].set_title(f"18S + anchored tx\n{meta['lineage_pair']}")

    axes[0, 1].imshow(rgb_18s)
    axes[0, 1].set_title(f"tinted cytoplasm  (σ={SIGMA_UM} µm linear diff)")

    im2 = axes[0, 2].imshow(soft_weight, cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].contour(binary_mask, levels=[0.5], colors="white", linewidths=0.8)
    axes[0, 2].set_title(f"cell mask = max(DAPI, 18S) > {CELL_MASK_TAU}\n(pre-segmentation, white = mask)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.02)

    axes[0, 3].imshow(rgb_ext)
    axes[0, 3].set_title(f"π after normalized-conv bleed\nσ_bleed={SIGMA_BLEED_PX:.1f} px")

    im4 = axes[1, 0].imshow(lam_max, cmap="magma")
    axes[1, 0].set_title("DiZenzo λ_max  (on bleed-extended π)")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.02)

    im5 = axes[1, 1].imshow(lam_max_nms, cmap="magma")
    axes[1, 1].set_title("after NMS (thinned)")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.02)

    rgb_over_tint = rgb_18s.copy()
    rgb_over_tint[edges_final] = (1.0, 1.0, 1.0)
    n_keep = int(edges_final.sum())
    axes[1, 2].imshow(rgb_over_tint)
    axes[1, 2].contour(binary_mask, levels=[0.5], colors="cyan", linewidths=0.5, alpha=0.6)
    axes[1, 2].set_title(f"edges (masked by cellprob>0) over tint\n{n_keep} edge px")

    rgb_over_18s = np.stack([s18_n, s18_n, s18_n], axis=-1)
    rgb_over_18s[edges_final] = (1.0, 0.0, 1.0)
    axes[1, 3].imshow(rgb_over_18s)
    axes[1, 3].contour(binary_mask, levels=[0.5], colors="cyan", linewidths=0.5, alpha=0.6)
    axes[1, 3].set_title("edges over raw 18S\n(cyan = cellprob>0 outline)")

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"{bid} — DiZenzo edges with normalized-convolution bleed",
                 y=1.005)
    plt.tight_layout()
    out_png = FIGS_DIR / f"{bid}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")


def main(bid):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    colors = np.stack([hex_to_rgb(LINEAGE_COLOURS[L]) for L in LINEAGES])

    data = make_pi_and_meta(ROIS_DIR / bid, gene_names, gl)
    print(f"[{bid}] π shape {data['pi'].shape}  ({data['meta']['lineage_pair']})")

    soft_weight, binary_mask = cell_mask_from_dapi_18s(data["dapi_n"], data["s18_n"])
    in_frac = float(binary_mask.mean())
    print(f"  cell mask = max(DAPI, 18S) > {CELL_MASK_TAU}  "
          f"covers {in_frac*100:.1f}% of crop")

    pi_ext = normalized_convolution(data["pi"], soft_weight, SIGMA_BLEED_PX)

    lam_max, theta = dizenzo_structure_tensor(pi_ext)
    print(f"  λ_max range: [{lam_max.min():.2e}, {lam_max.max():.2e}]")

    lam_max_nms = nms(lam_max, theta)
    nonzero = lam_max_nms[lam_max_nms > 0]
    if nonzero.size == 0:
        edges_bin = np.zeros_like(lam_max_nms, dtype=bool)
    else:
        t_low  = np.percentile(nonzero, T_LOW_PERC)
        t_high = np.percentile(nonzero, T_HIGH_PERC)
        edges_bin = hysteresis(lam_max_nms, t_low, t_high)
        print(f"  T_low={t_low:.2e}  T_high={t_high:.2e}  → {int(edges_bin.sum())} raw edge px")

    # Final mask: keep only edges where binary_mask is true
    edges_final = edges_bin & binary_mask
    n_kept = int(edges_final.sum())
    print(f"  after cellprob>0 output mask: {n_kept} edge px")

    render(bid, data, soft_weight, binary_mask, pi_ext, lam_max,
           lam_max_nms, edges_final, colors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    args = ap.parse_args()
    main(args.benchmark_id)
