"""Step 1: textbook DiZenzo edge detection on the 7-channel posterior field π_t(p).

Pipeline (per benchmark mini-ROI):
  1. Compute per-lineage point masses → σ=2 µm linear-conductance diffusion
     → Dirichlet posterior π_t(p) (no grey-abstain blend here; π is the
     mathematical object we want edges in).
  2. Sobel gradients per lineage channel.
  3. DiZenzo structure tensor: J^T·J where J is (2 × K) per pixel.
     λ_max = edge magnitude; eigenvector = gradient direction.
  4. Non-maximum suppression along the gradient direction (Canny-style).
  5. Hysteresis thresholding (two thresholds, connected-component propagate).

No confidence-gating or type-pair vetoing yet — those are layers 2 and 3.
This script outputs every intermediate step as a panel so we can see what
each stage contributes.

Output: figs/edges_dizenzo/<benchmark_id>.png
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
from skimage.morphology import skeletonize

from tint_cytoplasm_diffusion import (
    ALPHA, GREY, LINEAGE_COLOURS, LINEAGES, N_MIN_EVIDENCE,
    diffuse_anisotropic, hex_to_rgb, make_conductance, norm01,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
ROIS_DIR    = DATA / "benchmark_doublet_rois"
FIGS_DIR    = ROOT / "figs" / "edges_dizenzo"

PIXEL_SIZE_UM = 0.2125
SIGMA_UM      = 2.0
DT            = 0.2
SIGMA_PX      = SIGMA_UM / PIXEL_SIZE_UM
N_ITER        = int(round(SIGMA_PX ** 2 / DT))

# Pre-gradient smoothing (small, just to stabilise Sobel)
SIGMA_PRE_PX  = 1.0
# Post-gradient (structure-tensor) smoothing — the "neighbourhood" scale that
# determines which gradient direction dominates locally. Standard step in
# DiZenzo / Förstner / Harris-style structure tensor methods.
SIGMA_POST_PX = 3.0

# Hysteresis thresholds — set as percentiles of λ_max, tunable
T_HIGH_PERC = 95.0
T_LOW_PERC  = 80.0


# ── tinting (matches eval pipeline) ───────────────────────────────────────
def make_pi(roi_dir, gene_names, gl):
    s18 = tifffile.imread(roi_dir / "morphology_18S.tif")
    tx  = pd.read_parquet(roi_dir / "transcripts.parquet")
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

    # Dirichlet posterior (no grey blending — we want π for edge math)
    N = rho_d * 2.0 * np.pi * SIGMA_PX ** 2
    N_total    = N.sum(axis=0)
    pi_post    = (N + ALPHA) / (N_total + K * ALPHA)[None]
    confidence = N_total / (N_total + N_MIN_EVIDENCE)

    s18_n = norm01(s18)
    return {
        "s18":         s18, "s18_n": s18_n,
        "pi":          pi_post.astype(np.float32),   # (K, H, W), sums to 1
        "confidence":  confidence.astype(np.float32),
        "anch":        anch, "li_arr": li_arr,
        "meta":        meta,
    }


def tint_for_display(pi, confidence, colors):
    """Posterior-mean tint with grey-abstain blend (for visualization only)."""
    tint_post = np.einsum("khw,kc->hwc", pi, colors)
    out = (confidence[..., None] * tint_post
           + (1.0 - confidence[..., None]) * GREY[None, None, :])
    return out


# ── DiZenzo structure tensor & Canny pipeline ─────────────────────────────
def dizenzo_structure_tensor(pi, sigma_pre_px=SIGMA_PRE_PX, sigma_post_px=SIGMA_POST_PX):
    """Return (λ_max, λ_min, theta_grad) per pixel.

    pi: (K, H, W) probability field
    sigma_pre_px:  pre-Sobel smoothing per channel (noise reduction)
    sigma_post_px: smoothing applied to G_xx, G_yy, G_xy (the neighbourhood-
                   scale at which the structure tensor reports the dominant
                   gradient direction). Standard step in DiZenzo / Förstner /
                   Harris — without it, edges are per-pixel noisy specks.
    theta_grad: angle of the dominant gradient direction, in radians.
                The edge tangent is perpendicular to this.
    """
    K, H, W = pi.shape
    Jx = np.empty((K, H, W), dtype=np.float32)
    Jy = np.empty((K, H, W), dtype=np.float32)
    for k in range(K):
        p = gaussian_filter(pi[k], sigma=sigma_pre_px)
        # scipy.ndimage.sobel: axis=1 is x (cols), axis=0 is y (rows)
        Jx[k] = sobel(p, axis=1, mode="reflect") / 8.0
        Jy[k] = sobel(p, axis=0, mode="reflect") / 8.0

    G_xx = (Jx ** 2).sum(axis=0)
    G_yy = (Jy ** 2).sum(axis=0)
    G_xy = (Jx * Jy).sum(axis=0)

    # Structure-tensor smoothing (the key step we missed before)
    G_xx = gaussian_filter(G_xx, sigma=sigma_post_px)
    G_yy = gaussian_filter(G_yy, sigma=sigma_post_px)
    G_xy = gaussian_filter(G_xy, sigma=sigma_post_px)

    # Eigenvalues of [[G_xx, G_xy], [G_xy, G_yy]]
    trace = G_xx + G_yy
    det   = G_xx * G_yy - G_xy ** 2
    disc  = np.sqrt(np.maximum((trace / 2) ** 2 - det, 0.0))
    lam_max = trace / 2 + disc
    lam_min = trace / 2 - disc

    # Gradient direction (for the larger eigenvalue) in (x, y)
    # Standard formula: 2·θ = atan2(2·G_xy, G_xx − G_yy)
    theta_grad = 0.5 * np.arctan2(2 * G_xy, G_xx - G_yy + 1e-12)

    return lam_max.astype(np.float32), lam_min.astype(np.float32), theta_grad.astype(np.float32)


def nms(magnitude, theta_grad):
    """Non-max suppression along the gradient direction.

    For each pixel, compare to the two neighbours along ±theta_grad. Suppress
    if not a local max. Returns thinned magnitude.
    """
    # Quantise to 4 bins
    angle = (np.degrees(theta_grad) + 180) % 180  # [0, 180)
    out = magnitude.copy()
    # bin definitions and corresponding (dy, dx) offsets for one of two neighbours
    bins = [
        ((angle <  22.5) | (angle >= 157.5),  ( 0,  1)),   # 0°    (horizontal grad)
        ((angle >=  22.5) & (angle <  67.5),  (-1,  1)),   # 45°
        ((angle >=  67.5) & (angle < 112.5),  (-1,  0)),   # 90°   (vertical grad)
        ((angle >= 112.5) & (angle < 157.5),  (-1, -1)),   # 135°
    ]
    for mask, (dy, dx) in bins:
        n1 = np.roll(np.roll(magnitude,  dy, 0),  dx, 1)
        n2 = np.roll(np.roll(magnitude, -dy, 0), -dx, 1)
        suppress = mask & ((magnitude < n1) | (magnitude < n2))
        out[suppress] = 0
    return out


def hysteresis(magnitude, t_low, t_high):
    """Connect strong-edge seeds to weak-edge components."""
    strong = magnitude >= t_high
    weak   = magnitude >= t_low
    labels, _ = label(weak, structure=np.ones((3, 3), dtype=np.uint8))
    keep_ids = set(np.unique(labels[strong]).tolist())
    keep_ids.discard(0)
    out = np.isin(labels, list(keep_ids))
    return out


# ── plotting ──────────────────────────────────────────────────────────────
def render(bid, data, lam_max, lam_max_gated, lam_max_nms, edges_bin, gate, colors):
    s18_n      = data["s18_n"]
    pi         = data["pi"]
    confidence = data["confidence"]
    anch       = data["anch"]
    li_arr     = data["li_arr"]
    meta       = data["meta"]
    H, W = s18_n.shape

    tint = tint_for_display(pi, confidence, colors)
    rgb_18s = (s18_n[..., None] * tint).astype(np.float32)

    # Argmax categorical, gated by 18S·confidence (the same gate used for edges)
    argmax = pi.argmax(axis=0)
    argmax_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ki, L in enumerate(LINEAGES):
        argmax_rgb[argmax == ki] = hex_to_rgb(LINEAGE_COLOURS[L])
    argmax_rgb_gated = argmax_rgb * gate[..., None]   # dim by gate

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 18S + anchors
    axes[0, 0].imshow(s18_n, cmap="gray")
    for li, L in enumerate(LINEAGES):
        sub = anch[li_arr == li]
        if len(sub):
            axes[0, 0].scatter(sub["px"], sub["py"], s=10,
                                c=LINEAGE_COLOURS[L], edgecolor="none", alpha=0.6)
    axes[0, 0].set_title(f"18S + anchored transcripts\n{meta['lineage_pair']}")

    # Tinted RGB (the cytoplasm-tint visualization)
    axes[0, 1].imshow(rgb_18s)
    axes[0, 1].set_title(f"tinted cytoplasm  (σ={SIGMA_UM} µm linear diff)")

    # Argmax categorical, gated
    axes[0, 2].imshow(argmax_rgb_gated)
    axes[0, 2].set_title("argmax lineage  (gated by 18S · confidence)")

    # λ_max raw (no gating)
    im3 = axes[0, 3].imshow(lam_max, cmap="magma")
    axes[0, 3].set_title("DiZenzo λ_max raw")
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.02)

    # λ_max · gate
    im4 = axes[1, 0].imshow(lam_max_gated, cmap="magma")
    axes[1, 0].set_title("λ_max × (confidence · 18S)\n(gated edge magnitude)")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.02)

    # NMS on gated
    im5 = axes[1, 1].imshow(lam_max_nms, cmap="magma")
    axes[1, 1].set_title("after NMS on gated (thinned)")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.02)

    # Hysteresis: edges over tinted
    rgb_over = rgb_18s.copy()
    rgb_over[edges_bin] = (1.0, 1.0, 1.0)
    n_keep = int(edges_bin.sum())
    axes[1, 2].imshow(rgb_over)
    axes[1, 2].set_title(f"edges over tinted RGB  ({n_keep} px)")

    # Edges over 18S
    rgb18 = np.stack([s18_n, s18_n, s18_n], axis=-1)
    rgb18[edges_bin] = (1.0, 0.0, 1.0)   # magenta on grayscale
    axes[1, 3].imshow(rgb18)
    axes[1, 3].set_title("edges over raw 18S")

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"{bid} — DiZenzo edges, gated by confidence · 18S",
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

    data = make_pi(ROIS_DIR / bid, gene_names, gl)
    print(f"[{bid}] π shape {data['pi'].shape}  ({data['meta']['lineage_pair']})")

    lam_max, lam_min, theta = dizenzo_structure_tensor(data["pi"])
    print(f"  λ_max range:        [{lam_max.min():.2e}, {lam_max.max():.2e}]")

    # Step 2: gate λ_max by (confidence · 18S intensity)
    # Both factors per-pixel in [0, 1]; multiplied so an edge survives only
    # where the posterior is confident AND there's real cytoplasm signal.
    s18_n      = data["s18_n"]
    confidence = data["confidence"]
    gate       = (confidence * s18_n).astype(np.float32)
    lam_max_gated = lam_max * gate
    print(f"  gate range:         [{gate.min():.3f}, {gate.max():.3f}]  "
          f"(median {np.median(gate):.3f})")
    print(f"  λ_max·gate range:   [{lam_max_gated.min():.2e}, {lam_max_gated.max():.2e}]")

    lam_max_nms = nms(lam_max_gated, theta)
    nonzero = lam_max_nms[lam_max_nms > 0]
    if nonzero.size == 0:
        print("  NMS produced no edges; skipping hysteresis.")
        edges_bin = np.zeros_like(lam_max_nms, dtype=bool)
    else:
        t_low  = np.percentile(nonzero, T_LOW_PERC)
        t_high = np.percentile(nonzero, T_HIGH_PERC)
        edges_bin = hysteresis(lam_max_nms, t_low, t_high)
        print(f"  T_low={t_low:.2e}  T_high={t_high:.2e}  → {int(edges_bin.sum())} edge px")

    render(bid, data, lam_max, lam_max_gated, lam_max_nms, edges_bin, gate, colors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    args = ap.parse_args()
    main(args.benchmark_id)
