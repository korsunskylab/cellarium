"""Three-way comparison of edge-detection formulations on one mini-ROI:

  A. Structure tensor on  (π · 18S)         — gradient of product field
  B. Structure tensor on  π,  then × 18S    — pure type-gradient, output-weighted
  C. Structure tensor on  18S alone (K=1)   — intensity edges only, no lineage info

All three produce a per-pixel λ_max field that we put through identical NMS +
hysteresis. The output binary edge masks feed the downstream CP-SAM cut test.

Output: figs/edges_three_methods/<bid>.png  +  per-method binary edge maps
        saved alongside as <bid>_edges_<method>.tif for the eval pipeline.
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
FIGS_DIR    = ROOT / "figs" / "edges_three_methods"
EDGES_DIR   = ROOT / "data" / "edges_three_methods"

PIXEL_SIZE_UM = 0.2125
SIGMA_UM      = 2.0
DT            = 0.2
SIGMA_PX      = SIGMA_UM / PIXEL_SIZE_UM
N_ITER        = int(round(SIGMA_PX ** 2 / DT))

SIGMA_PRE_PX  = 1.0
SIGMA_POST_PX = 3.0

T_HIGH_PERC = 95.0
T_LOW_PERC  = 80.0


def make_pi(roi_dir, gene_names, gl):
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
    N        = rho_d * 2.0 * np.pi * SIGMA_PX ** 2
    N_total  = N.sum(axis=0)
    pi_post  = (N + ALPHA) / (N_total + K * ALPHA)[None]

    return {
        "s18": s18, "s18_n": norm01(s18),
        "dapi": dapi, "dapi_n": norm01(dapi),
        "pi": pi_post.astype(np.float32),
        "anch": anch, "li_arr": li_arr, "meta": meta,
    }


def dizenzo_lam_max(field, sigma_pre_px=SIGMA_PRE_PX, sigma_post_px=SIGMA_POST_PX):
    """Structure tensor for an arbitrary K-channel field.
    field: (K, H, W) or (H, W) — single-channel is auto-promoted to (1, H, W).
    Returns (λ_max, θ_grad).
    """
    if field.ndim == 2:
        field = field[None]
    K, H, W = field.shape
    Jx = np.empty((K, H, W), dtype=np.float32)
    Jy = np.empty((K, H, W), dtype=np.float32)
    for k in range(K):
        p = gaussian_filter(field[k], sigma=sigma_pre_px)
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
    lam_max = (trace / 2 + disc).astype(np.float32)
    theta   = (0.5 * np.arctan2(2 * G_xy, G_xx - G_yy + 1e-12)).astype(np.float32)
    return lam_max, theta


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


def edges_pipeline(lam_max, theta):
    """NMS + percentile hysteresis. Returns (binary edge mask, threshold info)."""
    nm = nms(lam_max, theta)
    nz = nm[nm > 0]
    if nz.size == 0:
        return np.zeros_like(nm, dtype=bool), (0.0, 0.0)
    t_low  = float(np.percentile(nz, T_LOW_PERC))
    t_high = float(np.percentile(nz, T_HIGH_PERC))
    return hysteresis(nm, t_low, t_high), (t_low, t_high)


def method_A(pi, s18_n):
    """gradient of (π · 18S)"""
    field = pi * s18_n[None]
    return dizenzo_lam_max(field)

def method_B(pi, s18_n):
    """gradient of π, then multiply λ_max by 18S"""
    lam, theta = dizenzo_lam_max(pi)
    return (lam * s18_n).astype(np.float32), theta

def method_C(s18_n):
    """gradient of 18S alone (K=1)"""
    return dizenzo_lam_max(s18_n)


def tint_for_display(pi, colors):
    return np.einsum("khw,kc->hwc", pi, colors)


def render(bid, data, results, colors):
    s18_n = data["s18_n"]; dapi_n = data["dapi_n"]
    anch  = data["anch"]; li_arr = data["li_arr"]; meta = data["meta"]

    tint = tint_for_display(data["pi"], colors)
    rgb_tint = (s18_n[..., None] * tint).astype(np.float32)
    rgb_dapi_18s = np.stack([s18_n, s18_n, dapi_n], axis=-1)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    # Row 0: reference panels
    axes[0, 0].imshow(rgb_dapi_18s)
    axes[0, 0].set_title("DAPI (blue) + 18S (yellow)")
    axes[0, 1].imshow(s18_n, cmap="gray")
    for li, L in enumerate(LINEAGES):
        sub = anch[li_arr == li]
        if len(sub):
            axes[0, 1].scatter(sub["px"], sub["py"], s=10,
                                c=LINEAGE_COLOURS[L], edgecolor="none", alpha=0.6)
    axes[0, 1].set_title(f"18S + anchored tx\n{meta['lineage_pair']}")
    axes[0, 2].imshow(rgb_tint)
    axes[0, 2].set_title("tinted cytoplasm  (σ=2 µm linear diffusion)")
    axes[0, 3].axis("off")
    axes[0, 3].text(0.05, 0.95, f"{bid}\n\nrank {meta['rank']}\n{meta['lineage_pair']}\n"
                                  f"top1={meta['top1_n']}  top2={meta['top2_n']}\n\n"
                                  f"σ_pre={SIGMA_PRE_PX} px\nσ_post={SIGMA_POST_PX} px\n"
                                  f"T_low={T_LOW_PERC}%  T_high={T_HIGH_PERC}%",
                     va="top", ha="left", family="monospace", fontsize=10,
                     transform=axes[0, 3].transAxes)

    # Rows 1, 2: A, B, C — λ_max heat, edges over tint, edges over 18S
    for ri, (name, (lam, theta, edges, thr)) in enumerate(results.items()):
        col0_ax = axes[ri + 1, 0]
        col1_ax = axes[ri + 1, 1]
        col2_ax = axes[ri + 1, 2]
        col3_ax = axes[ri + 1, 3]

        im = col0_ax.imshow(lam, cmap="magma")
        col0_ax.set_title(f"Method {name}: λ_max")
        plt.colorbar(im, ax=col0_ax, fraction=0.046, pad=0.02)

        col1_ax.imshow(nms(lam, theta), cmap="magma")
        col1_ax.set_title(f"{name}: after NMS")

        rgb_over_tint = rgb_tint.copy()
        rgb_over_tint[edges] = (1.0, 1.0, 1.0)
        col2_ax.imshow(rgb_over_tint)
        col2_ax.set_title(f"{name}: edges over tint  ({int(edges.sum())} px)")

        rgb_over_18s = np.stack([s18_n, s18_n, s18_n], axis=-1)
        rgb_over_18s[edges] = (1.0, 0.0, 1.0)   # magenta
        col3_ax.imshow(rgb_over_18s)
        col3_ax.set_title(f"{name}: edges over 18S")

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"{bid} — three edge methods: "
                 f"A=∇(π·18S)  B=∇π then ×18S  C=∇18S alone",
                 y=1.005)
    plt.tight_layout()
    out_png = FIGS_DIR / f"{bid}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_png}")


def main(bid):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    EDGES_DIR.mkdir(parents=True, exist_ok=True)

    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    colors = np.stack([hex_to_rgb(LINEAGE_COLOURS[L]) for L in LINEAGES])

    data = make_pi(ROIS_DIR / bid, gene_names, gl)
    print(f"[{bid}] π shape {data['pi'].shape}  ({data['meta']['lineage_pair']})")

    results = {}
    for name, fn in [("A", lambda: method_A(data["pi"], data["s18_n"])),
                     ("B", lambda: method_B(data["pi"], data["s18_n"])),
                     ("C", lambda: method_C(data["s18_n"]))]:
        lam, theta = fn()
        edges, thr = edges_pipeline(lam, theta)
        results[name] = (lam, theta, edges, thr)
        print(f"  Method {name}:  λ_max range [{lam.min():.2e}, {lam.max():.2e}]  "
              f"T_low={thr[0]:.2e}  T_high={thr[1]:.2e}  → {int(edges.sum())} edge px")
        # save binary edge mask for downstream eval
        tifffile.imwrite(EDGES_DIR / f"{bid}_edges_{name}.tif",
                          edges.astype(np.uint8), compression="zlib")

    render(bid, data, results, colors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_006")
    args = ap.parse_args()
    main(args.benchmark_id)
