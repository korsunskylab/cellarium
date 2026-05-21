"""Method E — multi-scale ridge detection on grey-abstained π_diff.

USER INSIGHT: the diffusion smooths the lineage transition across many pixels.
At single-pixel resolution the gradient is small; at the scale of the transition
itself, the same hill becomes a sharp, confident peak. Replace the fixed 3-pixel
Sobel with derivative-of-Gaussian at several σ, take the max across σ, NMS to
find the ridge, and weight by confidence — no global percentile threshold.

Pipeline:
  1. π_a, π_b from canonical make_pi
  2. apply grey-abstain (the bug we found earlier)
  3. compute |∇π_diff| via gaussian_gradient_magnitude at σ ∈ {1, 3, 6, 12} px
  4. multi-scale ridge: per-pixel max over σ; also record which σ won
  5. NMS along gradient direction at the winning σ
  6. final edge field = NMS · confidence (no hysteresis)
  7. cut 18S, run CP-SAM, compare to control

The user's bilateral-filter idea (limit diffusion across color boundaries) is
noted as a follow-up — not implemented in this iteration.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter, gaussian_gradient_magnitude, sobel, binary_dilation,
)
from skimage.segmentation import find_boundaries
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import make_pi, nms, SIGMA_PX as SIGMA_DIFFUSE_PX
from tint_cytoplasm_diffusion import (
    LINEAGES, LINEAGE_COLOURS, ALPHA, N_MIN_EVIDENCE, norm01,
)

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
BID = "DB_top100_003"
OUT_DIR = ROOT / "figs" / "method_E_multiscale_ridge"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

SCALES_PX = [1.0, 3.0, 6.0, 12.0]   # σ for derivative-of-Gaussian
SIGMA_CUT_PX = int(round(1.0 / PIXEL_SIZE_UM))   # 1 µm spread for the cut
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
    """Apply soft grey-abstain to π based on per-pixel anchor evidence."""
    K, H, W = pi.shape
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    rho_smooth = np.stack([gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    confidence = N_total / (N_total + n_min)        # 0..1
    pi_abst = confidence[None] * pi + (1.0 - confidence[None]) * (1.0 / K)
    return pi_abst, confidence


def multiscale_grad_mag(field, scales_px):
    """For each σ in scales_px, compute |∇(G_σ * field)|. Return (S, H, W) stack
    plus per-pixel argmax across σ and per-pixel max value."""
    out = []
    for s in scales_px:
        if s <= 0:
            gy = sobel(field, axis=0); gx = sobel(field, axis=1)
            mag = np.hypot(gx, gy)
        else:
            mag = gaussian_gradient_magnitude(field, sigma=s)
        out.append(mag.astype(np.float32))
    stack = np.stack(out, axis=0)   # (S, H, W)
    return stack, stack.argmax(axis=0), stack.max(axis=0)


def nms_at_scale(field, sigma_px):
    """NMS using the gradient direction at a given σ."""
    if sigma_px > 0:
        f = gaussian_filter(field, sigma=sigma_px)
    else:
        f = field
    gy = sobel(f, axis=0); gx = sobel(f, axis=1)
    theta = np.arctan2(gy, gx)
    mag = np.hypot(gx, gy)
    return nms(mag, theta), mag, theta


def main():
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
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape; K = pi.shape[0]
    print(f"  shape {s18.shape}, focal area {focal_region.sum()} px")

    # 1. grey-abstain
    pi_abst, confidence = grey_abstain_pi(pi, d["anch"], d["li_arr"],
                                            SIGMA_DIFFUSE_PX, N_MIN_EVIDENCE)
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    print(f"  median confidence inside focal: {float(np.median(confidence[focal_region])):.2f}")

    # 2. multi-scale gradient magnitude
    grad_stack, sigma_arg, grad_max = multiscale_grad_mag(pi_diff, SCALES_PX)
    sigmas_px = np.array(SCALES_PX)
    print(f"  scales tested (px): {SCALES_PX}")
    print(f"  inside focal, fraction of pixels winning at each σ:")
    for i, s in enumerate(SCALES_PX):
        n = int((sigma_arg[focal_region] == i).sum())
        print(f"    σ={s}px : {n} px ({100*n/focal_region.sum():.0f}%)")

    # 3. NMS at the dominant scale inside the focal cell — pick the σ that
    #    "wins" most often inside the focal as the best scale for this cell.
    sigma_focal = int(np.bincount(sigma_arg[focal_region]).argmax())
    s_best = SCALES_PX[sigma_focal]
    print(f"  best σ inside focal: {s_best}px")
    nms_field, mag_at_best, theta_at_best = nms_at_scale(pi_diff, s_best)

    # 4. Final edge field: NMS · confidence (no hysteresis threshold)
    edge_field = nms_field * confidence
    # Optional: normalize per-cell so the cut depth is comparable across cells
    in_focal_max = edge_field[focal_region].max() if edge_field[focal_region].max() > 0 else 1.0
    edge_field_norm = edge_field / in_focal_max
    edge_field_norm = np.clip(edge_field_norm, 0, 1)

    # 5. cut 18S
    cut_smooth = gaussian_filter(edge_field_norm, sigma=SIGMA_CUT_PX)
    if cut_smooth.max() > 0: cut_smooth = cut_smooth / cut_smooth.max() * DEPTH_MAX
    s18_cut = s18.astype(np.float32) * (1.0 - cut_smooth)

    # 6. CP-SAM control + method-E cut
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
        focal_area = focal_region.sum()
        if focal_area == 0: return 0, []
        lbls = np.unique(m_out[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted(
            [(int(L), int(((m_out == L) & focal_region).sum()) / focal_area) for L in lbls],
            key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]

    print("  running CP-SAM control + method-E cut…")
    m_ctrl = run_cp(s18); n_ctrl, cov_ctrl = count_focal(m_ctrl)
    m_E    = run_cp(s18_cut); n_E,    cov_E    = count_focal(m_E)
    print(f"  control: n_focal={n_ctrl} covers={[f'{c*100:.0f}%' for _, c in cov_ctrl]}")
    print(f"  method E: n_focal={n_E} covers={[f'{c*100:.0f}%' for _, c in cov_E]}")

    # 7. render diagnostic
    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]; focal_bound = find_boundaries(focal_local, mode="outer")
    def add_focal(ax, c="cyan", lw=1.5):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors=c, linewidths=lw)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    # Row 0: references
    axes[0,0].imshow(s18_n[crop], cmap="gray"); add_focal(axes[0,0])
    axes[0,0].set_title("REF: 18S")
    im = axes[0,1].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,1])
    axes[0,1].set_title(f"π_{lin_a} − π_{lin_b}  (abstained)")
    fig.colorbar(im, ax=axes[0,1], fraction=0.04, pad=0.02)
    im = axes[0,2].imshow(confidence[crop], cmap="viridis", vmin=0, vmax=1); add_focal(axes[0,2])
    axes[0,2].set_title("confidence (grey-abstain blend)")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04, pad=0.02)
    axes[0,3].imshow(s18_n[crop], cmap="gray")
    axes[0,3].imshow(label_overlay(masks[crop])); add_focal(axes[0,3])
    axes[0,3].set_title("CP-SAM WSI mask")

    # Row 1: gradient at each σ
    vmax_glob = max(grad_stack[:, focal_region].max(), 1e-6)
    for i, s in enumerate(SCALES_PX):
        im = axes[1, i].imshow(grad_stack[i, *crop], cmap="magma", vmax=vmax_glob); add_focal(axes[1, i])
        axes[1, i].set_title(f"|∇G_σ * π_diff|  σ={s}px ({s*PIXEL_SIZE_UM:.1f}µm)")

    # Row 2: scale arg, scale max, NMS, edge field
    im = axes[2,0].imshow(sigma_arg[crop], cmap="tab10", vmin=0, vmax=len(SCALES_PX)-1)
    add_focal(axes[2,0])
    axes[2,0].set_title("σ_arg (which scale wins at each pixel)")
    from matplotlib.patches import Patch
    handles = [Patch(color=plt.get_cmap("tab10")(i), label=f"σ={s}px") for i, s in enumerate(SCALES_PX)]
    axes[2,0].legend(handles=handles, fontsize=7, loc="upper right")
    im = axes[2,1].imshow(grad_max[crop], cmap="magma"); add_focal(axes[2,1])
    axes[2,1].set_title("multi-scale max |∇| (max over σ)")
    im = axes[2,2].imshow(nms_field[crop], cmap="magma"); add_focal(axes[2,2])
    axes[2,2].set_title(f"NMS-thinned ridge (at σ_focal={s_best}px)")
    im = axes[2,3].imshow(edge_field_norm[crop], cmap="magma", vmin=0, vmax=1); add_focal(axes[2,3])
    axes[2,3].set_title("FINAL edge field\n(NMS × confidence, per-cell normalized)")

    # Row 3: cut 18S and CP-SAM results
    axes[3,0].imshow(norm01(s18_cut[crop]), cmap="gray"); add_focal(axes[3,0])
    axes[3,0].set_title(f"cut 18S  (σ_cut={SIGMA_CUT_PX}px, depth={DEPTH_MAX})")
    axes[3,1].imshow(s18_n[crop], cmap="gray")
    axes[3,1].imshow(label_overlay(m_ctrl[crop])); add_focal(axes[3,1])
    axes[3,1].set_title(f"Control CP-SAM  n_focal={n_ctrl}")
    axes[3,2].imshow(norm01(s18_cut[crop]), cmap="gray")
    axes[3,2].imshow(label_overlay(m_E[crop])); add_focal(axes[3,2])
    axes[3,2].set_title(f"Method-E CP-SAM  n_focal={n_E}  (Δ={n_E-n_ctrl})")
    axes[3,3].axis("off")
    summary = (
        f"Method E — multi-scale ridge\n\n"
        f"Pipeline:\n"
        f"  1. grey-abstained π_diff\n"
        f"  2. |∇G_σ * π_diff| at σ ∈ {SCALES_PX} px\n"
        f"  3. multi-scale max (the 'peak of the hill')\n"
        f"  4. NMS at best σ inside focal (={s_best}px)\n"
        f"  5. edge = NMS × confidence (no hysteresis)\n"
        f"  6. Gaussian smooth + cut depth\n\n"
        f"  best σ inside focal: {s_best}px ({s_best*PIXEL_SIZE_UM:.1f}µm)\n"
        f"\n"
        f"CP-SAM result:\n"
        f"  Control  n_focal={n_ctrl}\n"
        f"    {[f'{c*100:.0f}%' for _, c in cov_ctrl]}\n"
        f"  Method E n_focal={n_E}\n"
        f"    {[f'{c*100:.0f}%' for _, c in cov_E]}\n"
    )
    axes[3,3].text(0.02, 0.98, summary, transform=axes[3,3].transAxes,
                    fontsize=10, family="monospace", va="top")

    plt.suptitle(
        f"Method E: multi-scale ridge detection (no global percentile threshold) — {BID}",
        fontsize=12, y=1.005)
    plt.tight_layout()
    out_png = OUT_DIR / f"{BID}_method_E.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
