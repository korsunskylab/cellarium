"""Sharpen the heterotypic edge signal at step (2,3).

The π_a − π_b posterior difference is clean but transitions smoothly across the
heterotypic boundary. The plain |∇(π_a−π_b)| is therefore broad and weak.

Test four alternative edge fields built from the same π:

  A. Plain gradient:          |∇(π_a − π_b)|                          (current)
  B. Sharpened gradient:      |∇ tanh(K · (π_a − π_b))|   with K big   (option 1a)
  C. Sign-of-π_diff boundary: boundary pixels of  sign(π_a − π_b)>0   (option 1b — sharpest possible)
  D. Argmax-π boundary restricted to {a,b}: boundary of (argmax_π=a)   (option 1c — sharp, but uses all K lineages)

The argmax-based ones are categorical → 1-pixel-wide boundary. Then optional
small Gaussian blur to control "cut width."

Output: figure comparing the four detection signals inside the focal doublet,
and the resulting CP-SAM mask count if we cut along each.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel, binary_dilation
from skimage.segmentation import find_boundaries
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import (
    make_pi, SIGMA_PRE_PX, SIGMA_POST_PX, SIGMA_PX as SIGMA_DIFFUSE_PX,
)
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
OUT_DIR = ROOT / "figs" / "enhance_pair_gradient"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

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


def gradient_magnitude(field, sigma_pre=SIGMA_PRE_PX, sigma_post=SIGMA_POST_PX):
    f = gaussian_filter(field, sigma=sigma_pre) if sigma_pre > 0 else field
    gy = sobel(f, axis=0); gx = sobel(f, axis=1)
    m = np.hypot(gx, gy)
    if sigma_post > 0:
        m = gaussian_filter(m, sigma=sigma_post)
    return m


def cut_18s(s18_raw, edge_field, sigma_cut_px=int(round(1.0/PIXEL_SIZE_UM)), depth_max=0.9):
    """Gaussian-smooth, scale to [0, depth_max], multiply 18S by (1 − cut)."""
    f = edge_field.astype(np.float32)
    if f.max() > 0: f = f / f.max()
    smoothed = gaussian_filter(f, sigma=sigma_cut_px) if sigma_cut_px > 0 else f
    if smoothed.max() > 0: smoothed = smoothed / smoothed.max() * depth_max
    return s18_raw.astype(np.float32) * (1.0 - smoothed)


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
    s18, dapi, pi = d["s18"], d["dapi"], d["pi"]; s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal); H, W = s18.shape
    K = pi.shape[0]
    print(f"  shape {s18.shape}, focal area {focal_region.sum()} px")

    # ----- CRITICAL FIX: apply grey-abstain to π before differentiation -----
    # The canonical `make_tint_from_rho` does this; `make_pi` in edges_three_methods
    # skipped it, so single-transcript pixels got strong posteriors → spurious edges.
    # N is the effective evidence per pixel after diffusion. We re-derive it from
    # the posterior + ALPHA (inverse of the Dirichlet update).
    # π_k = (N_k + α) / (N_total + Kα)  →  N_k = π_k · (N_total + Kα) − α
    # We don't have N_total handy, so reconstruct via the anchor density directly.
    # Simpler: pull rho from anch.
    from scipy.ndimage import gaussian_filter as gf
    H_, W_ = pi.shape[-2:]
    rho_pts = np.zeros((K, H_, W_), dtype=np.float32)
    li_arr = d["li_arr"]
    anch = d["anch"]
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    # rho already includes diffusion-anisotropic, so we just need the smoothed
    # density which is what make_pi used. Approximate by SIGMA_DIFFUSE_PX.
    rho_smooth = np.stack([gf(rho_pts[k], sigma=SIGMA_DIFFUSE_PX) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * SIGMA_DIFFUSE_PX ** 2
    N_total = N_eff.sum(axis=0)
    confidence = N_total / (N_total + N_MIN_EVIDENCE)  # 0..1
    pi_abst = confidence[None] * pi + (1.0 - confidence[None]) * (1.0 / K)
    print(f"  grey-abstain: median N_total={float(np.median(N_total)):.2f}, "
          f"median confidence={float(np.median(confidence)):.2f}")
    print(f"  inside focal: median confidence={float(np.median(confidence[focal_region])):.2f}")

    pi_diff      = (pi[ka]      - pi[kb]     ).astype(np.float32)  # original (buggy)
    pi_diff_abst = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)  # FIXED

    # NOTE: all four edge fields use the GREY-ABSTAINED π (pi_diff_abst).
    # The fix kills small blips because low-evidence pixels have π ≈ 1/K,
    # so π_diff ≈ 0 there and the gradient drops to 0.
    pdf = pi_diff_abst   # shorthand for the abstain-corrected π_diff

    # ----- A: plain gradient -----
    A_mag = gradient_magnitude(pdf)

    # ----- B: sharpened gradient — tanh contrast first -----
    K_SHARP = 8.0  # higher K → sharper transition
    pi_diff_sharp = np.tanh(K_SHARP * pdf)
    B_mag = gradient_magnitude(pi_diff_sharp)

    # ----- C: sign-of-π_diff boundary (zero-crossing of π_diff in pair-space) -----
    sign_field = (gaussian_filter(pdf, sigma=SIGMA_PRE_PX) > 0).astype(np.uint8)
    C_bin = find_boundaries(sign_field, mode="inner").astype(np.uint8)
    # Suppress boundary pixels where confidence is too low (those are abstain regions)
    C_bin = C_bin * (confidence > 0.5).astype(np.uint8)

    # ----- D: argmax-π boundary, restricted to the {a, b} pair, abstain-suppressed -----
    pi_argmax = pi_abst.argmax(axis=0)
    is_a = (pi_argmax == ka).astype(np.uint8)
    is_b = (pi_argmax == kb).astype(np.uint8)
    a_bound = find_boundaries(is_a, mode="outer")
    b_bound = find_boundaries(is_b, mode="outer")
    D_bin = (a_bound & b_bound).astype(np.uint8)
    D_bin = D_bin * (confidence > 0.5).astype(np.uint8)

    # All 4 cut fields (continuous)
    A_cut = cut_18s(s18, A_mag)
    B_cut = cut_18s(s18, B_mag)
    C_cut = cut_18s(s18, C_bin.astype(np.float32))
    D_cut = cut_18s(s18, D_bin.astype(np.float32))

    # ----- run CP-SAM control + each cut variant -----
    print("\nloading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_wsi(arr, key):
        return ((arr.astype(np.float32) - wsi_pct[key]["q_lo"]) /
                max(wsi_pct[key]["q_hi"] - wsi_pct[key]["q_lo"], 1e-6)).astype(np.float32)
    dapi_norm = norm_wsi(dapi, "DAPI")

    def run_cp(s18_image):
        s_norm = norm_wsi(s18_image, "18S")
        img = np.stack([dapi_norm, s_norm], axis=-1).astype(np.float32)
        masks_out, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return masks_out.astype(np.int32)

    def count_focal(masks_out):
        focal_area = focal_region.sum()
        if focal_area == 0: return 0, []
        lbls = np.unique(masks_out[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted(
            [(int(L), int(((masks_out == L) & focal_region).sum()) / focal_area) for L in lbls],
            key=lambda kv: -kv[1])
        n_at = sum(1 for _, c in covers if c >= 0.05)
        return n_at, covers[:5]

    results = {}
    for name, s18_image in [("Control", s18),
                              ("A_plain", A_cut), ("B_sharp", B_cut),
                              ("C_signboundary", C_cut), ("D_argmax_ab", D_cut)]:
        print(f"  {name}…")
        m_out = run_cp(s18_image)
        n_at, cov = count_focal(m_out)
        results[name] = dict(masks=m_out, n=n_at, cov=cov)
        print(f"    n_focal={n_at}, covers={[f'{c*100:.0f}%' for _, c in cov]}")

    # ----- render diagnostic figure (focused on focal cell) -----
    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]; focal_bound = find_boundaries(focal_local, mode="outer")
    def add_focal(ax, c="cyan", lw=1.5):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors=c, linewidths=lw)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    # Row 0: shared reference panels — show BUGGY vs FIXED π_diff side-by-side
    axes[0,0].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,0])
    axes[0,0].set_title(f"OLD: π_{lin_a}−π_{lin_b}  (no grey-abstain — buggy)")
    axes[0,1].imshow(pi_diff_abst[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,1])
    axes[0,1].set_title(f"NEW: π_{lin_a}−π_{lin_b}  (with grey-abstain)")
    im = axes[0,2].imshow(confidence[crop], cmap="viridis", vmin=0, vmax=1); add_focal(axes[0,2])
    axes[0,2].set_title("confidence = N_total/(N_total + n_min)\n(0=abstain, 1=trust)")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04, pad=0.02)
    axes[0,3].imshow(pi_diff_sharp[crop], cmap="RdBu_r", vmin=-1, vmax=1); add_focal(axes[0,3])
    axes[0,3].set_title(f"tanh(K=8·π_diff_abst)  — sharper transition")
    axes[0,4].imshow(pi_argmax[crop], cmap=plt.get_cmap("tab10"), vmin=0, vmax=len(LINEAGES)-1)
    add_focal(axes[0,4], c="white")
    axes[0,4].set_title("argmax π_abst (abstain pixels → fallback)")

    # Row 1: the 4 edge fields
    for i, (name, field, vmax) in enumerate([
        ("A: plain |∇π_diff|",                   A_mag,        np.percentile(A_mag, 99.5)),
        ("B: |∇ tanh(8·π_diff)|",                B_mag,        np.percentile(B_mag, 99.5)),
        ("C: sign(π_diff) boundary",             C_bin,        1.0),
        ("D: argmax_ab boundary",                D_bin,        1.0),
    ]):
        ax = axes[1, i]
        im = ax.imshow(field[crop], cmap="magma", vmax=vmax)
        add_focal(ax)
        n_inside = int(((field > 0) & focal_region).sum())
        ax.set_title(f"{name}\n{n_inside} active px inside focal")
    axes[1,4].axis("off")
    axes[1,4].text(0.0, 0.5,
        "Edge fields (row 1):\n\n"
        "A is the plain gradient — what we have now.\n"
        "B sharpens by tanh contrast before differentiation.\n"
        "C is the zero-crossing of π_diff: 1-px curve where\n"
        "    the dominant lineage switches between the pair.\n"
        "D is the boundary of argmax_π=a meeting argmax_π=b\n"
        "    — same idea but uses all K classes (other\n"
        "    lineages can grab pixels).\n",
        fontsize=9, family="monospace", va="center")

    # Row 2: CP-SAM with each cut
    for i, name in enumerate(["Control", "A_plain", "B_sharp", "C_signboundary", "D_argmax_ab"]):
        ax = axes[2, i]
        r = results[name]
        ax.imshow(s18_n[crop], cmap="gray")
        ax.imshow(label_overlay(r["masks"][crop]))
        add_focal(ax)
        cov_str = " ".join(f"{c*100:.0f}%" for _, c in r["cov"][:3])
        ax.set_title(f"{name}: n_focal={r['n']}\ncovers: {cov_str}")

    plt.suptitle(
        f"Heterotypic edge enhancement on {BID} (Fib × Mel)\n"
        f"Row 0 = references. Row 1 = four edge fields A/B/C/D. Row 2 = CP-SAM after each cut.",
        fontsize=12, y=1.005)
    plt.tight_layout()
    out_png = OUT_DIR / f"{BID}_enhancement_compare.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_png}")
    print(f"\n{'='*60}\nSUMMARY")
    for k, v in results.items():
        print(f"  {k:18s}  n_focal={v['n']}  covers={[f'{c*100:.0f}%' for _, c in v['cov']]}")


if __name__ == "__main__":
    main()
