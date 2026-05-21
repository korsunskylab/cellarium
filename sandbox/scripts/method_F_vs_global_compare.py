"""Method F (pair-specific) vs Method F-global (one-vs-rest) comparison.

Method F  uses (ka, kb) from the doublet table → segmentation-derived → not
allowed at WSI scale. The one-vs-rest variant is segmentation-agnostic:

    For each lineage k:  s_k = π[k] − max_{j≠k} π[j]
    edge_total = Σ_k |∇ tanh(K_sharp · s_k)|
    edge_global = edge_total × conf² × cell_evidence

In the pure-pair case this equals 2 × Method F's edge field, so it's a strict
generalization. Defaults match Method F: σ_cut=5, depth=0.99, ALPHA=10,
K_SHARP=8, N_MIN=3.

Per-doublet figure (3 rows × 4 cols):
  Row 0 (shared):  18S+anchors  |  argmax label  |  conf²  |  cell_evidence
  Row 1 (F pair):  edge_F       |  cut_F         |  s_cut_F      |  CP-SAM (F)
  Row 2 (global):  edge_global  |  cut_global    |  s_cut_global |  CP-SAM (global)
"""
from __future__ import annotations
import sys, json, traceback
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
OUT_DIR = ROOT / "figs" / "method_F_vs_global"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
SIGMA_CUT_PX = 5.0; DEPTH_MAX = 0.99
ANCHOR_DOT_SIZE = 18
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


def edge_field_pair(pi_abst, ka, kb, confidence, cell_ev, k_sharp=K_SHARP):
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    t = np.tanh(k_sharp * pi_diff)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g = np.hypot(gx, gy).astype(np.float32)
    return g * (confidence ** 2) * cell_ev, pi_diff


def edge_field_global(pi_abst, confidence, cell_ev, k_sharp=K_SHARP):
    """One-vs-rest edge field. For each lineage k:
       s_k = π[k] − max_{j≠k} π[j]
       edge_total = Σ_k |∇ tanh(K_sharp · s_k)|
    Compute the second-largest π per pixel once, then derive each k's
    "others_max" cheaply (= top1 if k != argmax, else top2)."""
    K, H, W = pi_abst.shape
    top_idx = np.argmax(pi_abst, axis=0).astype(np.int8)  # (H,W)
    top1_val = np.max(pi_abst, axis=0)
    # second-largest: mask top with -inf then re-argmax
    k_idx = np.arange(K, dtype=np.int8)[:, None, None]
    masked = np.where(k_idx == top_idx[None], -np.inf, pi_abst)
    top2_val = np.max(masked, axis=0)

    edge_total = np.zeros((H, W), dtype=np.float32)
    for k in range(K):
        others_max = np.where(top_idx == k, top2_val, top1_val)
        s_k = pi_abst[k] - others_max
        t_k = np.tanh(k_sharp * s_k).astype(np.float32)
        gy = sobel(t_k, axis=0); gx = sobel(t_k, axis=1)
        edge_total += np.hypot(gx, gy).astype(np.float32)
    return edge_total * (confidence ** 2) * cell_ev, top_idx, (top1_val - top2_val)


def label_overlay(mask, alpha=0.45):
    u = np.unique(mask); u = u[u != 0]
    cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], alpha)
    return out


def count_focal(m, focal_region):
    """EVAL-ONLY use of focal_region (allowed; cut field never sees it)."""
    a = focal_region.sum()
    if a == 0: return 0, []
    lbls = np.unique(m[focal_region]); lbls = lbls[lbls != 0]
    covers = sorted(
        [(int(L), int(((m == L) & focal_region).sum()) / a) for L in lbls],
        key=lambda kv: -kv[1]
    )
    return sum(1 for _, c in covers if c >= 0.05), covers[:5]


def smooth_and_clip(edge, sigma, depth):
    s = gaussian_filter(edge, sigma=sigma)
    if s.max() > 0:
        return (s / s.max()) * depth
    return s


def process_one(BID, m_sam, norm_dapi, norm_18s):
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
    cell_ev = np.maximum(dapi_n, s18_n).astype(np.float32)

    edge_F, pi_diff = edge_field_pair(pi_abst, ka, kb, confidence, cell_ev)
    edge_G, top_idx, margin = edge_field_global(pi_abst, confidence, cell_ev)

    cut_F = smooth_and_clip(edge_F, SIGMA_CUT_PX, DEPTH_MAX)
    cut_G = smooth_and_clip(edge_G, SIGMA_CUT_PX, DEPTH_MAX)
    s_cut_F = s18.astype(np.float32) * (1.0 - cut_F)
    s_cut_G = s18.astype(np.float32) * (1.0 - cut_G)

    # π_diff sign-change (visualization only) using known pair
    sign_field = (pi_diff > 0).astype(np.uint8)
    sign_boundary_focal = find_boundaries(sign_field, mode="inner") & focal_region

    def run_cp(s_img):
        img = np.stack([norm_dapi(dapi), norm_18s(s_img)], axis=-1).astype(np.float32)
        m, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m.astype(np.int32)

    m_ctrl = run_cp(s18);   n_c, cov_c = count_focal(m_ctrl, focal_region)
    m_F    = run_cp(s_cut_F); n_F, cov_F = count_focal(m_F, focal_region)
    m_G    = run_cp(s_cut_G); n_G, cov_G = count_focal(m_G, focal_region)

    # focal-centred crop
    ys, xs = np.where(focal_region); pad = 35
    y0 = max(ys.min()-pad, 0); y1 = min(ys.max()+pad, H)
    x0 = max(xs.min()-pad, 0); x1 = min(xs.max()+pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]

    in_crop = ((anch["py"].to_numpy() >= y0) & (anch["py"].to_numpy() < y1) &
               (anch["px"].to_numpy() >= x0) & (anch["px"].to_numpy() < x1))
    a_py = anch["py"].to_numpy()[in_crop] - y0
    a_px = anch["px"].to_numpy()[in_crop] - x0
    a_li = li_arr[in_crop]

    def overlays(ax, show_anchors=False):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.3)
        ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=1.7)
        if show_anchors:
            for k_lin, lin in enumerate(LINEAGES):
                m_lin = a_li == k_lin
                if not m_lin.any(): continue
                ax.scatter(a_px[m_lin], a_py[m_lin], s=ANCHOR_DOT_SIZE,
                            color=LINEAGE_COLOURS[lin], edgecolors="black", linewidths=0.4)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    # Row 0 — shared inputs / σ-independent intermediates
    ax = axes[0, 0]
    ax.imshow(norm01(s18[crop]), cmap="gray")
    overlays(ax, show_anchors=True)
    ax.set_title(f"18S + anchors  ({lin_a} × {lin_b})\n(yellow=focal, lime=π[{lin_a}]−π[{lin_b}] sign-change)")

    ax = axes[0, 1]
    # argmax label map — colour by lineage
    label_rgb = np.zeros((*top_idx.shape, 3), dtype=np.float32)
    for k_lin, lin in enumerate(LINEAGES):
        c = plt.matplotlib.colors.to_rgb(LINEAGE_COLOURS[lin])
        label_rgb[top_idx == k_lin] = c
    ax.imshow(label_rgb[crop])
    overlays(ax)
    ax.set_title("argmax(π_abst) — segmentation-agnostic label map\n(used by global edge field)")

    ax = axes[0, 2]
    ax.imshow((confidence ** 2)[crop], cmap="viridis", vmin=0, vmax=1)
    overlays(ax)
    ax.set_title("confidence²")

    ax = axes[0, 3]
    ax.imshow(cell_ev[crop], cmap="gray", vmin=0, vmax=1)
    overlays(ax)
    ax.set_title("cell_evidence = max(DAPI_n, 18S_n)")

    # Row 1 — Method F pair
    def edge_panel(ax, e, title):
        v = np.percentile(e[focal_region], 99) if focal_region.any() else 1.0
        ax.imshow(e[crop], cmap="hot", vmin=0, vmax=max(v, 1e-6))
        overlays(ax); ax.set_title(title)

    edge_panel(axes[1, 0], edge_F, f"Method F (pair) edge\n= |∇tanh(K·π_diff)| × conf² × cell_ev")
    axes[1, 1].imshow(cut_F[crop], cmap="hot", vmin=0, vmax=1)
    overlays(axes[1, 1]); axes[1, 1].set_title(f"cut_F (σ={SIGMA_CUT_PX}, depth={DEPTH_MAX})")
    axes[1, 2].imshow(norm01(s_cut_F[crop]), cmap="gray")
    overlays(axes[1, 2]); axes[1, 2].set_title("s_cut_F  (what CP-SAM sees)")
    axes[1, 3].imshow(norm01(s_cut_F[crop]), cmap="gray")
    axes[1, 3].imshow(label_overlay(m_F[crop]))
    axes[1, 3].contour((find_boundaries(m_F, mode="inner") & focal_region)[crop].astype(int),
                        levels=[0.5], colors="white", linewidths=1.4)
    overlays(axes[1, 3])
    cov_F_str = " ".join(f"{c*100:.0f}%" for _, c in cov_F[:3])
    delta_F = "+" if n_F > n_c else ("=" if n_F == n_c else "−")
    axes[1, 3].set_title(f"CP-SAM on s_cut_F  [{delta_F} vs ctrl={n_c}]\nn_focal={n_F}  {cov_F_str}")

    # Row 2 — Global one-vs-rest
    edge_panel(axes[2, 0], edge_G, "Global edge\n= Σ_k |∇tanh(K·s_k)| × conf² × cell_ev  (no pair)")
    axes[2, 1].imshow(cut_G[crop], cmap="hot", vmin=0, vmax=1)
    overlays(axes[2, 1]); axes[2, 1].set_title(f"cut_global (σ={SIGMA_CUT_PX}, depth={DEPTH_MAX})")
    axes[2, 2].imshow(norm01(s_cut_G[crop]), cmap="gray")
    overlays(axes[2, 2]); axes[2, 2].set_title("s_cut_global  (what CP-SAM sees)")
    axes[2, 3].imshow(norm01(s_cut_G[crop]), cmap="gray")
    axes[2, 3].imshow(label_overlay(m_G[crop]))
    axes[2, 3].contour((find_boundaries(m_G, mode="inner") & focal_region)[crop].astype(int),
                        levels=[0.5], colors="white", linewidths=1.4)
    overlays(axes[2, 3])
    cov_G_str = " ".join(f"{c*100:.0f}%" for _, c in cov_G[:3])
    delta_G = "+" if n_G > n_c else ("=" if n_G == n_c else "−")
    axes[2, 3].set_title(f"CP-SAM on s_cut_global  [{delta_G} vs ctrl={n_c}]\nn_focal={n_G}  {cov_G_str}")

    plt.suptitle(
        f"{BID} — Method F (pair-specific) vs Global one-vs-rest  "
        f"α={ALPHA}, K_sharp={K_SHARP}, σ={SIGMA_CUT_PX}, depth={DEPTH_MAX}",
        fontsize=13, y=1.003)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "benchmark_id": BID,
        "lineage_pair": row["lineage_pair"],
        "n_control": int(n_c),
        "n_F_pair": int(n_F),
        "n_F_global": int(n_G),
        "top1_control_pct": float(cov_c[0][1]*100) if cov_c else 0.0,
        "top1_F_pair_pct": float(cov_F[0][1]*100) if cov_F else 0.0,
        "top1_F_global_pct": float(cov_G[0][1]*100) if cov_G else 0.0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("benchmark_id", nargs="*",
                    help="If empty, process all approved doublets with cached ROI.")
    args = p.parse_args()

    if args.benchmark_id:
        ids = args.benchmark_id
    else:
        bench_ok = BENCH[BENCH["approved"].fillna(False)].copy() if "approved" in BENCH.columns else BENCH
        present = sorted([p.name for p in ROIS.iterdir() if p.is_dir()])
        ids = sorted(bench_ok[bench_ok["benchmark_id"].isin(present)]["benchmark_id"].tolist())
    print(f"processing {len(ids)} doublet(s)")

    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_dapi(arr):
        return ((arr.astype(np.float32) - wsi_pct["DAPI"]["q_lo"]) /
                max(wsi_pct["DAPI"]["q_hi"] - wsi_pct["DAPI"]["q_lo"], 1e-6)).astype(np.float32)
    def norm_18s(arr):
        return ((arr.astype(np.float32) - wsi_pct["18S"]["q_lo"]) /
                max(wsi_pct["18S"]["q_hi"] - wsi_pct["18S"]["q_lo"], 1e-6)).astype(np.float32)

    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    rows = []
    for i, BID in enumerate(ids):
        try:
            r = process_one(BID, m_sam, norm_dapi, norm_18s)
            rows.append(r)
            print(f"  [{i+1:2d}/{len(ids)}] {BID}: ctrl={r['n_control']}  "
                  f"F_pair={r['n_F_pair']}  F_global={r['n_F_global']}")
            pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
        except Exception as e:
            print(f"  [{i+1:2d}/{len(ids)}] {BID}: FAILED {e}")
            traceback.print_exc()

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        print()
        print(f"split-rate  control: {(df['n_control']>=2).sum()}/{len(df)}")
        print(f"            F_pair:  {(df['n_F_pair']>=2).sum()}/{len(df)}")
        print(f"            F_global: {(df['n_F_global']>=2).sum()}/{len(df)}")
        agree = (df['n_F_pair'] == df['n_F_global']).sum()
        print(f"F_pair vs F_global  same: {agree}/{len(df)}  "
              f"pair>global: {(df['n_F_pair']>df['n_F_global']).sum()}  "
              f"global>pair: {(df['n_F_global']>df['n_F_pair']).sum()}")


if __name__ == "__main__":
    main()
