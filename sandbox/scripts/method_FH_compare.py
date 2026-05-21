"""Method F vs Method H side-by-side comparison on all 4 test cells.

Same upstream pipeline (π posterior + ALPHA=10 + grey-abstain + cell_evidence
weighting). The only difference is WHERE the cut is placed:

  Method F: at the |∇ tanh(K · π_diff)| ridge — middle of the transition zone
            (sign-change line, π_diff = 0).

  Method H: at the boundary of {π_HOST > 0.3} — deep in the host's strong-
            dominance zone. Picks PRESERVE = most-compact connected
            argmax-π component, HOST = the other.

No blending or auto-selection. Run both on every cell, present results
side-by-side.

For each cell, one multi-panel figure showing:
  • 18S + anchors (BIG dots so transcripts are visible)
  • π_diff with sign-change line (F cut location)
  • H cut band (red→white line, dilated)
  • F cut depth field
  • H cut depth field
  • CP-SAM control / F / H results with anchors and contours
  • Compactness ranking + summary
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
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, label as cc_label
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
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
OUT_DIR = ROOT / "figs" / "method_FH_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125

ALPHA = 10.0
N_MIN = 3.0
K_SHARP = 8.0
SIGMA_CUT_PX = 2.0
DEPTH_MAX = 0.9
HOST_STRONG_THRESHOLD = 0.30
H_DILATE_PX = 2
ANCHOR_DOT_SIZE = 18    # big dots so the user can see anchor lineages clearly

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


def compactness(mask_bool):
    if mask_bool.sum() == 0: return 0.0
    rp = regionprops(mask_bool.astype(np.uint8))
    if not rp: return 0.0
    A = float(rp[0].area); P = float(rp[0].perimeter)
    if P <= 0: return 0.0
    return 4.0 * np.pi * A / (P * P)


def label_overlay(mask, alpha=0.45):
    u = np.unique(mask); u = u[u != 0]
    cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], alpha)
    return out


def cut_18s_from_band(s18, band_or_field, sigma_cut, depth_max):
    smoothed = gaussian_filter(band_or_field.astype(np.float32), sigma=sigma_cut)
    if smoothed.max() > 0: cut_field = smoothed / smoothed.max() * depth_max
    else: cut_field = smoothed
    return s18.astype(np.float32) * (1.0 - cut_field), cut_field


def run_cell(BID, m_sam, wsi_pct, gene_names, gl):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
    print(f"\n{'='*60}\n{BID}  ({lin_a} × {lin_b})\n{'='*60}")
    rdir = ROIS / BID
    d = make_pi(rdir, gene_names, gl)
    s18, dapi = d["s18"], d["dapi"]; s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    anch = d["anch"]; li_arr = d["li_arr"]
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape; K = d["pi"].shape[0]

    pi_abst, confidence = posterior_with_alpha(
        anch, li_arr, SIGMA_DIFFUSE_PX, H, W, K, alpha=ALPHA, n_min=N_MIN)
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    pi_argmax = pi_abst.argmax(axis=0)

    # ---- Method F: ridge of |∇ tanh(K·π_diff)| ----
    t = np.tanh(K_SHARP * pi_diff)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g = np.hypot(gx, gy).astype(np.float32)
    cell_ev = np.maximum(dapi_n, s18_n).astype(np.float32)
    edge_F = g * (confidence ** 2) * cell_ev
    s18_F, cut_F = cut_18s_from_band(s18, edge_F, SIGMA_CUT_PX, DEPTH_MAX)
    sign_field = (pi_diff > 0).astype(np.uint8)
    sign_boundary_focal = find_boundaries(sign_field, mode="inner") & focal_region

    # ---- Method H: red→white line ----
    # Identify pair components, pick most-compact PRESERVE, least-compact HOST
    pair_components = []
    for k, lineage in [(ka, lin_a), (kb, lin_b)]:
        terr = (pi_argmax == k) & focal_region
        lab, n = cc_label(terr)
        for i in range(1, n + 1):
            comp = (lab == i)
            if comp.sum() < 50: continue
            pair_components.append(dict(
                lineage=lineage, k=k, area=int(comp.sum()),
                compactness=compactness(comp), mask=comp))
    largest_a = max([c for c in pair_components if c["lineage"] == lin_a],
                     key=lambda c: c["area"], default=None)
    largest_b = max([c for c in pair_components if c["lineage"] == lin_b],
                     key=lambda c: c["area"], default=None)
    H_applicable = (largest_a is not None and largest_b is not None)
    if H_applicable:
        if largest_a["compactness"] > largest_b["compactness"]:
            preserve, host = largest_a, largest_b
        else:
            preserve, host = largest_b, largest_a
        host_strong = (pi_abst[host["k"]] > HOST_STRONG_THRESHOLD) & focal_region
        cut_band = find_boundaries(host_strong, mode="inner") & focal_region
        cut_band = binary_dilation(cut_band, iterations=H_DILATE_PX) & focal_region
        s18_H, cut_H = cut_18s_from_band(s18, cut_band, SIGMA_CUT_PX, DEPTH_MAX)
    else:
        preserve = host = None
        cut_band = np.zeros_like(focal_region)
        s18_H, cut_H = s18.astype(np.float32), np.zeros_like(focal_region, dtype=np.float32)

    # ---- CP-SAM ----
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
    m_F = run_cp(s18_F);  n_F, cov_F = count_focal(m_F)
    m_H = run_cp(s18_H);  n_H, cov_H = count_focal(m_H)
    print(f"  Control: n={n_c} {[f'{c*100:.0f}%' for _,c in cov_c]}")
    print(f"  F:       n={n_F} {[f'{c*100:.0f}%' for _,c in cov_F]}")
    print(f"  H:       n={n_H} {[f'{c*100:.0f}%' for _,c in cov_H]}")

    # ---- Render ----
    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]
    def overlays(ax, show_sign=True, show_preserve=False):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
        if show_sign:
            ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=1.5)
        if show_preserve and preserve is not None:
            ax.contour(preserve["mask"][crop].astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    sub = anch[(anch["px"]>=x0)&(anch["px"]<x1)&(anch["py"]>=y0)&(anch["py"]<y1)]
    def big_anchors(ax):
        for L in [lin_a, lin_b]:
            s = sub[sub["lineage"]==L]
            if len(s):
                ax.scatter(s["px"]-x0, s["py"]-y0, s=ANCHOR_DOT_SIZE,
                            c=LINEAGE_COLOURS[L], alpha=0.85,
                            edgecolors="black", linewidths=0.4)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    # Row 0: data
    axes[0,0].imshow(s18_n[crop], cmap="gray")
    big_anchors(axes[0,0]); overlays(axes[0,0], show_sign=False)
    handles = [Patch(color=LINEAGE_COLOURS[lin_a], label=f"{lin_a} ({(sub['lineage']==lin_a).sum()})"),
                Patch(color=LINEAGE_COLOURS[lin_b], label=f"{lin_b} ({(sub['lineage']==lin_b).sum()})")]
    axes[0,0].legend(handles=handles, fontsize=9, loc="upper right")
    axes[0,0].set_title("18S + anchors (large dots, pair lineages only)", fontsize=10)
    im = axes[0,1].imshow(pi_diff[crop], cmap="RdBu_r", vmin=-1, vmax=1)
    overlays(axes[0,1], show_sign=True)
    axes[0,1].set_title(f"π_{lin_a}−π_{lin_b}  (ALPHA={ALPHA})\nlime = F cut location (sign change)", fontsize=10)
    fig.colorbar(im, ax=axes[0,1], fraction=0.04)
    cmap = plt.get_cmap("tab10")
    axes[0,2].imshow(pi_argmax[crop], cmap=cmap, vmin=0, vmax=len(LINEAGES)-1)
    overlays(axes[0,2], show_sign=False, show_preserve=True)
    title_p = f"PRESERVE: {preserve['lineage']} (compact={preserve['compactness']:.2f})" if preserve else "n/a"
    title_h = f"HOST: {host['lineage']} (compact={host['compactness']:.2f})" if host else "n/a"
    axes[0,2].set_title(f"argmax-π. cyan = PRESERVE\n{title_p} / {title_h}", fontsize=9)
    # H cut band on top of 18S
    axes[0,3].imshow(s18_n[crop], cmap="gray")
    axes[0,3].contour(cut_band[crop].astype(int), levels=[0.5], colors="red", linewidths=1.8)
    overlays(axes[0,3], show_sign=True, show_preserve=True)
    axes[0,3].set_title("H cut band (red) overlaid on 18S\nlime=F cut loc, cyan=PRESERVE", fontsize=10)

    # Row 1: cut depth fields
    im = axes[1,0].imshow(cut_F[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX)
    overlays(axes[1,0], show_sign=True)
    axes[1,0].set_title("F cut depth field (sign-change ridge)", fontsize=10)
    fig.colorbar(im, ax=axes[1,0], fraction=0.04)
    im = axes[1,1].imshow(cut_H[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX)
    overlays(axes[1,1], show_sign=False, show_preserve=True)
    axes[1,1].set_title("H cut depth field (red→white ring)", fontsize=10)
    fig.colorbar(im, ax=axes[1,1], fraction=0.04)
    axes[1,2].imshow(norm01(s18_F[crop]), cmap="gray")
    overlays(axes[1,2], show_sign=True)
    axes[1,2].set_title("18S after F cut", fontsize=10)
    axes[1,3].imshow(norm01(s18_H[crop]), cmap="gray")
    overlays(axes[1,3], show_sign=False, show_preserve=True)
    axes[1,3].set_title("18S after H cut", fontsize=10)

    # Row 2: CP-SAM
    axes[2,0].imshow(s18_n[crop], cmap="gray")
    axes[2,0].imshow(label_overlay(m_ctrl[crop]))
    big_anchors(axes[2,0]); overlays(axes[2,0], show_sign=False)
    axes[2,0].set_title(f"CONTROL n={n_c}\n{[f'{c*100:.0f}%' for _,c in cov_c[:3]]}", fontsize=10)
    axes[2,1].imshow(norm01(s18_F[crop]), cmap="gray")
    axes[2,1].imshow(label_overlay(m_F[crop]))
    big_anchors(axes[2,1]); overlays(axes[2,1], show_sign=True)
    in_fb = find_boundaries(m_F, mode="inner") & focal_region
    axes[2,1].contour(in_fb[crop].astype(int), levels=[0.5], colors="white", linewidths=1.0)
    axes[2,1].set_title(f"Method F n={n_F}\n{[f'{c*100:.0f}%' for _,c in cov_F[:3]]}", fontsize=10)
    axes[2,2].imshow(norm01(s18_H[crop]), cmap="gray")
    axes[2,2].imshow(label_overlay(m_H[crop]))
    big_anchors(axes[2,2]); overlays(axes[2,2], show_sign=False, show_preserve=True)
    in_hb = find_boundaries(m_H, mode="inner") & focal_region
    axes[2,2].contour(in_hb[crop].astype(int), levels=[0.5], colors="white", linewidths=1.0)
    axes[2,2].set_title(f"Method H n={n_H}\n{[f'{c*100:.0f}%' for _,c in cov_H[:3]]}", fontsize=10)
    axes[2,3].axis("off")
    s_text = (
        f"{BID}\n  pair: {lin_a} × {lin_b}\n"
        f"  focal area: {focal_region.sum()} px\n\n"
        f"Compactness ranking:\n"
    )
    for c in sorted(pair_components, key=lambda c: -c["compactness"]):
        marker = "★ PRESERVE" if (preserve and c["lineage"]==preserve["lineage"] and c["area"]==preserve["area"]) else \
                  "  HOST" if (host and c["lineage"]==host["lineage"] and c["area"]==host["area"]) else "  "
        s_text += f"  {marker:11s}  {c['lineage']:>12s}  area={c['area']:>5d}  comp={c['compactness']:.3f}\n"
    s_text += (
        f"\nCP-SAM:\n"
        f"  Control: n={n_c}\n"
        f"  F:       n={n_F}  (Δ={n_F-n_c})\n"
        f"  H:       n={n_H}  (Δ={n_H-n_c})\n"
    )
    axes[2,3].text(0.02, 0.98, s_text, transform=axes[2,3].transAxes,
                    fontsize=10, family="monospace", va="top")

    plt.suptitle(
        f"{BID}  ({lin_a} × {lin_b}) — Method F (sign change) vs Method H (red→white)",
        fontsize=12, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_FH_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out.name}")
    return dict(BID=BID, n_ctrl=n_c, n_F=n_F, n_H=n_H,
                cov_ctrl=cov_c, cov_F=cov_F, cov_H=cov_H,
                preserve=preserve["lineage"] if preserve else None,
                host=host["lineage"] if host else None,
                preserve_compact=preserve["compactness"] if preserve else None,
                host_compact=host["compactness"] if host else None)


def main():
    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()

    cells = ["DB_top100_003", "DB_top100_015", "DB_top100_018", "DB_top100_098"]
    results = [run_cell(BID, m_sam, wsi_pct, gene_names, gl) for BID in cells]

    print("\n\n" + "="*72)
    print("SUMMARY (all 4 cells, F vs H):")
    print("="*72)
    print(f"{'cell':<18s} {'pair':<28s} {'preserve':>11s} {'host':>11s} "
          f"{'comp_p':>7s} {'comp_h':>7s} {'C':>3s} {'F':>3s} {'H':>3s}")
    print("-"*72)
    for r in results:
        row = BENCH[BENCH["benchmark_id"] == r["BID"]].iloc[0]
        print(f"{r['BID']:<18s} {row['lineage_pair']:<28s} "
              f"{str(r['preserve']):>11s} {str(r['host']):>11s} "
              f"{r['preserve_compact']:>7.3f} {r['host_compact']:>7.3f} "
              f"{r['n_ctrl']:>3d} {r['n_F']:>3d} {r['n_H']:>3d}")


if __name__ == "__main__":
    main()
