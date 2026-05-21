"""Render WSI | 1024-ROI | 2048-ROI mask comparison for each of the 68 approved
benchmark doublets.

Reads:
  - benchmark_doublets.parquet          (which doublets, centroids, lineage)
  - cpsam_whole_slide/masks.tif         (WSI mask)
  - morphology_focus_0002.ome.tif (18S) (background image for the WSI panel)
  - figs/doublet_survival_at_dev_size/masks/<benchmark_id>_{1024,2048}.npz
                                        (saved ROI runs)
  - figs/doublet_survival_at_dev_size/verdict.csv (verdict per doublet × size)

Produces:
  reports/doublet_masks_3way.pdf — one row per doublet, three columns:
      WSI mask | 1024-ROI mask | 2048-ROI mask
  All three panels share the same physical zoom window. Red contour = WSI
  doublet (the cps_id_at_bookmark) in every panel. Title color codes:
  green = KEEP (n_focal==1), red = SPLIT.

Run: python scripts/render_doublet_masks_3way.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_18S   = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS = DATA / "cpsam_whole_slide" / "masks.tif"
BENCH     = DATA / "benchmark_doublets.parquet"
MASKS_DIR = ROOT / "figs" / "doublet_survival_at_dev_size" / "masks"
VERDICT   = ROOT / "figs" / "doublet_survival_at_dev_size" / "verdict.csv"
OUT_PDF   = ROOT / "reports" / "doublet_masks_3way.pdf"

PIXEL_SIZE_UM = 0.2125
ZOOM_PX = 300            # display window in WSI px
DOUBLETS_PER_PAGE = 4    # → 12 panels/page; 17 pages for 68 doublets


def crop_centered_wsi_px(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_local(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def label_overlay(masks, alpha=0.55, seed=0):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n + 1))])
    shuf = perm[masks]
    rgba = plt.get_cmap("tab20")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def panel(ax, s18_disp, masks_disp, doublet_contour, title, title_color):
    ax.imshow(s18_disp, cmap="gray", interpolation="nearest")
    ax.imshow(label_overlay(masks_disp), interpolation="nearest")
    if doublet_contour is not None and doublet_contour.any():
        ax.contour(doublet_contour.astype(int), levels=[0.5],
                   colors="red", linewidths=1.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=title_color, pad=3)


def main():
    print("loading WSI + benchmark + verdict…")
    bench = pd.read_parquet(BENCH)
    bench = bench[bench["approved"] == True].sort_values("rank").reset_index(drop=True)
    verdict = pd.read_csv(VERDICT)
    s18_full   = tifffile.imread(WSI_18S, key=0)
    wsi_masks  = tifffile.imread(WSI_MASKS)
    print(f"  {len(bench)} approved doublets; WSI shape {wsi_masks.shape}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    half = ZOOM_PX // 2

    with PdfPages(OUT_PDF) as pdf:
        for page_start in range(0, len(bench), DOUBLETS_PER_PAGE):
            page_doublets = bench.iloc[page_start:page_start + DOUBLETS_PER_PAGE]
            n_rows = len(page_doublets)
            fig, axes = plt.subplots(n_rows, 3, figsize=(11, n_rows * 3.5))
            if n_rows == 1: axes = axes.reshape(1, -1)

            for i, (_, r) in enumerate(page_doublets.iterrows()):
                bid = r["benchmark_id"]; cps_id = int(r["cps_id_at_bookmark"])
                cx_px = int(round(r["x_um"] / PIXEL_SIZE_UM))
                cy_px = int(round(r["y_um"] / PIXEL_SIZE_UM))

                # ---- WSI panel ----
                s18_w, _    = crop_centered_wsi_px(s18_full,  cx_px, cy_px, ZOOM_PX)
                wsim_w, _   = crop_centered_wsi_px(wsi_masks, cx_px, cy_px, ZOOM_PX)
                focal_w     = (wsim_w == cps_id)
                # Relabel mask densely so colormap doesn't squash to one color
                # (WSI labels are sparse: e.g. 46182 next to 47029)
                uniq = np.unique(wsim_w); uniq = uniq[uniq != 0]
                remap = np.zeros(wsim_w.max() + 1, dtype=np.int32)
                for new_lbl, old_lbl in enumerate(uniq, start=1):
                    remap[int(old_lbl)] = new_lbl
                wsim_dense = remap[wsim_w]
                v1024 = verdict[(verdict["benchmark_id"] == bid) & (verdict["roi_size_px"] == 1024)]
                v2048 = verdict[(verdict["benchmark_id"] == bid) & (verdict["roi_size_px"] == 2048)]
                row_label = (f"{bid}  r{int(r['rank'])}  "
                              f"{r['lineage_pair'].replace(' × ', '×')}  "
                              f"cps_id={cps_id}")
                panel(axes[i, 0], norm_local(s18_w), wsim_dense, focal_w,
                       f"WSI mask\n{row_label}", "black")

                # ---- 1024-ROI panel ----
                ax = axes[i, 1]
                p1024 = MASKS_DIR / f"{bid}_1024.npz"
                if p1024.exists() and len(v1024):
                    arr = np.load(p1024)
                    masks = arr["masks"]; s18r = arr["s18"]; focal_r = arr["focal_region"]
                    H, W = masks.shape; cy_l = H // 2; cx_l = W // 2
                    zy0 = max(cy_l - half, 0); zy1 = min(cy_l + half, H)
                    zx0 = max(cx_l - half, 0); zx1 = min(cx_l + half, W)
                    nf = int(v1024.iloc[0]["n_focal"])
                    keep = bool(v1024.iloc[0]["stays_doublet"])
                    panel(ax, norm_local(s18r[zy0:zy1, zx0:zx1]),
                           masks[zy0:zy1, zx0:zx1], focal_r[zy0:zy1, zx0:zx1],
                           f"ROI 1024 px\nn_focal={nf}  {'KEEP' if keep else 'split'}",
                           "tab:green" if keep else "tab:red")
                else:
                    ax.set_title("ROI 1024 — pending", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])

                # ---- 2048-ROI panel ----
                ax = axes[i, 2]
                p2048 = MASKS_DIR / f"{bid}_2048.npz"
                if p2048.exists() and len(v2048):
                    arr = np.load(p2048)
                    masks = arr["masks"]; s18r = arr["s18"]; focal_r = arr["focal_region"]
                    H, W = masks.shape; cy_l = H // 2; cx_l = W // 2
                    zy0 = max(cy_l - half, 0); zy1 = min(cy_l + half, H)
                    zx0 = max(cx_l - half, 0); zx1 = min(cx_l + half, W)
                    nf = int(v2048.iloc[0]["n_focal"])
                    keep = bool(v2048.iloc[0]["stays_doublet"])
                    panel(ax, norm_local(s18r[zy0:zy1, zx0:zx1]),
                           masks[zy0:zy1, zx0:zx1], focal_r[zy0:zy1, zx0:zx1],
                           f"ROI 2048 px\nn_focal={nf}  {'KEEP' if keep else 'split'}",
                           "tab:green" if keep else "tab:red")
                else:
                    ax.set_title("ROI 2048 — pending", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])

            plt.suptitle(
                f"Approved benchmark doublets {page_start + 1}–{page_start + n_rows} of {len(bench)}  "
                f"— WSI mask  |  ROI 1024 mask  |  ROI 2048 mask  "
                f"(red contour = WSI doublet; same {ZOOM_PX}-px window across all 3 panels)",
                fontsize=10, y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
            print(f"  page {page_start // DOUBLETS_PER_PAGE + 1}/"
                  f"{(len(bench) + DOUBLETS_PER_PAGE - 1) // DOUBLETS_PER_PAGE} written")
    print(f"\nwrote {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
