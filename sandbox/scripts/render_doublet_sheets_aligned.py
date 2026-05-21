"""Re-render doublet-survival sheets so 1024 and 2048 panels show the SAME
physical WSI region for each doublet (not the same local pixel inside each
crop). Uses the WSI focal pixel as the center of the display window in both
ROIs, with bbox from each saved npz to convert to local coords.

Outputs three files:
  reports/doublet_masks_3way_aligned.pdf      — paginated, WSI | 1024 | 2048 per row
  reports/doublet_sheet_1024_aligned.png      — 8x9 grid, one panel per doublet
  reports/doublet_sheet_2048_aligned.png      — 8x9 grid, one panel per doublet
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_18S   = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS = DATA / "cpsam_whole_slide" / "masks.tif"
BENCH     = DATA / "benchmark_doublets.parquet"
MASKS_DIR = ROOT / "figs" / "doublet_survival_at_dev_size" / "masks"
VERDICT   = ROOT / "figs" / "doublet_survival_at_dev_size" / "verdict.csv"

OUT_PDF       = ROOT / "reports" / "doublet_masks_3way_aligned.pdf"
OUT_SHEET_1K  = ROOT / "reports" / "doublet_sheet_1024_aligned.png"
OUT_SHEET_2K  = ROOT / "reports" / "doublet_sheet_2048_aligned.png"

PIXEL_SIZE_UM = 0.2125
ZOOM_PX = 300


def norm_local(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def label_overlay(masks, alpha=0.55, seed=3):
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


def display_window_around_wsi_pixel(arr, bbox, cx_px, cy_px, half_zoom):
    """Crop arr to a (2*half_zoom)-square window centered on the WSI pixel
    (cx_px, cy_px). bbox is (x0, y0, x1, y1) of the arr in WSI coords."""
    cy_local = cy_px - bbox[1]
    cx_local = cx_px - bbox[0]
    H, W = arr.shape[-2:]
    zy0 = max(cy_local - half_zoom, 0); zy1 = min(cy_local + half_zoom, H)
    zx0 = max(cx_local - half_zoom, 0); zx1 = min(cx_local + half_zoom, W)
    if arr.ndim == 2:
        return arr[zy0:zy1, zx0:zx1], (cx_local - zx0, cy_local - zy0)
    return arr[..., zy0:zy1, zx0:zx1], (cx_local - zx0, cy_local - zy0)


def panel(ax, s18_disp, masks_disp, doublet_contour, star_xy, title, title_color):
    ax.imshow(s18_disp, cmap="gray", interpolation="nearest")
    ax.imshow(label_overlay(masks_disp), interpolation="nearest")
    if doublet_contour is not None and doublet_contour.any():
        ax.contour(doublet_contour.astype(int), levels=[0.5], colors="red", linewidths=1.4)
    if star_xy is not None:
        ax.plot(star_xy[0], star_xy[1], "*", color="yellow",
                markersize=12, markeredgecolor="black", markeredgewidth=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7, color=title_color, pad=2)


def main():
    print("loading WSI + benchmark + verdict…")
    bench = pd.read_parquet(BENCH)
    bench = bench[bench["approved"] == True].sort_values("rank").reset_index(drop=True)
    verdict = pd.read_csv(VERDICT)
    s18_full   = tifffile.imread(WSI_18S, key=0)
    wsi_masks  = tifffile.imread(WSI_MASKS)
    print(f"  {len(bench)} approved doublets")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    half = ZOOM_PX // 2

    # ============ 8x9 grids: one per ROI ============
    for roi_size, out_path in [(1024, OUT_SHEET_1K), (2048, OUT_SHEET_2K)]:
        n_rows = math.ceil(len(bench) / 8)
        fig, axes = plt.subplots(n_rows, 8, figsize=(8 * 2.4, n_rows * 2.4))
        axes = axes.flatten()
        for ax in axes: ax.axis("off")
        for i, (_, r) in enumerate(bench.iterrows()):
            bid = r["benchmark_id"]
            cps_id = int(r["cps_id_at_bookmark"])
            cx_px = int(round(r["x_um"] / PIXEL_SIZE_UM))
            cy_px = int(round(r["y_um"] / PIXEL_SIZE_UM))
            p = MASKS_DIR / f"{bid}_{roi_size}.npz"
            if not p.exists(): continue
            arr = np.load(p)
            masks = arr["masks"]; s18 = arr["s18"]; focal = arr["focal_region"]
            bbox = arr["bbox"]
            s18_disp, star = display_window_around_wsi_pixel(s18, bbox, cx_px, cy_px, half)
            masks_disp, _ = display_window_around_wsi_pixel(masks, bbox, cx_px, cy_px, half)
            focal_disp, _ = display_window_around_wsi_pixel(focal, bbox, cx_px, cy_px, half)
            ver = verdict[(verdict["benchmark_id"]==bid) & (verdict["roi_size_px"]==roi_size)]
            if len(ver) == 0:
                title = f"{bid} (no verdict)"; color = "gray"; nf = "?"
            else:
                nf = int(ver.iloc[0]["n_focal"])
                keep = bool(ver.iloc[0]["stays_doublet"])
                color = "tab:green" if keep else "tab:red"
            title = (f"{bid[-3:]} r{int(r['rank'])} "
                     f"{r['lineage_pair'].replace(' × ', '×')}\n"
                     f"nf={nf}  {'KEEP' if color=='tab:green' else 'split' if color=='tab:red' else '?'}")
            ax = axes[i]
            ax.axis("on")
            panel(ax, norm_local(s18_disp), masks_disp, focal_disp, star, title, color)
        plt.suptitle(f"Approved doublets — CP-SAM at ROI={roi_size}px "
                     f"({roi_size*PIXEL_SIZE_UM:.0f}µm), {ZOOM_PX}px display window "
                     f"centered on WSI focal pixel (yellow ★).  "
                     f"red contour = WSI doublet.  green title = KEEP, red title = split.",
                     fontsize=10, y=1.001)
        plt.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_path.name}  ({out_path.stat().st_size//1024} KB)")

    # ============ 3-way aligned PDF (WSI | 1024 | 2048) ============
    DOUBLETS_PER_PAGE = 4
    with PdfPages(OUT_PDF) as pdf:
        for ps in range(0, len(bench), DOUBLETS_PER_PAGE):
            page = bench.iloc[ps:ps + DOUBLETS_PER_PAGE]
            n_rows = len(page)
            fig, axes = plt.subplots(n_rows, 3, figsize=(11, n_rows * 3.5))
            if n_rows == 1: axes = axes.reshape(1, -1)
            for i, (_, r) in enumerate(page.iterrows()):
                bid = r["benchmark_id"]; cps_id = int(r["cps_id_at_bookmark"])
                cx_px = int(round(r["x_um"] / PIXEL_SIZE_UM))
                cy_px = int(round(r["y_um"] / PIXEL_SIZE_UM))

                # WSI panel: extract the same ZOOM_PX-square window directly from the WSI
                wzy0 = max(cy_px - half, 0); wzy1 = wzy0 + ZOOM_PX
                wzx0 = max(cx_px - half, 0); wzx1 = wzx0 + ZOOM_PX
                wzy1 = min(wzy1, s18_full.shape[0]); wzx1 = min(wzx1, s18_full.shape[1])
                s18_w   = s18_full[wzy0:wzy1, wzx0:wzx1]
                wsim_w  = wsi_masks[wzy0:wzy1, wzx0:wzx1]
                focal_w = (wsim_w == cps_id)
                star_wsi = (cx_px - wzx0, cy_px - wzy0)
                # densify labels for color contrast
                uniq = np.unique(wsim_w); uniq = uniq[uniq != 0]
                if len(uniq):
                    remap = np.zeros(int(wsim_w.max()) + 1, dtype=np.int32)
                    for new_lbl, old_lbl in enumerate(uniq, start=1):
                        remap[int(old_lbl)] = new_lbl
                    wsim_dense = remap[wsim_w]
                else:
                    wsim_dense = wsim_w.astype(np.int32)
                row_label = (f"{bid}  r{int(r['rank'])}  "
                              f"{r['lineage_pair'].replace(' × ', '×')}  cps={cps_id}")
                panel(axes[i, 0], norm_local(s18_w), wsim_dense, focal_w, star_wsi,
                       f"WSI mask\n{row_label}", "black")

                # 1024 + 2048 panels: display same WSI window using stored bbox
                for col, sz in enumerate((1024, 2048), start=1):
                    p = MASKS_DIR / f"{bid}_{sz}.npz"
                    if not p.exists():
                        axes[i, col].set_title(f"ROI {sz} — n/a"); axes[i, col].set_xticks([]); axes[i, col].set_yticks([])
                        continue
                    arr = np.load(p)
                    masks = arr["masks"]; s18 = arr["s18"]; focal = arr["focal_region"]; bbox = arr["bbox"]
                    s18d, star = display_window_around_wsi_pixel(s18, bbox, cx_px, cy_px, half)
                    masksd, _ = display_window_around_wsi_pixel(masks, bbox, cx_px, cy_px, half)
                    focald, _ = display_window_around_wsi_pixel(focal, bbox, cx_px, cy_px, half)
                    ver = verdict[(verdict["benchmark_id"]==bid) & (verdict["roi_size_px"]==sz)]
                    nf = int(ver.iloc[0]["n_focal"]) if len(ver) else None
                    keep = bool(ver.iloc[0]["stays_doublet"]) if len(ver) else False
                    color = "tab:green" if keep else "tab:red"
                    panel(axes[i, col], norm_local(s18d), masksd, focald, star,
                           f"ROI {sz}px  n_focal={nf}  {'KEEP' if keep else 'split'}",
                           color)
            plt.suptitle(
                f"Approved benchmark doublets {ps+1}–{ps+n_rows} of {len(bench)} — "
                f"WSI mask  |  ROI 1024 mask  |  ROI 2048 mask\n"
                f"Same {ZOOM_PX}-px WSI window in all three columns. "
                f"Red contour = WSI doublet. Yellow ★ = WSI focal pixel (centroid).",
                fontsize=10, y=1.005)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
            print(f"  page {ps // DOUBLETS_PER_PAGE + 1}/"
                  f"{math.ceil(len(bench) / DOUBLETS_PER_PAGE)} written")
    print(f"\nwrote {OUT_PDF}  ({OUT_PDF.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
