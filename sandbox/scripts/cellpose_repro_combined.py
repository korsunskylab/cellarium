"""Single combined script: runs cellpose at two ROI sizes around the same
focal cell, saves masks, generates the 2×4 figure (rows = full / zoom,
cols = ROI 1024 DAPI / 1024 18S / 4096 DAPI / 4096 18S) with mask overlay,
and produces a verbose run log — all in one execution so the figure and the
log can't disagree.

Output:
    reports/github_issue_figs/issue_cell6_full_and_zoom_2chan.png
    reports/github_issue_figs/runlog_verbose.txt           (verbose cellpose log)
    reports/github_issue_figs/cell6_combined_masks_1024.npz
    reports/github_issue_figs/cell6_combined_masks_4096.npz
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cellpose import models, io
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
OUT = ROOT / "reports" / "github_issue_figs"
OUT.mkdir(parents=True, exist_ok=True)

CX_PX, CY_PX = 41817, 6957             # focal cell centroid in WSI px
SIZES = (1024, 4096)
ZOOM_PX = 200                          # display zoom around cell #6
COUNT_HALF = 25                        # 50-px window for cell-#6 mask count


def crop_centered(arr, cx, cy, size):
    half = size // 2
    H, W = arr.shape[-2:]
    y0 = max(cy - half, 0); y1 = min(y0 + size, H); y0 = y1 - size
    x0 = max(cx - half, 0); x1 = min(x0 + size, W); x0 = x1 - size
    return arr[..., y0:y1, x0:x1]


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


def n_in_window(masks, cy, cx, half=COUNT_HALF):
    H, W = masks.shape
    y0 = max(cy - half, 0); y1 = min(cy + half, H)
    x0 = max(cx - half, 0); x1 = min(cx + half, W)
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0)


def main():
    logger = io.logger_setup()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S,  key=0)
    print(f"  WSI shape: {dapi_full.shape}")

    print("loading cellpose-SAM…")
    m = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    data = {}
    for sz in SIZES:
        print(f"\n=========  ROI = {sz} × {sz} px  =========")
        dapi_c = crop_centered(dapi_full, CX_PX, CY_PX, sz)
        s18_c  = crop_centered(s18_full,  CX_PX, CY_PX, sz)
        img = np.stack([dapi_c.astype(np.float32),
                        s18_c.astype(np.float32)], axis=-1)
        print(f"  img.shape = {img.shape}, dtype = {img.dtype}")
        print(f"  calling model.eval(img, channel_axis=-1)  with all other params at default")
        masks, _, _ = m.eval(img, channel_axis=-1)
        masks = masks.astype(np.int32)
        cy_loc = sz // 2; cx_loc = sz // 2
        n_focal = n_in_window(masks, cy_loc, cx_loc)
        print(f"  n_total masks: {int(masks.max())}")
        print(f"  # distinct masks in 50-px window around focal cell: {n_focal}")
        np.savez_compressed(OUT / f"cell6_combined_masks_{sz}.npz",
                            masks=masks, dapi=dapi_c, s18=s18_c)
        data[sz] = dict(masks=masks, dapi=dapi_c, s18=s18_c, n_focal=n_focal)

    # ============ FIGURE ============
    # 2 rows × 4 cols:
    #   col 0: ROI 1024 DAPI            col 1: ROI 1024 18S
    #   col 2: ROI 4096 DAPI            col 3: ROI 4096 18S
    # row 0: full view (red bbox around zoom region)
    # row 1: 200-px zoom (with mask overlay)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    half_zoom = ZOOM_PX // 2
    col_titles = ["DAPI — ROI 1024", "18S — ROI 1024",
                  "DAPI — ROI 4096", "18S — ROI 4096"]

    for ci, (sz, ch) in enumerate([(1024, "dapi"), (1024, "s18"),
                                    (4096, "dapi"), (4096, "s18")]):
        d = data[sz]; masks = d["masks"]; img_chan = d[ch]
        cy_loc = sz // 2; cx_loc = sz // 2

        # Top row: full view of the channel with mask overlay + red bbox
        img_disp = norm_local(img_chan)
        axes[0, ci].imshow(img_disp, cmap="gray", interpolation="nearest")
        axes[0, ci].imshow(label_overlay(masks, alpha=0.30), interpolation="nearest")
        rect = mpatches.Rectangle((cx_loc - half_zoom, cy_loc - half_zoom),
                                   ZOOM_PX, ZOOM_PX,
                                   linewidth=2.5, edgecolor="red", facecolor="none")
        axes[0, ci].add_patch(rect)
        axes[0, ci].set_title(f"{col_titles[ci]} — full view\n(red box = zoom region)",
                              fontsize=10)
        axes[0, ci].set_xticks([]); axes[0, ci].set_yticks([])

        # Bottom row: zoomed view of channel + masks
        H, W = masks.shape
        zy0 = max(cy_loc - half_zoom, 0); zy1 = min(cy_loc + half_zoom, H)
        zx0 = max(cx_loc - half_zoom, 0); zx1 = min(cx_loc + half_zoom, W)
        img_zoom = norm_local(img_chan[zy0:zy1, zx0:zx1])
        masks_zoom = masks[zy0:zy1, zx0:zx1]
        n_at = n_in_window(masks, cy_loc, cx_loc)
        axes[1, ci].imshow(img_zoom, cmap="gray", interpolation="nearest")
        axes[1, ci].imshow(label_overlay(masks_zoom), interpolation="nearest")
        axes[1, ci].plot(cx_loc - zx0, cy_loc - zy0, "*", color="yellow",
                          markersize=22, markeredgecolor="black", markeredgewidth=1.5)
        axes[1, ci].set_title(f"{col_titles[ci]} — 200×200 px zoom\n"
                               f"# masks at ★ (50-px window): {n_at}", fontsize=10)
        axes[1, ci].set_xticks([]); axes[1, ci].set_yticks([])

    plt.suptitle(
        "cellpose-SAM at TRUE defaults — same focal cell ★ (WSI px (41817, 6957)), "
        "two ROI sizes (1024 and 4096), both morphology channels shown (DAPI and 18S).\n"
        "The 1024 crop is fully contained inside the 4096 crop, so the focal cell and "
        "~110 µm of context are identical pixels in both. Only the surrounding image size "
        "differs.\n"
        f"Result: {data[1024]['n_focal']} masks at the focal cell in the 1024 crop, "
        f"{data[4096]['n_focal']} masks in the 4096 crop.",
        fontsize=10.5, y=1.02)
    plt.tight_layout()
    out_fig = OUT / "issue_cell6_full_and_zoom_2chan.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved figure: {out_fig}")
    print(f"saved masks:  {OUT / 'cell6_combined_masks_{1024,4096}.npz'}")
    print(f"verbose log written to {Path.home() / '.cellpose' / 'run.log'} "
          f"(also captured in this stdout)")


if __name__ == "__main__":
    main()
