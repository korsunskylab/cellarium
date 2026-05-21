"""Assemble the single GitHub-issue figure: 2 ROIs × {full view + zoom}.

Reads the npz files produced by scripts/make_github_issue_figures.py:
  issueB_cell6_size1024.npz, issueB_cell6_size4096.npz

Layout:
    +------------------+------------------+
    | FULL 1024 ROI    | FULL 4096 ROI    |  (red bbox around cell #6)
    +------------------+------------------+
    | ZOOM (cell #6)   | ZOOM (cell #6)   |  (with mask overlay)
    +------------------+------------------+

Caption includes # distinct masks at cell #6 in each.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
IN_DIR = ROOT / "reports" / "github_issue_figs"

PIXEL_SIZE_UM = 0.2125
CX_UM, CY_UM = 8886.2, 1478.4
CX_WSI_PX = int(round(CX_UM / PIXEL_SIZE_UM))
CY_WSI_PX = int(round(CY_UM / PIXEL_SIZE_UM))
ZOOM_PX = 200             # display window for the zoom row (in original px)


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


def n_masks_in_window(masks, cy, cx, half):
    H, W = masks.shape
    y0 = max(cy - half, 0); y1 = min(cy + half, H)
    x0 = max(cx - half, 0); x1 = min(cx + half, W)
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0)


def main():
    sizes = (1024, 4096)
    data = {}
    for sz in sizes:
        f = IN_DIR / f"issueB_cell6_size{sz}.npz"
        if not f.exists():
            print(f"  MISSING: {f.name} — run make_github_issue_figures.py first")
            return
        arr = np.load(f)
        data[sz] = dict(masks=arr["masks"], bbox=arr["bbox"], s18=arr["s18_raw"])

    # Figure: 2 rows × 2 cols
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    for col, sz in enumerate(sizes):
        d = data[sz]; masks = d["masks"]; bbox = d["bbox"]; s18 = d["s18"]
        # Local coords of the focal cell in this crop
        cy_loc = CY_WSI_PX - bbox[1]; cx_loc = CX_WSI_PX - bbox[0]
        half_zoom = ZOOM_PX // 2

        # --- Top row: FULL ROI with bbox around the zoom region ---
        s18_full_disp = norm_local(s18)
        axes[0, col].imshow(s18_full_disp, cmap="gray", interpolation="nearest")
        axes[0, col].imshow(label_overlay(masks, alpha=0.35), interpolation="nearest")
        # Red bbox around the zoom region
        rect = mpatches.Rectangle(
            (cx_loc - half_zoom, cy_loc - half_zoom),
            ZOOM_PX, ZOOM_PX,
            linewidth=2.5, edgecolor="red", facecolor="none")
        axes[0, col].add_patch(rect)
        axes[0, col].set_title(
            f"ROI = {sz} × {sz} px ({sz * PIXEL_SIZE_UM:.0f} µm) — full view\n"
            f"red box = zoom region around focal cell",
            fontsize=11)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        # --- Bottom row: ZOOM around the focal cell ---
        H, W = masks.shape
        zy0 = max(cy_loc - half_zoom, 0); zy1 = min(cy_loc + half_zoom, H)
        zx0 = max(cx_loc - half_zoom, 0); zx1 = min(cx_loc + half_zoom, W)
        s18_zoom = norm_local(s18[zy0:zy1, zx0:zx1])
        masks_zoom = masks[zy0:zy1, zx0:zx1]
        n_focal = n_masks_in_window(masks, cy_loc, cx_loc, 30)
        axes[1, col].imshow(s18_zoom, cmap="gray", interpolation="nearest")
        axes[1, col].imshow(label_overlay(masks_zoom), interpolation="nearest")
        # Star at focal cell centroid in the zoomed display
        axes[1, col].plot(cy_loc - zy0, cy_loc - zy0, marker="*", color="yellow",
                          markersize=24, markeredgecolor="black", markeredgewidth=1.5)
        # Use cx for x, cy for y — correct version:
        axes[1, col].lines[-1].set_xdata([cx_loc - zx0])
        axes[1, col].lines[-1].set_ydata([cy_loc - zy0])
        axes[1, col].set_title(
            f"zoom: 200 × 200 px around focal cell\n"
            f"# distinct masks in 60-px window at ★ : {n_focal}",
            fontsize=11)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    plt.suptitle(
        "Cellpose-SAM at default parameters — same input pixels, different ROI size, "
        "different result.\n"
        "Both crops are centered on the same focal cell (yellow ★, WSI px "
        f"({CX_WSI_PX}, {CY_WSI_PX})). The 1024-px crop is contained inside the\n"
        "4096-px crop — so the focal cell and its immediate context are identical "
        "in both. Only the surrounding image size differs.\n"
        "Call: `model.eval(img, channel_axis=-1)`  with `pretrained_model=\"cpsam\"` "
        "and all other parameters left at default.",
        fontsize=10.5, y=1.0)
    plt.tight_layout()
    out = IN_DIR / "issue_cell6_full_and_zoom.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
