"""Two spatial-only GitHub issue figures at **TRUE cellpose-SAM defaults**.

Default args from `cellpose.models.CellposeModel.eval`:
    augment=False, normalize=True, diameter=None,
    cellprob_threshold=0.0, flow_threshold=0.4, niter=None, bsize=256

These are the bare-minimum, no-customization settings. If the bug manifests
here, cellpose maintainers can't dismiss it as "you tuned the thresholds
weirdly."

ISSUE A — same location, different ROI size, different # masks:
    panel L: ROI=256 px at ring location
    panel R: ROI=4096 px at ring location

ISSUE B — same focal cell, different ROI size, different # masks at the cell:
    panel L: ROI=1024 px at focal-cell centroid
    panel R: ROI=4096 px at focal-cell centroid

Both figures show 18S intensity + cellpose mask overlay in a fixed display
window. Star marks the reference pixel. Caption shows # distinct masks in a
40-px (issue A) or 50-px (issue B) window at the star.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
OUT = ROOT / "reports" / "github_issue_figs"
OUT.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
CX_UM, CY_UM = 8886.2, 1478.4              # focal cell #6
RING_WSI_X, RING_WSI_Y = 41680, 6996       # WSI px of the onion-ring location

CPSAM_DEFAULTS = dict()    # empty → everything at cellpose defaults
# (augment=False, normalize=True, diameter=None,
#  cellprob_threshold=0.0, flow_threshold=0.4, niter=None, bsize=256)


def crop_centered_wsi_px(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def crop_centered_um(arr, cx_um, cy_um, size_px):
    return crop_centered_wsi_px(arr,
        int(round(cx_um / PIXEL_SIZE_UM)),
        int(round(cy_um / PIXEL_SIZE_UM)),
        size_px)


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


def n_labels_in_window(masks, cy, cx, half):
    H, W = masks.shape
    y0 = max(cy - half, 0); y1 = min(cy + half, H)
    x0 = max(cx - half, 0); x1 = min(cx + half, W)
    if y1 <= y0 or x1 <= x0: return 0
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0)


def run_at_default(m_sam, dapi_full, s18_full, cx_wsi_px, cy_wsi_px, size_px):
    """Cellpose-SAM at true defaults. Returns (masks, bbox, s18_raw)."""
    dapi_c, bbox = crop_centered_wsi_px(dapi_full, cx_wsi_px, cy_wsi_px, size_px)
    s18_c, _ = crop_centered_wsi_px(s18_full, cx_wsi_px, cy_wsi_px, size_px)
    img = np.stack([dapi_c.astype(np.float32), s18_c.astype(np.float32)], axis=-1)
    t0 = time.time()
    masks, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_DEFAULTS)
    dt = time.time() - t0
    print(f"    cellpose-SAM defaults, ROI={size_px}: n_total={masks.max()}, "
          f"runtime={dt:.1f}s")
    return masks.astype(np.int32), bbox, s18_c


def make_issue_A(m_sam, dapi_full, s18_full):
    """Onion ring: 256 vs 4096 at ring location, true cpsam defaults."""
    print("\n=== Issue A — onion ring location at cellpose defaults ===")
    pair = {}
    for sz in (256, 4096):
        masks, bbox, s18 = run_at_default(m_sam, dapi_full, s18_full,
                                           RING_WSI_X, RING_WSI_Y, sz)
        pair[sz] = (masks, bbox, s18)
        np.savez_compressed(OUT / f"issueA_ring_size{sz}.npz",
                            masks=masks, bbox=np.array(bbox), s18_raw=s18)

    ZOOM = 100; half = ZOOM // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, sz in zip(axes, (256, 4096)):
        masks, bbox, s18 = pair[sz]
        ly = RING_WSI_Y - bbox[1]; lx = RING_WSI_X - bbox[0]
        H, W = masks.shape
        zy0 = max(ly - half, 0); zy1 = min(ly + half, H)
        zx0 = max(lx - half, 0); zx1 = min(lx + half, W)
        n_lab = n_labels_in_window(masks, ly, lx, 20)
        ax.imshow(norm_local(s18[zy0:zy1, zx0:zx1]), cmap="gray", interpolation="nearest")
        ax.imshow(label_overlay(masks[zy0:zy1, zx0:zx1], seed=1), interpolation="nearest")
        ax.set_title(f"ROI = {sz} × {sz} px ({sz * PIXEL_SIZE_UM:.0f} µm)\n"
                     f"# distinct masks in 40-px window at ★ : {n_lab}",
                     fontsize=11)
        ax.plot(lx - zx0, ly - zy0, "*", color="yellow", markersize=22,
                markeredgecolor="black", markeredgewidth=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Identical pixel ★ (WSI px (41680, 6996)) inside both crops.\n"
                 "Cellpose-SAM at TRUE defaults: augment=False, normalize=True,\n"
                 "diameter=None, cellprob_threshold=0, flow_threshold=0.4. Only ROI size differs.",
                 fontsize=10.5, y=1.08)
    plt.tight_layout()
    fig.savefig(OUT / "issueA_onion_ring.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {OUT / 'issueA_onion_ring.png'}")


def make_issue_B(m_sam, dapi_full, s18_full):
    """Focal cell #6: ROI 1024 vs 4096, true cpsam defaults."""
    print("\n=== Issue B — focal cell at cellpose defaults ===")
    cx_px = int(round(CX_UM / PIXEL_SIZE_UM))
    cy_px = int(round(CY_UM / PIXEL_SIZE_UM))
    pair = {}
    for sz in (1024, 4096):
        masks, bbox, s18 = run_at_default(m_sam, dapi_full, s18_full,
                                           cx_px, cy_px, sz)
        pair[sz] = (masks, bbox, s18)
        np.savez_compressed(OUT / f"issueB_cell6_size{sz}.npz",
                            masks=masks, bbox=np.array(bbox), s18_raw=s18)

    DISP = 200; half = DISP // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, sz in zip(axes, (1024, 4096)):
        masks, bbox, s18 = pair[sz]
        ly = cy_px - bbox[1]; lx = cx_px - bbox[0]
        H, W = masks.shape
        zy0 = max(ly - half, 0); zy1 = min(ly + half, H)
        zx0 = max(lx - half, 0); zx1 = min(lx + half, W)
        n_at = n_labels_in_window(masks, ly, lx, 25)
        ax.imshow(norm_local(s18[zy0:zy1, zx0:zx1]), cmap="gray", interpolation="nearest")
        ax.imshow(label_overlay(masks[zy0:zy1, zx0:zx1], seed=3), interpolation="nearest")
        ax.set_title(f"ROI = {sz} × {sz} px ({sz * PIXEL_SIZE_UM:.0f} µm)\n"
                     f"# distinct masks in 50-px window at ★ : {n_at}",
                     fontsize=11)
        ax.plot(lx - zx0, ly - zy0, "*", color="yellow", markersize=22,
                markeredgecolor="black", markeredgewidth=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Identical pixel ★ (same focal cell) inside both crops.\n"
                 "Cellpose-SAM at TRUE defaults: augment=False, normalize=True,\n"
                 "diameter=None, cellprob_threshold=0, flow_threshold=0.4. Only ROI size differs.",
                 fontsize=10.5, y=1.08)
    plt.tight_layout()
    fig.savefig(OUT / "issueB_cell_count_inconsistency.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {OUT / 'issueB_cell_count_inconsistency.png'}")


def main():
    print("loading WSI imagery…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    make_issue_A(m_sam, dapi_full, s18_full)
    make_issue_B(m_sam, dapi_full, s18_full)


if __name__ == "__main__":
    main()
