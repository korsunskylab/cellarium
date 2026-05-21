"""Onion-ring stitching hypothesis test, focused on ONE specific onion ring near cell #6.

Procedure:
  1. Run baseline CP-SAM on 4096-px crop centered on cell #6
  2. Find the SPECIFIC onion-ring location near cell #6 (max distinct labels in window)
  3. Record its WSI-pixel coordinates
  4. For each test variant, zoom into the SAME WSI location, count distinct labels
     there, and visualize. Direct confirmation: is the ring still present?

Variants:
  baseline: augment=True (cellpose default, 50% overlap hardcoded), tile_overlap=0.1 (ignored)
  augment=False, tile_overlap=0.1: low overlap, no aug-averaging
  augment=False, tile_overlap=0.5: max overlap (cellpose-capped), no aug-averaging
  shifted crop: same content shifted +100,+100 px so cell sits at different position
                in cellpose's internal tile grid

Saves the actual masks (per variant) so the report can be regenerated without rerunning.
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
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = ROOT / "figs" / "onion_ring_stitching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
CROP_PX = 4096
SEARCH_PX = 500          # search-for-ring window around cell #6
RING_ZOOM_PX = 100       # zoom panel around the identified ring
SHIFT_PX = 100           # crop-bbox shift for shift test
CX_UM, CY_UM = 8886.2, 1478.4

CPSAM_KW = dict(diameter=None, niter=200,
                cellprob_threshold=-5.0, flow_threshold=0.0)


def crop_centered(arr, cx_um, cy_um, size_px, shift_px=(0, 0)):
    sx, sy = shift_px
    cx_px = int(round(cx_um / PIXEL_SIZE_UM)) - sx
    cy_px = int(round(cy_um / PIXEL_SIZE_UM)) - sy
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    a = arr.astype(np.float32)
    return np.clip((a - q_lo) / max(q_hi - q_lo, 1e-6), 0, 1).astype(np.float32)


def cellpose_tile_starts(L, bsize=256, tile_overlap=0.1, augment=True):
    if augment:
        ny = max(2, int(np.ceil(2. * L / bsize)))
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        ny = 1 if L <= bsize else int(np.ceil((1. + 2 * tile_overlap) * L / bsize))
    return np.linspace(0, L - bsize, ny).astype(int)


def find_onion_ring_in_window(masks, search_y0, search_y1, search_x0, search_x1, window=25, step=8):
    """Find the pixel inside (search_y0..y1, search_x0..x1) with the most distinct
    mask labels in a `window`-radius square. Returns (n_distinct, y, x)."""
    best = (0, 0, 0)
    for y in range(search_y0 + window, search_y1 - window, step):
        for x in range(search_x0 + window, search_x1 - window, step):
            patch = masks[y - window:y + window, x - window:x + window]
            n = len(np.unique(patch)) - (1 if 0 in patch else 0)
            if n > best[0]:
                best = (n, y, x)
    return best


def count_labels_at(masks, y_local, x_local, half_window=20):
    """Count distinct mask labels in a 2*half_window square centered at (y_local, x_local)."""
    H, W = masks.shape
    y0 = max(y_local - half_window, 0); y1 = min(y_local + half_window, H)
    x0 = max(x_local - half_window, 0); x1 = min(x_local + half_window, W)
    if y1 <= y0 or x1 <= x0: return 0
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0)


def label_overlay(masks, alpha=0.55):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(0).permutation(np.arange(1, n + 1))])
    shuf = perm[masks]
    rgba = plt.get_cmap("nipy_spectral")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def main():
    t_total = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(dapi_n, s18_n, **kw):
        k = dict(CPSAM_KW); k.update(kw)
        img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
        masks, _, _ = m_sam.eval(img, channel_axis=-1, **k)
        return masks.astype(np.int32)

    # === Baseline ===
    print("\n[1/4] baseline (augment=True)")
    dapi_c, bbox = crop_centered(dapi_full, CX_UM, CY_UM, CROP_PX)
    s18_c,  _    = crop_centered(s18_full,  CX_UM, CY_UM, CROP_PX)
    dapi_n = norm_with(dapi_c, **wsi_pct["DAPI"])
    s18_n  = norm_with(s18_c,  **wsi_pct["18S"])
    t0 = time.time()
    masks_baseline = run(dapi_n, s18_n, augment=True, tile_overlap=0.1)
    print(f"  n_cells={int(masks_baseline.max())} ({time.time()-t0:.1f}s)")

    # Find the SPECIFIC onion ring near cell #6 in the baseline
    cx_local = int(round(CX_UM / PIXEL_SIZE_UM)) - bbox[0]
    cy_local = int(round(CY_UM / PIXEL_SIZE_UM)) - bbox[1]
    print(f"  cell #6 local position in baseline crop: ({cx_local}, {cy_local})")
    # Search window: 500 px around focal cell
    sy0 = max(cy_local - SEARCH_PX // 2, 0); sy1 = sy0 + SEARCH_PX
    sx0 = max(cx_local - SEARCH_PX // 2, 0); sx1 = sx0 + SEARCH_PX
    n_ring_baseline, ring_y_baseline, ring_x_baseline = find_onion_ring_in_window(
        masks_baseline, sy0, sy1, sx0, sx1)
    print(f"  worst label density spot in 500-px search: ({ring_x_baseline}, {ring_y_baseline})  "
          f"with {n_ring_baseline} distinct labels in 25-px radius")
    # WSI-pixel coords of this specific ring
    ring_wsi_x = ring_x_baseline + bbox[0]
    ring_wsi_y = ring_y_baseline + bbox[1]
    print(f"  → WSI px: ({ring_wsi_x}, {ring_wsi_y})")

    # === Test 2: tile_overlap=0.1, augment=False ===
    print("\n[2/4] augment=False, tile_overlap=0.1 (low overlap)")
    t0 = time.time()
    masks_lowover = run(dapi_n, s18_n, augment=False, tile_overlap=0.1)
    print(f"  n_cells={int(masks_lowover.max())} ({time.time()-t0:.1f}s)")

    # === Test 3: tile_overlap=0.5, augment=False ===
    print("\n[3/4] augment=False, tile_overlap=0.5 (high overlap)")
    t0 = time.time()
    masks_highover = run(dapi_n, s18_n, augment=False, tile_overlap=0.5)
    print(f"  n_cells={int(masks_highover.max())} ({time.time()-t0:.1f}s)")

    # === Test 4: shifted crop ===
    print(f"\n[4/4] shifted crop by ({SHIFT_PX},{SHIFT_PX}) px (cell #6 at different tile-grid position)")
    dapi_s, bbox_s = crop_centered(dapi_full, CX_UM, CY_UM, CROP_PX, shift_px=(SHIFT_PX, SHIFT_PX))
    s18_s,  _      = crop_centered(s18_full,  CX_UM, CY_UM, CROP_PX, shift_px=(SHIFT_PX, SHIFT_PX))
    dapi_ns = norm_with(dapi_s, **wsi_pct["DAPI"])
    s18_ns  = norm_with(s18_s,  **wsi_pct["18S"])
    t0 = time.time()
    masks_shifted = run(dapi_ns, s18_ns, augment=True, tile_overlap=0.1)
    print(f"  n_cells={int(masks_shifted.max())} ({time.time()-t0:.1f}s)")

    # === Save masks (so the report can be regenerated without rerun) ===
    np.savez_compressed(OUT_DIR / "masks.npz",
                        baseline=masks_baseline, lowover=masks_lowover,
                        highover=masks_highover, shifted=masks_shifted,
                        baseline_bbox=np.array(bbox), shifted_bbox=np.array(bbox_s))

    # === Direct confirmation: count labels at the SAME WSI location in each variant ===
    # Translate WSI px of the baseline ring → local px in each variant's crop
    def wsi_to_local(wsi_x, wsi_y, var_bbox):
        return wsi_x - var_bbox[0], wsi_y - var_bbox[1]

    bx_loc, by_loc = wsi_to_local(ring_wsi_x, ring_wsi_y, bbox)
    sx_loc, sy_loc = wsi_to_local(ring_wsi_x, ring_wsi_y, bbox_s)
    n_at_baseline = count_labels_at(masks_baseline, by_loc, bx_loc)
    n_at_lowover  = count_labels_at(masks_lowover,  by_loc, bx_loc)
    n_at_highover = count_labels_at(masks_highover, by_loc, bx_loc)
    n_at_shifted  = count_labels_at(masks_shifted,  sy_loc, sx_loc)

    print(f"\n=== Distinct mask labels at the BASELINE ring's WSI location ===")
    print(f"  baseline:               {n_at_baseline}  (the ring we identified)")
    print(f"  augment=False, ov=0.1:  {n_at_lowover}")
    print(f"  augment=False, ov=0.5:  {n_at_highover}")
    print(f"  shifted crop:           {n_at_shifted}")

    # === Figure 1: baseline 500-px zoom with tile grid + ring location marked ===
    starts_aug = cellpose_tile_starts(CROP_PX, augment=True)
    starts_lo  = cellpose_tile_starts(CROP_PX, augment=False, tile_overlap=0.1)
    starts_hi  = cellpose_tile_starts(CROP_PX, augment=False, tile_overlap=0.5)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(s18_n[sy0:sy1, sx0:sx1], cmap="gray", extent=[sx0, sx1, sy1, sy0])
    ax.imshow(label_overlay(masks_baseline[sy0:sy1, sx0:sx1]), extent=[sx0, sx1, sy1, sy0])
    for x in starts_aug:
        if sx0 <= x <= sx1: ax.axvline(x, color="cyan", linewidth=0.8, alpha=0.7, linestyle="--")
        if sx0 <= x + 256 <= sx1: ax.axvline(x + 256, color="cyan", linewidth=0.4, alpha=0.5, linestyle=":")
    for y in starts_aug:
        if sy0 <= y <= sy1: ax.axhline(y, color="cyan", linewidth=0.8, alpha=0.7, linestyle="--")
        if sy0 <= y + 256 <= sy1: ax.axhline(y + 256, color="cyan", linewidth=0.4, alpha=0.5, linestyle=":")
    ax.plot(cx_local, cy_local, "x", color="yellow", markersize=20, markeredgewidth=3, label="cell #6 centroid")
    ax.plot(ring_x_baseline, ring_y_baseline, "o", color="red", markersize=15,
            markerfacecolor="none", markeredgewidth=3, label=f"onion ring ({n_ring_baseline} labels in 25-px)")
    ax.set_xlim(sx0, sx1); ax.set_ylim(sy1, sy0)
    ax.set_title(f"BASELINE 4096-px crop, 500-px zoom around cell #6\n"
                 f"cyan dashed = tile starts, cyan dotted = tile ends "
                 f"(bsize=256, augment=True → ~50% overlap, {len(starts_aug)} tiles per axis)")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_baseline_zoom_with_tilegrid.png", dpi=140, bbox_inches="tight")
    plt.close()

    # === Figure 2: 4-panel side-by-side at the SAME RING WSI LOCATION ===
    half = RING_ZOOM_PX // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    panels = [
        ("baseline\n(augment=True)",          masks_baseline, s18_n,  starts_aug, bbox,   bx_loc, by_loc, n_at_baseline),
        ("augment=False\noverlap=0.1",        masks_lowover,  s18_n,  starts_lo,  bbox,   bx_loc, by_loc, n_at_lowover),
        ("augment=False\noverlap=0.5",        masks_highover, s18_n,  starts_hi,  bbox,   bx_loc, by_loc, n_at_highover),
        (f"shifted +{SHIFT_PX}px crop\n(augment=True)", masks_shifted, s18_ns, starts_aug, bbox_s, sx_loc, sy_loc, n_at_shifted),
    ]
    for ax, (title, masks, s18i, starts, var_bbox, lx, ly, n_at) in zip(axes, panels):
        zy0 = max(ly - half, 0); zy1 = min(ly + half, masks.shape[0])
        zx0 = max(lx - half, 0); zx1 = min(lx + half, masks.shape[1])
        zoom_s = s18i[zy0:zy1, zx0:zx1]
        zoom_m = masks[zy0:zy1, zx0:zx1]
        ax.imshow(zoom_s, cmap="gray", extent=[zx0, zx1, zy1, zy0])
        ax.imshow(label_overlay(zoom_m), extent=[zx0, zx1, zy1, zy0])
        for x in starts:
            if zx0 <= x <= zx1: ax.axvline(x, color="cyan", linewidth=0.7, alpha=0.7, linestyle="--")
            if zx0 <= x + 256 <= zx1: ax.axvline(x + 256, color="cyan", linewidth=0.4, alpha=0.5, linestyle=":")
        for y in starts:
            if zy0 <= y <= zy1: ax.axhline(y, color="cyan", linewidth=0.7, alpha=0.7, linestyle="--")
            if zy0 <= y + 256 <= zy1: ax.axhline(y + 256, color="cyan", linewidth=0.4, alpha=0.5, linestyle=":")
        # Mark the ring location centre
        ax.plot(lx, ly, "+", color="red", markersize=22, markeredgewidth=2.5)
        ax.set_xlim(zx0, zx1); ax.set_ylim(zy1, zy0)
        flag = "🍩 RING PRESENT" if n_at >= 4 else "✓ normal cell" if n_at <= 2 else "↘ partial"
        ax.set_title(f"{title}\nlabels at this WSI location: {n_at}  {flag}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Same physical {RING_ZOOM_PX}-px window in WSI px ({ring_wsi_x}, {ring_wsi_y}) across 4 variants\n"
                 f"(red + = onion-ring location identified in baseline; cyan = cellpose tile boundaries)",
                 y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_specific_ring_across_variants.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Save results JSON
    results = {
        "focal_cell_um":   [CX_UM, CY_UM],
        "baseline_bbox":   list(bbox),
        "shifted_bbox":    list(bbox_s),
        "shift_px":        SHIFT_PX,
        "baseline_ring_wsi_px": [int(ring_wsi_x), int(ring_wsi_y)],
        "baseline_ring_local_px": [int(ring_x_baseline), int(ring_y_baseline)],
        "n_distinct_labels_in_25px_window": int(n_ring_baseline),
        "labels_at_same_wsi_loc": {
            "baseline":               int(n_at_baseline),
            "augment=False_ov=0.1":   int(n_at_lowover),
            "augment=False_ov=0.5":   int(n_at_highover),
            "shifted":                int(n_at_shifted),
        },
        "n_total_cells": {
            "baseline":               int(masks_baseline.max()),
            "augment=False_ov=0.1":   int(masks_lowover.max()),
            "augment=False_ov=0.5":   int(masks_highover.max()),
            "shifted":                int(masks_shifted.max()),
        },
        "tile_grid_starts_first10": {
            "augment=True":           [int(x) for x in starts_aug[:10]],
            "augment=False_ov=0.1":   [int(x) for x in starts_lo[:10]],
            "augment=False_ov=0.5":   [int(x) for x in starts_hi[:10]],
        },
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved figures + results to {OUT_DIR}")
    print(f"total runtime: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
