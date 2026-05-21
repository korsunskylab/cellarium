"""Run cellpose at 1024 and 2048 with controlled tile_overlap=0.05 (cell #6
in single tile). Save the predicted flow field (dY, dX) and cellprob. Then
compare them pixel-for-pixel in a window around the focal cell.

If the predicted flows at cell #6 are pixel-identical between 1024 and 2048,
the per-tile network prediction is unaffected by ROI size (as expected for a
ViT with no global state) and the 3-vs-2 n_focal gap must come from
downstream stuff: dynamics, mask grouping, or post-processing.

If the flows DIFFER at cell #6, then something in the per-tile-prediction +
stitching pipeline IS image-size-dependent (and we'd need to look deeper).

Saves arrays to figs/compare_flows/, plus a tiny figure that overlays the
two flow fields side-by-side and reports the max & mean abs difference.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS= DATA / "cpsam_whole_slide" / "masks.tif"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"

OUT_DIR = ROOT / "figs" / "compare_flows"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
CX_PX, CY_PX = 41817, 6957
COMPARE_HALF = 50    # 100-px window for pixel-wise comparison
ROI_SIZES = [1024, 2048]
CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=False, normalize=False,
                tile_overlap=0.05)


def crop_centered(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    """No clipping — matches cellpose normalize99 behavior."""
    a = arr.astype(np.float32)
    return ((a - q_lo) / max(q_hi - q_lo, 1e-6)).astype(np.float32)


def main():
    t0 = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    saved = {}
    for sz in ROI_SIZES:
        print(f"\n=== ROI = {sz} ===")
        dapi_c, _ = crop_centered(dapi_full, CX_PX, CY_PX, sz)
        s18_c,  _ = crop_centered(s18_full,  CX_PX, CY_PX, sz)
        dapi_n = norm_with(dapi_c, **wsi_pct["DAPI"])
        s18_n  = norm_with(s18_c,  **wsi_pct["18S"])
        img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
        t_r = time.time()
        masks, flows, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        dt = time.time() - t_r
        # flows: [rgb_flows, dP_flows, cellprob, ...]
        # flows[1] is shape (2, Ly, Lx) → (dY, dX); flows[2] is cellprob
        dP = flows[1].astype(np.float32)
        cellprob = flows[2].astype(np.float32)
        print(f"  n_total={int(masks.max())}, runtime={dt:.1f}s")
        print(f"  dP shape: {dP.shape}, cellprob shape: {cellprob.shape}")
        # Crop to focal window
        cy_loc = sz // 2; cx_loc = sz // 2
        h = COMPARE_HALF
        dP_focal = dP[:, cy_loc-h:cy_loc+h, cx_loc-h:cx_loc+h]
        cellprob_focal = cellprob[cy_loc-h:cy_loc+h, cx_loc-h:cx_loc+h]
        s18_focal = s18_n[cy_loc-h:cy_loc+h, cx_loc-h:cx_loc+h]
        saved[sz] = dict(dP_focal=dP_focal, cellprob_focal=cellprob_focal,
                          s18_focal=s18_focal,
                          masks_focal=masks[cy_loc-h:cy_loc+h, cx_loc-h:cx_loc+h])
        np.savez_compressed(OUT_DIR / f"flows_{sz}.npz",
                             dP=dP, cellprob=cellprob, masks=masks)
        print(f"  saved flows_{sz}.npz ({(dP.nbytes + cellprob.nbytes)/1e6:.0f} MB raw)")

    # --- pixel-wise comparison in the focal window ---
    dP_diff = saved[2048]["dP_focal"] - saved[1024]["dP_focal"]
    cellprob_diff = saved[2048]["cellprob_focal"] - saved[1024]["cellprob_focal"]
    dY_max_abs = float(np.abs(dP_diff[0]).max())
    dX_max_abs = float(np.abs(dP_diff[1]).max())
    cellprob_max_abs = float(np.abs(cellprob_diff).max())
    dY_mean_abs = float(np.abs(dP_diff[0]).mean())
    dX_mean_abs = float(np.abs(dP_diff[1]).mean())
    cellprob_mean_abs = float(np.abs(cellprob_diff).mean())

    print(f"\n=== pixel-wise comparison at focal cell ({2*COMPARE_HALF}×{2*COMPARE_HALF} window) ===")
    print(f"  dY (flow Y): max |diff| = {dY_max_abs:.4f}, mean |diff| = {dY_mean_abs:.5f}")
    print(f"  dX (flow X): max |diff| = {dX_max_abs:.4f}, mean |diff| = {dX_mean_abs:.5f}")
    print(f"  cellprob:    max |diff| = {cellprob_max_abs:.4f}, mean |diff| = {cellprob_mean_abs:.5f}")
    print(f"  cellprob range (1024): [{saved[1024]['cellprob_focal'].min():.2f}, {saved[1024]['cellprob_focal'].max():.2f}]")
    print(f"  cellprob range (2048): [{saved[2048]['cellprob_focal'].min():.2f}, {saved[2048]['cellprob_focal'].max():.2f}]")
    print(f"  dY range (1024): [{saved[1024]['dP_focal'][0].min():.2f}, {saved[1024]['dP_focal'][0].max():.2f}]")
    print(f"  dY range (2048): [{saved[2048]['dP_focal'][0].min():.2f}, {saved[2048]['dP_focal'][0].max():.2f}]")

    # --- figure: 3 rows × 4 cols ---
    # rows: 18S+masks / dY / dX / cellprob ; cols: 1024, 2048, abs-diff, signed-diff
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    # Row 0: 18S background with masks contours
    for i, sz in enumerate([1024, 2048]):
        axes[0, i].imshow(saved[sz]["s18_focal"], cmap="gray")
        axes[0, i].contour(saved[sz]["masks_focal"].astype(int), colors="white", linewidths=0.5)
        axes[0, i].set_title(f"ROI {sz}: 18S + mask contours")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    # Row 1: dY
    for i, sz in enumerate([1024, 2048]):
        v = max(abs(saved[sz]["dP_focal"][0]).max(), 1e-3)
        axes[1, i].imshow(saved[sz]["dP_focal"][0], cmap="RdBu_r", vmin=-v, vmax=v)
        axes[1, i].set_title(f"ROI {sz}: dY")
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    axes[1, 2].imshow(np.abs(dP_diff[0]), cmap="hot")
    axes[1, 2].set_title(f"|ΔdY|  (max={dY_max_abs:.3f}, mean={dY_mean_abs:.4f})")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    v = max(abs(dP_diff[0]).max(), 1e-3)
    axes[1, 3].imshow(dP_diff[0], cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1, 3].set_title("ΔdY (signed)")
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])
    # Row 2: dX
    for i, sz in enumerate([1024, 2048]):
        v = max(abs(saved[sz]["dP_focal"][1]).max(), 1e-3)
        axes[2, i].imshow(saved[sz]["dP_focal"][1], cmap="RdBu_r", vmin=-v, vmax=v)
        axes[2, i].set_title(f"ROI {sz}: dX")
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])
    axes[2, 2].imshow(np.abs(dP_diff[1]), cmap="hot")
    axes[2, 2].set_title(f"|ΔdX|  (max={dX_max_abs:.3f}, mean={dX_mean_abs:.4f})")
    axes[2, 2].set_xticks([]); axes[2, 2].set_yticks([])
    v = max(abs(dP_diff[1]).max(), 1e-3)
    axes[2, 3].imshow(dP_diff[1], cmap="RdBu_r", vmin=-v, vmax=v)
    axes[2, 3].set_title("ΔdX (signed)")
    axes[2, 3].set_xticks([]); axes[2, 3].set_yticks([])
    # Row 3: cellprob
    for i, sz in enumerate([1024, 2048]):
        axes[3, i].imshow(saved[sz]["cellprob_focal"], cmap="viridis")
        axes[3, i].set_title(f"ROI {sz}: cellprob")
        axes[3, i].set_xticks([]); axes[3, i].set_yticks([])
    axes[3, 2].imshow(np.abs(cellprob_diff), cmap="hot")
    axes[3, 2].set_title(f"|Δcellprob|  (max={cellprob_max_abs:.3f}, mean={cellprob_mean_abs:.4f})")
    axes[3, 2].set_xticks([]); axes[3, 2].set_yticks([])
    v = max(abs(cellprob_diff).max(), 1e-3)
    axes[3, 3].imshow(cellprob_diff, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[3, 3].set_title("Δcellprob (signed)")
    axes[3, 3].set_xticks([]); axes[3, 3].set_yticks([])

    plt.suptitle(f"Cell #6 ({2*COMPARE_HALF}×{2*COMPARE_HALF} px window) — "
                 f"cellpose-SAM with augment=False, tile_overlap=0.05, "
                 f"diameter=None, fixed WSI-percentile normalization\n"
                 f"(per-tile network prediction at cell #6 should be identical if image size doesn't matter; "
                 f"any non-zero diff means image-size affects per-tile output)",
                 fontsize=10, y=1.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "compare_flows.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {OUT_DIR/'compare_flows.png'}")
    print(f"total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
