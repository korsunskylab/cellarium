"""Minimal cellpose-SAM reproducer for the GitHub issue.

Loads the same focal cell at two ROI sizes (1024×1024 and 4096×4096) from
identical WSI pixels and runs cellpose-SAM at TRUE defaults. Captures a verbose
run log via `cellpose.io.logger_setup()` for the issue body.

Usage:
    python scripts/cellpose_repro_minimal.py > reports/github_issue_figs/runlog_verbose.txt 2>&1
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile
from cellpose import models, io

# Anonymisable inputs — pointers, not the data itself
ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"

# Same focal cell (pixel coordinates in the WSI). The 1024 crop is fully inside
# the 4096 crop, so the cell and its ~110 µm of context are identical pixels
# in both.
CX_PX, CY_PX = 41817, 6957


def crop_centered(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1]


def main():
    logger = io.logger_setup()

    print("loading 2-channel morphology image…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    print(f"  WSI shape: {dapi_full.shape}")

    print("loading cellpose-SAM…")
    m = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    for size_px in (1024, 4096):
        print(f"\n=========  ROI = {size_px} × {size_px} px  =========")
        dapi_c = crop_centered(dapi_full, CX_PX, CY_PX, size_px)
        s18_c  = crop_centered(s18_full,  CX_PX, CY_PX, size_px)
        img = np.stack([dapi_c.astype(np.float32),
                        s18_c.astype(np.float32)], axis=-1)
        print(f"  img.shape = {img.shape}, dtype = {img.dtype}")
        print(f"  calling model.eval(img, channel_axis=-1)  with all other "
              f"params at default")
        masks, _, _ = m.eval(img, channel_axis=-1)
        print(f"  n_total masks: {int(masks.max())}")
        # Count masks at the focal cell location (centre of the crop)
        cy_loc = masks.shape[0] // 2
        cx_loc = masks.shape[1] // 2
        H = W = masks.shape[0]
        y0 = max(cy_loc - 25, 0); y1 = min(cy_loc + 25, H)
        x0 = max(cx_loc - 25, 0); x1 = min(cx_loc + 25, W)
        patch = masks[y0:y1, x0:x1]
        n_at = len(np.unique(patch)) - (1 if 0 in patch else 0)
        print(f"  # distinct masks in 50-px window around focal cell: {n_at}")


if __name__ == "__main__":
    main()
