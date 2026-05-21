"""With diameter=None (confirmed: no rescaling in cpsam) and the focal cell
placed in a single tile (tile_overlap=0.05, augment=False), the 1024 vs 2048
ROI sizes still give different n_focal for cell #6. Tile averaging ruled out.
What else? Test normalization next.

Hold constant:
  augment=False, diameter=None, tile_overlap=0.05  (cell alone in 1 tile)
  cellprob_threshold=-5.0, flow_threshold=0.0, niter=200

Vary:
  normalize ∈ {True (cellpose default local %iles),
               False + WSI %iles (fixed for both ROIs),
               False + 1024-ROI %iles (fixed for both ROIs — uses small-ROI's stats)}
  roi ∈ {1024, 2048}
  focal cell ∈ {cell #6}

3 normalization × 2 ROIs = 6 runs. Adds <5 min sharing the GPU.

If normalization explains the 1024 vs 2048 gap:
  - With normalize=True (per-image local), 1024 vs 2048 should give different answers (we see this in merge_vs_scale `no_aug` data: 2 vs 1)
  - With normalize=False + FIXED percentiles applied to BOTH ROIs, the two ROIs see identical intensities for the focal cell and surrounding region → if normalization is the only mediator, both ROIs should give the SAME answer.
  - With normalize=False + 1024-ROI-derived percentiles, we explicitly *transfer* the small-ROI stats to the 2048 ROI — if normalization is the cause, the 2048 result should now match the 1024 result.

If 1024 vs 2048 still differ under FIXED normalization → normalization is NOT the cause; something else (flow stitching of NEIGHBORING tiles, image-area-dependent network behavior, …) is responsible.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS= DATA / "cpsam_whole_slide" / "masks.tif"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"

OUT_DIR = ROOT / "figs" / "normalization_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
CX_PX, CY_PX = 41817, 6957
CPS_FOCAL = 45463
ROI_SIZES = [1024, 2048]
MIN_COVER_FRAC = 0.05
CPSAM_KW_BASE = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                     flow_threshold=0.0, augment=False, tile_overlap=0.05)


def crop_centered(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    """Matches cellpose's normalize99 EXACTLY: no clipping, just (x-q1)/(q99-q1)."""
    a = arr.astype(np.float32)
    return ((a - q_lo) / max(q_hi - q_lo, 1e-6)).astype(np.float32)


def percentiles_of(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return float(qlo), float(qhi)


def count_focal_masks(masks, focal_region, min_cover=MIN_COVER_FRAC):
    focal_area = int(focal_region.sum())
    if focal_area == 0: return 0, []
    lbls = np.unique(masks[focal_region]); lbls = lbls[lbls != 0]
    out = []
    for L in lbls:
        cov = int(((masks == L) & focal_region).sum()) / focal_area
        if cov >= min_cover:
            out.append((int(L), float(cov)))
    out.sort(key=lambda kv: -kv[1])
    return len(out), out


def main():
    t0 = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_masks = tifffile.imread(WSI_MASKS)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())

    # Pre-compute the 1024-ROI's local %iles (these will be transferred to 2048)
    dapi_1024_c, _ = crop_centered(dapi_full, CX_PX, CY_PX, 1024)
    s18_1024_c,  _ = crop_centered(s18_full,  CX_PX, CY_PX, 1024)
    pct_1024 = dict(
        DAPI=dict(q_lo=percentiles_of(dapi_1024_c)[0], q_hi=percentiles_of(dapi_1024_c)[1]),
        S18 =dict(q_lo=percentiles_of(s18_1024_c )[0], q_hi=percentiles_of(s18_1024_c )[1]),
    )
    print(f"  1024-ROI local %iles: DAPI={pct_1024['DAPI']}, 18S={pct_1024['S18']}")
    print(f"  WSI %iles:            DAPI={wsi_pct['DAPI']}, 18S={wsi_pct['18S']}")

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(dapi_in, s18_in, normalize):
        kw = dict(CPSAM_KW_BASE); kw["normalize"] = normalize
        img = np.stack([dapi_in, s18_in], axis=-1).astype(np.float32)
        masks, _, _ = m_sam.eval(img, channel_axis=-1, **kw)
        return masks.astype(np.int32)

    results = []
    for sz in ROI_SIZES:
        dapi_c, _ = crop_centered(dapi_full, CX_PX, CY_PX, sz)
        s18_c,  _ = crop_centered(s18_full,  CX_PX, CY_PX, sz)
        wsi_c,  _ = crop_centered(wsi_masks, CX_PX, CY_PX, sz)
        focal_region = (wsi_c == CPS_FOCAL)

        # 3 normalization variants
        for norm_name, dapi_in, s18_in, normalize_flag in [
            ("local_per_image",
                dapi_c.astype(np.float32),  # cellpose normalizes internally
                s18_c.astype(np.float32),
                True),
            ("fixed_WSI_pcts",
                norm_with(dapi_c, **wsi_pct["DAPI"]),
                norm_with(s18_c,  **wsi_pct["18S"]),
                False),
            ("fixed_1024_pcts",
                norm_with(dapi_c, **pct_1024["DAPI"]),
                norm_with(s18_c,  **pct_1024["S18"]),
                False),
        ]:
            t_r = time.time()
            masks = run(dapi_in, s18_in, normalize_flag)
            dt = time.time() - t_r
            n_focal, covers = count_focal_masks(masks, focal_region)
            rec = dict(
                roi_size_px=sz, normalization=norm_name,
                n_total=int(masks.max()), n_focal=n_focal,
                covers=";".join(f"{c:.3f}" for _, c in covers[:5]),
                runtime_s=round(dt, 1),
            )
            results.append(rec)
            pd.DataFrame(results).to_csv(OUT_DIR / "verdict.csv", index=False)
            print(f"  ROI={sz}  norm={norm_name:<18s}  n_total={masks.max():>5d}  "
                  f"n_focal={n_focal}  covers=[{rec['covers']}]  ({dt:.1f}s)")

    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
