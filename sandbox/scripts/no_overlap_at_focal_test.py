"""Test the tile-averaging hypothesis directly.

User's prediction: if we place the focal cell in the CENTER of a single
cellpose tile (no overlap zone touching the cell), then 1024 and 2048 ROI
sizes should give the SAME segmentation answer for that cell.

Setup:
  - augment=False (so cellpose actually respects tile_overlap; with augment=True
    it forces ~50% overlap regardless)
  - tile_overlap=0.05 (minimum allowed by cellpose's clamping; this puts the
    cell at the exact center of a single tile with no overlap-zone coverage at
    cell #6)
  - Compare to the existing tile_overlap=0.1 baseline where the cell DOES fall
    in an overlap zone at 2048 (sometimes; depending on size).

Two focal cells:
  A. cell #6 at WSI px (41817, 6957) — small (~85 px), focal merge case
  B. onion-ring location at WSI px (41680, 6996) — onion-ring case

For each, sweep tile_overlap ∈ {0.05, 0.1, 0.2, 0.5} at ROI ∈ {1024, 2048}
with augment=False, normalize=False + WSI percentiles, niter=200,
cellprob_threshold=-5, flow_threshold=0.

Outputs:
  figs/no_overlap_test/verdict.csv
  figs/no_overlap_test/runlog.txt  (when run with > redirect)
  figs/no_overlap_test/tile_geometry.txt
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

OUT_DIR = ROOT / "figs" / "no_overlap_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
FOCAL = {
    "cell6": dict(wsi_x=41817, wsi_y=6957, cps_id=45463, name="cell #6 (merge case)"),
    "ring":  dict(wsi_x=41680, wsi_y=6996, cps_id=None,  name="onion-ring location"),
}
ROI_SIZES = [1024, 2048]
TILE_OVERLAPS = [0.05, 0.1, 0.2, 0.5]
MIN_COVER_FRAC = 0.05
RING_WIN_HALF = 20    # for ring location (no WSI mask), use 40-px window
CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=False, normalize=False)


def crop_centered(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    a = arr.astype(np.float32)
    return np.clip((a - q_lo) / max(q_hi - q_lo, 1e-6), 0, 1).astype(np.float32)


def tile_starts(L, bsize=256, tile_overlap=0.1, augment=False):
    """Replicates cellpose's `make_tiles` for the augment=False branch."""
    if augment:
        ny = max(2, int(np.ceil(2. * L / bsize)))
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        ny = 1 if L <= bsize else int(np.ceil((1. + 2 * tile_overlap) * L / bsize))
    return np.linspace(0, max(L - bsize, 0), ny).astype(int), ny


def cell_tile_membership(focal_local_px, L, bsize, tile_overlap, augment):
    """Which tiles contain focal_local_px? Is it in an overlap region?"""
    starts, ny = tile_starts(L, bsize, tile_overlap, augment)
    tiles = [(s, s + bsize) for s in starts]
    containing = [i for i, (lo, hi) in enumerate(tiles) if lo <= focal_local_px < hi]
    in_overlap = len(containing) > 1
    # Position of cell within each containing tile
    positions = [focal_local_px - tiles[i][0] for i in containing]
    return dict(starts=starts.tolist(), tiles=tiles, containing=containing,
                in_overlap=in_overlap, positions=positions)


def count_focal_masks_wsi(masks, focal_region, min_cover=MIN_COVER_FRAC):
    """When we have the WSI mask: count CP-SAM masks covering ≥5% of the WSI region."""
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


def count_focal_masks_at_pixel(masks, y, x, half=RING_WIN_HALF):
    """When we don't have a WSI mask: count distinct masks in a window."""
    H, W = masks.shape
    y0 = max(y - half, 0); y1 = min(y + half, H)
    x0 = max(x - half, 0); x1 = min(x + half, W)
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0), []


def main():
    t0 = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_masks = tifffile.imread(WSI_MASKS)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())

    # ---- tile geometry sanity check (print to txt) ----
    geom = []
    for fk, f in FOCAL.items():
        for sz in ROI_SIZES:
            focal_local = sz // 2
            for ov in TILE_OVERLAPS:
                info = cell_tile_membership(focal_local, sz, 256, ov, augment=False)
                geom.append({
                    "focal": fk, "roi": sz, "tile_overlap": ov,
                    "focal_local_px": focal_local,
                    "n_tiles_per_axis": len(info["starts"]),
                    "tiles_containing": info["containing"],
                    "in_overlap": info["in_overlap"],
                    "positions_in_tile": info["positions"],
                })
    geom_df = pd.DataFrame(geom)
    (OUT_DIR / "tile_geometry.txt").write_text(geom_df.to_string(index=False))
    print("\n=== tile geometry ===")
    print(geom_df.to_string(index=False))

    print("\nloading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(dapi_in, s18_in, **kw):
        k = dict(CPSAM_KW); k.update(kw)
        img = np.stack([dapi_in, s18_in], axis=-1).astype(np.float32)
        masks, _, _ = m_sam.eval(img, channel_axis=-1, **k)
        return masks.astype(np.int32)

    results = []
    for fk, f in FOCAL.items():
        cx_px, cy_px = f["wsi_x"], f["wsi_y"]
        for sz in ROI_SIZES:
            dapi_c, bbox = crop_centered(dapi_full, cx_px, cy_px, sz)
            s18_c, _ = crop_centered(s18_full, cx_px, cy_px, sz)
            wsi_c, _ = crop_centered(wsi_masks, cx_px, cy_px, sz)
            dapi_n = norm_with(dapi_c, **wsi_pct["DAPI"])
            s18_n  = norm_with(s18_c,  **wsi_pct["18S"])
            focal_local_x = cx_px - bbox[0]
            focal_local_y = cy_px - bbox[1]
            focal_region = (wsi_c == f["cps_id"]) if f["cps_id"] is not None else None
            for ov in TILE_OVERLAPS:
                t_run = time.time()
                masks = run(dapi_n, s18_n, tile_overlap=ov)
                dt = time.time() - t_run
                if focal_region is not None:
                    n_focal, covers = count_focal_masks_wsi(masks, focal_region)
                else:
                    n_focal, covers = count_focal_masks_at_pixel(
                        masks, focal_local_y, focal_local_x)
                info_x = cell_tile_membership(focal_local_x, sz, 256, ov, augment=False)
                rec = dict(
                    focal=fk, roi_size_px=sz, tile_overlap=ov,
                    n_total=int(masks.max()), n_focal=n_focal,
                    in_tile_overlap_x=info_x["in_overlap"],
                    n_tiles_per_axis=len(info_x["starts"]),
                    runtime_s=round(dt, 1),
                )
                results.append(rec)
                print(f"  {fk:8s}  ROI={sz}  ov={ov:.2f}  "
                      f"in_overlap={info_x['in_overlap']}  "
                      f"n_tiles_axis={len(info_x['starts'])}  "
                      f"n_focal={n_focal}  n_total={masks.max()}  ({dt:.1f}s)")
                pd.DataFrame(results).to_csv(OUT_DIR / "verdict.csv", index=False)

    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
