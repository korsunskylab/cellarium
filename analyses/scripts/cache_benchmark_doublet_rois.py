"""Materialize each approved benchmark doublet as a self-contained mini-ROI.

For every cell in `benchmark_doublets.parquet` with `approved=True`, write a
directory at `data/.../benchmark_doublet_rois/<benchmark_id>/` containing the
same artifact set as the existing top-level ROIs (ROI1–ROI_new_1), plus the
CP-SAM mask crop from the new whole-slide segmentation:

    morphology_DAPI.tif, morphology_18S.tif,
    morphology_ATP1A1_CD45_ECad.tif, morphology_aSMA_Vim.tif
    cells_cpsam_masks.tif        — uint32, cropped from cpsam_whole_slide/masks.tif
    cells_10x_masks.tif          — uint32, cropped from cells.zarr["masks"]["1"]
    cells_10x_polygons.parquet   — 10X polygons with centroid in bbox
    transcripts.parquet          — qv≥20 transcripts inside bbox
    metadata.json                — bbox (px+um), benchmark annotation, source cps_id

Identity in metadata is preserved both spatially (x_um, y_um, bbox_*_um) and
by snapshot (cps_id_at_bookmark + cpsam_version) so the ROI survives any
future re-segmentation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"

BENCHMARK_PARQUET = DATA / "benchmark_doublets.parquet"
CPSAM_MASKS       = DATA / "cpsam_whole_slide" / "masks.tif"
CELLS_ZARR        = DATA / "cells.zarr.zip"
TX_ZARR           = DATA / "transcripts.zarr.zip"
MORPHO_DIR        = DATA / "morphology_focus"
OUT_DIR           = DATA / "benchmark_doublet_rois"

CHANNEL_FILES = [MORPHO_DIR / f"morphology_focus_{i:04d}.ome.tif" for i in range(4)]
CHANNEL_NAMES = ["DAPI", "ATP1A1_CD45_ECad", "18S", "aSMA_Vim"]

PIXEL_SIZE_UM = 0.2125
PAD_UM        = 20.0
TILE_SIZE_UM  = 250.0
QV_MIN        = 20.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading benchmark parquet: {BENCHMARK_PARQUET}")
    bm = pd.read_parquet(BENCHMARK_PARQUET)
    approved = bm[bm["approved"]].reset_index(drop=True)
    log(f"  {len(approved)} approved cells (of {len(bm)} ranked)")

    # ── Eager-load morphology channels (4 × 1.7 GB = 6.8 GB on 128 GB box) ─
    log("loading morphology channels into memory...")
    morpho_pages = []
    for p in CHANNEL_FILES:
        t0 = time.time()
        morpho_pages.append(tifffile.imread(p, key=0))
        log(f"  {p.name}: {time.time()-t0:.1f}s")
    H, W = morpho_pages[0].shape
    log(f"  slide: {H} × {W} px")

    log(f"loading CP-SAM masks (3.5 GB)...")
    t0 = time.time()
    cpsam_masks = tifffile.imread(CPSAM_MASKS)
    log(f"  done in {time.time()-t0:.1f}s  shape={cpsam_masks.shape} dtype={cpsam_masks.dtype}")

    log("opening cells.zarr (10X masks + polygons)...")
    cells_root  = zarr.open(zarr.ZipStore(CELLS_ZARR, mode="r"), mode="r")
    masks_10x   = cells_root["masks"]["1"]    # lazy
    cell_summary = cells_root["cell_summary"][:]   # (N_cells, ≥2) — x_um, y_um, ...
    poly_grp    = cells_root["polygon_sets"]["1"]
    poly_verts  = poly_grp["vertices"][:]
    poly_nverts = poly_grp["num_vertices"][:]
    poly_cellidx = poly_grp["cell_index"][:]
    log(f"  cells.zarr loaded: {len(cell_summary)} 10X cells")

    log("opening transcripts.zarr (lazy)...")
    tx_root = zarr.open(zarr.ZipStore(TX_ZARR, mode="r"), mode="r")
    grid0   = tx_root["grids"]["0"]
    gene_names = list(tx_root.attrs["gene_names"])
    log(f"  panel: {len(gene_names)} genes")

    # ── Per-cell: crop + write ────────────────────────────────────────────
    pad_px = int(round(PAD_UM / PIXEL_SIZE_UM))
    log(f"writing {len(approved)} mini-ROIs (pad={PAD_UM} µm = {pad_px} px)...")
    for i, row in approved.iterrows():
        bid = row["benchmark_id"]
        out = OUT_DIR / bid
        out.mkdir(parents=True, exist_ok=True)

        x0_um = float(row["bbox_x0_um"]) - PAD_UM
        x1_um = float(row["bbox_x1_um"]) + PAD_UM
        y0_um = float(row["bbox_y0_um"]) - PAD_UM
        y1_um = float(row["bbox_y1_um"]) + PAD_UM
        x0_px = max(int(round(x0_um / PIXEL_SIZE_UM)), 0)
        x1_px = min(int(round(x1_um / PIXEL_SIZE_UM)), W)
        y0_px = max(int(round(y0_um / PIXEL_SIZE_UM)), 0)
        y1_px = min(int(round(y1_um / PIXEL_SIZE_UM)), H)
        # Snap back to exact µm of the crop window after clamping
        x0_um = x0_px * PIXEL_SIZE_UM; x1_um = x1_px * PIXEL_SIZE_UM
        y0_um = y0_px * PIXEL_SIZE_UM; y1_um = y1_px * PIXEL_SIZE_UM

        # 1) Morphology channels
        for zi, name in enumerate(CHANNEL_NAMES):
            crop = morpho_pages[zi][y0_px:y1_px, x0_px:x1_px]
            tifffile.imwrite(out / f"morphology_{name}.tif", crop, compression="zlib")

        # 2) CP-SAM mask
        cpsam_crop = cpsam_masks[y0_px:y1_px, x0_px:x1_px]
        tifffile.imwrite(out / "cells_cpsam_masks.tif",
                         cpsam_crop.astype(np.uint32), compression="zlib")
        n_cpsam_in_crop = len(np.unique(cpsam_crop)) - (1 if 0 in cpsam_crop else 0)

        # 3) 10X mask
        m10x_crop = np.asarray(masks_10x[y0_px:y1_px, x0_px:x1_px]).astype(np.uint32)
        tifffile.imwrite(out / "cells_10x_masks.tif", m10x_crop, compression="zlib")
        n_10x_in_crop = len(np.unique(m10x_crop)) - (1 if 0 in m10x_crop else 0)

        # 4) 10X polygons whose centroid lies in bbox
        cx = cell_summary[:, 0]; cy = cell_summary[:, 1]
        in_bbox = ((cx >= x0_um) & (cx <= x1_um) &
                   (cy >= y0_um) & (cy <= y1_um))
        cell_idx_set = set(int(v) for v in np.where(in_bbox)[0])
        recs = []
        for k, cell_id in enumerate(poly_cellidx):
            ci = int(cell_id)
            if ci not in cell_idx_set: continue
            nv = int(poly_nverts[k])
            xy = poly_verts[k, : nv * 2].reshape(nv, 2)
            recs.append({
                "cell_idx":      ci,
                "x_centroid_um": float(cx[ci]),
                "y_centroid_um": float(cy[ci]),
                "x_vertices_um": xy[:, 0].astype(np.float32).tolist(),
                "y_vertices_um": xy[:, 1].astype(np.float32).tolist(),
            })
        pd.DataFrame(recs).to_parquet(out / "cells_10x_polygons.parquet")

        # 5) Transcripts: scan overlapping tiles, filter by bbox + qv + valid
        i0 = int(x0_um // TILE_SIZE_UM); i1 = int(x1_um // TILE_SIZE_UM)
        j0 = int(y0_um // TILE_SIZE_UM); j1 = int(y1_um // TILE_SIZE_UM)
        locs, gids, qvs, valids = [], [], [], []
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                key = f"{ii},{jj}"
                if key not in grid0: continue
                tile = grid0[key]
                locs  .append(tile["location"][:])
                gids  .append(tile["gene_identity"][:].squeeze(-1))
                qvs   .append(tile["quality_score"][:].squeeze(-1))
                valids.append(tile["valid"][:].squeeze(-1))
        if locs:
            loc = np.concatenate(locs, axis=0)
            gid = np.concatenate(gids, axis=0)
            qv  = np.concatenate(qvs,  axis=0)
            valid = np.concatenate(valids, axis=0)
            keep = ((loc[:, 0] >= x0_um) & (loc[:, 0] <= x1_um) &
                    (loc[:, 1] >= y0_um) & (loc[:, 1] <= y1_um) &
                    (valid > 0) & (qv >= QV_MIN))
            df_tx = pd.DataFrame({
                "x_um":   loc[keep, 0].astype(np.float32),
                "y_um":   loc[keep, 1].astype(np.float32),
                "z_um":   loc[keep, 2].astype(np.float32),
                "gene_id": gid[keep].astype(np.uint16),
                "qv":      qv[keep].astype(np.float32),
            })
        else:
            df_tx = pd.DataFrame(columns=["x_um", "y_um", "z_um", "gene_id", "qv"])
        df_tx.to_parquet(out / "transcripts.parquet")

        # 6) Metadata
        meta = {
            "benchmark_id":       bid,
            "rank":               int(row["rank"]),
            "cps_id_at_bookmark": int(row["cps_id_at_bookmark"]),
            "cpsam_version":      str(row["cpsam_version"]),
            "is_triplet":         bool(row["is_triplet"]),
            "lineage_pair":       str(row["lineage_pair"]),
            "top1_lin":           str(row["top1_lin"]),
            "top1_n":             int(row["top1_n"]),
            "top2_lin":           str(row["top2_lin"]),
            "top2_n":             int(row["top2_n"]),
            "n_anchors":          int(row["n_anchors"]),
            "centroid_um":        {"x": float(row["x_um"]), "y": float(row["y_um"])},
            "bbox_cell_um":       {"x0": float(row["bbox_x0_um"]),
                                   "x1": float(row["bbox_x1_um"]),
                                   "y0": float(row["bbox_y0_um"]),
                                   "y1": float(row["bbox_y1_um"])},
            "bbox_crop_um":       {"x0": x0_um, "x1": x1_um, "y0": y0_um, "y1": y1_um},
            "bbox_crop_px":       {"x0": x0_px, "x1": x1_px, "y0": y0_px, "y1": y1_px},
            "pad_um":             PAD_UM,
            "pixel_size_um":      PIXEL_SIZE_UM,
            "shape_yx":           [y1_px - y0_px, x1_px - x0_px],
            "channels":           CHANNEL_NAMES,
            "n_transcripts_qv20": int(len(df_tx)),
            "n_cpsam_cells_in_crop": int(n_cpsam_in_crop),
            "n_10x_cells_in_crop":   int(n_10x_in_crop),
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if (i + 1) % 10 == 0:
            log(f"  wrote {i+1}/{len(approved)}: {bid}  "
                f"({y1_px-y0_px}×{x1_px-x0_px} px, {len(df_tx)} tx)")

    log(f"done in {(time.time()-t_start):.1f}s. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
