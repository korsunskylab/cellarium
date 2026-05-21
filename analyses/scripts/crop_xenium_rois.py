"""Crop ROIs from the Xenium Prime 5K Skin dataset and save self-contained snapshots.

Each ROI directory will contain:
  morphology_DAPI.tif, morphology_ATP1A1_CD45_ECad.tif,
  morphology_18S.tif, morphology_aSMA_Vim.tif         — uint16 channel images
  cells_10x_masks.tif        — uint32 label image (10X's official cell masks)
  cells_10x_polygons.parquet — cell_idx, centroid, vertices (microns)
  transcripts.parquet        — x_um, y_um, z_um, gene_id, qv (qv≥20, valid)
  metadata.json              — bbox, pixel size, counts
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import zarr, tifffile

DATA = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/data/Xenium_Prime_Human_Skin_FFPE_xe_outs")
PIXEL_SIZE = 0.2125

ROIS = {
    "ROI1": dict(x0_um=960.0,  x1_um=1220.0, y0_um=2930.0, y1_um=3060.0),
    "ROI2": dict(x0_um=640.0,  x1_um=1040.0, y0_um=3370.0, y1_um=3580.0),
    # 2026-05-06 additions — surveyed candidates with low 10X assignment, morphologically distinct
    "ROI3": dict(x0_um=8250.0, x1_um=8500.0, y0_um=1000.0, y1_um=1250.0),  # C1: striated/fibrous, dense
    "ROI4": dict(x0_um=1500.0, x1_um=1750.0, y0_um=2250.0, y1_um=2500.0),  # C3: dense mixed cellular
    "ROI5": dict(x0_um=6750.0, x1_um=7000.0, y0_um=3000.0, y1_um=3250.0),  # C4: dense round (epithelial/tumor)
    "ROI6": dict(x0_um=1250.0, x1_um=1500.0, y0_um=3500.0, y1_um=3750.0),  # C6: striated, 58% assignment
    # 2026-05-13 — doublet-rich ROI from notebook 01 scan; picked for 15 Keratinocyte × Myeloid
    # doublets (epithelial × immune, the heterotypic mix pipelines/MVP0 needs).
    "ROI_new_1": dict(x0_um=750.0, x1_um=1000.0, y0_um=4000.0, y1_um=4250.0),
}

CHANNEL_FILES = [DATA / "morphology_focus" / f"morphology_focus_{i:04d}.ome.tif" for i in range(4)]
CHANNEL_NAMES = ["DAPI", "ATP1A1_CD45_ECad", "18S", "aSMA_Vim"]


def main():
    # Open zarr stores once; reuse across ROIs
    cells_store = zarr.ZipStore(DATA / "cells.zarr.zip", mode="r")
    cells_root  = zarr.open(cells_store, mode="r")
    cell_summary = cells_root["cell_summary"][:]
    poly_grp     = cells_root["polygon_sets"]["1"]      # cell boundaries (set 1)
    poly_verts   = poly_grp["vertices"][:]              # (N, 50)
    poly_nverts  = poly_grp["num_vertices"][:]
    poly_cellidx = poly_grp["cell_index"][:]
    cell_id_arr  = cells_root["cell_id"][:]
    print(f"opened cells.zarr: {len(cell_summary)} cells total")

    tx_store = zarr.ZipStore(DATA / "transcripts.zarr.zip", mode="r")
    tx_root  = zarr.open(tx_store, mode="r")
    grid0    = tx_root["grids"]["0"]
    TILE_SIZE = 250.0

    # Load full morphology pages once (big — ~1.7 GB each), then crop both ROIs from in-memory
    print("loading 4 morphology pages (~6.8 GB total)...")
    full_pages = []
    for i, p in enumerate(CHANNEL_FILES):
        t0 = time.time()
        full_pages.append(tifffile.imread(p, key=0))
        print(f"  {p.name}: {time.time()-t0:.1f}s")

    for roi_name, bbox in ROIS.items():
        print(f"\n=== {roi_name} ===  x[{bbox['x0_um']}, {bbox['x1_um']}] y[{bbox['y0_um']}, {bbox['y1_um']}]")
        out_dir = DATA / roi_name
        # Skip if already cropped (all four artifacts present)
        required = ["morphology_DAPI.tif", "morphology_18S.tif",
                    "cells_10x_masks.tif", "transcripts.parquet", "metadata.json"]
        if out_dir.exists() and all((out_dir / f).exists() for f in required):
            print(f"  → already cropped, skipping")
            continue
        out_dir.mkdir(exist_ok=True)

        x0_px = int(round(bbox["x0_um"] / PIXEL_SIZE))
        x1_px = int(round(bbox["x1_um"] / PIXEL_SIZE))
        y0_px = int(round(bbox["y0_um"] / PIXEL_SIZE))
        y1_px = int(round(bbox["y1_um"] / PIXEL_SIZE))

        # 1. Morphology channels
        for i, name in enumerate(CHANNEL_NAMES):
            crop = full_pages[i][y0_px:y1_px, x0_px:x1_px]
            tifffile.imwrite(out_dir / f"morphology_{name}.tif", crop, compression="zlib")
            print(f"  morphology_{name}.tif: shape={crop.shape} range=[{crop.min()}, {crop.max()}]")

        # 2. 10X cell mask (label image at the same 0.2125 µm/px)
        masks_10x = cells_root["masks"]["1"][y0_px:y1_px, x0_px:x1_px].astype(np.uint32)
        tifffile.imwrite(out_dir / "cells_10x_masks.tif", masks_10x, compression="zlib")
        n_in_mask = int((masks_10x > 0).sum() and (len(np.unique(masks_10x)) - 1))
        print(f"  cells_10x_masks.tif: {n_in_mask} cells in mask")

        # 3. 10X cell polygons (filter by centroid in bbox)
        cx = cell_summary[:, 0]; cy = cell_summary[:, 1]
        in_bbox = ((cx >= bbox["x0_um"]) & (cx <= bbox["x1_um"]) &
                   (cy >= bbox["y0_um"]) & (cy <= bbox["y1_um"]))
        cell_idx_in_bbox = set(int(v) for v in np.where(in_bbox)[0])
        poly_records = []
        for row, cell_id in enumerate(poly_cellidx):
            ci = int(cell_id)
            if ci not in cell_idx_in_bbox:
                continue
            nv = int(poly_nverts[row])
            xy = poly_verts[row, : nv * 2].reshape(nv, 2)
            poly_records.append({
                "cell_idx": ci,
                "x_centroid_um": float(cx[ci]),
                "y_centroid_um": float(cy[ci]),
                "x_vertices_um": xy[:, 0].astype(np.float32).tolist(),
                "y_vertices_um": xy[:, 1].astype(np.float32).tolist(),
            })
        df_poly = pd.DataFrame(poly_records)
        df_poly.to_parquet(out_dir / "cells_10x_polygons.parquet")
        print(f"  cells_10x_polygons.parquet: {len(df_poly)} cells")

        # 4. Transcripts (overlapping tiles → exact filter)
        i0 = int(bbox["x0_um"] // TILE_SIZE); i1 = int(bbox["x1_um"] // TILE_SIZE)
        j0 = int(bbox["y0_um"] // TILE_SIZE); j1 = int(bbox["y1_um"] // TILE_SIZE)
        all_loc, all_gid, all_qv, all_valid = [], [], [], []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                key = f"{i},{j}"
                if key not in grid0:
                    continue
                tile = grid0[key]
                all_loc.append(tile["location"][:])
                all_gid.append(tile["gene_identity"][:].squeeze(-1))
                all_qv .append(tile["quality_score"][:].squeeze(-1))
                all_valid.append(tile["valid"][:].squeeze(-1))
        loc   = np.concatenate(all_loc,   axis=0)
        gid   = np.concatenate(all_gid,   axis=0)
        qv    = np.concatenate(all_qv,    axis=0)
        valid = np.concatenate(all_valid, axis=0)
        keep = ((loc[:, 0] >= bbox["x0_um"]) & (loc[:, 0] <= bbox["x1_um"]) &
                (loc[:, 1] >= bbox["y0_um"]) & (loc[:, 1] <= bbox["y1_um"]) &
                (valid > 0) & (qv >= 20))
        df_tx = pd.DataFrame({
            "x_um":  loc[keep, 0].astype(np.float32),
            "y_um":  loc[keep, 1].astype(np.float32),
            "z_um":  loc[keep, 2].astype(np.float32),
            "gene_id": gid[keep].astype(np.uint16),
            "qv":      qv[keep].astype(np.float32),
        })
        df_tx.to_parquet(out_dir / "transcripts.parquet")
        print(f"  transcripts.parquet: {len(df_tx)} transcripts (qv≥20)")

        # 5. Metadata
        meta = {
            "roi_name": roi_name,
            "bbox_um": bbox,
            "pixel_size_um": PIXEL_SIZE,
            "pixel_bbox": {"x0_px": x0_px, "x1_px": x1_px, "y0_px": y0_px, "y1_px": y1_px},
            "shape_yx": [y1_px - y0_px, x1_px - x0_px],
            "channels": CHANNEL_NAMES,
            "n_cells_in_mask": n_in_mask,
            "n_cells_polygon_in_bbox": len(df_poly),
            "n_transcripts_qv20": len(df_tx),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  metadata.json")


if __name__ == "__main__":
    main()
