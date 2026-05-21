"""Save the manually-approved doublet/triplet cells as a benchmark cache.

Identity is stored as **centroid + bbox in microns** so the benchmark survives
re-segmentation (cps_id will change if CP-SAM is re-run; spatial coordinates are
stable). The current cps_id is preserved as a convenience column, tagged with
the cpsam-run version.

Outputs:
    data/.../benchmark_doublets.parquet      — labelled positives + unapproved
    figs/benchmark_doublets/                  — copies of the approved-cell PNGs
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import center_of_mass, find_objects

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TOP100      = DATA / "cells_doublet_composition_whole_slide_top100.parquet"
MASKS_TIF   = DATA / "cpsam_whole_slide" / "masks.tif"
OUT_PARQUET = DATA / "benchmark_doublets.parquet"
SRC_FIGS    = ROOT / "figs" / "whole_slide_doublets"
DST_FIGS    = ROOT / "figs" / "benchmark_doublets"

CPSAM_VERSION = "cpsam_whole_slide_2026-05-14"
APPROVED_DATE = "2026-05-14"
PIXEL_SIZE_UM = 0.2125

# Approval list from the 2026-05-14 review of top-100 (one-indexed ranks).
APPROVED_RANKS = [
    3, 5, 6, 7, 9, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 50, 53, 54, 55, 56, 57, 58, 64, 66, 67, 68, 69, 70, 71,
    73, 74, 75, 77, 80, 82, 84, 86, 87, 88, 89, 90, 91, 92, 93,
    97, 98, 99, 100,
]
TRIPLET_RANKS = [64]


def main() -> None:
    print(f"loading top-100 parquet: {TOP100}")
    df = pd.read_parquet(TOP100).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df = df.rename(columns={"cps_id": "cps_id_at_bookmark"})
    print(f"  {len(df)} cells")

    print(f"loading CP-SAM masks: {MASKS_TIF}")
    masks = tifffile.imread(MASKS_TIF)
    print(f"  masks: {masks.shape} dtype={masks.dtype}  n_cells={int(masks.max())}")

    print("computing per-cell centroid + bbox (microns) for the 100 ranked cells...")
    bboxes = find_objects(masks)

    rows = []
    for _, r in df.iterrows():
        cps = int(r["cps_id_at_bookmark"])
        bb = bboxes[cps - 1]
        if bb is None:
            rows.append({"x_um": np.nan, "y_um": np.nan,
                         "bbox_x0_um": np.nan, "bbox_x1_um": np.nan,
                         "bbox_y0_um": np.nan, "bbox_y1_um": np.nan,
                         "area_px": 0})
            continue
        sy, sx = bb
        sub = masks[sy, sx] == cps
        if not sub.any():
            rows.append({"x_um": np.nan, "y_um": np.nan,
                         "bbox_x0_um": np.nan, "bbox_x1_um": np.nan,
                         "bbox_y0_um": np.nan, "bbox_y1_um": np.nan,
                         "area_px": 0})
            continue
        cy_local, cx_local = center_of_mass(sub)
        cy_px = sy.start + cy_local
        cx_px = sx.start + cx_local
        rows.append({
            "x_um":       cx_px * PIXEL_SIZE_UM,
            "y_um":       cy_px * PIXEL_SIZE_UM,
            "bbox_x0_um": sx.start * PIXEL_SIZE_UM,
            "bbox_x1_um": sx.stop  * PIXEL_SIZE_UM,
            "bbox_y0_um": sy.start * PIXEL_SIZE_UM,
            "bbox_y1_um": sy.stop  * PIXEL_SIZE_UM,
            "area_px":    int(sub.sum()),
        })
    geom = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), geom], axis=1)

    # Approval annotations
    approved_set = set(APPROVED_RANKS)
    triplet_set  = set(TRIPLET_RANKS)
    df["approved"]   = df["rank"].isin(approved_set)
    df["is_triplet"] = df["rank"].isin(triplet_set)
    df["cpsam_version"] = CPSAM_VERSION
    df["approved_date"] = np.where(df["approved"], APPROVED_DATE, "")

    # Stable id: just the rank for now (within this top-100 export), to make
    # cross-references in writing/discussion painless.
    df.insert(0, "benchmark_id", df["rank"].apply(lambda r: f"DB_top100_{r:03d}"))

    # PNG filename (mirrors find_whole_slide_doublets naming)
    df["png_filename"] = df.apply(
        lambda r: (
            f"rank_{int(r['rank']):03d}_"
            f"{r['lineage_pair'].replace(' × ', 'x').replace(' ', '')}"
            f"_cps{int(r['cps_id_at_bookmark'])}_top2_{int(r['top2_n'])}"
            f"_ratio{r['doublet_ratio']:.2f}.png"
        ),
        axis=1,
    )

    n_approved = int(df["approved"].sum())
    n_triplet  = int(df["is_triplet"].sum())
    print(f"\napproved: {n_approved} / {len(df)}  (triplets: {n_triplet})")
    print(f"unapproved (left in top-100, not manually confirmed): "
          f"{len(df) - n_approved}")
    print(f"\nlineage-pair breakdown of approved:")
    print(df.loc[df['approved'], 'lineage_pair'].value_counts().to_string())

    df.to_parquet(OUT_PARQUET)
    print(f"\nwrote {OUT_PARQUET}")
    print(f"  cols: {df.columns.tolist()}")

    # Copy approved PNGs to a curated dir for easy review
    DST_FIGS.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for _, r in df[df["approved"]].iterrows():
        src = SRC_FIGS / r["png_filename"]
        if src.exists():
            shutil.copy2(src, DST_FIGS / r["png_filename"])
            n_copied += 1
    print(f"copied {n_copied} approved-PNG files → {DST_FIGS}")


if __name__ == "__main__":
    main()
