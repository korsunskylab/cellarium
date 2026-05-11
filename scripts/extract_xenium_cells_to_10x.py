"""Convert Xenium cells.zarr + cell_feature_matrix.zarr → 10X-format dir + cells_meta.parquet.

R-side reads this via Seurat::Read10X() (or Matrix::readMM + read.table directly)
plus arrow::read_parquet() for per-cell centroids/areas.

One-time conversion; rerun only if you re-download the Xenium output.

Usage: python extract_xenium_cells_to_10x.py <data_root> [<out_dir>]
       (out_dir defaults to <data_root>/cells_extracted)

Writes to <out_dir>/:
    barcodes.tsv.gz       cell barcodes, one per line, gzipped
    features.tsv.gz       (feature_id, feature_symbol, feature_type) TSV, gzipped
    matrix.mtx.gz         sparse genes × cells, MatrixMarket, gzipped (10X convention)
    cells_meta.parquet    per-cell: cell_id, x_um, y_um, area_um2, nucleus_x_um,
                          nucleus_y_um, nucleus_area_um2, n_counts, n_features
"""
import sys
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csc_matrix
from scipy.io import mmwrite


def main(data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cells_zarr = zarr.open(str(data_root / "cells.zarr.zip"), mode="r")
    feat_zarr  = zarr.open(str(data_root / "cell_feature_matrix.zarr.zip"), mode="r")
    feat       = feat_zarr["cell_features"]

    n_cells    = int(feat.attrs["number_cells"])
    n_features = int(feat.attrs["number_features"])
    feature_ids    = list(feat.attrs["feature_ids"])
    feature_keys   = list(feat.attrs["feature_keys"])     # gene symbols
    feature_types  = list(feat.attrs["feature_types"])
    assert len(feature_ids) == n_features == len(feature_keys) == len(feature_types)
    print(f"matrix: {n_cells} cells × {n_features} features")

    # CSC: indptr length n_features+1 marks column starts; indices are cell rows
    data    = feat["data"][:]
    indices = feat["indices"][:]
    indptr  = feat["indptr"][:]
    # Build as cells × genes (transpose of native CSC, which is genes × cells)
    # Actually — Xenium stores it as features × cells (indptr along features); each
    # column is a feature, indices are cell rows. So csc_matrix((data, indices, indptr),
    # shape=(n_cells, n_features)) gives cells-as-rows. But 10X mtx convention is
    # genes × cells, so we transpose at write time below.
    mat_cells_x_genes = csc_matrix((data, indices, indptr),
                                    shape=(n_cells, n_features), dtype=np.int32)
    print(f"sparse matrix built: cells×genes = {mat_cells_x_genes.shape}, "
          f"nnz = {mat_cells_x_genes.nnz:,}")

    # ─── barcodes.tsv.gz ────────────────────────────────────────────────
    # cell_id in zarr is (N, 2): integer barcode + segmentation method id.
    # Use a stable string barcode = "cell_<id>_<seg>" so re-runs are deterministic.
    cell_id_arr = cells_zarr["cell_id"][:]
    barcodes    = [f"cell_{int(a)}_{int(b)}" for a, b in cell_id_arr]
    with gzip.open(out_dir / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(barcodes) + "\n")
    print(f"wrote {out_dir / 'barcodes.tsv.gz'} ({len(barcodes)} barcodes)")

    # ─── features.tsv.gz ────────────────────────────────────────────────
    with gzip.open(out_dir / "features.tsv.gz", "wt") as f:
        for fid, key, ftype in zip(feature_ids, feature_keys, feature_types):
            f.write(f"{fid}\t{key}\t{ftype}\n")
    print(f"wrote {out_dir / 'features.tsv.gz'} ({n_features} features)")

    # ─── matrix.mtx.gz ──────────────────────────────────────────────────
    # 10X convention: features × cells. scipy.io.mmwrite handles sparse → MatrixMarket.
    mat_genes_x_cells = mat_cells_x_genes.T.tocsc()
    buf = io.BytesIO()
    mmwrite(buf, mat_genes_x_cells, field="integer", precision=0)
    with gzip.open(out_dir / "matrix.mtx.gz", "wb") as f:
        f.write(buf.getvalue())
    print(f"wrote {out_dir / 'matrix.mtx.gz'} ({n_features} × {n_cells}, "
          f"{mat_genes_x_cells.nnz:,} nnz)")

    # ─── cells_meta.parquet ─────────────────────────────────────────────
    cs = cells_zarr["cell_summary"][:]
    col_names = list(cells_zarr["cell_summary"].attrs["column_names"])
    meta = pd.DataFrame(cs, columns=col_names)
    meta.insert(0, "barcode", barcodes)
    meta["n_counts"]   = np.asarray(mat_cells_x_genes.sum(axis=1)).ravel().astype(np.int64)
    meta["n_features"] = (mat_cells_x_genes > 0).sum(axis=1).A1.astype(np.int32)
    # Rename to the conventions the R notebook expects
    meta = meta.rename(columns={
        "cell_centroid_x":    "x_um",
        "cell_centroid_y":    "y_um",
        "cell_area":          "area_um2",
        "nucleus_centroid_x": "nucleus_x_um",
        "nucleus_centroid_y": "nucleus_y_um",
        "nucleus_area":       "nucleus_area_um2",
    })
    meta.to_parquet(out_dir / "cells_meta.parquet")
    print(f"wrote {out_dir / 'cells_meta.parquet'} "
          f"({len(meta)} rows, cols: {list(meta.columns)})")

    print("\ndone — load in R with:")
    print(f'  mat   <- Seurat::Read10X("{out_dir}")               # genes × cells')
    print(f'  meta  <- arrow::read_parquet("{out_dir / "cells_meta.parquet"}")')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    data_root = Path(sys.argv[1])
    out_dir   = Path(sys.argv[2]) if len(sys.argv) >= 3 else data_root / "cells_extracted"
    main(data_root, out_dir)
