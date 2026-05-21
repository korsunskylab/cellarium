"""Save per-ROI labeled-cell tables: for each ROI directory under
data/<dataset>/ROI*/, write `cells_labeled.parquet` containing the
lineage and fine cell-type labels for every 10X cell whose mask sits
inside that ROI.

Run once after `analyses/01_celltype_ground_truth.ipynb` has produced
clustering results. The downstream notebooks then read each ROI's
`cells_labeled.parquet` directly instead of loading the full-slide
cells table + applying the cluster→lineage mapping every time.

Inputs:
  data/<dataset>/cells_extracted/cells_meta.parquet
      (barcode, x_um, y_um, ...) — one row per cells.zarr row, in zarr order
  data/<dataset>/celltype_analysis_A/cells_with_clusters.parquet
      (barcode, cluster, UMAP_1, UMAP_2) — seeded clustering output
  data/<dataset>/ROI{N}/cells_10x_masks.tif — per-ROI 10X label image

Outputs:
  data/<dataset>/ROI{N}/cells_labeled.parquet
      cols: mask_id, barcode, x_um, y_um, cluster, cell_type_fine, cell_type_lineage

Mask values are 1-based row indices into cells.zarr (0 is background).
cells_meta.parquet was built in cells.zarr row order, so mask value V
maps to cells_meta.iloc[V-1].

Usage:
  python scripts/label_roi_cells.py <data_root>

Notes:
- The cluster→fine and fine→lineage mappings are hardcoded here to match
  analyses/01_celltype_ground_truth §5. If you re-cluster with different
  parameters, update both files in sync.
- A cell present in an ROI mask but missing from `cells_with_clusters`
  (because it failed notebook 02's QC) is included with NA labels.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile


CLUSTER_TO_FINE = {
    "0":  "Melanoma_main",      "1":  "Melanoma_MITFhigh", "2":  "Macrophage_TAM",
    "3":  "Melanoma_Sphase",    "4":  "Tcell",             "5":  "Melanoma_other",
    "6":  "Fibroblast",         "7":  "Endothelial",       "8":  "Myeloid_inflammatory",
    "9":  "myCAF",              "10": "Melanoma_G2M",      "11": "Plasma",
    "12": "Keratinocyte_basal", "13": "Melanoma_lowdef",   "14": "Keratinocyte_suprabasal",
}

FINE_TO_LINEAGE = {
    "Melanoma_main":           "Melanoma",
    "Melanoma_MITFhigh":       "Melanoma",
    "Melanoma_Sphase":         "Melanoma",
    "Melanoma_G2M":            "Melanoma",
    "Melanoma_other":          "Melanoma",
    "Melanoma_lowdef":         "Melanoma",
    "Macrophage_TAM":          "Myeloid",
    "Myeloid_inflammatory":    "Myeloid",
    "Tcell":                   "Tcell",
    "Plasma":                  "Plasma",
    "Fibroblast":              "Fibroblast",
    "myCAF":                   "Fibroblast",
    "Endothelial":             "Endothelial",
    "Keratinocyte_basal":      "Keratinocyte",
    "Keratinocyte_suprabasal": "Keratinocyte",
}


def main(data_root: Path) -> None:
    meta_path = data_root / "cells_extracted" / "cells_meta.parquet"
    cwc_path  = data_root / "celltype_analysis_A" / "cells_with_clusters.parquet"
    for p in (meta_path, cwc_path):
        if not p.exists():
            sys.exit(f"missing {p}")

    cells_meta = pd.read_parquet(meta_path)[["barcode", "x_um", "y_um"]]
    cwc_raw = pd.read_parquet(cwc_path)
    # Optional doublet columns — present after notebook 01 §6 (DoubletFinder).
    keep_cols = ["barcode", "cluster"] + [
        c for c in ("doublet_score", "doublet_class", "doublet_lineage_pair")
        if c in cwc_raw.columns
    ]
    cwc = cwc_raw[keep_cols].copy()
    cwc["cluster"]           = cwc["cluster"].astype(str)
    cwc["cell_type_fine"]    = cwc["cluster"].map(CLUSTER_TO_FINE)
    cwc["cell_type_lineage"] = cwc["cell_type_fine"].map(FINE_TO_LINEAGE)
    assert not cwc["cell_type_fine"].isna().any(),    "cluster_to_fine missing entries"
    assert not cwc["cell_type_lineage"].isna().any(), "fine_to_lineage missing entries"

    barcode_to_labels = cwc.set_index("barcode")

    for roi_dir in sorted(data_root.glob("ROI*")):
        mask_path = roi_dir / "cells_10x_masks.tif"
        if not mask_path.exists():
            print(f"  skip {roi_dir.name} (no cells_10x_masks.tif)")
            continue
        mask = tifffile.imread(mask_path)
        mask_ids = np.setdiff1d(np.unique(mask), [0]).astype(int)
        rows = mask_ids - 1                              # mask val = cells.zarr row + 1
        meta_sub = cells_meta.iloc[rows].reset_index(drop=True)

        labels_sub = barcode_to_labels.reindex(meta_sub["barcode"]).reset_index(drop=True)

        out_dict = {
            "mask_id":           mask_ids,
            "barcode":           meta_sub["barcode"].values,
            "x_um":              meta_sub["x_um"].values,
            "y_um":              meta_sub["y_um"].values,
            "cluster":           labels_sub["cluster"].values,
            "cell_type_fine":    labels_sub["cell_type_fine"].values,
            "cell_type_lineage": labels_sub["cell_type_lineage"].values,
        }
        # Carry through any doublet columns present in cells_with_clusters
        for c in ("doublet_score", "doublet_class", "doublet_lineage_pair"):
            if c in labels_sub.columns:
                out_dict[c] = labels_sub[c].values
        out = pd.DataFrame(out_dict)
        out_path = roi_dir / "cells_labeled.parquet"
        out.to_parquet(out_path, index=False)

        n_labeled = out["cell_type_lineage"].notna().sum()
        print(f"  {roi_dir.name}: {len(out)} cells in mask, {n_labeled} labeled "
              f"({100 * n_labeled / max(len(out), 1):.1f}%) → {out_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(Path(sys.argv[1]))
