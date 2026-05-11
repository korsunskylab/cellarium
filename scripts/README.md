# scripts

Helpers called by the notebooks. Two categories:

## Preprocessing (run once per dataset)

| Script | What it produces |
|---|---|
| `crop_xenium_rois.py` | Per-ROI directories under `data/<dataset>/ROI{1..6}/` from the full Xenium output bundle. Each ROI has the 4 morphology channels, the 10X cell mask + polygons, the filtered transcripts (qv ≥ 20), and a metadata.json. |
| `extract_xenium_cells_to_10x.py` | Converts `cells.zarr.zip` + `cell_feature_matrix.zarr.zip` to a standard 10X-format directory (`matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`) plus a `cells_meta.parquet` with per-cell centroids/areas. Called automatically from `workflow/01_celltype_ground_truth.ipynb §1` via `system2()` if `cells_extracted/` doesn't already exist. |
| `survey_xenium_rois.py` | Slide-level survey for picking informative ROIs (cell density, transcript count, marker enrichment). |
| `preview_roi_candidates.py` | Renders candidate ROIs as mini-views so you can eyeball before committing. Outputs to `figs/_roi_candidates_*.png`. |

## Runtime (called by notebooks during a session)

| Script | Called by | Purpose |
|---|---|---|
| `run_cpsam_on_dir.py` | `workflow/03_mrna_gradients.ipynb §7` via `system2()` to the omnipose env's Python binary | R-side writes DAPI + modified 18S to a TIFF pair → this script runs CP-SAM (max-recall config) → writes back `masks.tif` (uint16 label image), `cellprob.tif` + `flow_mag.tif` (uint16 with scaling metadata), `scaling.json`. R reads them back to close the loop without leaving the notebook. |

All preprocess scripts are idempotent — safe to re-run on a new dataset.
