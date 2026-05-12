# workflow

The canonical pipeline. Run these in order to reproduce the boundary-prior segmentation results.

| # | Notebook | Kernel | Inputs | Outputs |
|---|---|---|---|---|
| 01 | `01_celltype_ground_truth.ipynb` | R | 10X Xenium output (cells.zarr + cell_feature_matrix.zarr) | `gene_celltype_counts_{fine,lineage}.parquet`, `gene_labels_lineage.parquet` |
| 02 | `02_mrna_gradients.ipynb` | R | `transcripts.parquet` per ROI + outputs from 01 | `boundary_score_per_tx.parquet`, `boundary_likelihood.tif`, cpsam roundtrip outputs |
| 03 | `03_gap_intervention_test.ipynb` | Python | morphology TIFFs + 10X masks from any ROI | none — Step-0 validation |

## Dependency chain

```
01 celltype_ground_truth ──→ 02 mrna_gradients
                              (consumes 01's outputs)

03 gap_intervention_test (independent step-0 validation; no data dep on 01/02)
```

## What each notebook does

### 01. Cell-type ground truth (R)

Loads the full 10X-segmented cell × gene matrix (~112k cells × 5k genes) via a one-time zarr → 10X-mtx extraction (`scripts/extract_xenium_cells_to_10x.py`, called from §1 via `system2()`). Runs Seurat with seeded, BLAS-single-threaded determinism (`SEED=42`, `OMP_NUM_THREADS=1`) for stable cluster IDs across re-runs. Annotates 15 fine clusters into 7 biological lineages (Melanoma, Myeloid, Tcell, Plasma, Fibroblast, Endothelial, Keratinocyte). Builds per-gene counts × cell-type matrices and assigns each gene either a single lineage label (strict-specific) or "ambiguous". Exports three parquets to `<DATA_ROOT>`.

### 02. mRNA gradients (R)

The active workstream. Loads the count matrix from notebook 01, column-normalizes to P(gene | type), z-scores per gene to get a centered embedding, joins to transcripts via the gene→label mapping, and runs Tessera (Delaunay mesh pruned at 3 µm → field estimation → bilateral anisotropic smoothing) to produce a per-transcript boundary score. §4 is a stub for the score → mask conversion (currently a placeholder; user-designed conversion goes here). §5–§6 visualize + export. §7 closes the loop via `scripts/run_cpsam_on_dir.py` (called via `system2()` to the omnipose env's Python) so you can iterate on the boundary mask end-to-end in R.

### 03. Gap intervention test (Python)

Step-0 pre-flight that validated the cut-on-18S idea before the rest of the pipeline was worth building. Phase A runs a controlled (σ × intensity-factor) sweep on a synthetic merged-blob ellipse and finds the operating window where a Gaussian dim profile causes CP-SAM to split. Phase B repeats on a real merged doublet in ROI1, with a 2-D Gaussian cut placed at the midpoint of the two 10X-reference centroids. Found design rule: **narrow + deep cuts split cleanly; wide + shallow cuts confuse CP-SAM**.
