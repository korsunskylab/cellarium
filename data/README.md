# data/

Xenium spatial transcriptomics data + per-ROI crops + derived outputs from the workflow notebooks.

Most of this directory is **gitignored** — only the per-ROI crops + a few small derived files are tracked.

## Layout

```
data/<dataset>/
├── cells.zarr.zip               # 10X cell segmentation + boundaries        (gitignored, ~100 MB)
├── cell_feature_matrix.zarr.zip # 10X cells × genes matrix                  (gitignored, ~55 MB)
├── transcripts.zarr.zip         # full-slide transcripts                    (gitignored, ~3 GB)
├── analysis.zarr.zip            # 10X precomputed analysis                  (gitignored)
├── morphology_focus/            # full-slide morphology TIFFs               (gitignored, ~1.7 GB)
├── experiment.xenium            # experiment metadata
├── gene_panel.json              # gene panel (ID → symbol)
│
├── ROI{1..6}/                   # per-ROI crops (TRACKED; ~10 MB each)
│   ├── morphology_{DAPI,18S,ATP1A1_CD45_ECad,aSMA_Vim}.tif   # 4 morphology channels
│   ├── cells_10x_masks.tif      # 10X label image at native resolution
│   ├── cells_10x_polygons.parquet
│   ├── transcripts.parquet       # filtered to qv ≥ 20
│   └── metadata.json
│
├── cells_extracted/             # workflow/01 §1 output                     (gitignored, regen via scripts/extract_xenium_cells_to_10x.py)
│   ├── matrix.mtx.gz
│   ├── features.tsv.gz
│   ├── barcodes.tsv.gz
│   └── cells_meta.parquet
│
├── gene_celltype_counts_{fine,lineage}.parquet                              (gitignored, workflow/01 output)
├── gene_labels_lineage.parquet                                              (gitignored, workflow/01 output)
│
├── celltype_analysis*/          # exploratory Seurat outputs                (gitignored)
│
└── ROI*/boundary_prior/         # workflow/03 outputs                       (gitignored)
    ├── boundary_score_per_tx.parquet
    └── boundary_likelihood.tif
```

## What's tracked vs. gitignored

**Tracked** (small; useful as a starting checkpoint without needing the full Xenium dataset):
- `ROI{1..6}/*` — the 6 pre-cropped ROIs (~60 MB total)
- `experiment.xenium`, `gene_panel.json` — metadata
- `analysis_summary.html`

**Gitignored** (large; regenerable):
- `cells.zarr.zip`, `cell_feature_matrix.zarr.zip`, `transcripts.zarr.zip`, `analysis.zarr.zip`
- `morphology_focus/`
- `cells_extracted/` — regen with `scripts/extract_xenium_cells_to_10x.py`
- `celltype_analysis*/` — exploratory outputs from `workflow/01`
- `gene_celltype_counts_*.parquet`, `gene_labels_lineage.parquet` — main outputs of `workflow/01`
- `ROI*/boundary_prior/` — outputs of `workflow/03`
- `ROI*/cpsam_roundtrip/` — temp directory for `workflow/03 §7` (R↔Python cpsam loop)

## Regeneration commands

```bash
# (1) From a full Xenium output bundle → per-ROI crops:
python scripts/crop_xenium_rois.py

# (2) From zarr → 10X mtx + cells_meta.parquet (one-time, ~1 min):
python scripts/extract_xenium_cells_to_10x.py data/<dataset>

# (3) Cell-type ground truth → gene_celltype_counts_*.parquet + gene_labels_lineage.parquet:
# Run workflow/01_celltype_ground_truth.ipynb end-to-end.

# (4) mRNA gradients → boundary_score_per_tx.parquet + boundary_likelihood.tif:
# Run workflow/03_mrna_gradients.ipynb end-to-end.
```
