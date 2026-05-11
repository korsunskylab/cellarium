# cellarium

Improving cell segmentation on Xenium spatial transcriptomics by augmenting the morphology image with a transcript-derived **boundary-likelihood prior** before running CP-SAM (cellpose 4). The hypothesis: where mRNA gradients indicate a cell-type transition, dim the 18S morphology channel along that line so CP-SAM is more likely to split heterotypic neighbors that would otherwise be merged.

The repo is organized into three tiers of notebooks plus supporting scripts:

```
cellarium/
├── workflow/   # canonical pipeline — run these in order to reproduce results
├── tools/      # parameterized utilities for exploring a new dataset
├── sandbox/    # archived one-off experiments that informed pipeline decisions
├── scripts/    # preprocessing + runtime helpers invoked by the notebooks
├── .claude/    # nbformat builders for the notebooks (Claude scaffolding; not user-facing)
├── data/       # Xenium output + per-ROI crops + derived outputs (mostly gitignored)
└── figs/       # standalone diagnostic figures
```

Each subfolder has its own README. Start with `workflow/README.md` to reproduce the pipeline; `tools/README.md` to explore a new dataset; `sandbox/README.md` to understand why we made specific design decisions.

## Top-level pipeline narrative

1. **`workflow/01_celltype_ground_truth.ipynb`** (R) — Ingest the full 10X-segmented cells (~112k), QC, cluster via Seurat + Harmony, annotate clusters into 15 fine + 7 lineage cell-type labels, build a gene × cell-type count matrix, and emit per-gene lineage labels (one of the 7 types or "ambiguous").
2. **`workflow/02_label_smoothing_methods.ipynb`** (R) — Synthetic test bed comparing naive K-NN pooling vs anchored label propagation vs Potts-model Gibbs sampling at varying anchor densities. Outcome: label propagation wins at the realistic anchor density (~14% of genes are single-type-specific in this Xenium panel).
3. **`workflow/03_mrna_gradients.ipynb`** (R) — Per-transcript embedding from the count matrix in (1), filtered by labels, processed via Tessera (mesh + gradient + smoothing) to produce a per-transcript boundary score. Exports `boundary_likelihood.tif` at morphology resolution for the Python pipeline.
4. **`workflow/04_gap_intervention_test.ipynb`** (Python) — Step-0 validation that a Gaussian dim cut on 18S forces CP-SAM to split a merged doublet. Operating window mapped on synthetic + real ROI data.

The integration step (apply boundary mask from notebook 03 to the 18S channel + re-run CP-SAM) is set up at the end of notebook 03 (`§7 cpsam_roundtrip`).

## Recommended CP-SAM settings

For Xenium 5K tissue work in this pipeline, the validated **max-recall** config gives ~93% transcript-to-cell assignment vs ~75% for 10X defaults across 6 ROIs of the Human Skin Melanoma slide:

```python
from cellpose import models

m = models.CellposeModel(gpu=True, pretrained_model="cpsam")
img_2ch = np.stack([dapi, s18], axis=-1)  # DAPI + 18S, percentile-normalised
masks, flows, _ = m.eval(
    img_2ch, channel_axis=-1, diameter=None,
    cellprob_threshold=-5.0,   # very permissive; saturates at -4
    flow_threshold=0.0,        # disable flow-coherence filter
    augment=True,
)
```

The bias is intentional toward over-segmentation: false positives are filterable downstream by Baysor's transcript voting, false negatives are not.

## Environment

Python (cpsam, cellpose-omni, omnipose) — used by Python notebooks + by `scripts/run_cpsam_on_dir.py`:

```bash
mamba create -n omnipose python=3.11 pip -y
mamba activate omnipose
pip install cellpose==4.1.1 omnipose==0.4.4 jupyterlab tifffile zarr pandas pyarrow pooch plotly anywidget
python -m ipykernel install --user --name omnipose --display-name "Python (omnipose)"
```

R env (Seurat 5, harmony, presto, tessera, arrow, tiff) — used by all R workflow notebooks. R kernel registered with Jupyter (`IRkernel::installspec()`) so .ipynb files with `kernelspec.name = "ir"` open with the R kernel.

## Regenerating ROIs from source

If you have the original Xenium output bundle locally:

```bash
python scripts/crop_xenium_rois.py     # writes ROI1..ROI6 to data/<dataset>/ROI*/
```
