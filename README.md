# cellarium

Improving cell segmentation on Xenium spatial transcriptomics by augmenting the morphology image with a transcript-derived **boundary-likelihood prior** before running Cellpose-SAM. The hypothesis: where mRNA gradients indicate a cell-type transition, dim the 18S morphology channel along that line so CP-SAM is more likely to split heterotypic neighbors that would otherwise be merged.

## Repository layout

```
cellarium/
├── analyses/                  # notebooks grounded in a specific dataset, each producing a data product
├── sandbox/                   # exploratory iteration space — new ideas, dead-end documentation
├── pipelines/                 # production-ready code
│   └── V0/                    # active pipeline: boundary-prior + CP-SAM
├── data/                      # Xenium output + per-ROI crops (mostly gitignored)
├── documentation/             # methods write-ups
├── scratch/                   # dev figures and intermediates (gitignored)
├── .claude/                   # AI tooling: notebook builders + role definitions
│
├── CLAUDE.md                  # workflow conventions for the AI assistant
├── FIGURE_STANDARDS.md        # figure conventions
├── findings_registry.yaml     # durable findings affecting downstream notebooks
└── README.md                  # this file
```

Each tier (`analyses/`, `sandbox/`, `pipelines/`) has a `CLAUDE.md` with tier-specific conventions.

## Where to start

- **Working with the AI assistant** — read `CLAUDE.md` first; it has the workflow.
- **Active pipeline development** — `pipelines/V0/`. The boundary-prior V0 module is a self-contained folder with `lib.py`, numbered entry-point notebooks, and diagnostic scripts. New pipeline versions follow this pattern.
- **Final analyses on specific datasets** — `analyses/`. Each notebook documents its dataset, output product, and any `findings_registry.yaml` entries it implements.
- **Trying a new idea** — `sandbox/`. Topic-named notebooks; scripts grouped into conceptual subfolders.

## Recommended Cellpose-SAM settings

For Xenium 5K tissue work in this pipeline, the validated **max-recall** config gives ~93% transcript-to-cell assignment vs ~75% for 10X defaults, across 6 ROIs of the Human Skin Melanoma slide:

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

The bias is intentional toward over-segmentation: false positives are filterable downstream by Baysor's transcript voting; false negatives are not.

## Environment

Python (Cellpose 4 + cellpose-omni). Used by all Python notebooks and pipeline scripts:

```bash
mamba create -n omnipose python=3.11 pip -y
mamba activate omnipose
pip install cellpose==4.1.1 cellpose-omni==0.9.1 jupyterlab tifffile zarr pandas pyarrow pooch plotly anywidget
python -m ipykernel install --user --name omnipose --display-name "Python (omnipose)"
```

R (Seurat 5, harmony, presto, tessera, arrow, tiff). Used by R analysis notebooks. Register the kernel with `IRkernel::installspec()` so `.ipynb` files with `kernelspec.name = "ir"` open with the R kernel.

## Regenerating ROIs from source

If you have the original Xenium output bundle locally:

```bash
python analyses/scripts/crop_xenium_rois.py     # writes ROI1..ROI6 to data/<dataset>/ROI*/
```
