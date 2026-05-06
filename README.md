# cellarium

Cell-segmentation and transcript-assignment experiments on Xenium spatial transcriptomics data, focused on improving over 10X's default segmentation by re-running cellpose 4 (`cpsam`) with permissive thresholds.

## What's here

- `scripts/` — Python builders that generate the notebooks below from source. Re-run any builder if you want to regenerate a notebook.
- `notebooks/` — executed Jupyter notebooks with embedded plots:
  - `omnipose_install_check.ipynb` — synthetic-data sanity check that the env runs cellpose 4 + omnipose side-by-side on Apple Silicon (MPS).
  - `cp_unet_vs_sam.ipynb` — architecture comparison: legacy cyto2 U-Net vs cpsam (SAM backbone), same post-processing, same synthetic data.
  - `cpsam_intensity_split_test.ipynb` — small experiment probing whether cpsam uses intensity contrast or shape cues to split touching cells.
  - `xenium_roi_crop_cpsam.ipynb` — the main per-ROI workspace. Toggle `ROI = "ROI1"` … `"ROI6"` at the top to switch. Compares 10X cell calls vs cpsam at varied threshold settings; visualises cellprob, flow direction, flow magnitude (incl. interactive plotly); plots cprob vs mRNA density to inform threshold choice.
  - `multi_roi_comparison.ipynb` — runs cpsam MAX RECALL on all 6 ROIs and compares to 10X with a slide-level overview, top-10 gene tables, and side-by-side mask + transcript-assignment grids.
- `figs/` — survey outputs (slide-level cell density, candidate-ROI mini-views).
- `data/Xenium_Prime_Human_Skin_FFPE_xe_outs/ROI{1..6}/` — six pre-cropped ROIs from the Human Skin Melanoma FFPE 5K slide. Each contains the 4 morphology channels (`DAPI`, `ATP1A1_CD45_ECad`, `18S`, `aSMA_Vim`) as TIFFs, the 10X cell mask, 10X cell polygons (parquet), filtered transcripts (parquet, qv≥20), and metadata.

The full source dataset is *not* tracked (~3 GB). The crop scripts are kept in `scripts/` so the ROIs can be regenerated from source if needed.

## Environment

Python 3.11 conda env with `cellpose==4.1.1`, `cellpose-omni==0.9.1`, `omnipose==0.4.4`. `torch==2.11`, MPS-enabled (Apple Silicon GPU).

```bash
mamba create -n omnipose python=3.11 pip -y
mamba activate omnipose
pip install cellpose==4.1.1 omnipose==0.4.4 jupyterlab tifffile zarr pandas pyarrow pooch plotly anywidget
python -m ipykernel install --user --name omnipose --display-name "Python (omnipose)"
```

## Recommended cpsam settings

For Xenium 5K tissue work in this pipeline, MAX RECALL config gives ~93% transcript-to-cell assignment vs ~75% for 10X defaults across the 6 ROIs:

```python
from cellpose import models

m = models.CellposeModel(gpu=True, pretrained_model="cpsam")
img_2ch = np.stack([dapi, s18], axis=-1)  # DAPI + 18S, percentile-normalised
masks, flows, _ = m.eval(
    img_2ch, channel_axis=-1, diameter=None,
    cellprob_threshold=-5.0,   # very permissive; saturates at -4
    flow_threshold=0.0,        # disable flow-coherence filter (otherwise it cancels cellprob)
    augment=True,
)
```

Bias toward over-segmentation: false positives are filterable downstream by Baysor's transcript voting, false negatives are not.

## Regenerating ROIs from source

If you have the original Xenium output bundle locally:

```bash
python scripts/crop_xenium_rois.py     # writes ROI1..ROI6 to data/<dataset>/ROI*/
```
