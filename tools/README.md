# tools

Parameterized utilities. Re-runnable on a new dataset; not part of the production pipeline.

| Notebook | Kernel | What it does |
|---|---|---|
| `xenium_roi_crop_cpsam.ipynb` | Python | Per-ROI workspace. Toggle `ROI = "ROI1"`…`"ROI6"` at the top. Loads the four morphology channels (DAPI, ATP1A1+CD45+ECad, 18S, αSMA+Vim), the 10X cell mask, the 10X polygons, and the filtered transcripts. Runs CP-SAM (DAPI + 18S) at varied thresholds. Visualizes cellprob, flow direction (HSV), flow magnitude, and a cellprob-vs-mRNA-density curve to inform threshold choice. Compares 10X vs CP-SAM segmentation side-by-side. |
| `multi_roi_comparison.ipynb` | Python | Runs CP-SAM max-recall across all 6 ROIs simultaneously. Slide-level overview, top-10 marker gene tables, side-by-side mask + transcript-assignment grids. Useful for sanity-checking pipeline behavior across tissue contexts before committing to a particular ROI for downstream work. |
