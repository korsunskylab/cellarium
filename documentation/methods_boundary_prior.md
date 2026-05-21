# Methods — V0 boundary-prior 18S cropping

## Lineage-specific transcript anchors

Each gene was assigned to a single lineage label based on its expression specificity across a reference set of seven dermal lineages (Endothelial, Fibroblast, Keratinocyte, Melanoma, Myeloid, Plasma, Tcell). For each gene *g* we computed the conditional probability *P*(lineage | *g*) and the ratio between the top and runner-up lineage probability. Genes whose top assignment was unambiguous were retained as lineage-specific markers; all other genes (≈72% of the panel) were labelled "ambiguous" and excluded from downstream anchor construction.

Xenium transcripts were filtered to per-transcript quality score ≥ 20 and `valid == 1`. Each retained transcript whose gene mapped to one of the seven lineages became an "anchor" at its (*x*, *y*) pixel position (pixel size = 0.2125 µm). Across the whole-slide image, this yielded 1.3 × 10⁷ anchors from 7.4 × 10⁷ raw transcripts.

## Per-pixel lineage posterior

For each lineage *k*, the anchor density ρ<sub>k</sub>(*x*, *y*) was computed by binning anchors to integer pixel positions and convolving with an isotropic Gaussian of σ<sub>diff</sub> = 2 µm (≈ 9.4 px). The effective per-pixel count is

> *N*<sub>k</sub>(*x*, *y*) = 2π σ<sub>diff</sub>² · ρ<sub>k</sub>(*x*, *y*)

so that each anchor contributes a unit mass to its local neighbourhood. The summed count is *N*<sub>total</sub> = Σ<sub>k</sub> *N*<sub>k</sub>.

A Dirichlet posterior with concentration α = 10 was applied:

> π<sub>k</sub><sup>post</sup>(*x*, *y*) = (*N*<sub>k</sub> + α) / (*N*<sub>total</sub> + *K*·α),  with *K* = 7.

To prevent unreliable posteriors in sparse-anchor regions, we used a "grey-abstain" mixture with the uniform prior. Defining a per-pixel confidence

> *c*(*x*, *y*) = *N*<sub>total</sub> / (*N*<sub>total</sub> + *N*<sub>min</sub>),  with *N*<sub>min</sub> = 3,

the final posterior is

> π<sub>k</sub>(*x*, *y*) = *c* · π<sub>k</sub><sup>post</sup> + (1 − *c*) · (1 / *K*).

In regions with abundant anchors (*c* → 1) this converges to the Dirichlet posterior; in regions with few or no anchors (*c* → 0) it collapses to a uniform distribution, suppressing spurious edge signal downstream.

## Per-pixel edge field

To detect heterotypic boundaries without prior knowledge of cell identity, we used a segmentation-agnostic one-vs-rest formulation. For each lineage *k*, define a signed competition margin

> *s*<sub>k</sub>(*x*, *y*) = π<sub>k</sub> − max<sub>j ≠ k</sub> π<sub>j</sub>.

Note that *s*<sub>k</sub> changes sign exactly where lineage *k* transitions from locally dominant to non-dominant — i.e., at lineage boundaries. Each lineage's contribution to the edge field is the gradient magnitude of a tanh-sharpened margin:

> *e*<sub>k</sub>(*x*, *y*) = ‖∇ tanh(*K*<sub>sharp</sub> · *s*<sub>k</sub>)‖₂,  with *K*<sub>sharp</sub> = 8,

computed using Sobel operators. The combined edge field is

> *E*(*x*, *y*) = Σ<sub>k</sub> *e*<sub>k</sub>(*x*, *y*).

Heterotypic boundaries are detected from both sides (each of the two adjacent lineages contributes a sign change) and are therefore emphasised relative to homotypic transitions.

## Cell-evidence weighting

To restrict edges to cellular regions without consuming any prior segmentation, we weighted the edge field by morphology signal only. DAPI and 18S whole-slide images were 1st–99th percentile clipped to [0, 1] to give *D*<sub>n</sub> and *S*<sub>n</sub>. The cell-evidence gate is

> ε(*x*, *y*) = max(*D*<sub>n</sub>, *S*<sub>n</sub>),

which is non-zero wherever any nuclear (DAPI) or cytoplasmic (18S) signal is present. Combined with the per-pixel confidence, the weighted edge field is

> *E**(*x*, *y*) = *E* · *c*² · ε.

The confidence-squared term up-weights regions where the lineage posterior is sharply defined (many anchors) and zeros out empty space; ε excludes background pixels independent of any segmentation.

## Cut field

The weighted edge field was spatially smoothed with σ<sub>cut</sub> = 5 px (≈ 1.06 µm) and rescaled to a maximum of *d*<sub>max</sub> = 0.99:

> *u*(*x*, *y*) = clip(*G*<sub>σcut</sub> ∗ *E** / max(*G*<sub>σcut</sub> ∗ *E**), 0, *d*<sub>max</sub>).

The cut field *u* takes values in [0, *d*<sub>max</sub>] with peaks at predicted heterotypic boundaries and ~0 elsewhere.

## 18S modification ("boundary-prior 18S")

The raw 18S image *S* was attenuated by the cut field:

> *S**(*x*, *y*) = *S*(*x*, *y*) · (1 − *u*(*x*, *y*)).

This dims the 18S signal at predicted heterotypic boundaries while leaving non-boundary regions unchanged. Because *u* depends only on raw transcripts and raw morphology, the modification is segmentation-agnostic and can be applied before any cell-segmentation pass.

## Cell segmentation

Segmentation was performed with Cellpose-SAM (`cpsam` pretrained model, Cellpose 4.1.1) on a two-channel image (DAPI, 18S) using the validated "max-recall" settings: `cellprob_threshold = −5.0`, `flow_threshold = 0.0`, `diameter = None` (auto), `augment = True`, `niter = 200`. Each channel was independently 1st–99th percentile clipped to [0, 1] before stacking.

For the apples-to-apples comparison between unmodified and boundary-prior 18S, two segmentation passes were performed with identical hyperparameters and identical DAPI input:

- **Control:** Cellpose-SAM on (DAPI, *S*).
- **Boundary-prior:** Cellpose-SAM on (DAPI, *S**).

The only difference between the two passes is the second channel.

## Implementation

The pipeline was implemented in Python 3.11. Anchor density convolution and Sobel gradients were computed with `scipy.ndimage.gaussian_filter` and `scipy.ndimage.sobel`. Image I/O used `tifffile` (BigTIFF for whole-slide outputs); transcripts were read from the Xenium per-FOV `transcripts.zarr.zip` index. Cellpose-SAM inference ran on Apple Silicon (Metal Performance Shaders backend).

Reference V0 implementation: `pipelines/V0/lib.py` with companion notebooks `00_run_pipeline.ipynb` (single ROI walkthrough) and `01_batch_dev_doublets.ipynb` (batch across 68 development doublets).

## Parameter summary

| Symbol | Value | Description |
|---|---|---|
| pixel size | 0.2125 µm/px | Xenium morphology resolution |
| σ<sub>diff</sub> | 2 µm (≈ 9.4 px) | Gaussian width for anchor density |
| α | 10 | Dirichlet posterior concentration |
| *N*<sub>min</sub> | 3 | Grey-abstain anchor count threshold |
| *K*<sub>sharp</sub> | 8 | tanh sharpening on competition margin |
| σ<sub>cut</sub> | 5 px (≈ 1.06 µm) | Spatial smoothing of the edge field |
| *d*<sub>max</sub> | 0.99 | Maximum 18S attenuation depth |
| qv threshold | 20 | Transcript quality score floor |
| `cellprob_threshold` | −5.0 | Cellpose-SAM |
| `flow_threshold` | 0.0 | Cellpose-SAM |
| `niter` | 200 | Cellpose-SAM dynamics steps |
| `augment` | True | Cellpose-SAM test-time augmentation |
