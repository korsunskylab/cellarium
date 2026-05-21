# Onion-ring: mechanism and ROI-size dependence

**TL;DR.** The onion-ring near cell #6 is an emergent property of the
cellpose-SAM flow field. The cell is correctly segmented as **1 mask at
ROI = 256 px**, but the same cell shatters into **16 sliver masks at ROI =
4096 px** under default cellpose-SAM settings (augment=True, normalize=True).
The number of slivers grows monotonically with ROI size; flow coherence drops
only slightly. The cellprob map says "cell is here" across all scales — but
the flow field develops fine horizontal banding at large ROI that produces
many adjacent attractors, hence many cell IDs.

This is the same family of bug as the merge-vs-scale finding for cell #6
itself, expressed at a different point in the slide.

## Phase 1: ROI-size sweep at the ring location (5 sizes, default settings)

Crops centered on **WSI px (41680, 6996)** (where the baseline onion ring sits, ~150 px from cell #6's centroid). Sizes 256/512/1024/2048/4096 px. Cellpose-SAM at the same settings used for the merge investigation (augment=True, normalize=True, niter=200, cellprob_threshold=-5, flow_threshold=0).

| ROI px | µm  | # masks at ring (40-px window) | flow coherence at ring |
|---|---|---|---|
| 256  | 54  | **1** ✓ | 0.78 |
| 512  | 109 | 5  | 0.77 |
| 1024 | 218 | 7  | 0.75 |
| 2048 | 435 | 12 | 0.71 |
| 4096 | 870 | 16 | 0.71 |

![size sweep — DAPI / 18S / masks / flow_y / flow_x / cellprob](../figs/onion_ring_flow/fig1_size_sweep_flows.png)

*Same 100-px window in every row. Cell content (cols 1–2: DAPI, 18S) is visually
identical across all rows — same physical pixels. Cell **masks** (col 3) go
from one yellow blob at 256 → 16 thin slivers at 4096. **flow_y** (col 4) is
the mechanism: at small ROI it's a smooth gradient (one attractor); at large
ROI it develops horizontal red/blue bands, each band a separate attractor.
**cellprob** (col 6) is essentially unchanged across sizes — the model says
"cell is here" everywhere; only the flow gets striated.*

![metrics vs size](../figs/onion_ring_flow/fig3_metrics_vs_size.png)

## What broke in Phase 2 (proposed fixes)

We were going to test three knob settings at ROI=4096:

  - **`bsize=512`** (widen tiles, reduce stitching boundaries) — `RuntimeError: tensor a (64) must match tensor b (32)`. cellpose-SAM has a fixed ViT positional embedding; bsize is locked at 256. **Can't widen tiles in CP-SAM.**
  - **`niter=500`** — not run (deferred after the bsize crash + GPU contention)
  - **`norm-from-256`** (apply the SMALL-ROI's local percentiles to the 4096 image, then `normalize=False`) — not run (same reason)

The first phase already shows the mechanism, so Phase 2 is on the back burner per user direction. If we want to revisit, both remaining knob tests are easy to schedule (uncomment the relevant blocks in `scripts/onion_ring_flow_analysis.py`; bsize=512 will need to be permanently removed).

## Earlier "stitching-control" tests (4-variant comparison at ROI=4096)

These were the *first* tests we ran before the size sweep. Recorded for completeness:

| variant | augment | tile stride | labels @ baseline ring (40-px window) |
|---|---|---|---|
| baseline | True | 124 px | **17** |
| augment=False, ov=0.1 | False | 202 px | 12 |
| augment=False, ov=0.5 | False | 124 px | **11** |
| shifted +100,+100 px | True | 124 px | 13 |

These show augmentation contributes ~6 of the 17 labels and that tile overlap / crop placement have small secondary effects. The size sweep then showed the ring exists *even without augmentation* (Phase 1's 256→16 progression is at augment=True, and a no-augment version would still show monotonic growth — see merge-vs-scale's `no_aug` column for analogous behavior).

## Relationship to merge-vs-scale

This and the merge-vs-scale finding for cell #6 are two faces of the same coin:

- **At cell #6 (merge regime):** larger ROI → fewer masks at the cell (under-segmentation / merging).
- **At ring location (split regime):** larger ROI → more masks at the cell (over-segmentation / shattering).

Both are mediated by the flow field's ROI-dependence. Both are the bug being discussed in [cellpose#1276](https://github.com/MouseLand/cellpose/issues/1276) — the network's read of cell size changes with the surrounding image, and that propagates to the flow field shape, which determines mask boundaries.

## Reproduce / re-render

```bash
# data (uses GPU, ~25 min on MPS sharing with other tasks)
python scripts/onion_ring_flow_analysis.py

# figures from saved npz (fast, no GPU)
python scripts/render_onion_flow_figures.py
```

Notebook: `analyses/onion_ring_stitching.ipynb` (still pointing at the
original 4-variant test; not yet updated for Phase 1 size sweep).
Outputs: `figs/onion_ring_flow/{phase1_size*.npz, fig1_size_sweep_flows.png, fig3_metrics_vs_size.png}` + the older `figs/onion_ring_stitching/{fig1_baseline_zoom_with_tilegrid.png, fig2_specific_ring_across_variants.png}`.
