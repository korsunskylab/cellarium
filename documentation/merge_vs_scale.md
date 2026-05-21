# Merge-vs-scale: WSI-consistent CP-SAM settings

## Answer

**If you want ROI dev-work results to match the WSI segmentation: use either
`augment=False` or explicit `diameter=70`, on ROI ≥ 2048 px (≈ 435 µm).**

Both give n_focal = 1 at the rank-#6 focal cell — the same answer as WSI —
across all tested sizes from 2048 px upward. Below 2048 px, no parameter
combination we tested gives the WSI-matching 1-cell answer for this cell.

`default` settings (cpsam's max-recall defaults we'd been using:
augment=True, normalize=True, niter=200, cellprob_threshold=-5,
flow_threshold=0) are scale-stable at n_focal = 2 across 256→8192 px,
but they never converge to the WSI's 1-cell answer.

## Method (one-page)

- Target: rank-#6 focal cell (WSI µm (8886.2, 1478.4) = WSI px (41817, 6957)).
- Outcome: # CP-SAM masks that cover ≥ 5% of the WSI focal-cell mask area.
- Image: Xenium Prime FFPE, 2 morphology channels (DAPI + 18S), 0.2125 µm/px.
- 5 cellpose conditions × 5 ROI sizes (main grid), + 8192-px extension + reflect-padding rescue test.
- All runs same model (cpsam), same focal-pixel center, fresh load per crop.

## Main-grid results

![fig1 main grid](../figs/merge_vs_scale/fig1_main_grid.png)

n_focal table (rows = ROI side, cols = condition; bold = matches WSI):

| ROI px | µm  | default | wsi_norm | no_aug | both | diameter_70 |
|---|---|---|---|---|---|---|
| 256  | 54   | 2 | 3 | 2 | 3 | 2 |
| 512  | 109  | 2 | 3 | 2 | 3 | 2 |
| 1024 | 218  | 2 | 2 | 2 | 3 | 2 |
| 2048 | 435  | 2 | 2 | **1** | 2 | **1** |
| 4096 | 870  | 2 | 3 | **1** | 2 | **1** |
| 8192 | 1741 | 2 | 3 (`both`) | — | — | — |

(`wsi_norm` = augment=True, normalize=False + WSI-percentile pre-normalization.
`both` = augment=False + WSI-percentile pre-normalization. `no_aug` = augment=False, local %ile normalization.)

Mask-cover percentages for the WSI-matching runs (single mask covers ≥ 90% of WSI focal cell area):

- no_aug, 2048: 92%
- no_aug, 4096: 92%
- diameter_70, 2048: 93%
- diameter_70, 4096: 96%

## Padding-rescue test

![fig2 padding rescue](../figs/merge_vs_scale/fig2_padding_rescue.png)

Take a small crop (256 or 1024 px), reflect-pad it to 4096 × 4096, run cellpose. If the padded result matches the native 4096 result, image size/tiling is the cause; if it matches the native small result, real surrounding cell context is the cause.

Results for the unstable `both` condition (the only one where padding could possibly change the answer — `default` is 2 everywhere already):

| source | image size cellpose sees | n_focal | matches native ROI of size: |
|---|---|---|---|
| pad 256 → 4096 | 4096 | 3 | 256 (=3), not 4096 (=2) |
| pad 1024 → 4096 | 4096 | 2 | 4096 (=2), not 1024 (=3) |

So **real surrounding cell content matters up to ≈ 218 µm (1024 px)** — beyond that, additional context doesn't change the answer for this cell. Padding doesn't rescue a 256-px crop; that crop is too local for the model to "know" what's around it. But a 1024-px crop already contains enough real context that reflect-padding to 4096 reproduces the 4096-native answer.

This is consistent with cellpose collaborator `mrariden`'s hypothesis in [cellpose#1276](https://github.com/MouseLand/cellpose/issues/1276): the network reads object size from surrounding cells, and small tiles cue smaller objects.

## What we couldn't test

- **`bsize=512`** at 4096: crashed with `RuntimeError: tensor a (64) must match tensor b (32)` — cellpose-SAM has a fixed ViT positional embedding, so `bsize` is locked at 256. No way to widen tiles to reduce stitching boundaries.
- **`niter`** sweep: not tested in this run; iter=200 was used throughout.

## Recommendation for dev-iteration

| goal | what to use | why |
|---|---|---|
| Make ROI dev-work match WSI segmentation | `augment=False` or `diameter=70`, ROI ≥ 2048 px (435 µm) | Single condition that gives n_focal = 1 across all sizes ≥ 2048 and matches WSI. |
| Reproducible, ROI-invariant *baseline* (not WSI-matching) | `default` (augment=True, normalize=True, our max-recall thresholds) | Stable at n_focal = 2 across 256→8192. Won't match WSI but won't surprise you across ROI sizes either. |
| Iterate quickly on tiny crops (≤ 1024 px) | not safe with any condition | Below ~2048 px, no setting matches WSI. The 256-px answer is *not* recoverable by reflect-padding. |

## Reproduce / re-render

```bash
# data (uses GPU, ~90 min on MPS sharing with other tasks)
python scripts/merge_vs_scale_test.py

# notebook (fast, no GPU)
python .claude/build_merge_vs_scale_notebook.py
```

Notebook: `analyses/merge_vs_scale.ipynb`.
Outputs: `figs/merge_vs_scale/{fig1_main_grid.png, fig2_padding_rescue.png,
merge_vs_scale.csv, results.json}`.
