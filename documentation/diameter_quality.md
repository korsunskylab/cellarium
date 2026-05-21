# Diameter sweep — independent quality metric (anchor purity)

**TL;DR.** Cellpose-SAM `diameter` does *not* meaningfully change segmentation
**quality** for any lineage. It changes detection count and cellprob, but the
independent biology metric (anchor purity) is essentially flat across 15–200 µm.
**Recommendation: leave `diameter=None` (auto). Don't tune per lineage.**

**Caveat.** Cellprob *rises* with diameter while purity does not — so high
cellprob does **not** mean better segmentation. This is the
"cellpose-encoded vs. independent" confound we set out to test.

## What we measured

A 500 × 500 µm crop auto-selected for 3-lineage density (Tcells + Fibroblasts +
Melanoma, plus everything else present). 209,350 anchored transcripts in the
crop; ~3,068 WSI cells with confident lineage labels.

For each `diameter ∈ {15, 30, 70, 120, 200, 500}` px (3.2 / 6.4 / 14.9 / 25.5 /
42.5 / 106 µm — spanning well below to well above biological cell size), we ran
CP-SAM, assigned each detected cell its dominant transcript lineage, and computed:

  - **purity** = (transcripts of dominant lineage) / (total anchored transcripts in cell)  ← *independent of cellpose*
  - **cellprob** at WSI cell location ← *cellpose-encoded; baseline*
  - mean cell area (µm²)
  - # distinct lineages per cell

(`d=5 px` was attempted but skipped — at 1 µm cellpose tries to find ~5M
sub-cellular masks and never finishes within a useful time budget.)

## What the data shows

![diameter sweep](../figs/diameter_extreme_sweep/fig1_diameter_extreme_sweep.png)

### Purity (independent biology) is flat

| diameter (µm) | Melanoma | T cell | Fibroblast | Myeloid | Plasma | Endothelial |
|---|---|---|---|---|---|---|
| 3 | 0.72 | 0.50 | 0.61 | 0.56 | 0.60 | 0.59 |
| 6 (default-ish) | 0.72 | 0.51 | 0.61 | 0.56 | 0.61 | 0.59 |
| 15 | 0.72 | 0.50 | 0.60 | 0.55 | 0.59 | 0.58 |
| 26 | 0.72 | 0.49 | 0.59 | 0.53 | 0.59 | 0.58 |
| 43 | 0.71 | 0.46 | 0.56 | 0.50 | 0.54 | 0.55 |
| 106 | 0.71 | 0.52 | 0.57 | 0.49 | 0.53 | 0.58 |

- **Melanoma purity is exactly flat** at 0.71–0.72 across the full 3→106 µm range.
- Most other lineages decline ~5 percentage points between 6 µm and 43 µm — small
  and gradual, not a sharp optimum.
- T cells consistently have the **lowest purity** (~0.50) regardless of
  diameter — they're hard to segment cleanly because they're small and
  often adjacent to other cells.

### Cellprob is misleading

| diameter (µm) | Melanoma cellprob | T cell cellprob | T cell purity |
|---|---|---|---|
| 6 | 1.10 | 0.85 | 0.51 |
| 26 | 1.31 | 1.67 | 0.49 |
| 43 | 1.50 | **2.06** ← highest | **0.46** ← lowest |
| 106 | 1.43 | -0.77 | 0.52 |

At d=43 µm, cellpose's **highest-confidence T-cell calls have the
lowest biological purity**. If we'd tuned diameter to maximize cellprob we
would have picked the worst setting. This vindicates the user's call to use an
independent metric.

### Detection count drops at extreme diameters

At d=106 µm, T cell detection drops from 707 → 53 (92% loss) and Fibroblast
from 459 → 165 (64% loss) — too-large diameter just misses small cells.
Detection at d=15 µm starts to recover (T cells back to 680/707 = 96%),
confirming there's nothing to gain at d=3 µm.

## Recommendation

- Use `diameter=None` (cellpose-SAM auto-estimates ~30 px = 6 µm here, in the
  flat-purity region).
- Do **not** route cells through per-lineage diameter heads (no purity gain).
- Stop trusting `cellprob` as a quality proxy when comparing different
  parameter settings — use anchor purity or another independent metric instead.

The one place diameter *does* matter (per the merge-vs-scale test):
explicit `diameter=70` changes the focal-cell mask count at ROI ≥ 2048 px,
matching WSI behavior. So `diameter` interacts with ROI size in a separate
way — see `reports/merge_vs_scale.md` for that.

## Reproduce / re-render

```bash
# data (uses GPU, ~20 min on MPS sharing with other tasks)
python scripts/diameter_extreme_sweep.py

# notebook (fast, no GPU)
python .claude/build_diameter_quality_notebook.py
```

Notebook: `analyses/diameter_quality.ipynb`.
Outputs: `figs/diameter_extreme_sweep/{fig1_diameter_extreme_sweep.png,
per_lineage_per_diameter.csv, results.json}`.
