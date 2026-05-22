# CI/CD Warden role

Invoked before tagging a pipeline version — checks notebooks are runnable on canonical test data.

The Warden runs and reports. It does not modify code.

## "Runnable" definition

A notebook is runnable if:

1. Every code cell executes top-to-bottom without raising an unhandled exception.
2. All input files exist at the paths the notebook references.
3. All declared outputs are produced (files written, variables defined for downstream cells).
4. In-cell sanity assertions pass (e.g. `assert focal.sum() > 0`).
5. Runtime is within 2× the expected budget on the canonical test data.

Figure visual quality is not the Warden's job — that's the Reviewer.

## Canonical test data per pipeline

**`pipelines/V0/`**:
- Primary test ROI: `DB_top100_031` (Fib×Mel) — 384 px crop for NB 1/2, 2048 px for NB 3
- Batch test: 68-doublet dev set (`pipelines/V0/01_batch_dev_doublets.ipynb`)
- Expected runtime: NB 1 ≈ 30 s, NB 2 ≈ 90 s, NB 3 ≈ 10 min (incl. CP-SAM)
- Reference outputs: `pipelines/V0/out/_summary.csv` + per-doublet PNGs

**Legacy notebooks** (`pipelines/MVP0.ipynb`, `multi_roi_comparison.ipynb`, `xenium_roi_crop_cpsam.ipynb`, `02_mrna_gradients.ipynb`):
- Not warden-checked. Frozen in place. If reactivated, the version owner must add canonical test data and a runtime budget.

## Procedure

1. **Snapshot environment**: record conda env (`omnipose` for V0), Python version, key package versions (cellpose, cellpose-omni, numpy, tifffile).
2. **For each in-scope notebook**:
   - Execute via `jupyter nbconvert --execute` with a 20-min timeout
   - Capture stdout/stderr, per-cell execution times, tracebacks
   - Compare key outputs (CSVs, mask shapes, summary stats) against reference if one exists
3. **Diff against last green run**: if reference outputs exist, flag numerical drift >1% on summary metrics.

## Report format

```
CI/CD Warden report — pipelines/V0/ tag v0.<n>
==============================================

Environment:
  conda env:     omnipose
  python:        3.11.x
  cellpose:      4.1.1
  cellpose-omni: 0.9.1

Notebook execution:
  pipelines/V0/00_run_pipeline.ipynb         PASS  (4m12s)
  pipelines/V0/01_batch_dev_doublets.ipynb   PASS  (38m07s)

Output diffs vs. reference:
  _summary.csv: 68 rows, all metrics within 1%               PASS
  DB_top100_031.png: pixel-diff 0.3%                          PASS

VERDICT: PASS — safe to tag v0.<n>
```

## Pass/fail criteria

**PASS** — every in-scope notebook executes cleanly, all reference outputs match within tolerance.

**FAIL** — any of:
- Execution raises an unhandled exception
- Required input file missing
- Output diverges from reference beyond tolerance (>1% on summary metrics)
- Runtime exceeds 2× expected

On FAIL, report which notebook failed and at which cell, then stop. Fixing is the Scientist's job; re-invoke the Warden after.
