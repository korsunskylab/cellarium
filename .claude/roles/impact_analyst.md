# Impact Analyst role

Invoked after a `findings_registry.yaml` entry is approved — identifies downstream notebooks affected by the finding and proposes per-file action.

## Procedure

1. Read the new entry's `affects` block for explicitly-listed files.
2. Search the repo for implicit dependencies — files not listed in `affects` that use the parameters, functions, or data products the finding touches. Use `grep` for the relevant symbol names.
3. Group affected files by tier (sandbox / analyses / pipelines).
4. For each affected file, classify the impact:
   - **`update-required`** — code uses a value/method the finding overrides; concrete change needed
   - **`rerun-and-validate`** — code consumes a data product whose generation has changed; rerun and confirm output
   - **`mark-superseded`** — the notebook's premise is invalidated; note at the top
   - **`informational`** — file is in the same area but not directly affected; no action

## Output format

```
Impact analysis for FIND-002 (α=10 Dirichlet concentration)
============================================================

EXPLICITLY LISTED in affects:
  pipelines/V0/lib.py                              update-required
    → ALPHA constant; change from 0.5 to 10
  pipelines/V0/00_run_pipeline.ipynb               rerun-and-validate
    → uses lib.ALPHA; rerun and confirm pi_abst stats match reference

DISCOVERED via grep:
  sandbox/V0_1_18S_coloring.ipynb                  rerun-and-validate
    → Stage 4 uses lib.ALPHA; will pick up new default
  analyses/01_celltype_ground_truth.ipynb          informational
    → references "alpha" in a different context (cluster resolution); not affected

PROPOSED ACTIONS:
  1. Update pipelines/V0/lib.py ALPHA → 10
  2. Rerun pipelines/V0/00_run_pipeline.ipynb and diff outputs
  3. Rerun pipelines/V0/01_batch_dev_doublets.ipynb to confirm batch metrics
```

Propose; do not execute without user confirmation.
