# Pipelines rules

Production-ready code. New pipeline versions follow the `pipelines/V0/` module pattern.

## Module pattern (template)

A pipeline version lives in its own self-contained folder under `pipelines/`. Required structure:

```
pipelines/V<n>/
├── README.md              # what this version does, what changed from V<n-1>, test data to use
├── lib.py                 # all reusable functions, constants, defaults
├── 00_run_pipeline.ipynb  # canonical end-to-end on a single ROI / test case
├── 01_batch_*.ipynb       # batch evaluation across the dev set
├── _*.py                  # diagnostic / single-shot inspection scripts (leading underscore)
└── out/                   # regenerable outputs (gitignored)
```

- **`lib.py`** is the source of truth for parameters and recipes. Notebooks orchestrate; `lib.py` contains.
- **Numbered notebooks** (`00_`, `01_`, …) are the user-facing entry points.
- **`_*.py` scripts** are diagnostic — never imported, run standalone.
- **`out/`** is regenerable and gitignored.

## Required README.md per version folder

A short README at `pipelines/V<n>/README.md` stating:

1. What this version does (one paragraph).
2. What changed from the previous version (bullet list of substantive diffs).
3. What test data to use to verify it works (specific ROI or benchmark file).

## Legacy notebooks at `pipelines/` top level

The following are legacy and remain in place but should not be modified for new functionality:

- `MVP0.ipynb`
- `multi_roi_comparison.ipynb`
- `xenium_roi_crop_cpsam.ipynb`
- `02_mrna_gradients.ipynb`

**Do not add new notebooks at the `pipelines/` top level.** New work goes in a versioned subfolder (`pipelines/V1/`, `pipelines/V2/`, …).
