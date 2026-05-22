# Analyses rules

Each notebook is grounded in a specific dataset and ends with a direct data product (a gene signature, a set of extracted ROIs, a QC table, etc.).

## Naming

Numbered prefixes (`01_`, `02_`, …) reflecting logical sequence within the analyses tier. Topic name follows the number. Example: `01_celltype_ground_truth.ipynb`.

## Required top-of-notebook documentation

Every analysis notebook starts with a markdown cell stating:

1. **Dataset** — which dataset is used (path + one-line description).
2. **Output product** — what this notebook produces, where it lands, and what downstream code consumes it.
3. **Findings** — which `findings_registry.yaml` entries this notebook implements or depends on. Omit the section entirely if none apply.

## Reproducibility

The output product must be regenerable end-to-end from the dataset listed at the top, with no manual intermediate steps.
