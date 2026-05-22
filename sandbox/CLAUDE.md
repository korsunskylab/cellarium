# Sandbox rules

Iteration space. Standalone notebooks and scripts where new ideas are tested. **No production code here.**

## Notebooks

- **Naming**: topic-only, no numeric prefix. Examples: `cp_unet_vs_sam.ipynb`, `gap_intervention_test.ipynb`, `pc_gradient_field.ipynb`.
- **Dead-end notebooks**: do not delete. Write a top-cell comment stating what was tried, what didn't work, and why — so future-you doesn't reopen the same dead end.
- **Notebooks that produce a real finding**: state the finding clearly at the bottom (one or two sentences). This is what the Knowledge Cartographer reads when proposing a `findings_registry.yaml` entry.

## Scripts

Scripts belong in a conceptual subfolder under `sandbox/scripts/`, **never directly in the flat `scripts/` root**.

Current clusters:

- `method_F*/`
- `method_G*/`
- `tint_cytoplasm*/`
- `cellpose_repro*/`
- `render*/`
- `compare*/`
- `onion*/`
- `edges*/`
- `cpsam*/`

A new script goes into the most relevant existing subfolder. If none fits, create a new named subfolder — do not drop it into the flat root.

**Current state**: existing scripts are currently in a flat `sandbox/scripts/` directory. They get folded into subfolders opportunistically as they're touched, not in a bulk reorg.
