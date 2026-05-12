# .claude/

Claude's scaffolding for generating the `.ipynb` notebooks programmatically. **Not part of the user workflow.**

Each `build_*_notebook.py` is an `nbformat`-based script that, when run, writes a freshly-generated `.ipynb` to its destination in `workflow/`, `tools/`, or `sandbox/`. The purpose was to let Claude do large notebook refactors in plain Python source rather than editing JSON .ipynb files directly.

| Builder | Output |
|---|---|
| `build_celltype_gt_notebook.py`            | `workflow/01_celltype_ground_truth.ipynb` |
| `build_mrna_gradients_notebook.py`         | `workflow/02_mrna_gradients.ipynb` |
| `build_gap_intervention_test_notebook.py`  | `workflow/03_gap_intervention_test.ipynb` |
| `build_label_smoothing_methods_notebook.py` | `sandbox/label_smoothing_methods.ipynb` |
| `build_xenium_roi_crop_notebook.py`        | `tools/xenium_roi_crop_cpsam.ipynb` |
| `build_multi_roi_comparison_notebook.py`   | `tools/multi_roi_comparison.ipynb` |
| `build_omnipose_test_notebook.py`          | `sandbox/omnipose_install_check.ipynb` |
| `build_cpsam_intensity_test_notebook.py`   | `sandbox/cpsam_intensity_split_test.ipynb` |
| `build_cp_unet_vs_sam_notebook.py`         | `sandbox/cp_unet_vs_sam.ipynb` |

**Cost of running a builder**: it overwrites the `.ipynb` cleanly, wiping any executed cell outputs the user had saved. So builders are appropriate for refactors, not casual edits. For small edits to a notebook that already has valuable outputs, edit the `.ipynb` directly (in Jupyter) and don't re-run the builder.
