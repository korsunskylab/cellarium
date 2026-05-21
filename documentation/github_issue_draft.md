# Draft NEW cellpose GitHub issue

**Repo:** https://github.com/MouseLand/cellpose
**Title:** Inconsistent cellpose-SAM segmentation between sub-ROI and full image at default parameters
**Labels suggested:** bug

This is related to [#1276](https://github.com/MouseLand/cellpose/issues/1276) but the workarounds suggested there (switch from cellpose-SAM to `cyto3`; fill in an empty channel) don't apply to our setup — cellpose-SAM, both morphology channels populated with real signal, no custom training, default params. Posting a fresh issue with a controlled same-pixel reproduction.

---

## Issue body

### Before you fill out this form

- [x] Yes, I reviewed the FAQ on the cellpose ReadTheDocs
- [x] Yes, I searched for related previous GH issues — closest is [#1276](https://github.com/MouseLand/cellpose/issues/1276) (and the related [#1312](https://github.com/MouseLand/cellpose/issues/1312)), but the workarounds in that thread (switch to `cyto3`; fill in an empty channel) don't apply to my case: I'm running stock cellpose-SAM with two populated morphology channels and no custom training.

### Describe the bug

I want to develop a post-processing method using small crops out of a large WSI, then deploy it on the full image. My assumption was that whatever cellpose returns for a given cell in a small ROI should match what it returns for that same cell when more surrounding image is included.

Cellpose-SAM at all-default parameters does not satisfy this — the same focal cell gets segmented differently depending on the size of the surrounding crop. Two crops centered on the same WSI pixel, one fully contained inside the other, give qualitatively different masks at that pixel:

- ROI = 1024 × 1024 px → focal cell is split into **2 distinct masks**
- ROI = 4096 × 4096 px → focal region is merged into **1 large mask** spanning multiple neighbours (along with bits of 3 adjacent cells touching the 50-px count window)

### To Reproduce

```python
import numpy as np, tifffile
from cellpose import models, io
logger = io.logger_setup()

# Two morphology channels from the same WSI
dapi_full = tifffile.imread("morphology_focus_0000.ome.tif", key=0)
s18_full  = tifffile.imread("morphology_focus_0002.ome.tif", key=0)

# Crop the SAME focal cell at two ROI sizes (centered on the same WSI pixel).
CX_PX, CY_PX = 41817, 6957
def crop_centered(arr, cx, cy, size):
    half = size // 2; H, W = arr.shape[-2:]
    y0 = max(cy - half, 0); y1 = min(y0 + size, H); y0 = y1 - size
    x0 = max(cx - half, 0); x1 = min(x0 + size, W); x0 = x1 - size
    return arr[..., y0:y1, x0:x1]

m = models.CellposeModel(gpu=True, pretrained_model="cpsam")
for size_px in (1024, 4096):
    dapi_c = crop_centered(dapi_full, CX_PX, CY_PX, size_px)
    s18_c  = crop_centered(s18_full,  CX_PX, CY_PX, size_px)
    img = np.stack([dapi_c.astype(np.float32),
                    s18_c.astype(np.float32)], axis=-1)
    masks, _, _ = m.eval(img, channel_axis=-1)          # all params at default
    print(f"ROI={size_px}: n_total={masks.max()}")
```

The 1024 crop is fully contained inside the 4096 crop, so the focal cell and its ~110 µm of context are identical pixel-for-pixel in both. Only the surrounding image size differs.

### Run log

<details><summary>Verbose log</summary>

```
cellpose version:   4.1.1
platform:           darwin
python version:     3.11.15
torch version:      2.11.0
** TORCH MPS version installed and working. **
>>>> using GPU (MPS)
>>>> loading model /Users/.../models/cpsam

=========  ROI = 1024 × 1024 px  =========
  img.shape = (1024, 1024, 2), dtype = float32
  calling model.eval(img, channel_axis=-1)  with all other params at default
  n_total masks: 309
  # distinct masks in 50-px window around focal cell: 2

=========  ROI = 4096 × 4096 px  =========
  img.shape = (4096, 4096, 2), dtype = float32
  calling model.eval(img, channel_axis=-1)  with all other params at default
  n_total masks: 5868
  # distinct masks in 50-px window around focal cell: 4
```

(`# distinct masks in 50-px window` includes parts of 3 adjacent cells whose masks touch the window at ROI=4096. The dominant mask covering the focal cell itself is a single large merged mask spanning ~5 nuclei — visible in the bottom-right two panels of the figure below.)

</details>

### Screenshots

![cell6 full and zoom — both morphology channels](github_issue_figs/issue_cell6_full_and_zoom_2chan.png)

*Columns 1–2: ROI 1024 (DAPI then 18S); columns 3–4: ROI 4096 (DAPI then 18S). Top row is the full ROI; red box marks the 200×200 px window we zoom into. Bottom row is that zoom with cellpose masks overlaid. The yellow ★ marks the focal pixel in all four panels — same WSI pixel in every panel.*

### What's different from #1276

Adding because the workarounds in that thread don't fit my case:

- I'm running stock `cellpose-SAM`, not a custom-trained model (rules out the training-data hypothesis from the OP).
- Both my channels (`DAPI`, `18S`) carry real signal; histograms are non-trivial in both. Not the empty-channel case `@Zailushang211` reported.
- Switching to `cyto3` (which worked for `@m-albert`) isn't an option for me — cellpose-SAM is producing better cell delineation overall in our setting, we'd just like it to be scale-stable.

### Additional tests we ran (in case useful for narrowing the fix)

Same focal pixel, swept ROI ∈ {256, 512, 1024, 2048, 4096, 8192} px under 5 parameter combinations. # masks covering the focal cell:

| ROI (px / µm) | default | augment=True + fixed-percentile norm | augment=False | augment=False + fixed-percentile norm | augment=True, explicit diameter=70 |
|---|---|---|---|---|---|
| 256 / 54 | 2 | 3 | 2 | 3 | 2 |
| 512 / 109 | 2 | 3 | 2 | 3 | 2 |
| 1024 / 218 | 2 | 2 | 2 | 3 | 2 |
| 2048 / 435 | 2 | 2 | **1** | 2 | **1** |
| 4096 / 870 | 2 | 3 | **1** | 2 | **1** |
| 8192 / 1741 | 2 | 3 | (not tested) | (not tested) | (not tested) |

Two ways to get a consistent "1 mask at the focal cell" answer once ROI ≥ 2048 px (≈435 µm):

1. `augment=False`
2. `diameter=70` explicit

Below 2048 px no combination we tested converges to 1 mask. This is consistent with `@mrariden`'s hypothesis in #1276 that the network reads object size from the surrounding image — but in our case the result *also* doesn't stabilize at the larger end: cellpose-SAM at default is monotonically stuck at 2 across 256→8192 px, and never converges to the 1-mask answer that explicit-diameter gives.

We also tried `bsize=512` to reduce tile-stitching boundaries, but cellpose-SAM has a fixed ViT positional embedding (`RuntimeError: tensor a (64) vs b (32)`), so `bsize` is effectively locked at 256.

### Environment

- cellpose 4.1.1, `pretrained_model="cpsam"`
- Apple Silicon MPS backend (macOS)
- Image: Xenium Prime FFPE spatial transcriptomics, 2 morphology channels (DAPI + 18S), 0.2125 µm/px

### Questions

1. Is there a documented mechanism for the scale-dependence we observe (auto-diameter? per-tile normalization? something else)?
2. What's the recommended workflow for iterating on small ROIs and having results match a whole-image run with cellpose-SAM?
3. Is there a parameter combination that's truly scale-invariant for cellpose-SAM, given that `bsize` is fixed?

Happy to share the raw `.ome.tif` files (large, ~7 GB) or just a 4096-px crop around this focal cell.

---

## Local notes (not part of the issue body)

- Figure: `reports/github_issue_figs/issue_cell6_full_and_zoom_2chan.png`
- Repro: `scripts/cellpose_repro_combined.py` (one shot — runs cellpose with `logger_setup`, saves masks, renders the figure)
- Masks npz: `reports/github_issue_figs/cell6_combined_masks_{1024,4096}.npz`
- Focal cell coordinates in our WSI: (8886.2, 1478.4) µm  →  WSI px (41817, 6957)
- Source of the 5×6 sweep table: `figs/merge_vs_scale/merge_vs_scale.csv` (from `scripts/merge_vs_scale_test.py`)
