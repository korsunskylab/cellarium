# sandbox

Archived one-off experiments whose findings now live in code or memory. Not re-run as part of the workflow; kept for reference / to document why specific design decisions were made.

| Notebook | Kernel | Question it answered |
|---|---|---|
| `omnipose_install_check.ipynb` | Python | "Does my conda env actually run cellpose 4 + omnipose side-by-side on Apple Silicon (MPS)?" Yes — both `cpsam` (cellpose 4.1.1) and `cyto2_omni` (cellpose-omni 0.9.1) run on MPS, no conflict from co-installing both packages. |
| `cp_unet_vs_sam.ipynb` | Python | "Does cpsam (SAM backbone) outperform legacy cyto2 (U-Net) on Xenium-like data?" Yes — cpsam is more robust on round/elongated/super-thin/dense-colony shapes. Use cpsam by default; reach for cyto2_omni only for dense bacterial colonies. |
| `cpsam_intensity_split_test.ipynb` | Python | "Does cpsam use intensity contrast or shape cues to split touching cells?" Both, but it relies more on shape than intensity. Two-disk synthetic geometries always split regardless of intensity contrast. This is why the Step-0 cut intervention works — we need to perturb the *shape* signal (via 18S dimming) rather than just create local intensity dips. |
