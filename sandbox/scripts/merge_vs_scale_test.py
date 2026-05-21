"""Find a CP-SAM configuration that gives ROI-scale results consistent with WSI.

GOAL (from user): "find a solution to how do I get consistent results with WSI
segmentation… I just want consistent behavior. I accept the possibility that
there is a minimum ROI size that I need to run CP-SAM on to get WSI-comparable
results."

We test the rank-#6 focal cell, where:
  256/512 px  → 3 masks cover focal region (over-split)
  1024-4096   → 2 masks cover focal region (correctly split fibro vs melanoma)
  WSI         → 1 mask covers focal region (merged doublet — production answer)

Hypotheses (paired conditions):
  H1 norm-shift   → small ROIs renormalize differently. Fix: WSI %iles + normalize=False
  H2 augment      → 4-rot TTA stochastically merges/splits. Fix: augment=False
  H3 tile-stitch  → more tiles → more cross-tile merging. Fix: bsize=512 (fewer tiles)
  H4 auto-diameter→ diameter heuristic varies with ROI size. Fix: explicit diameter
  H5 cell-context → real surrounding cells modulate flows near focal. Test: REFLECT-PAD
                    small ROI to 4096 (adds image area but no new real cells; if behavior
                    matches the large-ROI version, the cause is image-size/tiling, not
                    real context)
  H6 image-size   → cellpose runs different code paths for small vs large. Tested by
                    reflect-padding (H5) — if padded 256→4096 matches native-4096, it's
                    size-not-context.

Conditions (per size):
  (a) DEFAULT      : augment=True,  normalize=True,  diameter=None,  bsize=256  (recreates bug)
  (b) WSI-NORM     : augment=True,  normalize=False, diameter=None,  bsize=256  (H1)
  (c) NO-AUG       : augment=False, normalize=True,  diameter=None,  bsize=256  (H2)
  (d) BOTH         : augment=False, normalize=False, diameter=None,  bsize=256  (H1+H2)
  (e) DIAMETER=70  : augment=True,  normalize=True,  diameter=70,    bsize=256  (H4)

Extra runs (one-offs, at size=4096 only unless noted):
  (f) BSIZE=512    : augment=False, normalize=True,  diameter=None,  bsize=512  (H3)
  (g) PAD_TO_4096  : take small crop, reflect-pad to 4096×4096, run with (a) and (d) (H5)

Sizes tested in main grid:   256, 512, 1024, 2048, 4096
Size 8192 (one-shot for a, d): confirms whether trend continues toward WSI

Decision criteria (final report):
  - If any condition gives n_focal=1 (matches WSI) across ALL sizes → that's the fix
  - If any condition gives the SAME n_focal across all sizes (even if ≠ 1) → that's
    "consistent behavior" the user can dev against
  - If padding rescues small ROI to match large-ROI behavior → size/tiling is the cause,
    use padding before any small-ROI dev work
  - If nothing flattens the curve → report minimum ROI size that gives WSI-matching answer
"""
from __future__ import annotations
import json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS = DATA / "cpsam_whole_slide" / "masks.tif"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = ROOT / "figs" / "merge_vs_scale"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
CX_UM, CY_UM = 8886.2, 1478.4
CPS_FOCAL = 45463
MAIN_SIZES_PX = [256, 512, 1024, 2048, 4096]
EXTRA_SIZES_PX = [8192]
PAD_FROM_SIZES = [256, 1024]
PAD_TO_SIZE = 4096
MIN_COVER_FRAC = 0.05
CPSAM_KW_BASE = dict(diameter=None, niter=200,
                     cellprob_threshold=-5.0, flow_threshold=0.0)

CSV_PATH = OUT_DIR / "merge_vs_scale.csv"
JSON_PATH = OUT_DIR / "results.json"


def crop_centered(arr, cx_um, cy_um, size_px):
    cx_px = int(round(cx_um / PIXEL_SIZE_UM))
    cy_px = int(round(cy_um / PIXEL_SIZE_UM))
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    a = arr.astype(np.float32)
    return np.clip((a - q_lo) / max(q_hi - q_lo, 1e-6), 0, 1).astype(np.float32)


def norm_local(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def count_focal_masks(masks, focal_region, min_cover=MIN_COVER_FRAC):
    focal_area = int(focal_region.sum())
    if focal_area == 0: return 0, []
    labels_in_focal = np.unique(masks[focal_region])
    labels_in_focal = labels_in_focal[labels_in_focal != 0]
    out = []
    for L in labels_in_focal:
        cov = int(((masks == L) & focal_region).sum()) / focal_area
        if cov >= min_cover:
            out.append((int(L), float(cov)))
    out.sort(key=lambda kv: -kv[1])
    return len(out), out


def append_result(rec, all_results):
    all_results.append(rec)
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    JSON_PATH.write_text(json.dumps({
        "focal_cell_um":  [CX_UM, CY_UM],
        "wsi_cps_focal":  CPS_FOCAL,
        "results":        all_results,
    }, indent=2))


def get_image_for_condition(condition, dapi_c, s18_c, wsi_pct):
    """Return (dapi_in, s18_in) normalized for the named condition."""
    if condition in ("default", "no_aug", "diameter_70", "bsize_512"):
        return norm_local(dapi_c), norm_local(s18_c)
    elif condition in ("wsi_norm", "both"):
        return (norm_with(dapi_c, **wsi_pct["DAPI"]),
                norm_with(s18_c,  **wsi_pct["18S"]))
    raise ValueError(condition)


def cpsam_kwargs_for(condition):
    base = dict(CPSAM_KW_BASE)
    if condition == "default":
        base.update(augment=True,  normalize=True)
    elif condition == "wsi_norm":
        base.update(augment=True,  normalize=False)
    elif condition == "no_aug":
        base.update(augment=False, normalize=True)
    elif condition == "both":
        base.update(augment=False, normalize=False)
    elif condition == "diameter_70":
        base.update(augment=True,  normalize=True, diameter=70)
    elif condition == "bsize_512":
        base.update(augment=False, normalize=True, bsize=512)
    else:
        raise ValueError(condition)
    return base


def pad_reflect_centered(arr, target_size):
    """Reflect-pad 2-D array so its center stays the center of the larger output."""
    src = arr.shape[-1]
    pad_total = target_size - src
    if pad_total <= 0:
        return arr
    pad_lo = pad_total // 2
    pad_hi = pad_total - pad_lo
    if arr.ndim == 2:
        return np.pad(arr, ((pad_lo, pad_hi), (pad_lo, pad_hi)), mode="reflect")
    return np.pad(arr, ((0, 0), (pad_lo, pad_hi), (pad_lo, pad_hi)), mode="reflect")


def pad_zeros_centered(arr, target_size):
    """Zero-pad mask to put original center at center of larger output."""
    src = arr.shape[-1]
    pad_total = target_size - src
    if pad_total <= 0:
        return arr
    pad_lo = pad_total // 2
    pad_hi = pad_total - pad_lo
    return np.pad(arr, ((pad_lo, pad_hi), (pad_lo, pad_hi)), mode="constant")


def main():
    t_total = time.time()
    print("loading WSI imagery + masks…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_masks_full = tifffile.imread(WSI_MASKS)
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    print(f"  DAPI shape={dapi_full.shape} dtype={dapi_full.dtype}")
    print(f"  WSI mask shape={wsi_masks_full.shape}, focal area in WSI mask: "
          f"{int((wsi_masks_full == CPS_FOCAL).sum())} px")

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(dapi, s18, **kw):
        k = dict(CPSAM_KW_BASE); k.update(kw)
        img = np.stack([dapi, s18], axis=-1).astype(np.float32)
        masks, _, _ = m_sam.eval(img, channel_axis=-1, **k)
        return masks.astype(np.int32)

    all_results = []

    # ============================================================
    # PHASE 1: Main grid — 5 conditions × MAIN_SIZES_PX
    # ============================================================
    conditions_main = ["default", "wsi_norm", "no_aug", "both", "diameter_70"]
    print(f"\n{'='*72}\nPHASE 1: main grid — {len(conditions_main)} conds × {len(MAIN_SIZES_PX)} sizes\n{'='*72}")
    for sp in MAIN_SIZES_PX:
        dapi_c, bbox = crop_centered(dapi_full, CX_UM, CY_UM, sp)
        s18_c,  _    = crop_centered(s18_full,  CX_UM, CY_UM, sp)
        wsi_c,  _    = crop_centered(wsi_masks_full, CX_UM, CY_UM, sp)
        focal_region = (wsi_c == CPS_FOCAL)
        focal_area = int(focal_region.sum())
        print(f"\n--- size={sp}px ({sp*PIXEL_SIZE_UM:.0f} µm), focal {focal_area} px ---")
        for cond in conditions_main:
            try:
                dapi_in, s18_in = get_image_for_condition(cond, dapi_c, s18_c, wsi_pct)
                kw = cpsam_kwargs_for(cond)
                t0 = time.time()
                masks = run(dapi_in, s18_in, **kw)
                dt = time.time() - t0
                n_total = int(masks.max())
                n_focal, covers = count_focal_masks(masks, focal_region)
                rec = dict(
                    phase="main_grid", size_px=sp, size_um=sp * PIXEL_SIZE_UM,
                    condition=cond, pad_from_px=None,
                    n_total=n_total, n_focal=n_focal,
                    covers=";".join(f"{c:.3f}" for _, c in covers[:5]),
                    runtime_s=round(dt, 1),
                )
                append_result(rec, all_results)
                print(f"  {cond:<14s} n_total={n_total:>5d}  n_focal={n_focal}  "
                      f"covers=[{', '.join(f'{c*100:.0f}%' for _, c in covers[:5])}]  ({dt:.1f}s)")
            except Exception as e:
                print(f"  {cond:<14s} FAILED: {e}")
                traceback.print_exc()

    # ============================================================
    # PHASE 2: size=8192 with default + both (extend toward WSI scale)
    # ============================================================
    print(f"\n{'='*72}\nPHASE 2: size=8192 with default + both\n{'='*72}")
    for sp in EXTRA_SIZES_PX:
        dapi_c, _ = crop_centered(dapi_full, CX_UM, CY_UM, sp)
        s18_c,  _ = crop_centered(s18_full,  CX_UM, CY_UM, sp)
        wsi_c,  _ = crop_centered(wsi_masks_full, CX_UM, CY_UM, sp)
        focal_region = (wsi_c == CPS_FOCAL)
        focal_area = int(focal_region.sum())
        print(f"\n--- size={sp}px ({sp*PIXEL_SIZE_UM:.0f} µm), focal {focal_area} px ---")
        for cond in ["default", "both"]:
            try:
                dapi_in, s18_in = get_image_for_condition(cond, dapi_c, s18_c, wsi_pct)
                kw = cpsam_kwargs_for(cond)
                t0 = time.time()
                masks = run(dapi_in, s18_in, **kw)
                dt = time.time() - t0
                n_total = int(masks.max())
                n_focal, covers = count_focal_masks(masks, focal_region)
                rec = dict(
                    phase="extend_8192", size_px=sp, size_um=sp * PIXEL_SIZE_UM,
                    condition=cond, pad_from_px=None,
                    n_total=n_total, n_focal=n_focal,
                    covers=";".join(f"{c:.3f}" for _, c in covers[:5]),
                    runtime_s=round(dt, 1),
                )
                append_result(rec, all_results)
                print(f"  {cond:<14s} n_total={n_total:>5d}  n_focal={n_focal}  "
                      f"covers=[{', '.join(f'{c*100:.0f}%' for _, c in covers[:5])}]  ({dt:.1f}s)")
            except Exception as e:
                print(f"  {cond:<14s} FAILED: {e}")
                traceback.print_exc()

    # ============================================================
    # PHASE 3: PADDING RESCUE — pad small ROI to 4096 via reflection
    # ============================================================
    print(f"\n{'='*72}\nPHASE 3: padding rescue — reflect-pad small ROI to {PAD_TO_SIZE}px\n{'='*72}")
    for src_sp in PAD_FROM_SIZES:
        dapi_c, _ = crop_centered(dapi_full, CX_UM, CY_UM, src_sp)
        s18_c,  _ = crop_centered(s18_full,  CX_UM, CY_UM, src_sp)
        wsi_c,  _ = crop_centered(wsi_masks_full, CX_UM, CY_UM, src_sp)
        # Pad image with reflection, pad mask with zeros (no real cells outside crop)
        dapi_p = pad_reflect_centered(dapi_c, PAD_TO_SIZE)
        s18_p  = pad_reflect_centered(s18_c,  PAD_TO_SIZE)
        wsi_p  = pad_zeros_centered(wsi_c,    PAD_TO_SIZE)
        focal_region = (wsi_p == CPS_FOCAL)
        focal_area = int(focal_region.sum())
        print(f"\n--- pad {src_sp}→{PAD_TO_SIZE}px ({src_sp*PIXEL_SIZE_UM:.0f}→"
              f"{PAD_TO_SIZE*PIXEL_SIZE_UM:.0f} µm), focal {focal_area} px ---")
        for cond in ["default", "both"]:
            try:
                # NOTE: pre-normalized image arrays must use the ALREADY-PADDED tile,
                # because normalize_local depends on global %iles over the input.
                if cond == "default":
                    dapi_in, s18_in = norm_local(dapi_p), norm_local(s18_p)
                else:  # both → uses WSI percentiles (no local normalization)
                    dapi_in = norm_with(dapi_p, **wsi_pct["DAPI"])
                    s18_in  = norm_with(s18_p,  **wsi_pct["18S"])
                kw = cpsam_kwargs_for(cond)
                t0 = time.time()
                masks = run(dapi_in, s18_in, **kw)
                dt = time.time() - t0
                n_total = int(masks.max())
                n_focal, covers = count_focal_masks(masks, focal_region)
                rec = dict(
                    phase="pad_rescue", size_px=PAD_TO_SIZE,
                    size_um=PAD_TO_SIZE * PIXEL_SIZE_UM,
                    condition=cond, pad_from_px=src_sp,
                    n_total=n_total, n_focal=n_focal,
                    covers=";".join(f"{c:.3f}" for _, c in covers[:5]),
                    runtime_s=round(dt, 1),
                )
                append_result(rec, all_results)
                print(f"  pad{src_sp}→{PAD_TO_SIZE} {cond:<10s} n_total={n_total:>5d}  "
                      f"n_focal={n_focal}  covers=[{', '.join(f'{c*100:.0f}%' for _, c in covers[:5])}]  ({dt:.1f}s)")
            except Exception as e:
                print(f"  pad{src_sp}→{PAD_TO_SIZE} {cond:<10s} FAILED: {e}")
                traceback.print_exc()

    # ============================================================
    # PHASE 4: BSIZE=512 at size=4096 (test tile-stitching hypothesis)
    # ============================================================
    print(f"\n{'='*72}\nPHASE 4: bsize=512 at size=4096 (H3 tile-stitch)\n{'='*72}")
    sp = 4096
    dapi_c, _ = crop_centered(dapi_full, CX_UM, CY_UM, sp)
    s18_c,  _ = crop_centered(s18_full,  CX_UM, CY_UM, sp)
    wsi_c,  _ = crop_centered(wsi_masks_full, CX_UM, CY_UM, sp)
    focal_region = (wsi_c == CPS_FOCAL)
    for cond in ["bsize_512"]:
        try:
            dapi_in, s18_in = get_image_for_condition(cond, dapi_c, s18_c, wsi_pct)
            kw = cpsam_kwargs_for(cond)
            t0 = time.time()
            masks = run(dapi_in, s18_in, **kw)
            dt = time.time() - t0
            n_total = int(masks.max())
            n_focal, covers = count_focal_masks(masks, focal_region)
            rec = dict(
                phase="bsize_test", size_px=sp, size_um=sp * PIXEL_SIZE_UM,
                condition=cond, pad_from_px=None,
                n_total=n_total, n_focal=n_focal,
                covers=";".join(f"{c:.3f}" for _, c in covers[:5]),
                runtime_s=round(dt, 1),
            )
            append_result(rec, all_results)
            print(f"  {cond:<14s} n_total={n_total:>5d}  n_focal={n_focal}  "
                  f"covers=[{', '.join(f'{c*100:.0f}%' for _, c in covers[:5])}]  ({dt:.1f}s)")
        except Exception as e:
            print(f"  {cond:<14s} FAILED: {e}")
            traceback.print_exc()

    # ============================================================
    # Final plots
    # ============================================================
    df = pd.DataFrame(all_results)
    print(f"\n\nFINAL RESULTS:\n{df.to_string(index=False)}")

    # Plot 1: main grid n_focal vs size
    main_df = df[df["phase"].isin(["main_grid", "extend_8192"])]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {"default": "#1f77b4", "wsi_norm": "#ff7f0e", "no_aug": "#2ca02c",
              "both": "#d62728", "diameter_70": "#9467bd"}
    for cond in main_df["condition"].unique():
        sub = main_df[main_df["condition"] == cond].sort_values("size_um")
        c = colors.get(cond, "gray")
        axes[0].plot(sub["size_um"], sub["n_focal"], "o-", label=cond, color=c,
                     linewidth=2, markersize=8)
        axes[1].plot(sub["size_um"], sub["n_total"], "o-", label=cond, color=c,
                     linewidth=2, markersize=8)
    axes[0].axhline(1, color="black", linestyle=":", alpha=0.6, label="WSI → 1 cell")
    axes[0].set_xscale("log"); axes[0].set_xlabel("ROI size (µm)")
    axes[0].set_ylabel("# CP-SAM masks covering focal WSI cell")
    axes[0].set_title("THE BUG: monotonic 3→2→1 merging across ROI size\n"
                       "(if any line flattens, that condition is the fix)")
    axes[0].set_ylim(0, max(main_df["n_focal"].max() + 1, 4))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xscale("log"); axes[1].set_xlabel("ROI size (µm)")
    axes[1].set_ylabel("total cells in ROI")
    axes[1].set_title("Total cells detected (scales with area)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_main_grid.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Plot 2: padding rescue
    pad_df = df[df["phase"] == "pad_rescue"]
    if not pad_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for cond in ["default", "both"]:
            sub_main = main_df[main_df["condition"] == cond].sort_values("size_um")
            ax.plot(sub_main["size_um"], sub_main["n_focal"], "o-",
                    label=f"{cond} (native ROI)", color=colors[cond],
                    linewidth=2, markersize=8, alpha=0.7)
            sub_pad = pad_df[pad_df["condition"] == cond]
            for _, r in sub_pad.iterrows():
                ax.plot(r["pad_from_px"] * PIXEL_SIZE_UM, r["n_focal"], "*",
                        color=colors[cond], markersize=18, markeredgecolor="black",
                        markeredgewidth=1.5,
                        label=f"{cond} (pad {int(r['pad_from_px'])}→{PAD_TO_SIZE}px)")
        ax.axhline(1, color="black", linestyle=":", alpha=0.6, label="WSI → 1 cell")
        ax.set_xscale("log")
        ax.set_xlabel("ROI size (µm) — stars = padded crop (image area = "
                      f"{PAD_TO_SIZE * PIXEL_SIZE_UM:.0f}µm)")
        ax.set_ylabel("# CP-SAM masks covering focal WSI cell")
        ax.set_title("Padding rescue test: if star (padded) matches the dot at "
                     f"{PAD_TO_SIZE*PIXEL_SIZE_UM:.0f}µm,\n"
                     "image SIZE/tiling is the cause (real cell context is not needed)")
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "fig2_padding_rescue.png", dpi=140, bbox_inches="tight")
        plt.close()

    print(f"\nfigures + table saved to {OUT_DIR}")
    print(f"total runtime: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
