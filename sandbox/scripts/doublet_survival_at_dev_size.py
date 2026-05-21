"""For each of the 68 approved benchmark doublets, run CP-SAM on a small
ROI centered on the doublet, count how many CP-SAM masks cover the original
WSI doublet, and produce visual + CSV verdict so the user can pick which
doublets are useful for dev-size iteration.

A doublet "stays a doublet" at dev size if exactly 1 CP-SAM mask covers the
WSI doublet area at that ROI (i.e. cellpose is *not* breaking the merged pair
into two cells at that ROI size, so the user has a doublet to attack with
their boundary-prior intervention).

Settings (per user choice): augment=True, normalize=False + WSI %iles,
diameter=None, niter=200, cellprob_threshold=-5, flow_threshold=0
— our max-recall pipeline.

ROI sizes (per user choice): 1024 px (218 µm) and 2048 px (435 µm).

Outputs:
    figs/doublet_survival_at_dev_size/
        verdict.csv                          (one row per doublet × size)
        sheet_1024.png                       (8×9 grid, one panel per doublet)
        sheet_2048.png                       (8×9 grid, one panel per doublet)
        masks/<benchmark_id>_<size>.npz     (per-doublet CP-SAM masks)
"""
from __future__ import annotations
import json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cellpose import models
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI    = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S     = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS   = DATA / "cpsam_whole_slide" / "masks.tif"
WSI_PCT_JSON= DATA / "wsi_morphology_percentiles.json"
BENCH_PARQ  = DATA / "benchmark_doublets.parquet"

OUT_DIR = ROOT / "figs" / "doublet_survival_at_dev_size"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "masks").mkdir(exist_ok=True)

PIXEL_SIZE_UM = 0.2125
ROI_SIZES_PX  = [1024, 2048]
MIN_COVER_FRAC = 0.05               # CP-SAM mask counts if it covers ≥5% of WSI doublet area
ZOOM_PX_FOR_PANEL = 250             # display window per panel
CSV_PATH = OUT_DIR / "verdict.csv"

CPSAM_KW = dict(diameter=None, niter=200,
                cellprob_threshold=-5.0, flow_threshold=0.0,
                augment=True, normalize=False)


def crop_centered_um(arr, cx_um, cy_um, size_px):
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


def label_overlay(masks, alpha=0.55, seed=0):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n + 1))])
    shuf = perm[masks]
    rgba = plt.get_cmap("tab20")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


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


def main():
    t_total = time.time()
    print("loading benchmark + WSI…")
    df = pd.read_parquet(BENCH_PARQ)
    df = df[df["approved"] == True].reset_index(drop=True)
    print(f"  {len(df)} approved doublets")

    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S,  key=0)
    wsi_masks = tifffile.imread(WSI_MASKS)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(dapi_n, s18_n, **kw):
        k = dict(CPSAM_KW); k.update(kw)
        img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
        masks, _, _ = m_sam.eval(img, channel_axis=-1, **k)
        return masks.astype(np.int32)

    # Resume support: skip rows already in CSV
    done_keys = set()
    if CSV_PATH.exists():
        prev = pd.read_csv(CSV_PATH)
        done_keys = set(zip(prev["benchmark_id"], prev["roi_size_px"]))
        results = prev.to_dict("records")
        print(f"  resuming: {len(done_keys)} (benchmark_id, size) rows already computed")
    else:
        results = []

    for _, row in df.iterrows():
        bench_id = row["benchmark_id"]
        cps_id   = int(row["cps_id_at_bookmark"])
        cx_um, cy_um = float(row["x_um"]), float(row["y_um"])
        pair    = str(row["lineage_pair"])
        rank    = int(row["rank"])

        for sz in ROI_SIZES_PX:
            if (bench_id, sz) in done_keys: continue
            t0 = time.time()
            dapi_c, bbox = crop_centered_um(dapi_full, cx_um, cy_um, sz)
            s18_c,  _    = crop_centered_um(s18_full,  cx_um, cy_um, sz)
            wsi_c,  _    = crop_centered_um(wsi_masks, cx_um, cy_um, sz)
            focal_region = (wsi_c == cps_id)
            focal_area = int(focal_region.sum())
            dapi_n = norm_with(dapi_c, **wsi_pct["DAPI"])
            s18_n  = norm_with(s18_c,  **wsi_pct["18S"])
            try:
                masks = run(dapi_n, s18_n)
            except Exception as e:
                print(f"  {bench_id:>14s} sz={sz} FAILED: {e}")
                continue
            n_focal, covers = count_focal_masks(masks, focal_region)
            np.savez_compressed(OUT_DIR / "masks" / f"{bench_id}_{sz}.npz",
                                 masks=masks, focal_region=focal_region,
                                 dapi=dapi_c, s18=s18_c, bbox=np.array(bbox))
            dt = time.time() - t0
            rec = dict(
                benchmark_id=bench_id, rank=rank, cps_id=cps_id,
                lineage_pair=pair, x_um=cx_um, y_um=cy_um,
                roi_size_px=sz, roi_size_um=sz * PIXEL_SIZE_UM,
                wsi_focal_area_px=focal_area,
                n_focal=n_focal,
                covers=";".join(f"{c:.3f}" for _, c in covers[:5]),
                stays_doublet=(n_focal == 1),
                runtime_s=round(dt, 1),
            )
            results.append(rec)
            pd.DataFrame(results).to_csv(CSV_PATH, index=False)
            print(f"  {bench_id:>14s} sz={sz}  n_focal={n_focal}  "
                  f"stays={'Y' if n_focal == 1 else 'N'}  ({dt:.1f}s)")

    # ===================== Render summary sheets =====================
    df_out = pd.DataFrame(results)
    print("\n=== summary ===")
    for sz in ROI_SIZES_PX:
        sub = df_out[df_out["roi_size_px"] == sz]
        n_kept = int(sub["stays_doublet"].sum())
        print(f"  ROI={sz}px:  {n_kept}/{len(sub)} doublets stay merged (n_focal=1)")

    n_rows = math.ceil(len(df) / 8)        # 8 cols
    half_zoom = ZOOM_PX_FOR_PANEL // 2

    for sz in ROI_SIZES_PX:
        fig, axes = plt.subplots(n_rows, 8, figsize=(8 * 2.5, n_rows * 2.5))
        axes = axes.flatten()
        sub = df_out[df_out["roi_size_px"] == sz].sort_values("rank").reset_index(drop=True)
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        for i, r in sub.iterrows():
            ax = axes[i]
            bench_id = r["benchmark_id"]
            arr_path = OUT_DIR / "masks" / f"{bench_id}_{sz}.npz"
            if not arr_path.exists(): continue
            arr = np.load(arr_path)
            masks = arr["masks"]; s18 = arr["s18"]; focal = arr["focal_region"]
            H, W = masks.shape
            cy = H // 2; cx = W // 2
            zy0 = max(cy - half_zoom, 0); zy1 = min(cy + half_zoom, H)
            zx0 = max(cx - half_zoom, 0); zx1 = min(cx + half_zoom, W)
            # show normalized 18S + masks
            s18_disp = (s18[zy0:zy1, zx0:zx1] - s18.min()) / max(s18.max() - s18.min(), 1)
            ax.imshow(s18_disp, cmap="gray", interpolation="nearest")
            ax.imshow(label_overlay(masks[zy0:zy1, zx0:zx1]), interpolation="nearest")
            # Red WSI doublet outline
            focal_z = focal[zy0:zy1, zx0:zx1]
            if focal_z.any():
                ax.contour(focal_z.astype(int), levels=[0.5], colors="red", linewidths=1.8)
            color = "green" if r["stays_doublet"] else "tab:red"
            short_pair = r["lineage_pair"].replace(" × ", "×")
            ax.set_title(f"{bench_id[-3:]} r{int(r['rank'])} {short_pair}\n"
                         f"n_focal={int(r['n_focal'])}  "
                         f"{'KEEP' if r['stays_doublet'] else 'split'}",
                         fontsize=7, color=color, pad=2)
        for j in range(len(sub), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f"68 approved doublets — CP-SAM at ROI={sz}px ({sz * PIXEL_SIZE_UM:.0f}µm) "
                     f"with augment=True (our max-recall settings)\n"
                     f"red contour = WSI doublet mask;  KEEP (green title) = n_focal==1;  "
                     f"split (red title) = CP-SAM broke the doublet at this ROI",
                     fontsize=11, y=1.0)
        plt.tight_layout()
        out_fig = OUT_DIR / f"sheet_{sz}.png"
        fig.savefig(out_fig, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"saved {out_fig}")

    print(f"\ntotal runtime: {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
