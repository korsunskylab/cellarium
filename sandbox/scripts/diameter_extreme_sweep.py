"""Diameter sweep with extreme values + per-lineage mask quality, in microns.

Tests CP-SAM at 7 diameters spanning well below and well above biological cell sizes:
   diameter (px)  → diameter (µm)  → biological scale
       5           1.06             sub-nuclear (absurd)
      15           3.19             smaller than smallest cell (T cells ~6-10 µm)
      30           6.38             T cell scale  (cellpose-SAM default-ish reference)
      70          14.88             fibroblast / model-calibrated for our 70-px cells
     120          25.50             large melanoma
     200          42.50             beyond all biology
     500         106.25             absurdly large

For each diameter × lineage (WSI-assigned via anchor overlap):
  1. mean cellprob inside WSI cell mask region — measures *model confidence*
  2. n_cells detected at that diameter and assigned to that lineage — measures *detection rate*
  3. mean cell area in µm² — measures *mask size* (does big diameter merge small cells into big blobs?)

Same 500 µm × 500 µm test region as the multiscale notebook (auto-selected
for 3-lineage density: T-cells + Fibroblasts + Melanoma).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import zarr
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS = DATA / "cpsam_whole_slide" / "masks.tif"
TX_ZARR  = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR  = ROOT / "figs" / "diameter_extreme_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
CROP_UM       = 500.0
CROP_PX       = int(round(CROP_UM / PIXEL_SIZE_UM))   # 2353 px

DIAMETERS_PX = [15, 30, 70, 120, 200, 500]   # d=5 dropped (infeasible: ~5M sub-cellular masks)

LINEAGES = ["Melanoma", "Myeloid", "Tcell", "Plasma", "Fibroblast", "Endothelial", "Keratinocyte"]
LINEAGE_FOCUS = ["Tcell", "Fibroblast", "Melanoma"]
MIN_ANCHORS = 5
CPSAM_KW = dict(niter=200, cellprob_threshold=-5.0, flow_threshold=0.0, augment=True)


def crop_at(arr, cx_um, cy_um, size_px):
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


def find_3lineage_region(gene_names, gname_to_lin):
    """Replicates the region-search from the multiscale notebook."""
    WSI_H_PX, WSI_W_PX = 20294, 42748
    WSI_H_UM, WSI_W_UM = WSI_H_PX * PIXEL_SIZE_UM, WSI_W_PX * PIXEL_SIZE_UM
    n_x = int(WSI_W_UM // CROP_UM); n_y = int(WSI_H_UM // CROP_UM)
    counts = np.zeros((n_y, n_x, len(LINEAGE_FOCUS)), dtype=np.int32)
    lin_focus_idx = {L: i for i, L in enumerate(LINEAGE_FOCUS)}
    zroot = zarr.open(TX_ZARR, mode="r")
    grid0 = zroot["grids"]["0"]
    for key in grid0.keys():
        tile = grid0[key]
        loc = tile["location"][:]; gid = tile["gene_identity"][:].squeeze(-1)
        qv  = tile["quality_score"][:].squeeze(-1); valid = tile["valid"][:].squeeze(-1)
        keep = (valid > 0) & (qv >= 20)
        if not keep.any(): continue
        x = loc[keep, 0]; y = loc[keep, 1]; g = gid[keep]
        for xi, yi, gi in zip(x, y, g):
            L = gname_to_lin.get(gene_names[gi])
            if L not in lin_focus_idx: continue
            gx = int(xi // CROP_UM); gy = int(yi // CROP_UM)
            if 0 <= gy < n_y and 0 <= gx < n_x:
                counts[gy, gx, lin_focus_idx[L]] += 1
    min_of_three = counts.min(axis=-1)
    gy_best, gx_best = np.unravel_index(min_of_three.argmax(), min_of_three.shape)
    return gx_best * CROP_UM + CROP_UM/2, gy_best * CROP_UM + CROP_UM/2, counts[gy_best, gx_best]


def load_anchored_tx_in_crop(zroot, gene_names, gname_to_lin, x0, y0, x1, y1):
    TILE = 250.0
    i0 = int(x0 // TILE); i1 = int(x1 // TILE)
    j0 = int(y0 // TILE); j1 = int(y1 // TILE)
    grid0 = zroot["grids"]["0"]
    xs, ys, lins = [], [], []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            key = f"{i},{j}"
            if key not in grid0: continue
            tile = grid0[key]
            loc = tile["location"][:]; gid = tile["gene_identity"][:].squeeze(-1)
            qv  = tile["quality_score"][:].squeeze(-1); valid = tile["valid"][:].squeeze(-1)
            keep = (valid > 0) & (qv >= 20)
            if not keep.any(): continue
            xc = loc[keep, 0]; yc = loc[keep, 1]; gi = gid[keep]
            in_crop = (xc >= x0) & (xc < x1) & (yc >= y0) & (yc < y1)
            for xi, yi, g in zip(xc[in_crop], yc[in_crop], gi[in_crop]):
                L = gname_to_lin.get(gene_names[g])
                if L and L != "ambiguous":
                    xs.append(xi); ys.append(yi); lins.append(L)
    return pd.DataFrame({"x_um": xs, "y_um": ys, "lineage": lins})


def assign_lineages_and_purity(masks, anch_df, x0_um, y0_um, min_anchors=MIN_ANCHORS):
    """Per-cell dominant lineage AND purity (fraction of anchors belonging to dominant lineage).

    Purity is the INDEPENDENT quality metric — it uses biology (transcript lineage
    assignments) and does NOT depend on cellpose's cellprob/confidence.
    A cleanly-segmented cell has anchors mostly of one lineage (purity ≈ 1).
    A merged or fragmented cell has anchors mixed across lineages (purity ≈ 1/K).

    Returns dict {cell_id: {'lineage', 'purity', 'n_anchors', 'n_lineages_present'}}.
    Cells with < min_anchors total → "low_n" lineage and NaN purity.
    """
    if masks.max() == 0 or len(anch_df) == 0:
        return {}
    py = np.rint((anch_df["y_um"].values - y0_um) / PIXEL_SIZE_UM).astype(int)
    px = np.rint((anch_df["x_um"].values - x0_um) / PIXEL_SIZE_UM).astype(int)
    H, W = masks.shape
    valid = (py >= 0) & (py < H) & (px >= 0) & (px < W)
    py = py[valid]; px = px[valid]; lin = anch_df["lineage"].values[valid]
    cell_ids = masks[py, px]
    keep = cell_ids > 0
    df_a = pd.DataFrame({"cell_id": cell_ids[keep], "lineage": lin[keep]})
    if len(df_a) == 0: return {}
    counts = df_a.groupby(["cell_id", "lineage"]).size().unstack(fill_value=0)
    out = {}
    for cid in counts.index:
        row = counts.loc[cid]
        total = int(row.sum())
        if total < min_anchors:
            out[int(cid)] = {"lineage": "low_n", "purity": float("nan"),
                              "n_anchors": total, "n_lineages_present": int((row > 0).sum())}
        else:
            dom = row.idxmax(); dom_count = int(row.max())
            out[int(cid)] = {"lineage": dom, "purity": dom_count / total,
                              "n_anchors": total, "n_lineages_present": int((row > 0).sum())}
    return out


def assign_lineages(masks, anch_df, x0_um, y0_um, min_anchors=MIN_ANCHORS):
    """Back-compat shim — returns {cell_id: lineage_str}."""
    full = assign_lineages_and_purity(masks, anch_df, x0_um, y0_um, min_anchors)
    return {cid: v["lineage"] for cid, v in full.items()}


def main():
    t0 = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_masks_full = tifffile.imread(WSI_MASKS)
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())

    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    gname_to_lin = gl.to_dict()

    print("finding 3-lineage region…")
    cx_um, cy_um, lc = find_3lineage_region(gene_names, gname_to_lin)
    print(f"  region centroid: ({cx_um:.0f}, {cy_um:.0f}) µm, "
          f"Tcell/Fibro/Mel counts: {lc.tolist()}")

    dapi_c, bbox = crop_at(dapi_full, cx_um, cy_um, CROP_PX)
    s18_c,  _    = crop_at(s18_full,  cx_um, cy_um, CROP_PX)
    wsi_c,  _    = crop_at(wsi_masks_full, cx_um, cy_um, CROP_PX)
    dapi_n = norm_with(dapi_c, **wsi_pct["DAPI"])
    s18_n  = norm_with(s18_c,  **wsi_pct["18S"])
    x0_um, y0_um = bbox[0] * PIXEL_SIZE_UM, bbox[1] * PIXEL_SIZE_UM
    x1_um, y1_um = bbox[2] * PIXEL_SIZE_UM, bbox[3] * PIXEL_SIZE_UM

    print("loading anchored tx in crop…")
    anch_df = load_anchored_tx_in_crop(zroot, gene_names, gname_to_lin, x0_um, y0_um, x1_um, y1_um)
    print(f"  {len(anch_df)} anchored tx")

    print("WSI lineage assignment…")
    wsi_int = wsi_c.astype(np.int32)
    wsi_cell_lin = assign_lineages(wsi_int, anch_df, x0_um, y0_um)
    wsi_lineage_counts = pd.Series(list(wsi_cell_lin.values())).value_counts().to_dict()
    print(f"  WSI cells per lineage: {wsi_lineage_counts}")

    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    per_diam = []
    for d_px in DIAMETERS_PX:
        d_um = d_px * PIXEL_SIZE_UM
        print(f"\n[d={d_px} px = {d_um:.2f} µm] running CP-SAM…")
        t_run = time.time()
        img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
        masks, flows, _ = m_sam.eval(img, channel_axis=-1, diameter=d_px, **CPSAM_KW)
        dt = time.time() - t_run
        masks = masks.astype(np.int32)
        cellprob = flows[2].astype(np.float32)
        n_cells = int(masks.max())
        # Per-cell areas (px → µm²)
        cell_areas_px = np.bincount(masks.ravel())[1:] if n_cells > 0 else np.array([])
        cell_areas_um2 = cell_areas_px * (PIXEL_SIZE_UM ** 2)
        # Per-cell lineage assignment + purity (independent quality metric)
        cell_full = assign_lineages_and_purity(masks, anch_df, x0_um, y0_um)
        cell_lin = {cid: v["lineage"] for cid, v in cell_full.items()}
        # WSI-anchored cellprob: sample cellprob at WSI cell mask regions, per WSI lineage
        per_diam.append({
            "diameter_px":      d_px,
            "diameter_um":      float(d_um),
            "runtime_s":        dt,
            "n_cells":          n_cells,
            "masks":            masks,
            "cellprob":         cellprob,
            "cell_lineages":    cell_lin,
            "cell_full":        cell_full,        # purity etc.
            "cell_areas_um2":   cell_areas_um2,
        })
        print(f"  n_cells={n_cells}, runtime={dt:.1f}s, "
              f"mean cell area = {cell_areas_um2.mean() if n_cells > 0 else 0:.1f} µm²")

    # Per-diameter, per-lineage analytics
    rows_per_lin = []
    for r in per_diam:
        masks = r["masks"]; cellprob = r["cellprob"]
        cell_lin = r["cell_lineages"]; cell_full = r["cell_full"]
        # WSI-anchored cellprob per lineage
        for L in LINEAGES:
            wsi_cells_L = [cid for cid, l in wsi_cell_lin.items() if l == L]
            if not wsi_cells_L:
                continue
            cellprobs_at_L = []
            for wcid in wsi_cells_L:
                region = (wsi_int == wcid)
                if region.any():
                    cellprobs_at_L.append(float(cellprob[region].mean()))
            # Cells detected at this diameter assigned to this lineage
            detected_cids = [cid for cid, ll in cell_lin.items() if ll == L]
            areas_for_L = ([float(r["cell_areas_um2"][cid - 1]) for cid in detected_cids
                            if 0 < cid <= len(r["cell_areas_um2"])])
            purities_for_L = [cell_full[cid]["purity"] for cid in detected_cids
                              if cid in cell_full and not np.isnan(cell_full[cid]["purity"])]
            n_lineages_L = [cell_full[cid]["n_lineages_present"] for cid in detected_cids
                            if cid in cell_full]
            rows_per_lin.append({
                "diameter_px":              r["diameter_px"],
                "diameter_um":              r["diameter_um"],
                "lineage":                  L,
                "n_wsi_cells":              len(wsi_cells_L),
                "n_detected_assigned":      len(detected_cids),
                "mean_cellprob_at_wsi":     float(np.mean(cellprobs_at_L)) if cellprobs_at_L else np.nan,
                "mean_area_um2":            float(np.mean(areas_for_L)) if areas_for_L else np.nan,
                "median_area_um2":          float(np.median(areas_for_L)) if areas_for_L else np.nan,
                "mean_purity":              float(np.mean(purities_for_L)) if purities_for_L else np.nan,
                "mean_n_lineages_per_cell": float(np.mean(n_lineages_L)) if n_lineages_L else np.nan,
            })
    df = pd.DataFrame(rows_per_lin)
    df.to_csv(OUT_DIR / "per_lineage_per_diameter.csv", index=False)
    print(f"\nresults DataFrame saved ({len(df)} rows)")
    print(df.to_string(index=False))

    # ── Figure: 6-panel summary including PURITY (independent quality metric) ──
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    lineage_palette = plt.get_cmap("tab10")
    lineages_with_data = [L for L in LINEAGES if (df["lineage"] == L).sum() > 0
                          and (df[df["lineage"] == L]["n_wsi_cells"].iloc[0] >= 3)]

    # Panel 1: Detection count
    for i, L in enumerate(lineages_with_data):
        sub = df[df["lineage"] == L]
        axes[0, 0].plot(sub["diameter_um"], sub["n_detected_assigned"], "o-",
                         label=f"{L} (n_WSI={int(sub['n_wsi_cells'].iloc[0])})",
                         color=lineage_palette(i))
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("diameter (µm) — log scale")
    axes[0, 0].set_ylabel("# cells detected & assigned this lineage")
    axes[0, 0].set_title("Detection count by lineage × diameter")
    axes[0, 0].axvspan(5, 30, alpha=0.10, color="green", label="biological cell-size range")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    # Panel 2: Mean cellprob (WSI-anchored)
    for i, L in enumerate(lineages_with_data):
        sub = df[df["lineage"] == L]
        axes[0, 1].plot(sub["diameter_um"], sub["mean_cellprob_at_wsi"], "o-",
                         label=L, color=lineage_palette(i))
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("diameter (µm) — log scale")
    axes[0, 1].set_ylabel("mean cellprob inside WSI cell region")
    axes[0, 1].set_title("Cellprob (WSI-anchored) by lineage × diameter")
    axes[0, 1].axvspan(5, 30, alpha=0.10, color="green")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # Panel 3: Mean detected cell area (µm²)
    for i, L in enumerate(lineages_with_data):
        sub = df[df["lineage"] == L].dropna(subset=["mean_area_um2"])
        axes[1, 0].plot(sub["diameter_um"], sub["mean_area_um2"], "o-",
                         label=L, color=lineage_palette(i))
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("diameter (µm) — log scale")
    axes[1, 0].set_ylabel("mean mask area (µm²) — log scale")
    axes[1, 0].set_title("Detected cell area by lineage × diameter\n(if mean area grows with diameter → saturation/merging confirmed)")
    axes[1, 0].axvspan(5, 30, alpha=0.10, color="green")
    # Reference biological cell areas
    for area_um2, name in [(28, "T-cell typical ~28 µm²"),
                            (177, "Fibroblast ~177 µm²"),
                            (314, "Melanoma ~314 µm²")]:
        axes[1, 0].axhline(area_um2, color="grey", linewidth=0.8, alpha=0.5, linestyle=":")
        axes[1, 0].text(5, area_um2 * 1.1, name, fontsize=7, color="grey")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    # Panel 4: PURITY by lineage (independent quality metric — not from cellpose)
    for i, L in enumerate(lineages_with_data):
        sub = df[df["lineage"] == L].dropna(subset=["mean_purity"])
        axes[1, 1].plot(sub["diameter_um"], sub["mean_purity"], "o-",
                         label=L, color=lineage_palette(i))
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("diameter (µm) — log scale")
    axes[1, 1].set_ylabel("mean anchor purity (dominant lineage / total)")
    axes[1, 1].set_title("INDEPENDENT QUALITY METRIC: anchor purity per cell\n"
                          "high purity ≈ 1: cell contains one lineage; low ≈ 1/K: mixed/merged")
    axes[1, 1].axvspan(5, 30, alpha=0.10, color="green")
    axes[1, 1].axhline(1.0, color="grey", linewidth=0.5, alpha=0.5)
    axes[1, 1].axhline(1/7, color="grey", linewidth=0.5, alpha=0.5, linestyle=":")
    axes[1, 1].text(5, 1/7 + 0.02, "uniform across 7 lineages = 1/7", fontsize=7, color="grey")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)

    # Panel 5: Mean # lineages present per cell (related quality)
    for i, L in enumerate(lineages_with_data):
        sub = df[df["lineage"] == L].dropna(subset=["mean_n_lineages_per_cell"])
        axes[1, 2].plot(sub["diameter_um"], sub["mean_n_lineages_per_cell"], "o-",
                         label=L, color=lineage_palette(i))
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel("diameter (µm) — log scale")
    axes[1, 2].set_ylabel("mean # distinct lineages per detected cell")
    axes[1, 2].set_title("# lineages per cell (lower = cleaner; higher = mixed/merged)\n"
                          "should be ≈1 for well-segmented cells")
    axes[1, 2].axvspan(5, 30, alpha=0.10, color="green")
    axes[1, 2].axhline(1, color="grey", linewidth=0.5, alpha=0.5)
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(alpha=0.3)

    # Add a runtime panel at top-right (replacing what was there before)
    # Note: axes[0, 2] now holds runtime + total count
    diameters_um_ordered = [r["diameter_um"] for r in per_diam]
    runtimes = [r["runtime_s"] for r in per_diam]
    n_cells_total = [r["n_cells"] for r in per_diam]
    ax2 = axes[0, 2]
    ax2.plot(diameters_um_ordered, runtimes, "o-", color="black", label="CP-SAM runtime (s)")
    ax2.set_xscale("log")
    ax2.set_xlabel("diameter (µm) — log scale")
    ax2.set_ylabel("CP-SAM runtime (s)", color="black")
    ax_r = ax2.twinx()
    ax_r.plot(diameters_um_ordered, n_cells_total, "s-", color="red", label="total n_cells")
    ax_r.set_ylabel("total cells detected in 500 µm crop", color="red")
    ax2.set_title("Runtime + total cell count by diameter")
    ax2.axvspan(5, 30, alpha=0.10, color="green")
    ax2.grid(alpha=0.3)

    plt.suptitle(f"CP-SAM diameter sweep with extreme values (500×500 µm crop, "
                  f"{len([k for k,v in wsi_cell_lin.items() if v != 'low_n'])} WSI cells with lineage)",
                  y=1.001)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_diameter_extreme_sweep.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── Save a JSON summary too ──
    summary = {
        "region_centroid_um": [float(cx_um), float(cy_um)],
        "crop_bbox_wsi_px": list(bbox),
        "n_anchored_tx_in_crop": int(len(anch_df)),
        "wsi_lineage_counts": {k: int(v) for k, v in wsi_lineage_counts.items()},
        "diameters_tested_px": DIAMETERS_PX,
        "diameters_tested_um": [d * PIXEL_SIZE_UM for d in DIAMETERS_PX],
        "per_diameter_summary": [
            {"diameter_px": r["diameter_px"], "diameter_um": r["diameter_um"],
             "n_cells": r["n_cells"], "runtime_s": r["runtime_s"],
             "mean_cell_area_um2": float(r["cell_areas_um2"].mean()) if r["n_cells"] > 0 else 0.0}
            for r in per_diam
        ],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntotal runtime: {(time.time()-t0)/60:.1f} min")
    print(f"figures + CSV + JSON saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
