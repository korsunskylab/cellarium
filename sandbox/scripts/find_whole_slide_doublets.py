"""Find heterotypic doublet candidates across the whole slide using CP-SAM masks.

Strategy mirrors `analyses/cell_doublet_composition.ipynb` but at slide scale:

  1. Load gene → lineage map (new Bayesian P(t|g) rule from notebook 01).
  2. Stream transcripts.zarr tiles; for each anchored transcript, look up its
     CP-SAM cell_id from the whole-slide masks via vectorized fancy indexing.
  3. Build the per-cell composition matrix (cells × 7 lineages).
  4. Score each cell by `top2_n` (absolute count of the 2nd-most-abundant
     lineage among its anchored transcripts) — same metric as the per-ROI run.
  5. Exclude Keratinocyte × Melanoma pairs (markers too similar to be a
     reliable spatial-merger indicator).
  6. Save `cells_doublet_composition_whole_slide.parquet` cache.
  7. Plot top 100 candidates: 2-panel DAPI + 18S + anchor scatter, one cell
     plus 10 µm padding each direction. Filenames sort by rank.

Outputs:
    data/.../cells_doublet_composition_whole_slide.parquet
    figs/whole_slide_doublets/rank_NNN_<lineagepair>_cps<id>_top2<n>.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import find_objects

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
MASKS_TIF   = DATA / "cpsam_whole_slide" / "masks.tif"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
DAPI_TIF    = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
S18_TIF     = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"

OUT_PARQUET = DATA / "cells_doublet_composition_whole_slide.parquet"
FIGS_DIR    = ROOT / "figs" / "whole_slide_doublets"

# ── Config ────────────────────────────────────────────────────────────────
PIXEL_SIZE_UM = 0.2125
LINEAGES = ["Melanoma", "Myeloid", "Tcell", "Plasma", "Fibroblast", "Endothelial", "Keratinocyte"]
LINEAGE_COLOURS = {
    "Melanoma":     "#E41A1C",
    "Myeloid":      "#FF7F00",
    "Tcell":        "#377EB8",
    "Plasma":       "#984EA3",
    "Fibroblast":   "#4DAF4A",
    "Endothelial":  "#00CED1",
    "Keratinocyte": "#F781BF",
}
EXCLUDE_PAIRS = {("Keratinocyte", "Melanoma")}   # canonical order: sorted alphabetical
PAD_UM        = 10.0                              # crop padding around each cell
PAD_PX        = int(round(PAD_UM / PIXEL_SIZE_UM))   # ~47 px
N_PLOTS       = 100
QV_MIN        = 20.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Step 1: anchor map ────────────────────────────────────────────────────
def load_anchor_map() -> tuple[np.ndarray, dict[str, int]]:
    """Return (gene_id → lineage_idx or -1) using the canonical zarr gene_names order."""
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    gl = pd.read_parquet(GENE_LABELS)
    lin_to_idx = {L: i for i, L in enumerate(LINEAGES)}
    gid_to_lin = np.full(len(gene_names), -1, dtype=np.int8)
    n_anchored = 0
    for gene, label in zip(gl["gene"], gl["label"]):
        if label == "ambiguous" or label not in lin_to_idx:
            continue
        if gene not in name_to_idx:
            continue
        gid_to_lin[name_to_idx[gene]] = lin_to_idx[label]
        n_anchored += 1
    log(f"anchored: {n_anchored} genes / {len(gene_names)} panel  "
        f"({100*n_anchored/len(gene_names):.1f}%)")
    return gid_to_lin, lin_to_idx


# ── Step 2: stream transcripts → per-cell × per-lineage counts ────────────
def build_composition_matrix(gid_to_lin: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return counts[cell_id, lineage_idx]. cell_id 0 = background and is skipped."""
    n_cells = int(masks.max())
    K = len(LINEAGES)
    counts = np.zeros((n_cells + 1, K), dtype=np.int32)   # +1 to index by cell_id directly
    H, W = masks.shape

    zroot = zarr.open(TX_ZARR, mode="r")
    grid = zroot["grids"]["0"]
    tile_keys = list(grid.keys())
    log(f"streaming {len(tile_keys)} transcript tiles...")

    n_tx_total = 0
    n_anch_total = 0
    n_in_cell_total = 0
    for ti, key in enumerate(tile_keys):
        tile = grid[key]
        loc   = tile["location"][:]                 # (N, 3) float32
        gid   = tile["gene_identity"][:].squeeze(-1)   # (N,) uint16
        qv    = tile["quality_score"][:].squeeze(-1)
        valid = tile["valid"][:].squeeze(-1)
        n_tx_total += len(loc)

        # Filter: valid + qv≥20 + anchored gene
        lin = gid_to_lin[gid]                       # (N,) int8 (-1 if not anchored)
        keep = (valid > 0) & (qv >= QV_MIN) & (lin >= 0)
        if not keep.any():
            continue
        x_um = loc[keep, 0]
        y_um = loc[keep, 1]
        lin = lin[keep]
        n_anch_total += len(lin)

        # → pixel coords
        px = np.rint(x_um / PIXEL_SIZE_UM).astype(np.int64)
        py = np.rint(y_um / PIXEL_SIZE_UM).astype(np.int64)
        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not in_bounds.all():
            px = px[in_bounds]; py = py[in_bounds]; lin = lin[in_bounds]
        cell_id = masks[py, px]
        in_cell = cell_id > 0
        if not in_cell.any():
            continue
        cell_id = cell_id[in_cell]; lin = lin[in_cell]
        n_in_cell_total += len(lin)

        # Accumulate via np.add.at (handles repeated (cell, lineage) tuples)
        np.add.at(counts, (cell_id, lin), 1)

        if (ti + 1) % 50 == 0:
            log(f"  tile {ti+1}/{len(tile_keys)}: cum_tx={n_tx_total:,}  "
                f"cum_anch={n_anch_total:,}  cum_in_cell={n_in_cell_total:,}")

    log(f"DONE streaming  total_tx={n_tx_total:,}  anchored={n_anch_total:,}  "
        f"in_cpsam_cell={n_in_cell_total:,} ({100*n_in_cell_total/max(n_anch_total,1):.1f}%)")
    return counts


# ── Step 3: per-cell scoring + filter ─────────────────────────────────────
def score_cells(counts: np.ndarray) -> pd.DataFrame:
    """Skip row 0 (background); return long-form per-cell metrics."""
    K = len(LINEAGES)
    n_cells = counts.shape[0] - 1
    cell_ids = np.arange(1, n_cells + 1)
    cnt = counts[1:]                                # (n_cells, K)

    total = cnt.sum(axis=1)
    # Top-2 lineages per cell
    sorted_idx = np.argsort(-cnt, axis=1)          # (n_cells, K) desc
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1]
    top1_n = cnt[np.arange(n_cells), top1_idx]
    top2_n = cnt[np.arange(n_cells), top2_idx]

    lin_arr = np.array(LINEAGES)
    top1_lin = lin_arr[top1_idx]
    top2_lin = lin_arr[top2_idx]

    # Canonical pair: alphabetical to dedupe
    pair_sorted = np.sort(np.stack([top1_lin, top2_lin], axis=1), axis=1)
    lineage_pair = np.array([" × ".join(p) for p in pair_sorted])

    df = pd.DataFrame({
        "cps_id":        cell_ids,
        "n_anchors":     total.astype(np.int32),
        "top1_lin":      top1_lin,
        "top1_n":        top1_n.astype(np.int32),
        "top2_lin":      top2_lin,
        "top2_n":        top2_n.astype(np.int32),
        "lineage_pair":  lineage_pair,
        "doublet_score": top2_n.astype(np.int32),
        "doublet_ratio": (top2_n / np.maximum(top1_n, 1)).astype(np.float32),
    })
    for i, L in enumerate(LINEAGES):
        df[f"n_{L}"] = cnt[:, i].astype(np.int32)
    return df


def filter_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Drop background-y or low-evidence cells, exclude Kerat×Melanoma, rank by top2_n."""
    canon_excl = {" × ".join(sorted(p)) for p in EXCLUDE_PAIRS}
    keep = (df["top2_n"] >= 2) & (~df["lineage_pair"].isin(canon_excl))
    df = df[keep].copy()
    df = df.sort_values(["top2_n", "doublet_ratio"], ascending=[False, False]).reset_index(drop=True)
    return df


# ── Step 4: plotting ──────────────────────────────────────────────────────
def make_plots(top: pd.DataFrame, masks: np.ndarray) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"computing per-label bboxes via find_objects (this is the slow step)...")
    t0 = time.time()
    bboxes = find_objects(masks)
    log(f"  find_objects: {time.time()-t0:.1f}s ({sum(b is not None for b in bboxes)} bboxes)")

    log("loading whole-slide DAPI + 18S into memory (3.4 GB)...")
    t0 = time.time()
    dapi_full = tifffile.imread(DAPI_TIF, key=0)
    log(f"  DAPI: {time.time()-t0:.1f}s")
    t0 = time.time()
    s18_full = tifffile.imread(S18_TIF, key=0)
    log(f"  18S : {time.time()-t0:.1f}s")
    H, W = masks.shape

    # Load anchored transcripts once for plotting (per-cell scatter overlay)
    log("loading anchored transcripts for plotting overlay...")
    t0 = time.time()
    gid_to_lin, _ = load_anchor_map()
    zroot = zarr.open(TX_ZARR, mode="r")
    grid = zroot["grids"]["0"]
    anch_x: list[np.ndarray] = []
    anch_y: list[np.ndarray] = []
    anch_l: list[np.ndarray] = []
    for key in grid.keys():
        tile = grid[key]
        loc   = tile["location"][:]
        gid   = tile["gene_identity"][:].squeeze(-1)
        qv    = tile["quality_score"][:].squeeze(-1)
        valid = tile["valid"][:].squeeze(-1)
        lin = gid_to_lin[gid]
        keep = (valid > 0) & (qv >= QV_MIN) & (lin >= 0)
        if not keep.any():
            continue
        anch_x.append(loc[keep, 0])
        anch_y.append(loc[keep, 1])
        anch_l.append(lin[keep])
    A_x = np.concatenate(anch_x)
    A_y = np.concatenate(anch_y)
    A_l = np.concatenate(anch_l)
    log(f"  loaded {len(A_x):,} anchored tx in {time.time()-t0:.1f}s")

    # Spatial index: precompute pixel coords for fast bbox lookup
    A_px = np.rint(A_x / PIXEL_SIZE_UM).astype(np.int64)
    A_py = np.rint(A_y / PIXEL_SIZE_UM).astype(np.int64)

    log(f"rendering {len(top)} plots → {FIGS_DIR}")
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        cps_id = int(row.cps_id)
        bb = bboxes[cps_id - 1]
        if bb is None:
            log(f"  rank {rank}: cps_id={cps_id} has no bbox; skipping")
            continue
        sy, sx = bb
        y0 = max(sy.start - PAD_PX, 0)
        y1 = min(sy.stop  + PAD_PX, H)
        x0 = max(sx.start - PAD_PX, 0)
        x1 = min(sx.stop  + PAD_PX, W)

        dapi_crop = dapi_full[y0:y1, x0:x1]
        s18_crop  = s18_full[y0:y1, x0:x1]
        cell_mask = (masks[y0:y1, x0:x1] == cps_id)

        ext = [x0 * PIXEL_SIZE_UM, x1 * PIXEL_SIZE_UM,
               y1 * PIXEL_SIZE_UM, y0 * PIXEL_SIZE_UM]   # (left, right, bottom, top) for origin='upper'
        x0u, x1u = ext[0], ext[1]
        y0u, y1u = ext[3], ext[2]

        # Norm crops to [0,1] per panel
        def norm(im):
            lo, hi = np.percentile(im, [1, 99])
            return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        dapi_d = norm(dapi_crop)
        s18_d  = norm(s18_crop)

        # Anchors inside the crop window
        win = (A_px >= x0) & (A_px < x1) & (A_py >= y0) & (A_py < y1)
        win_x = A_x[win]; win_y = A_y[win]; win_l = A_l[win]
        # Subset to anchors inside the focal cell
        in_cell = masks[A_py[win], A_px[win]] == cps_id

        # Left: DAPI (blue) + 18S (yellow) RGB overlay. Yellow = R+G channels,
        # blue = B channel; DAPI+18S co-located reads as white. Background is black.
        rgb_overlay = np.stack([s18_d, s18_d, dapi_d], axis=-1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_overlay, extent=ext, aspect="equal")
        axes[0].contour(cell_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                        extent=ext, origin="upper")
        axes[0].set_xlim(x0u, x1u); axes[0].set_ylim(y1u, y0u)
        axes[0].set_title(f"DAPI (blue) + 18S (yellow)  cps_id={cps_id}")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(s18_d, cmap="gray", extent=ext, aspect="equal")
        axes[1].contour(cell_mask, levels=[0.5], colors="cyan", linewidths=0.8,
                        extent=ext, origin="upper")
        # Faint outside-cell anchors (context)
        outside = ~in_cell
        for li, L in enumerate(LINEAGES):
            sel = outside & (win_l == li)
            if sel.any():
                axes[1].scatter(win_x[sel], win_y[sel], s=8,
                                c=LINEAGE_COLOURS[L], alpha=0.35,
                                edgecolor="none")
        # Bold inside-cell anchors (evidence)
        for li, L in enumerate(LINEAGES):
            sel = in_cell & (win_l == li)
            if sel.any():
                axes[1].scatter(win_x[sel], win_y[sel], s=28,
                                c=LINEAGE_COLOURS[L],
                                label=f"{L} (n={int(sel.sum())})",
                                edgecolor="black", linewidth=0.4, alpha=0.95)
        axes[1].set_xlim(x0u, x1u); axes[1].set_ylim(y1u, y0u)
        axes[1].set_title(f"18S  {row.top1_lin}({row.top1_n}) × "
                          f"{row.top2_lin}({row.top2_n})  ratio={row.doublet_ratio:.2f}")
        axes[1].legend(loc="upper right", fontsize=7, framealpha=0.85)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        pair_slug = row.lineage_pair.replace(" × ", "x").replace(" ", "")
        fname = (f"rank_{rank:03d}_{pair_slug}_cps{cps_id}_top2_{row.top2_n}"
                 f"_ratio{row.doublet_ratio:.2f}.png")
        plt.suptitle(f"#{rank}/{len(top)}  ({row.lineage_pair})  "
                     f"n_anchors={row.n_anchors}", y=1.005)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / fname, dpi=130, bbox_inches="tight")
        plt.close()

        if rank % 10 == 0:
            log(f"  plotted {rank}/{len(top)}")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.time()
    log(f"loading CP-SAM masks: {MASKS_TIF}")
    masks = tifffile.imread(MASKS_TIF)
    log(f"  masks: {masks.shape} dtype={masks.dtype}  n_cells={int(masks.max())}")

    gid_to_lin, _ = load_anchor_map()
    counts = build_composition_matrix(gid_to_lin, masks)

    log("scoring per-cell composition...")
    df = score_cells(counts)
    log(f"  {len(df):,} cells scored.  doublet_score distribution:")
    log("  " + str(df["doublet_score"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict()))

    log("filtering + ranking...")
    df_ranked = filter_and_rank(df)
    log(f"  after filter (top2_n≥2, no Kerat×Melanoma): {len(df_ranked):,} cells")
    log(f"  top 12 lineage pairs (top 100):")
    log("  " + str(df_ranked.head(N_PLOTS)["lineage_pair"].value_counts().head(12).to_dict()))

    log(f"writing parquet → {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET)
    top100_path = OUT_PARQUET.parent / "cells_doublet_composition_whole_slide_top100.parquet"
    df_ranked.head(N_PLOTS).to_parquet(top100_path)
    log(f"writing top-100 parquet → {top100_path}")

    log("plotting top 100...")
    make_plots(df_ranked.head(N_PLOTS), masks)

    log(f"done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
