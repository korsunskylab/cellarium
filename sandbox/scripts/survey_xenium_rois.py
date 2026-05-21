"""Survey the Xenium slide to find ROIs that:
  1. Have sufficient cell density (10X called >= MIN_CELLS cells)
  2. Have a low 10X transcript-assignment fraction (where 10X did poorly)
  3. Are morphologically distinct from each other (varied cell-size/density profiles)

Outputs:
  - slide-level scatter of all tiles colored by assignment fraction
  - mini-views of top candidates with cell outlines
  - printed list of candidate (x, y) bboxes the user can copy into the crop script
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import zarr, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import tifffile

DATA       = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/data/Xenium_Prime_Human_Skin_FFPE_xe_outs")
PIXEL_SIZE = 0.2125
TILE_UM    = 250.0     # zarr's natural tile size — re-using
MIN_CELLS  = 40        # tile must have at least this many cells

EXISTING_ROIS = {
    "ROI1": (960.0, 1220.0, 2930.0, 3060.0),
    "ROI2": (640.0, 1040.0, 3370.0, 3580.0),
}

cells_root = zarr.open(zarr.ZipStore(DATA / "cells.zarr.zip", mode="r"), mode="r")
tx_root    = zarr.open(zarr.ZipStore(DATA / "transcripts.zarr.zip", mode="r"), mode="r")
cell_summary = cells_root["cell_summary"][:]
masks_full   = cells_root["masks"]["1"]
gc           = tx_root["gene_category"][:]
grid0        = tx_root["grids"]["0"]
H_PX, W_PX   = masks_full.shape

print(f"surveying slide ({W_PX*PIXEL_SIZE:.0f} × {H_PX*PIXEL_SIZE:.0f} µm)…")

# ─── 1) per-tile statistics ─────────────────────────────────────────────
records = []
for key in grid0.group_keys():
    i, j = map(int, key.split(","))
    x0, y0 = i * TILE_UM, j * TILE_UM
    x1, y1 = x0 + TILE_UM, y0 + TILE_UM
    if x1 > W_PX*PIXEL_SIZE or y1 > H_PX*PIXEL_SIZE:
        continue

    # Transcripts in tile
    tile = grid0[key]
    loc = tile["location"][:]
    qv  = tile["quality_score"][:].squeeze(-1)
    val = tile["valid"][:].squeeze(-1)
    gid = tile["gene_identity"][:].squeeze(-1)
    is_gene = gc[gid, 0]
    keep = (val > 0) & (qv >= 20) & is_gene & \
           (loc[:, 0] >= x0) & (loc[:, 0] < x1) & \
           (loc[:, 1] >= y0) & (loc[:, 1] < y1)
    if keep.sum() < 200:
        continue
    loc = loc[keep]

    # 10X cells in tile (by centroid)
    cell_x = cell_summary[:, 0]; cell_y = cell_summary[:, 1]
    cell_area = cell_summary[:, 2]
    in_tile = (cell_x >= x0) & (cell_x < x1) & (cell_y >= y0) & (cell_y < y1)
    n_cells = int(in_tile.sum())
    if n_cells < MIN_CELLS:
        continue

    # Mask label per transcript — read just this tile's mask block
    px0 = int(round(x0 / PIXEL_SIZE)); px1 = min(W_PX, int(round(x1 / PIXEL_SIZE)))
    py0 = int(round(y0 / PIXEL_SIZE)); py1 = min(H_PX, int(round(y1 / PIXEL_SIZE)))
    mask_block = masks_full[py0:py1, px0:px1]
    tx_px = (loc[:, 0] / PIXEL_SIZE).astype(int) - px0
    tx_py = (loc[:, 1] / PIXEL_SIZE).astype(int) - py0
    inb   = (tx_px >= 0) & (tx_px < mask_block.shape[1]) & \
            (tx_py >= 0) & (tx_py < mask_block.shape[0])
    inside = (mask_block[tx_py[inb], tx_px[inb]] > 0).sum()
    assign_frac = inside / max(inb.sum(), 1)

    # Morphology proxies that are cheap to compute (no image reads)
    median_cell_area = float(np.median(cell_area[in_tile]))
    cell_density     = n_cells / (TILE_UM * TILE_UM)  # cells per µm²

    records.append({
        "x0_um": x0, "y0_um": y0,
        "n_cells": n_cells,
        "n_tx": int(keep.sum()),
        "tx_density": float(keep.sum()) / (TILE_UM * TILE_UM),  # tx per µm²
        "assign_frac": assign_frac,
        "median_cell_area_um2": median_cell_area * (PIXEL_SIZE ** 2),  # convert to µm²
        "cell_density": cell_density,
    })

df = pd.DataFrame(records).sort_values("assign_frac")
print(f"\nsurveyed {len(df)} populated tiles")
print(f"assign_frac quantiles: 10%={df.assign_frac.quantile(0.1):.2f}  "
      f"50%={df.assign_frac.quantile(0.5):.2f}  90%={df.assign_frac.quantile(0.9):.2f}")
print(f"median cell area (µm²) quantiles: "
      f"10%={df.median_cell_area_um2.quantile(0.1):.0f}  "
      f"50%={df.median_cell_area_um2.quantile(0.5):.0f}  "
      f"90%={df.median_cell_area_um2.quantile(0.9):.0f}")

# ─── 2) Pick candidates: low-assign tiles, spatially+morphologically diverse ──
# Filter to bottom 30% of assign_frac
bad = df[df.assign_frac <= df.assign_frac.quantile(0.30)].copy()

# Drop tiles too close to existing ROIs
def too_close(x0, y0, existing, min_dist_um=400):
    for name, (ex0, ex1, ey0, ey1) in existing.items():
        cx_e, cy_e = (ex0+ex1)/2, (ey0+ey1)/2
        cx_t, cy_t = x0+TILE_UM/2, y0+TILE_UM/2
        if np.hypot(cx_e-cx_t, cy_e-cy_t) < min_dist_um:
            return True
    return False

bad = bad[~bad.apply(lambda r: too_close(r.x0_um, r.y0_um, EXISTING_ROIS), axis=1)]

# Farthest-point sampling on (x, y, log10(median_cell_area), tx_density)
# scaled so spatial and morphology contribute roughly equally
features = bad[["x0_um", "y0_um"]].to_numpy().astype(float)
# Normalize x, y to [0, 1]
features[:, 0] /= 9100.0
features[:, 1] /= 4400.0
# Add morphology proxies, normalized
size_z = (np.log10(bad.median_cell_area_um2.to_numpy()) - 1.5) / 1.0  # roughly center + scale
dens_z = (bad.cell_density.to_numpy() - bad.cell_density.median()) / max(bad.cell_density.std(), 1e-6)
features = np.column_stack([features * 1.5, size_z * 0.7, dens_z * 0.5])

# Greedy farthest-first: pick 6 candidates
N_PICK = 6
picked_idx = [0]  # start with the worst-assignment tile
while len(picked_idx) < min(N_PICK, len(bad)):
    dists = np.linalg.norm(features[:, None, :] - features[picked_idx][None, :, :], axis=-1).min(axis=1)
    dists[picked_idx] = -1
    picked_idx.append(int(np.argmax(dists)))

candidates = bad.iloc[picked_idx].sort_values("y0_um").reset_index(drop=True)
print(f"\nselected {len(candidates)} candidate tiles:")
for k, r in candidates.iterrows():
    print(f"  candidate {k+1}: "
          f"x[{r.x0_um:.0f}, {r.x0_um+TILE_UM:.0f}] y[{r.y0_um:.0f}, {r.y0_um+TILE_UM:.0f}]  "
          f"  cells={int(r.n_cells)}  tx={int(r.n_tx):,}  "
          f"assign={r.assign_frac:.2%}  med_area={r.median_cell_area_um2:.0f}µm²")

# ─── 3) Slide overview ──────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sc = ax[0].scatter(df.x0_um + TILE_UM/2, df.y0_um + TILE_UM/2,
                    c=df.assign_frac, s=18, cmap="RdYlGn", vmin=0.4, vmax=0.95)
ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)")
ax[0].set_title("10X transcript-assignment fraction per 250µm tile")
ax[0].invert_yaxis(); ax[0].set_aspect("equal")
fig.colorbar(sc, ax=ax[0], label="assign_frac")
# Mark existing ROIs
for name, (x0, x1, y0, y1) in EXISTING_ROIS.items():
    ax[0].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fc="none", ec="blue", lw=1.5))
    ax[0].text(x0, y0-30, name, color="blue", fontsize=9)
# Mark candidates
for k, r in candidates.iterrows():
    ax[0].add_patch(plt.Rectangle((r.x0_um, r.y0_um), TILE_UM, TILE_UM,
                                   fc="none", ec="black", lw=2))
    ax[0].text(r.x0_um, r.y0_um-30, f"C{k+1}", color="black", fontsize=10, fontweight="bold")

# Right: assign_frac histogram with quantiles + candidate values
ax[1].hist(df.assign_frac, bins=40, color="0.7")
ax[1].axvline(df.assign_frac.quantile(0.30), color="red", ls="--", label="30th %ile (cutoff)")
ax[1].axvline(df.assign_frac.median(), color="black", ls="--", label="median")
ax[1].set_xlabel("assign_frac per tile"); ax[1].set_ylabel("# tiles")
ax[1].set_title("Distribution of 10X assignment fraction across slide")
ax[1].legend()
for k, r in candidates.iterrows():
    ax[1].axvline(r.assign_frac, color="black", lw=0.6)
plt.tight_layout()
out_dir = Path(__file__).parent.parent / "figs"; out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "_roi_candidates_overview.png", dpi=110, bbox_inches="tight")
print(f"\nsaved overview: {out_dir / '_roi_candidates_overview.png'}")

# ─── 4) Mini-views of each candidate (cells + transcript dots only — no morphology) ──
fig2, ax2 = plt.subplots(2, 3, figsize=(16, 10))
for k, r in candidates.iterrows():
    a = ax2.flat[k]
    x0, y0 = r.x0_um, r.y0_um
    x1, y1 = x0 + TILE_UM, y0 + TILE_UM

    # Cells (centroids only — fast)
    in_tile = (cell_summary[:, 0] >= x0) & (cell_summary[:, 0] < x1) & \
              (cell_summary[:, 1] >= y0) & (cell_summary[:, 1] < y1)
    a.scatter(cell_summary[in_tile, 0], cell_summary[in_tile, 1], s=12, c="steelblue",
              alpha=0.6, edgecolors="none", label=f"cells (n={int(in_tile.sum())})")

    # Transcript dots
    key = f"{int(x0/TILE_UM)},{int(y0/TILE_UM)}"
    if key in grid0:
        tile = grid0[key]
        loc = tile["location"][:]; qv = tile["quality_score"][:].squeeze(-1)
        val = tile["valid"][:].squeeze(-1); gid = tile["gene_identity"][:].squeeze(-1)
        keep = (val > 0) & (qv >= 20) & gc[gid, 0] & \
               (loc[:, 0] >= x0) & (loc[:, 0] < x1) & \
               (loc[:, 1] >= y0) & (loc[:, 1] < y1)
        a.scatter(loc[keep, 0], loc[keep, 1], s=0.4, c="0.5", alpha=0.4, edgecolors="none")
    a.set_xlim(x0, x1); a.set_ylim(y1, y0); a.set_aspect("equal")
    a.set_title(f"C{k+1}  x[{int(x0)},{int(x1)}] y[{int(y0)},{int(y1)}]\n"
                f"cells={int(r.n_cells)}, tx={int(r.n_tx):,}, assign={r.assign_frac:.0%}, "
                f"med_area={r.median_cell_area_um2:.0f}µm²", fontsize=9)
    a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)")

# Hide unused subplot
for k in range(len(candidates), 6):
    ax2.flat[k].axis("off")

plt.tight_layout()
fig2.savefig(out_dir / "_roi_candidates_minimaps.png", dpi=110, bbox_inches="tight")
print(f"saved minimaps: {out_dir / '_roi_candidates_minimaps.png'}")

# ─── 5) Save candidate definitions for later use ──────────────────────────
candidates_dict = {
    f"C{k+1}": {
        "x0_um": float(r.x0_um),
        "x1_um": float(r.x0_um + TILE_UM),
        "y0_um": float(r.y0_um),
        "y1_um": float(r.y0_um + TILE_UM),
        "n_cells_10x": int(r.n_cells),
        "n_tx_10x": int(r.n_tx),
        "assign_frac_10x": float(r.assign_frac),
        "median_cell_area_um2": float(r.median_cell_area_um2),
    }
    for k, r in candidates.iterrows()
}
import json
with open(out_dir / "_roi_candidates.json", "w") as f:
    json.dump(candidates_dict, f, indent=2)
print(f"saved candidate defs: {out_dir / '_roi_candidates.json'}")
