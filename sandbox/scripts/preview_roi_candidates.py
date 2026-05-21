"""Render DAPI+18S composite mini-views of each candidate ROI for visual inspection."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import json, time
import numpy as np, matplotlib.pyplot as plt, tifffile, zarr

DATA       = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/data/Xenium_Prime_Human_Skin_FFPE_xe_outs")
PIXEL_SIZE = 0.2125
NB_DIR     = Path(__file__).parent.parent / "figs"

with open(NB_DIR / "_roi_candidates.json") as f:
    candidates = json.load(f)

# Read DAPI and 18S at full slide; chunked decode of compressed tiles
print("loading DAPI page (channel 0)...")
t0 = time.time()
dapi_full = tifffile.imread(DATA / "morphology_focus" / "morphology_focus_0000.ome.tif", key=0)
print(f"  {time.time()-t0:.1f}s, shape {dapi_full.shape}")

print("loading 18S page (channel 2)...")
t0 = time.time()
s18_full = tifffile.imread(DATA / "morphology_focus" / "morphology_focus_0002.ome.tif", key=0)
print(f"  {time.time()-t0:.1f}s, shape {s18_full.shape}")

# Cell centroids and transcripts (for overlay)
cells_root = zarr.open(zarr.ZipStore(DATA / "cells.zarr.zip", mode="r"), mode="r")
cell_summary = cells_root["cell_summary"][:]
tx_root      = zarr.open(zarr.ZipStore(DATA / "transcripts.zarr.zip", mode="r"), mode="r")
gc           = tx_root["gene_category"][:]
grid0        = tx_root["grids"]["0"]
TILE_UM      = 250.0


def normalize_for_display(img, low=1, high=99):
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def composite(dapi, s18):
    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = s18; rgb[..., 1] = s18; rgb[..., 2] = dapi
    return np.clip(rgb, 0, 1)


fig, ax = plt.subplots(2, 3, figsize=(17, 8))
ax = ax.flat
for k, (name, c) in enumerate(candidates.items()):
    x0, x1 = c["x0_um"], c["x1_um"]
    y0, y1 = c["y0_um"], c["y1_um"]
    ix0, ix1 = int(round(x0 / PIXEL_SIZE)), int(round(x1 / PIXEL_SIZE))
    iy0, iy1 = int(round(y0 / PIXEL_SIZE)), int(round(y1 / PIXEL_SIZE))

    dapi = normalize_for_display(dapi_full[iy0:iy1, ix0:ix1])
    s18  = normalize_for_display(s18_full[iy0:iy1,  ix0:ix1])
    img = composite(dapi, s18)

    a = ax[k]
    a.imshow(img, aspect="equal", extent=[x0, x1, y1, y0])

    # Overlay cell centroids and transcript dots — fast, no mask reads
    in_tile = (cell_summary[:, 0] >= x0) & (cell_summary[:, 0] < x1) & \
              (cell_summary[:, 1] >= y0) & (cell_summary[:, 1] < y1)
    a.scatter(cell_summary[in_tile, 0], cell_summary[in_tile, 1], s=4,
              c="cyan", alpha=0.5, edgecolors="none")

    key = f"{int(x0/TILE_UM)},{int(y0/TILE_UM)}"
    if key in grid0:
        tile = grid0[key]
        loc = tile["location"][:]; qv = tile["quality_score"][:].squeeze(-1)
        val = tile["valid"][:].squeeze(-1); gid = tile["gene_identity"][:].squeeze(-1)
        keep = (val > 0) & (qv >= 20) & gc[gid, 0] & \
               (loc[:, 0] >= x0) & (loc[:, 0] < x1) & \
               (loc[:, 1] >= y0) & (loc[:, 1] < y1)
        a.scatter(loc[keep, 0], loc[keep, 1], s=0.4, c="0.8", alpha=0.4, edgecolors="none")

    a.set_xlim(x0, x1); a.set_ylim(y1, y0)
    a.set_title(f"{name}: x[{int(x0)},{int(x1)}] y[{int(y0)},{int(y1)}]\n"
                f"cells={c['n_cells_10x']}, tx={c['n_tx_10x']:,}, "
                f"assign={c['assign_frac_10x']:.0%}", fontsize=10)
    a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)")

plt.tight_layout()
out = NB_DIR / "_roi_candidates_morphology.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
