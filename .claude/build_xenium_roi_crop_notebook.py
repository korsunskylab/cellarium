import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Xenium 5K ROI: 10X vs CP-SAM segmentation (DAPI + 18S)

Load a pre-cropped ROI from `data/Xenium_Prime_Human_Skin_FFPE_xe_outs/ROI{1,2}/`, visualize the morphology channels and 10X-called cells/transcripts, then run CP-SAM on DAPI+18S and compare to 10X.

**ROIs available:**
- `ROI1` — x ∈ [960, 1220] µm, y ∈ [2930, 3060] µm  (260 × 130 µm, ~158 cells)
- `ROI2` — x ∈ [640, 1040] µm, y ∈ [3370, 3580] µm  (400 × 210 µm, ~469 cells)
- `ROI3` — x ∈ [8250, 8500] µm, y ∈ [1000, 1250] µm  (250 × 250 µm, ~656 cells, striated/fibrous)
- `ROI4` — x ∈ [1500, 1750] µm, y ∈ [2250, 2500] µm  (250 × 250 µm, ~536 cells, dense mixed cellular)
- `ROI5` — x ∈ [6750, 7000] µm, y ∈ [3000, 3250] µm  (250 × 250 µm, ~384 cells, dense round)
- `ROI6` — x ∈ [1250, 1500] µm, y ∈ [3500, 3750] µm  (250 × 250 µm, ~233 cells, striated, lowest 10X assignment)

To switch ROIs, change the `ROI` variable in the next cell."""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load pre-cropped ROI"))

cells.append(nbf.v4.new_code_cell("""import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import json, time
import numpy as np, pandas as pd, tifffile, torch
import matplotlib.pyplot as plt

# ─── pick ROI ─────────────────────────────────────────────────────────────
ROI = "ROI1"      # ← change to "ROI2" to load the second ROI
# ─────────────────────────────────────────────────────────────────────────

DATA = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/data/Xenium_Prime_Human_Skin_FFPE_xe_outs")
ROI_DIR = DATA / ROI

with open(ROI_DIR / "metadata.json") as f:
    meta = json.load(f)
print(json.dumps(meta, indent=2))

x0_um, x1_um = meta["bbox_um"]["x0_um"], meta["bbox_um"]["x1_um"]
y0_um, y1_um = meta["bbox_um"]["y0_um"], meta["bbox_um"]["y1_um"]
extent       = [x0_um, x1_um, y1_um, y0_um]   # for matplotlib imshow
print(f"\\nMPS available: {torch.backends.mps.is_available()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Load all 4 morphology channels
channels = [tifffile.imread(ROI_DIR / f"morphology_{name}.tif") for name in meta["channels"]]
dapi = channels[0]
s18  = channels[2]
print("channel shapes:", [c.shape for c in channels])

# Load 10X cell masks (label image at the same 0.2125 µm/px) and polygons
masks_10x  = tifffile.imread(ROI_DIR / "cells_10x_masks.tif")
df_poly    = pd.read_parquet(ROI_DIR / "cells_10x_polygons.parquet")
df_tx      = pd.read_parquet(ROI_DIR / "transcripts.parquet")
print(f"10X masks: {masks_10x.shape}, n_cells_in_mask={len(np.unique(masks_10x))-1}")
print(f"10X polygons: {len(df_poly)} cells")
print(f"transcripts:  {len(df_tx):,} transcripts (qv≥20)")

# Compact 10X label range (cells outside bbox get label 0; renumber the rest)
labs_10x = np.unique(masks_10x); labs_10x = labs_10x[labs_10x != 0]
remap_10x = np.zeros(int(masks_10x.max()) + 1, dtype=np.int64)
for new, old in enumerate(labs_10x, start=1):
    remap_10x[old] = new
m_10x_compact = remap_10x[masks_10x.astype(np.int64)]
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Show all 4 morphology channels"))

cells.append(nbf.v4.new_code_cell("""def normalize_for_display(img, low=1, high=99):
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

fig, ax = plt.subplots(2, 2, figsize=(14, 7))
for i, (a, name, ch) in enumerate(zip(ax.flat, meta["channels"], channels)):
    a.imshow(normalize_for_display(ch), cmap="gray", aspect="equal", extent=extent)
    a.set_title(f"channel {i}: {name}")
    a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)")
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. DAPI + 18S two-color composite (blue + yellow)"))

cells.append(nbf.v4.new_code_cell("""dapi_n = normalize_for_display(dapi)
s18_n  = normalize_for_display(s18)

def blue_yellow_composite(dapi, s18):
    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = s18                  # R  (R+G = yellow)
    rgb[..., 1] = s18                  # G
    rgb[..., 2] = dapi                 # B  (DAPI alone = blue)
    return np.clip(rgb, 0, 1)

composite = blue_yellow_composite(dapi_n, s18_n)
plt.figure(figsize=(11, 5.5))
plt.imshow(composite, aspect="equal", extent=extent)
plt.title("DAPI (blue) + 18S (yellow)")
plt.xlabel("x (µm)"); plt.ylabel("y (µm)")
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. 10X cells in this ROI (cell outlines on DAPI)"))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(11, 5.5))
ax.imshow(dapi_n, cmap="gray", aspect="equal", extent=extent)
for _, row in df_poly.iterrows():
    ax.plot(row["x_vertices_um"], row["y_vertices_um"], color="cyan", lw=0.8)
ax.set_xlim(x0_um, x1_um); ax.set_ylim(y1_um, y0_um)
ax.set_title(f"10X cell outlines  —  n={len(df_poly)}")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Transcripts (gray dots on DAPI)"))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(11, 5.5))
ax.imshow(dapi_n, cmap="gray", aspect="equal", extent=extent)
ax.scatter(df_tx["x_um"], df_tx["y_um"], s=1.5, c="0.6", alpha=0.55, edgecolors="none")
ax.set_xlim(x0_um, x1_um); ax.set_ylim(y1_um, y0_um)
ax.set_title(f"transcripts  —  {len(df_tx):,} dots (qv≥20)")
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Run CP-SAM on DAPI + 18S

cellpose 4 cpsam, 2-channel input, MPS GPU. Default `cellprob_threshold=0.0`."""))

cells.append(nbf.v4.new_code_cell("""from cellpose import models as cp_modern

m_sam = cp_modern.CellposeModel(gpu=True, pretrained_model="cpsam")
print("cpsam loaded on:", m_sam.device)

img_2ch = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
t0 = time.time()
masks_sam, flows_sam, _ = m_sam.eval(
    img_2ch, channel_axis=-1, diameter=None, niter=200,
    cellprob_threshold=0.0, flow_threshold=0.4,
)
print(f"cpsam DAPI+18S: n={int(masks_sam.max())} cells in {time.time()-t0:.2f}s")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6b. CP-SAM cellprob, flow direction, and flow magnitude

Three diagnostics from `flows`:
- `flows[2]` = **cellprob** (log-odds): per-pixel "is this a cell pixel?"
- `flows[1]` = **dY, dX**: per-pixel flow vectors that the dynamics step follows
- magnitude(`flows[1]`) tells you *how confidently* the network is steering each pixel toward a cell center

A cell with high cellprob that doesn't appear in the masks usually fails at the *flow* stage: either the flows are flat (no convergence basin) or they point out of the cell into a neighbor, so the dynamics absorb those pixels into the neighbor's mask."""))

cells.append(nbf.v4.new_code_cell("""# ─── zoom bbox (microns) for high-res flow plot ──────────────────────────
# Set to None to skip the zoom; otherwise (xz0, xz1, yz0, yz1) — should be inside the ROI
ZOOM_BBOX = (1100.0, 1180.0, 2980.0, 3060.0)   # adjust to a missed cell of interest
# ────────────────────────────────────────────────────────────────────────

cellprob = flows_sam[2]              # (H, W) float, log-odds
flow_yx  = flows_sam[1]              # (2, H, W) — dY, dX
flow_mag = np.linalg.norm(flow_yx, axis=0)
flow_ang = np.arctan2(flow_yx[0], flow_yx[1])  # radians, for HSV color

print(f"cellprob range: [{cellprob.min():.2f}, {cellprob.max():.2f}]  (log-odds)")
print(f"  pixels above thr=0.0:  {(cellprob > 0).mean()*100:.1f}%")
print(f"  pixels above thr=-1.0: {(cellprob > -1).mean()*100:.1f}%")
print(f"  pixels above thr=-2.0: {(cellprob > -2).mean()*100:.1f}%")

# 2x2 grid: composite, cellprob, flow direction (HSV), flow magnitude
H_im, W_im = composite.shape[:2]
panel_w = 7.0
panel_h = panel_w * (H_im / W_im)
fig, ax = plt.subplots(2, 2, figsize=(panel_w * 2 + 0.6, panel_h * 2 + 0.4))
ax[0, 0].imshow(composite, aspect="equal", extent=extent)
ax[0, 0].set_title("DAPI (blue) + 18S (yellow)"); ax[0, 0].axis("off")

im = ax[0, 1].imshow(cellprob, cmap="magma", aspect="equal", extent=extent)
ax[0, 1].set_title("cellprob (log-odds)"); ax[0, 1].axis("off")
fig.colorbar(im, ax=ax[0, 1], fraction=0.046)

# HSV flow-direction map — saturation modulated by magnitude (so flat regions show as muted)
import matplotlib.colors as mcolors
hsv = np.zeros((*flow_ang.shape, 3), dtype=np.float32)
hsv[..., 0] = (flow_ang + np.pi) / (2 * np.pi)               # hue ∈ [0, 1]
hsv[..., 1] = np.clip(flow_mag / max(flow_mag.max(), 1e-6), 0, 1)
hsv[..., 2] = (cellprob > 0).astype(np.float32) * 0.95 + 0.05
rgb = mcolors.hsv_to_rgb(hsv)
ax[1, 0].imshow(rgb, aspect="equal", extent=extent)
ax[1, 0].set_title("flow direction (HSV; saturation = magnitude, value = is-cell)")
ax[1, 0].axis("off")

im = ax[1, 1].imshow(flow_mag, cmap="viridis", aspect="equal", extent=extent)
ax[1, 1].set_title("flow magnitude")
ax[1, 1].axis("off")
fig.colorbar(im, ax=ax[1, 1], fraction=0.046)

# Mark the zoom box on each panel for orientation
if ZOOM_BBOX is not None:
    xz0, xz1, yz0, yz1 = ZOOM_BBOX
    for a in ax.flat:
        a.add_patch(plt.Rectangle((xz0, yz0), xz1-xz0, yz1-yz0,
                                   fill=False, edgecolor="white", lw=1.5, linestyle="--"))
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interactive zoom (plotly)

3 linked panels (composite, cellprob, flow magnitude) with shared pan/zoom — drag a box or scroll-wheel to zoom into any region of interest. Flow arrows are overlaid on the composite at a coarse stride (every 8 px). To inspect a specific cell, just zoom in with the mouse; the arrows stay anchored to their pixels."""))

cells.append(nbf.v4.new_code_cell("""import plotly.graph_objects as go
from plotly.subplots import make_subplots

H_im, W_im = composite.shape[:2]

# Convert composite (float [0,1] RGB) → uint8 RGB for plotly
comp_u8 = (composite * 255).astype(np.uint8)

# Subsample arrows so plotly stays responsive — one arrow per `arrow_stride` pixels
arrow_stride = 8
yy_idx, xx_idx = np.mgrid[0:H_im:arrow_stride, 0:W_im:arrow_stride]
yy_um = y0_um + yy_idx * meta["pixel_size_um"]
xx_um = x0_um + xx_idx * meta["pixel_size_um"]
dy_sub = flow_yx[0, ::arrow_stride, ::arrow_stride]
dx_sub = flow_yx[1, ::arrow_stride, ::arrow_stride]
mag_sub = flow_mag[::arrow_stride, ::arrow_stride]
# Length of arrows in microns (scale them to be visible at default zoom)
arrow_scale_um = 1.5
xs_lines, ys_lines = [], []
for k in range(yy_um.size):
    x0, y0 = float(xx_um.flat[k]), float(yy_um.flat[k])
    x1 = x0 + float(dx_sub.flat[k]) * arrow_scale_um
    y1 = y0 + float(dy_sub.flat[k]) * arrow_scale_um
    xs_lines += [x0, x1, None]
    ys_lines += [y0, y1, None]

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("DAPI(blue)+18S(yellow)+flow", "cellprob (log-odds)", "flow magnitude"),
    horizontal_spacing=0.04,
)

fig.add_trace(go.Image(z=comp_u8, x0=x0_um, dx=meta["pixel_size_um"],
                        y0=y0_um, dy=meta["pixel_size_um"]),
              row=1, col=1)
fig.add_trace(go.Scatter(x=xs_lines, y=ys_lines, mode="lines",
                         line=dict(color="rgba(255,255,255,0.7)", width=1),
                         hoverinfo="skip", showlegend=False),
              row=1, col=1)

fig.add_trace(go.Heatmap(z=cellprob, colorscale="magma",
                          x0=x0_um, dx=meta["pixel_size_um"],
                          y0=y0_um, dy=meta["pixel_size_um"],
                          colorbar=dict(title="log-odds", x=0.66, len=0.9)),
              row=1, col=2)

fig.add_trace(go.Heatmap(z=flow_mag, colorscale="viridis",
                          x0=x0_um, dx=meta["pixel_size_um"],
                          y0=y0_um, dy=meta["pixel_size_um"],
                          colorbar=dict(title="|flow|", x=1.0, len=0.9)),
              row=1, col=3)

# Force every panel's x and y axes to track panel 1's axes so zooming syncs
fig.update_xaxes(matches="x", row=1, col=2)
fig.update_xaxes(matches="x", row=1, col=3)
fig.update_yaxes(matches="y", row=1, col=2)
fig.update_yaxes(matches="y", row=1, col=3)
# y inverted (image coords), equal-aspect on each panel
fig.update_yaxes(autorange="reversed", row=1, col=1)
fig.update_yaxes(autorange="reversed", row=1, col=2)
fig.update_yaxes(autorange="reversed", row=1, col=3)
fig.update_yaxes(scaleanchor="x",  scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=1, col=3)

fig.update_layout(height=520, width=1500,
                  margin=dict(l=10, r=10, t=40, b=10),
                  dragmode="zoom")
fig.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Where to set `cellprob_threshold`? Use mRNA density.

The natural way to pick a threshold: bin pixels by cellprob, compute the mRNA density (transcripts per pixel) in each bin. Real cells are mRNA-rich, background is sparse — the cellprob value where density drops to a baseline is your cutoff. Below that, you're keeping pixels with no transcript signal."""))

cells.append(nbf.v4.new_code_cell("""# For each transcript, get the cellprob at that pixel
tx_px = ((df_tx["x_um"].to_numpy() - x0_um) / meta["pixel_size_um"]).astype(int)
tx_py = ((df_tx["y_um"].to_numpy() - y0_um) / meta["pixel_size_um"]).astype(int)
H_, W_ = cellprob.shape
inb = (tx_px >= 0) & (tx_px < W_) & (tx_py >= 0) & (tx_py < H_)
tx_cprob = cellprob[tx_py[inb], tx_px[inb]]

cprob_bin_edges   = np.linspace(cellprob.min() - 0.01, cellprob.max() + 0.01, 60)
cprob_bin_centers = 0.5 * (cprob_bin_edges[1:] + cprob_bin_edges[:-1])

pixel_counts, _ = np.histogram(cellprob.flatten(), bins=cprob_bin_edges)
tx_counts,    _ = np.histogram(tx_cprob,           bins=cprob_bin_edges)
density = tx_counts / np.maximum(pixel_counts, 1)   # transcripts per pixel

# Cumulative: at threshold T, what fraction of transcripts are kept and what area is kept?
sort_idx = np.argsort(cellprob.flatten())
cprob_sorted = cellprob.flatten()[sort_idx]
total_pixels = cprob_sorted.size

# For each threshold T (taken at percentile points), compute fraction of transcripts kept
ts = np.linspace(cellprob.min(), cellprob.max(), 100)
frac_tx, frac_area = [], []
for T in ts:
    keep_pixel = cellprob > T
    frac_area.append(keep_pixel.mean())
    if inb.sum() > 0:
        frac_tx.append((tx_cprob > T).sum() / inb.sum())
    else:
        frac_tx.append(0.0)
frac_tx, frac_area = np.array(frac_tx), np.array(frac_area)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Left: density curve — transcripts/pixel as function of cprob bin
ax[0].plot(cprob_bin_centers, density, marker="o", ms=4, lw=1.5, color="C0")
ax[0].set_xlabel("cellprob (log-odds)")
ax[0].set_ylabel("transcripts per pixel  in this cprob bin")
ax[0].set_title("mRNA density as a function of cellprob")
ax[0].axvline(0.0,  ls="--", color="black", alpha=0.5, label="cellprob=0 (default)")
ax[0].axvline(-0.5, ls="--", color="C1",    alpha=0.7, label="cellprob=-0.5")
ax[0].axvline(-2.0, ls="--", color="C2",    alpha=0.7, label="cellprob=-2.0")
ax[0].axvline(-3.0, ls="--", color="C3",    alpha=0.7, label="cellprob=-3.0")
ax[0].legend(loc="upper left", fontsize=8)
ax[0].grid(alpha=0.3)

# Right: trade-off — for each threshold, % transcripts kept vs % area kept
ax[1].plot(ts, frac_tx*100,   color="C0", lw=2, label="% transcripts kept")
ax[1].plot(ts, frac_area*100, color="C3", lw=2, label="% area kept")
ax[1].set_xlabel("cellprob threshold")
ax[1].set_ylabel("%")
ax[1].set_title("Threshold trade-off")
ax[1].axvline(0.0,  ls="--", color="black", alpha=0.5)
ax[1].axvline(-0.5, ls="--", color="C1",    alpha=0.7)
ax[1].axvline(-2.0, ls="--", color="C2",    alpha=0.7)
ax[1].axvline(-3.0, ls="--", color="C3",    alpha=0.7)
ax[1].legend(loc="lower left", fontsize=8)
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# Background density baseline = density in the lowest-cprob bins (definitely no cells there)
bg_bins = density[cprob_bin_centers < -5]
bg_density = bg_bins.mean() if len(bg_bins) else 0.0
print(f"background mRNA density (cprob<-5):  {bg_density:.4f} tx/pixel")
print(f"max mRNA density (highest cprob bin): {density.max():.4f} tx/pixel  (~{density.max()/max(bg_density,1e-6):.0f}× background)")
for T in [0.0, -0.5, -2.0, -3.0]:
    sel = cprob_bin_centers >= T
    if sel.sum():
        d_above = (density[sel] * pixel_counts[sel]).sum() / max(pixel_counts[sel].sum(), 1)
        print(f"  thr={T:5.1f}:  area={100*(cellprob>T).mean():5.1f}%   "
              f"tx kept={100*(tx_cprob>T).sum()/max(inb.sum(),1):5.1f}%   "
              f"avg density={d_above:.4f} tx/px")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2D density of cellprob vs flow magnitude

Where real cells live in (cellprob, |flow|) space. Each pixel contributes one point. Dashed lines mark threshold settings."""))

cells.append(nbf.v4.new_code_cell("""# 2D histogram over all pixels (subsample to keep it fast)
from matplotlib.colors import LogNorm

stride = 2
cp_flat  = cellprob[::stride, ::stride].flatten()
mag_flat = flow_mag[::stride, ::stride].flatten()

# Log bins on |flow| axis — flow magnitudes span several orders of magnitude
mag_floor = 1e-2
mag_safe  = np.maximum(mag_flat, mag_floor)
y_bins = np.logspace(np.log10(mag_floor), np.log10(mag_safe.max()), 80)
x_bins = np.linspace(cp_flat.min(), cp_flat.max(), 80)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
h = ax[0].hist2d(cp_flat, mag_safe, bins=[x_bins, y_bins], cmap="viridis", norm=LogNorm())
ax[0].set_xlabel("cellprob (log-odds)"); ax[0].set_ylabel("|flow|  (log scale)")
ax[0].set_yscale("log")
ax[0].set_title("2D density: cellprob vs flow magnitude (all pixels)")
ax[0].axvline(0.0,   ls="--", color="white", alpha=0.7, label="cellprob=0 (default)")
ax[0].axvline(-0.5,  ls="--", color="cyan",  alpha=0.7, label="cellprob=-0.5")
ax[0].axvline(-2.0,  ls="--", color="orange",alpha=0.7, label="cellprob=-2.0")
ax[0].legend(loc="upper left", fontsize=8)
plt.colorbar(h[3], ax=ax[0], label="pixel count (log)")

# Restricted to pixels inside 10X cells
in_cell = (m_10x_compact[::stride, ::stride] > 0).flatten()
h2 = ax[1].hist2d(cp_flat[in_cell], mag_safe[in_cell],
                   bins=[x_bins, y_bins], cmap="magma", norm=LogNorm())
ax[1].set_xlabel("cellprob (log-odds)"); ax[1].set_ylabel("|flow|  (log scale)")
ax[1].set_yscale("log")
ax[1].set_title("Same, restricted to pixels inside 10X cells")
ax[1].axvline(0.0,   ls="--", color="white", alpha=0.7)
ax[1].axvline(-0.5,  ls="--", color="cyan",  alpha=0.7)
ax[1].axvline(-2.0,  ls="--", color="orange",alpha=0.7)
plt.colorbar(h2[3], ax=ax[1], label="pixel count (log)")
plt.tight_layout(); plt.show()

# Quick stats
in_cell_full = (m_10x_compact > 0)
print(f"Pixels inside 10X cells: {in_cell_full.sum():,}")
print(f"  median cellprob inside: {np.median(cellprob[in_cell_full]):.2f}")
print(f"  median |flow| inside  : {np.median(flow_mag[in_cell_full]):.2f}")
print(f"  fraction of in-cell pixels with cellprob<0: {(cellprob[in_cell_full] < 0).mean()*100:.1f}%")
print(f"  fraction of in-cell pixels with cellprob<-2:{(cellprob[in_cell_full] < -2).mean()*100:.1f}%")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. Side-by-side: 10X vs CP-SAM (DAPI + 18S)

Top row: cell masks overlaid on the DAPI+18S composite.
Bottom row: transcripts as gray dots on the same composite (no cell overlay).
"""))

cells.append(nbf.v4.new_code_cell("""from skimage.segmentation import find_boundaries

def shuffle_labels(masks, seed=0):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return masks
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n + 1))])
    return perm[masks]

def label_overlay(masks, seed, alpha_fill=0.45, edge_alpha=1.0, edge_color=(1, 1, 1)):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0:
        return np.zeros((*masks.shape, 4), dtype=np.float32)
    shuf = shuffle_labels(masks, seed=seed)
    cmap = plt.get_cmap("nipy_spectral")
    rgba = cmap(shuf / max(n, 1))
    rgba[..., 3] = alpha_fill * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges, 0] = edge_color[0]; rgba[edges, 1] = edge_color[1]
    rgba[edges, 2] = edge_color[2]; rgba[edges, 3] = edge_alpha
    return rgba

panels = [
    ("10X cells",            m_10x_compact, 7),
    ("CP-SAM (DAPI + 18S)",  masks_sam,    11),
]

def transcripts_inside(masks):
    \"\"\"Boolean array: for each transcript, True if it lands inside any cell mask.\"\"\"
    H, W = masks.shape
    px = ((df_tx["x_um"].to_numpy() - x0_um) / meta["pixel_size_um"]).astype(int)
    py = ((df_tx["y_um"].to_numpy() - y0_um) / meta["pixel_size_um"]).astype(int)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    inside = np.zeros(len(df_tx), dtype=bool)
    inside[inb] = masks[py[inb], px[inb]] > 0
    return inside

H_im, W_im = composite.shape[:2]
panel_w = 7.5
panel_h = panel_w * (H_im / W_im)
fig, ax = plt.subplots(2, 2, figsize=(panel_w * 2 + 0.5, panel_h * 2 + 0.4))
for j, (title, masks, seed) in enumerate(panels):
    ax[0, j].imshow(composite,                       aspect="equal", extent=extent)
    ax[0, j].imshow(label_overlay(masks, seed=seed), aspect="equal", extent=extent)
    ax[0, j].set_title(f"{title}  (n = {int(masks.max())})")
    ax[0, j].set_xlabel("x (µm)"); ax[0, j].set_ylabel("y (µm)")

    inside = transcripts_inside(masks)
    n_in, n_out = int(inside.sum()), int((~inside).sum())
    ax[1, j].imshow(composite, aspect="equal", extent=extent)
    # gray dots = inside a cell, red dots = orphan transcripts
    ax[1, j].scatter(df_tx.loc[inside,  "x_um"], df_tx.loc[inside,  "y_um"],
                     s=1.5, c="0.6", alpha=0.5, edgecolors="none", label=f"inside cell ({n_in:,})")
    ax[1, j].scatter(df_tx.loc[~inside, "x_um"], df_tx.loc[~inside, "y_um"],
                     s=2.5, c="red",  alpha=0.7, edgecolors="none", label=f"orphan ({n_out:,})")
    ax[1, j].set_xlim(x0_um, x1_um); ax[1, j].set_ylim(y1_um, y0_um)
    ax[1, j].set_title(f"transcripts: {100*n_in/(n_in+n_out):.1f}% assigned to {title.split()[0]}")
    ax[1, j].legend(loc="lower right", fontsize=8, markerscale=2)
    ax[1, j].set_xlabel("x (µm)"); ax[1, j].set_ylabel("y (µm)")
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 8. Tuning thresholds — two non-monotonic knobs

Counter-intuitive fact: **lowering `cellprob_threshold` does not always mean more cells.** Two filters compete:

1. *`cellprob_threshold`* (log-odds; default `0.0`) — pixel-level "is this a cell" filter. Lower = more pixels pass.
2. After mask reconstruction, cellpose drops cells that are larger than `max_size_fraction=0.4` (40% of the image) or fail the *flow-coherence* check controlled by `flow_threshold` (default `0.4`; higher = more permissive).

So when you drop `cellprob_threshold` too far, adjacent cells' pixel-masks bleed together, the flow-following step merges them into one giant blob, and that blob gets filtered out by `max_size_fraction`. Net effect: fewer cells. The sweep below spans both directions to make this visible."""))

cells.append(nbf.v4.new_code_cell("""def run_eval(cellprob_thr, flow_thr=0.0):
    \"\"\"NOTE: flow_thr=0.0 (off) by default — otherwise the merge-blob filter masks
    the actual cellprob effect at low thresholds.\"\"\"
    masks, _, _ = m_sam.eval(
        img_2ch, channel_axis=-1, diameter=None, niter=200,
        cellprob_threshold=cellprob_thr, flow_threshold=flow_thr,
    )
    return masks

# cellprob_threshold sweep at flow_threshold=0 (off) so cprob effect is monotonic
cp_sweep = [-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
cp_results = {thr: run_eval(thr) for thr in cp_sweep}

# flow_threshold sweep at very-low cellprob so we can see flow's actual effect
# (with default cellprob, the cellprob filter would mask flow's role)
flow_sweep = [0.0, 0.4, 0.8, 1.0, 2.0]
flow_results = {ft: run_eval(-5.0, ft) for ft in flow_sweep}

print("# cellprob_threshold sweep (flow_threshold=0.0):")
for thr, m in cp_results.items():
    print(f"   {thr:5.1f}  →  {int(m.max()):4d} cells")

print("\\n# flow_threshold sweep (cellprob_threshold=-5.0):")
for ft, m in flow_results.items():
    print(f"   {ft:5.1f}  →  {int(m.max()):4d} cells")
"""))

cells.append(nbf.v4.new_code_cell("""# Helper: 2-row sweep plot — top row masks, bottom row transcripts (gray=inside, red=outside)
def plot_sweep_2row(configs, seed_offset=30, panel_w=6.0):
    \"\"\"configs: list of (title, masks). Renders a (2 × len(configs)) grid.
    Panel height is derived from the ROI's actual aspect to avoid whitespace.\"\"\"
    n = len(configs)
    H_im, W_im = composite.shape[:2]
    panel_h = panel_w * (H_im / W_im)
    fig, ax = plt.subplots(2, n, figsize=(panel_w * n, panel_h * 2 + 0.6), squeeze=False)
    for k, (title, masks) in enumerate(configs):
        ax[0, k].imshow(composite,                              aspect="equal", extent=extent)
        ax[0, k].imshow(label_overlay(masks, seed=seed_offset+k), aspect="equal", extent=extent)
        ax[0, k].set_title(f"{title}  (n = {int(masks.max())})")
        ax[0, k].set_xlabel("x (µm)"); ax[0, k].set_ylabel("y (µm)")

        inside = transcripts_inside(masks)
        n_in, n_out = int(inside.sum()), int((~inside).sum())
        ax[1, k].imshow(composite, aspect="equal", extent=extent)
        ax[1, k].scatter(df_tx.loc[inside,  "x_um"], df_tx.loc[inside,  "y_um"],
                          s=1.2, c="0.6", alpha=0.45, edgecolors="none")
        ax[1, k].scatter(df_tx.loc[~inside, "x_um"], df_tx.loc[~inside, "y_um"],
                          s=2.2, c="red",  alpha=0.7, edgecolors="none")
        ax[1, k].set_xlim(x0_um, x1_um); ax[1, k].set_ylim(y1_um, y0_um)
        ax[1, k].set_title(f"transcripts: {100*n_in/(n_in+n_out):.1f}% in cells")
        ax[1, k].set_xlabel("x (µm)"); ax[1, k].set_ylabel("y (µm)")
    plt.tight_layout(); plt.show()

# transcripts_inside is defined later in the side-by-side cell; redefine here for use up here too
def transcripts_inside(masks):
    H, W = masks.shape
    px = ((df_tx["x_um"].to_numpy() - x0_um) / meta["pixel_size_um"]).astype(int)
    py = ((df_tx["y_um"].to_numpy() - y0_um) / meta["pixel_size_um"]).astype(int)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    inside = np.zeros(len(df_tx), dtype=bool)
    inside[inb] = masks[py[inb], px[inb]] > 0
    return inside

# Aggressive / negative-control end (cprob=-5, -4, -3)
plot_sweep_2row([(f"cellprob_thr = {thr}", cp_results[thr]) for thr in cp_sweep if thr <= -3], seed_offset=30)
# Mainline sweep (cprob=-2 → +1)
plot_sweep_2row([(f"cellprob_thr = {thr}", cp_results[thr]) for thr in cp_sweep if thr >= -2], seed_offset=33)
"""))

cells.append(nbf.v4.new_code_cell("""plot_sweep_2row([(f"flow_thr = {ft}", flow_results[ft]) for ft in flow_sweep], seed_offset=40)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 8c. Eval-pipeline knobs — augmentation, niter, max-recall configs

When cellprob and flow magnitude both look fine but a cell still isn't recovered, the issue is in the eval pipeline:

- **`augment=True`**: runs 4 rotations of the image and averages predictions. Often recovers cells the single-pass eval misses.
- **`tile_overlap`** (default 0.1): fraction of tile-size overlap between tiles. Increasing it gives the network more context near tile boundaries.
- **`niter`** (default auto): max iterations for the flow-following dynamics. Cells whose flows take more steps to converge are more likely to fail.
- **MAX RECALL**: lower `cellprob_threshold` *and* `flow_threshold` simultaneously — false positives are filterable downstream by Baysor based on transcript voting."""))

cells.append(nbf.v4.new_code_cell("""def run_eval3(**kw):
    base = dict(channel_axis=-1, diameter=None, niter=200,
                cellprob_threshold=-0.5, flow_threshold=0.0)
    base.update(kw)
    masks, _, _ = m_sam.eval(img_2ch, **base)
    return masks

H_im, W_im = composite.shape[:2]

pipeline_runs = {
    "default eval (cprob=-0.5, flow=0)":       run_eval3(),
    "augment=True":                             run_eval3(augment=True),
    # Max-recall: lower both thresholds simultaneously + augment
    "MAX RECALL (cprob=-2, flow=0, augment)":
        run_eval3(cellprob_threshold=-2.0, flow_threshold=0.0, augment=True),
    "MAX RECALL (cprob=-3, flow=0, augment)":
        run_eval3(cellprob_threshold=-3.0, flow_threshold=0.0, augment=True),
    "MAX RECALL (cprob=-4, flow=0, augment)":
        run_eval3(cellprob_threshold=-4.0, flow_threshold=0.0, augment=True),
    "MAX RECALL (cprob=-5, flow=0, augment) [recommended]":
        run_eval3(cellprob_threshold=-5.0, flow_threshold=0.0, augment=True),
}
for name, m in pipeline_runs.items():
    print(f"  {name:40s}  →  {int(m.max()):4d} cells")
"""))

cells.append(nbf.v4.new_code_cell("""# Visualize the most promising of the new runs
plot_sweep_2row(
    [(name, masks) for name, masks in pipeline_runs.items()],
    seed_offset=70,
)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Pinpoint a missed cell — what label does CP-SAM give that pixel?

Set `PROBE_XY_UM` to a point inside the cell you suspect was missed, and this cell prints the mask label at that location across every sweep result. If the cell shows **0** everywhere → it's truly being filtered. If it shows **a nonzero label** in some configs → the cell IS being called but merged with a neighbor (the label is the neighbor's id)."""))

cells.append(nbf.v4.new_code_cell("""# ─── pinpoint a single pixel inside the suspected missed cell ────────────
PROBE_XY_UM = (1140.0, 3010.0)   # adjust to a missed cell location in microns
# ────────────────────────────────────────────────────────────────────────

px = int(round((PROBE_XY_UM[0] - x0_um) / meta["pixel_size_um"]))
py = int(round((PROBE_XY_UM[1] - y0_um) / meta["pixel_size_um"]))
print(f"probe at ({PROBE_XY_UM[0]:.0f}, {PROBE_XY_UM[1]:.0f}) µm  →  pixel ({py}, {px})\\n")
print(f"  cellprob at that pixel  : {cellprob[py, px]:.2f}  (log-odds)")
print(f"  flow magnitude          : {flow_mag[py, px]:.2f}")
print(f"  10X mask label here     : {int(m_10x_compact[py, px])}")
print(f"\\n  CP-SAM mask label across configs:")
all_runs = {
    "default (cprob=0, flow=0.4)":     masks_sam,
    "cprob=-0.5, flow=0.0":            pipeline_runs["default eval (cprob=-0.5, flow=0)"],
    "augment=True":                    pipeline_runs["augment=True"],
    "MAX RECALL (cprob=-2)":           pipeline_runs["MAX RECALL (cprob=-2, flow=0, augment)"],
    "MAX RECALL (cprob=-3)":           pipeline_runs["MAX RECALL (cprob=-3, flow=0, augment)"],
    "MAX RECALL (cprob=-5) [best]":    pipeline_runs["MAX RECALL (cprob=-5, flow=0, augment) [recommended]"],
}
for name, m in all_runs.items():
    lbl = int(m[py, px])
    note = "MISSED" if lbl == 0 else f"called as cell #{lbl}"
    print(f"    {name:30s}  →  {note}")

# Visualize a 60×60 µm crop around the probe with a crosshair so we can see it
hp_um = 30.0
zx0, zx1 = PROBE_XY_UM[0]-hp_um, PROBE_XY_UM[0]+hp_um
zy0, zy1 = PROBE_XY_UM[1]-hp_um, PROBE_XY_UM[1]+hp_um
zix0 = max(0, int(round((zx0 - x0_um) / meta["pixel_size_um"])))
zix1 = min(W_im, int(round((zx1 - x0_um) / meta["pixel_size_um"])))
ziy0 = max(0, int(round((zy0 - y0_um) / meta["pixel_size_um"])))
ziy1 = min(H_im, int(round((zy1 - y0_um) / meta["pixel_size_um"])))
zoom_extent2 = [zx0, zx1, zy1, zy0]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(composite[ziy0:ziy1, zix0:zix1], aspect="equal", extent=zoom_extent2)
ax[0].plot(PROBE_XY_UM[0], PROBE_XY_UM[1], "+", color="white", ms=15, mew=2)
ax[0].set_title("composite (probe ✚)")
ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)")

im = ax[1].imshow(cellprob[ziy0:ziy1, zix0:zix1], cmap="magma", aspect="equal", extent=zoom_extent2)
ax[1].plot(PROBE_XY_UM[0], PROBE_XY_UM[1], "+", color="white", ms=15, mew=2)
ax[1].set_title("cellprob"); fig.colorbar(im, ax=ax[1], fraction=0.046)

best_masks = pipeline_runs["MAX RECALL (cprob=-5, flow=0, augment) [recommended]"]
ax[2].imshow(composite[ziy0:ziy1, zix0:zix1], aspect="equal", extent=zoom_extent2)
ax[2].imshow(label_overlay(best_masks[ziy0:ziy1, zix0:zix1], seed=88), aspect="equal", extent=zoom_extent2)
ax[2].plot(PROBE_XY_UM[0], PROBE_XY_UM[1], "+", color="white", ms=15, mew=2)
ax[2].set_title("masks (augment=True + no tiling)")
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 9. Quick numerical comparison vs 10X"))

cells.append(nbf.v4.new_code_cell("""from scipy.optimize import linear_sum_assignment

def per_cell_iou(pred, truth):
    plabs = [int(l) for l in np.unique(pred)  if l != 0]
    tlabs = [int(l) for l in np.unique(truth) if l != 0]
    if not plabs or not tlabs:
        return np.array([])
    iou = np.zeros((len(tlabs), len(plabs)), dtype=np.float32)
    for i, T in enumerate(tlabs):
        tm = truth == T
        for j, P in enumerate(plabs):
            inter = (tm & (pred == P)).sum()
            if inter == 0: continue
            union = (tm | (pred == P)).sum()
            iou[i, j] = inter / union if union else 0.0
    ri, ci = linear_sum_assignment(-iou)
    return np.array([iou[i, j] for i, j in zip(ri, ci)])

def pct_tx_assigned(masks):
    \"\"\"Fraction of transcripts whose (x_um, y_um) lands inside any cell mask.\"\"\"
    tx_x_px = ((df_tx["x_um"].to_numpy() - x0_um) / meta["pixel_size_um"]).astype(int)
    tx_y_px = ((df_tx["y_um"].to_numpy() - y0_um) / meta["pixel_size_um"]).astype(int)
    H, W = masks.shape
    inb = (tx_x_px >= 0) & (tx_x_px < W) & (tx_y_px >= 0) & (tx_y_px < H)
    inside = (masks[tx_y_px[inb], tx_x_px[inb]] > 0).sum()
    return 100.0 * inside / max(inb.sum(), 1)

print(f"10X cells: {int(m_10x_compact.max())},  10X tx-assignment: {pct_tx_assigned(m_10x_compact):.1f}%\\n")

hdr = f"{'thr':>6s}  {'n_cells':>8s}  {'matched':>8s}  {'mean IoU':>9s}  {'≥0.5':>5s}  {'≥0.7':>5s}  {'tx%':>6s}"
print(f"# cellprob_threshold sweep (flow_threshold=0.0)")
print(hdr)
for thr, masks in cp_results.items():
    ious = per_cell_iou(masks.astype(np.int64), m_10x_compact)
    print(f"  {thr:4.1f}    {int(masks.max()):6d}    {len(ious):6d}    {ious.mean():7.3f}    "
          f"{(ious>=0.5).mean()*100:4.0f}%   {(ious>=0.7).mean()*100:4.0f}%   {pct_tx_assigned(masks):5.1f}%")

print(f"\\n# flow_threshold sweep (cellprob_threshold=-5.0)")
print(hdr)
for ft, masks in flow_results.items():
    ious = per_cell_iou(masks.astype(np.int64), m_10x_compact)
    print(f"  {ft:4.1f}    {int(masks.max()):6d}    {len(ious):6d}    {ious.mean():7.3f}    "
          f"{(ious>=0.5).mean()*100:4.0f}%   {(ious>=0.7).mean()*100:4.0f}%   {pct_tx_assigned(masks):5.1f}%")

print(f"\\n# eval-pipeline + max-recall configs")
print(f"{'config':>50s}  {'n_cells':>8s}  {'matched':>8s}  {'mean IoU':>9s}  {'≥0.5':>5s}  {'≥0.7':>5s}  {'tx%':>6s}")
for name, masks in pipeline_runs.items():
    ious = per_cell_iou(masks.astype(np.int64), m_10x_compact)
    print(f"  {name[:48]:>48s}    {int(masks.max()):6d}    {len(ious):6d}    {ious.mean():7.3f}    "
          f"{(ious>=0.5).mean()*100:4.0f}%   {(ious>=0.7).mean()*100:4.0f}%   {pct_tx_assigned(masks):5.1f}%")
"""))

nb.cells = cells

out = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/tools/xenium_roi_crop_cpsam.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote:", out)
