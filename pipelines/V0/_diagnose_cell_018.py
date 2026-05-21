"""V0 diagnostic on DB_top100_018 — exhibit every intermediate.

Why this script exists: V0 batch produced 0/27 improvements after 45 min.
Need to know whether the cut is (a) placed in the right location, (b) deep
enough after CP-SAM's per-channel renormalisation, or (c) missing entirely.

Saves a single big figure with annotated intermediates.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lib

BID = "DB_top100_018"
CROP_PX = 2048
OUT_PNG = HERE / "_diagnose_cell_018.png"

bench = pd.read_parquet(lib.DATA / "benchmark_doublets.parquet")
row = bench[bench["benchmark_id"] == BID].iloc[0]
cx_um, cy_um = float(row["x_um"]), float(row["y_um"])
cps_focal = int(row["cps_id_at_bookmark"])
lin_a, lin_b = row["lineage_pair"].split(" × ")
ka, kb = lib.LIN_TO_IDX[lin_a], lib.LIN_TO_IDX[lin_b]
print(f"{BID}: {lin_a} (k={ka}) × {lin_b} (k={kb})  focal={cps_focal}")

with tifffile.TiffFile(lib.DAPI_TIF) as tf:
    wsi_H, wsi_W = tf.series[0].shape[-2:]
y0, y1, x0, x1 = lib.make_roi_bbox_centred(cx_um, cy_um, CROP_PX, wsi_H, wsi_W)

dapi, s18 = lib.load_morphology(y0, y1, x0, x1)
print(f"DAPI range: [{dapi.min()}, {dapi.max()}]  mean={dapi.mean():.1f}")
print(f"18S range:  [{s18.min()}, {s18.max()}]   mean={s18.mean():.1f}")
print(f"DAPI == 18S?  pixels equal: {(dapi == s18).sum()}/{dapi.size}  fraction: {(dapi == s18).mean():.3f}")

py, px, li = lib.load_anchors_in_bbox(y0, y1, x0, x1, lib.load_lineage_label_map())
H, W = s18.shape
pi_abst, confidence, _ = lib.lineage_posterior(py, px, li, H, W)
edge_total, top_idx, margin = lib.edge_global_field(pi_abst)

percentiles = lib.load_wsi_percentiles()
evidence = lib.cell_evidence(dapi, s18, percentiles)
cut, weighted_edge = lib.cut_field_from_edge(edge_total, confidence, evidence)
s_cut = lib.apply_cut(s18, cut)

# Pre- and post-renormalisation channel statistics (what CP-SAM actually sees)
def fresh_clip_stats(arr, label):
    rng = np.random.default_rng(0)
    sample = arr.ravel() if arr.size < 10_000_000 else arr.ravel()[rng.integers(0, arr.size, 10_000_000)]
    q_lo, q_hi = np.percentile(sample, [1.0, 99.0])
    print(f"  {label}: fresh 1-99 percentile = [{q_lo:.0f}, {q_hi:.0f}]  span={q_hi-q_lo:.0f}")
    return float(q_lo), float(q_hi)

print("\n--- Per-channel renormalisation that CP-SAM applies ---")
dapi_lo, dapi_hi = fresh_clip_stats(dapi, "DAPI         ")
s18_lo,  s18_hi  = fresh_clip_stats(s18,  "raw 18S      ")
scut_lo, scut_hi = fresh_clip_stats(s_cut, "s_cut        ")

# What fraction of 18S pixels has cut > 0.5? > 0.8?
print(f"\n--- Cut footprint ---")
print(f"  cut > 0.1 : {(cut > 0.1).mean()*100:.2f}% of pixels")
print(f"  cut > 0.5 : {(cut > 0.5).mean()*100:.2f}% of pixels")
print(f"  cut > 0.9 : {(cut > 0.9).mean()*100:.2f}% of pixels")
print(f"  cut max = {cut.max():.3f}")

# WSI focal-mask footprint
wsi_mask_crop = tifffile.imread(lib.DATA / "cpsam_whole_slide" / "masks.tif")[y0:y1, x0:x1]
focal = (wsi_mask_crop == cps_focal)
print(f"\nFocal cell {cps_focal} footprint in 2048 crop: {focal.sum()} px ({focal.mean()*100:.2f}%)")

# Argmax label inside the focal cell — what does V0 *think* this cell is made of?
print(f"\n--- Argmax composition inside focal cell ---")
for k in range(lib.K):
    n_in_focal = ((top_idx == k) & focal).sum()
    if n_in_focal > 0:
        frac = n_in_focal / focal.sum()
        print(f"  {lib.LINEAGES[k]:13}  {frac*100:5.1f}%  (max π_k inside focal = {pi_abst[k][focal].max():.3f})")

# How much of the cut field actually lands INSIDE the focal cell?
cut_in_focal = cut[focal]
print(f"\nCut field inside focal cell:")
print(f"  max  = {cut_in_focal.max():.3f}")
print(f"  mean = {cut_in_focal.mean():.3f}")
print(f"  fraction > 0.5 = {(cut_in_focal > 0.5).mean()*100:.2f}%")

# === FIGURE ==========================================================
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

# Focal-zoom crop
ys, xs = np.where(focal)
pad = 30
zy0 = max(ys.min() - pad, 0); zy1 = min(ys.max() + pad, H)
zx0 = max(xs.min() - pad, 0); zx1 = min(xs.max() + pad, W)
zcrop = (slice(zy0, zy1), slice(zx0, zx1))
focal_z = focal[zcrop]
focal_bound = find_boundaries(focal_z, mode="outer")

def overlay(ax, show_anchors=False):
    ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.2)
    if show_anchors:
        in_z = ((py >= zy0) & (py < zy1) & (px >= zx0) & (px < zx1))
        for k_lin in range(lib.K):
            m = in_z & (li == k_lin)
            if m.any():
                ax.scatter(px[m] - zx0, py[m] - zy0, s=12,
                           color=plt.get_cmap("tab10").colors[k_lin],
                           edgecolors="black", linewidths=0.3)
    ax.set_xticks([]); ax.set_yticks([])

# Row 0: raw inputs
axes[0,0].imshow(np.log1p(dapi[zcrop]), cmap="gray"); overlay(axes[0,0])
axes[0,0].set_title("DAPI (log1p) + focal contour")
axes[0,1].imshow(np.log1p(s18[zcrop]), cmap="gray"); overlay(axes[0,1])
axes[0,1].set_title("raw 18S (log1p) + focal contour")
axes[0,2].imshow(np.log1p(s18[zcrop]), cmap="gray"); overlay(axes[0,2], show_anchors=True)
axes[0,2].set_title(f"18S + anchors (focal lineages: {lin_a}, {lin_b})")
axes[0,3].imshow(evidence[zcrop], cmap="gray", vmin=0, vmax=1); overlay(axes[0,3])
axes[0,3].set_title(f"cell_evidence = max(DAPI_n, 18S_n)\nmean inside focal = {evidence[focal].mean():.3f}")

# Row 1: per-lineage posterior (the doublet's two lineages + others-collapsed + argmax)
axes[1,0].imshow(pi_abst[ka][zcrop], cmap="viridis", vmin=0, vmax=pi_abst.max())
overlay(axes[1,0]); axes[1,0].set_title(f"π[{lin_a}]  max in focal = {pi_abst[ka][focal].max():.3f}")
axes[1,1].imshow(pi_abst[kb][zcrop], cmap="viridis", vmin=0, vmax=pi_abst.max())
overlay(axes[1,1]); axes[1,1].set_title(f"π[{lin_b}]  max in focal = {pi_abst[kb][focal].max():.3f}")
# π_diff for the pair
pi_diff = pi_abst[ka] - pi_abst[kb]
vm = max(abs(pi_diff[zcrop].min()), abs(pi_diff[zcrop].max()))
axes[1,2].imshow(pi_diff[zcrop], cmap="RdBu_r", vmin=-vm, vmax=vm)
sign_change = find_boundaries((pi_diff > 0).astype(np.uint8), mode="inner")
axes[1,2].contour(sign_change[zcrop].astype(int), levels=[0.5], colors="lime", linewidths=1.5)
overlay(axes[1,2]); axes[1,2].set_title(f"π[{lin_a}] − π[{lin_b}]\nlime = pair sign-change")
# Argmax label map (V0 one-vs-rest "lineage of this pixel")
lin_cmap = ListedColormap([plt.get_cmap("tab10").colors[k] for k in range(lib.K)])
axes[1,3].imshow(top_idx[zcrop], cmap=lin_cmap, vmin=-0.5, vmax=lib.K-0.5)
overlay(axes[1,3]); axes[1,3].set_title("argmax(π) — V0 lineage label per pixel")

# Row 2: edge field + cut
axes[2,0].imshow(margin[zcrop], cmap="magma", vmin=0, vmax=margin[zcrop].max())
overlay(axes[2,0]); axes[2,0].set_title(f"margin = π_top1 − π_top2\nmax in focal = {margin[focal].max():.3f}")
v_edge = np.percentile(edge_total, 99.5)
axes[2,1].imshow(edge_total[zcrop], cmap="hot", vmin=0, vmax=v_edge)
overlay(axes[2,1]); axes[2,1].set_title(f"edge_total = Σ_k |∇ tanh(K·s_k)|\nmean = {edge_total.mean():.3f}  max = {edge_total.max():.2f}")
v_w = np.percentile(weighted_edge, 99.5)
axes[2,2].imshow(weighted_edge[zcrop], cmap="hot", vmin=0, vmax=v_w)
overlay(axes[2,2]); axes[2,2].set_title(f"weighted_edge = edge · c² · evidence\nmean = {weighted_edge.mean():.3f}")
axes[2,3].imshow(cut[zcrop], cmap="hot", vmin=0, vmax=1)
overlay(axes[2,3])
axes[2,3].set_title(f"cut_field (σ={lib.SIGMA_CUT_PX}, d={lib.DEPTH_MAX})\nmean in focal = {cut[focal].mean():.3f}")

# Row 3: what CP-SAM actually sees — pre- and post-normalisation
def clip01(arr, q_lo, q_hi):
    a = arr.astype(np.float32)
    a = (a - q_lo) / max(q_hi - q_lo, 1e-6)
    return np.clip(a, 0, 1)
dapi_norm = clip01(dapi, dapi_lo, dapi_hi)
s18_norm  = clip01(s18,  s18_lo,  s18_hi)
scut_norm = clip01(s_cut, scut_lo, scut_hi)
axes[3,0].imshow(s18_norm[zcrop], cmap="gray", vmin=0, vmax=1); overlay(axes[3,0])
axes[3,0].set_title(f"control 18S (post-CP-SAM normalisation)\nq_hi={s18_hi:.0f}")
axes[3,1].imshow(scut_norm[zcrop], cmap="gray", vmin=0, vmax=1); overlay(axes[3,1])
axes[3,1].set_title(f"s_cut (post-CP-SAM normalisation)\nq_hi={scut_hi:.0f}  Δ={s18_hi-scut_hi:.0f}")
diff = (scut_norm - s18_norm)[zcrop]
vmd = max(abs(diff.min()), abs(diff.max()), 1e-3)
axes[3,2].imshow(diff, cmap="RdBu_r", vmin=-vmd, vmax=vmd)
overlay(axes[3,2])
axes[3,2].set_title(f"s_cut_norm − 18S_norm  (RED=cut dimmed it)\nmax dim = {-diff.min():.3f}  max brighten = {diff.max():.3f}")
# Verify dimming amount at the strongest cut pixel
cut_max_y, cut_max_x = np.unravel_index(cut.argmax(), cut.shape)
in_focal = focal[cut_max_y, cut_max_x]
axes[3,3].imshow(cut[zcrop], cmap="hot", vmin=0, vmax=1, alpha=0.5)
axes[3,3].imshow(s18_norm[zcrop], cmap="gray", vmin=0, vmax=1, alpha=0.5)
overlay(axes[3,3])
axes[3,3].set_title(f"cut field overlaid on 18S_norm\nstrongest cut at ({cut_max_y},{cut_max_x}) inside_focal={in_focal}")

plt.suptitle(
    f"{BID} V0 diagnostic — α={lib.ALPHA}, K_sharp={lib.K_SHARP}, σ_diff={lib.SIGMA_DIFFUSE_PX:.2f}, "
    f"σ_cut={lib.SIGMA_CUT_PX}, depth={lib.DEPTH_MAX}\n"
    f"pi_abst range [{pi_abst.min():.3f}, {pi_abst.max():.3f}]  "
    f"|  raw18S q_hi {s18_hi:.0f} → s_cut q_hi {scut_hi:.0f} ({(scut_hi/s18_hi-1)*100:+.0f}%)",
    fontsize=13, y=1.002)
plt.tight_layout()
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
print(f"\nsaved {OUT_PNG}")
