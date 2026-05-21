"""Render onion-ring-flow Phase-1 figures from saved npz files.

(Phase 2 was bsize=512/niter=500/norm_from_256; bsize=512 crashed on
cellpose-SAM's fixed positional embedding, so we only have Phase 1 data.)

Produces:
  figs/onion_ring_flow/fig1_size_sweep_flows.png
  figs/onion_ring_flow/fig3_metrics_vs_size.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

OUT_DIR = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/figs/onion_ring_flow")
SIZES_PX = [256, 512, 1024, 2048, 4096]
RING_HALF_WIN = 20
ZOOM_PX = 100


def norm_local(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def label_overlay(masks, alpha=0.55):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(0).permutation(np.arange(1, n + 1))])
    shuf = perm[masks]
    rgba = plt.get_cmap("nipy_spectral")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def count_labels_at(masks, y, x, half=RING_HALF_WIN):
    H, W = masks.shape
    y0 = max(y - half, 0); y1 = min(y + half, H)
    x0 = max(x - half, 0); x1 = min(x + half, W)
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0)


def flow_coherence_at(flow_y, flow_x, y, x, half=RING_HALF_WIN):
    H, W = flow_x.shape
    y0 = max(y - half, 0); y1 = min(y + half, H)
    x0 = max(x - half, 0); x1 = min(x + half, W)
    fx = flow_x[y0:y1, x0:x1].astype(np.float32)
    fy = flow_y[y0:y1, x0:x1].astype(np.float32)
    mag = np.sqrt(fx * fx + fy * fy)
    if mag.max() < 1e-6: return 0.0
    ux = fx / (mag + 1e-6); uy = fy / (mag + 1e-6)
    return float(np.sqrt(ux.mean() ** 2 + uy.mean() ** 2))


# Fig 1: size sweep master grid
half_z = ZOOM_PX // 2
cols = ["DAPI (raw)", "18S (raw)", "masks (on 18S)", "flow_y", "flow_x", "cellprob"]
fig, axes = plt.subplots(len(SIZES_PX), len(cols), figsize=(3 * len(cols), 3 * len(SIZES_PX)))

records = []
for i, sp in enumerate(SIZES_PX):
    arr = np.load(OUT_DIR / f"phase1_size{sp}.npz")
    masks = arr["masks"]; flow_y = arr["flow_y"]; flow_x = arr["flow_x"]
    cellprob = arr["cellprob"]; dapi_r = arr["dapi_raw"]; s18_r = arr["s18_raw"]
    lx, ly = arr["ring_local"]
    H, W = masks.shape
    zy0 = max(ly - half_z, 0); zy1 = min(ly + half_z, H)
    zx0 = max(lx - half_z, 0); zx1 = min(lx + half_z, W)
    n_lab = count_labels_at(masks, ly, lx)
    coh = flow_coherence_at(flow_y, flow_x, ly, lx)
    records.append((sp, n_lab, coh))
    axes[i, 0].imshow(norm_local(dapi_r[zy0:zy1, zx0:zx1]), cmap="gray", interpolation="nearest")
    axes[i, 0].set_ylabel(f"size={sp} px\nlabels at ring = {n_lab}\nflow coh = {coh:.2f}",
                          fontsize=9, rotation=0, ha="right", va="center")
    axes[i, 1].imshow(norm_local(s18_r[zy0:zy1, zx0:zx1]), cmap="gray", interpolation="nearest")
    axes[i, 2].imshow(norm_local(s18_r[zy0:zy1, zx0:zx1]), cmap="gray", interpolation="nearest")
    axes[i, 2].imshow(label_overlay(masks[zy0:zy1, zx0:zx1]), interpolation="nearest")
    vmax = max(abs(flow_y[zy0:zy1, zx0:zx1]).max(), 1e-3)
    axes[i, 3].imshow(flow_y[zy0:zy1, zx0:zx1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    vmax = max(abs(flow_x[zy0:zy1, zx0:zx1]).max(), 1e-3)
    axes[i, 4].imshow(flow_x[zy0:zy1, zx0:zx1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[i, 5].imshow(cellprob[zy0:zy1, zx0:zx1], cmap="viridis", interpolation="nearest")
    for j in range(len(cols)):
        axes[i, j].plot(lx - zx0, ly - zy0, "+", color="red", markersize=14, markeredgewidth=2)
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    if i == 0:
        for j, c in enumerate(cols):
            axes[i, j].set_title(c, fontsize=11)
plt.suptitle("PHASE 1 — same 100-px window at WSI ring (41680, 6996) across ROI sizes; red + = ring center\n"
             "default cpsam settings (augment=True, normalize=True, niter=200)",
             fontsize=12, y=1.0)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_size_sweep_flows.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"saved {OUT_DIR / 'fig1_size_sweep_flows.png'}")

# Fig 3: metric trends vs size
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sps = [r[0] for r in records]; nlabs = [r[1] for r in records]; cohs = [r[2] for r in records]
axes[0].plot(sps, nlabs, "o-", color="#d62728", linewidth=2, markersize=10)
axes[0].set_xscale("log"); axes[0].set_xlabel("ROI size (px)")
axes[0].set_ylabel("# distinct CP-SAM masks at ring (40-px window)")
axes[0].axhline(1, color="grey", linestyle=":", alpha=0.6, label="goal: 1")
axes[0].set_title("Onion ring severity grows monotonically with ROI size")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(sps, cohs, "o-", color="#2ca02c", linewidth=2, markersize=10)
axes[1].set_xscale("log"); axes[1].set_xlabel("ROI size (px)")
axes[1].set_ylabel("flow coherence (0..1)")
axes[1].set_title("Flow alignment at ring degrades only slightly")
axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_metrics_vs_size.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"saved {OUT_DIR / 'fig3_metrics_vs_size.png'}")

print("\nrecords:")
for sp, nl, co in records:
    print(f"  size={sp}: labels_at_ring={nl}, flow_coherence={co:.3f}")
