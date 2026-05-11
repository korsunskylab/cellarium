"""Builder for workflow/04_gap_intervention_test.ipynb.

Step 0 of the boundary-prior initiative — validates the gap-cut intervention
on the 18S channel before the upstream gradient pipeline is built.

Phase A: single-ellipse merged-blob, 1-channel 18S (no DAPI). Sweep
         (width × intensity_factor); show full mask grid + heatmap.
Phase B: ROI1 merged doublet (cpsam mask containing >=2 10X cells),
         tight bbox + neighbors masked out, sweep cut along inferred
         10X boundary. Same full-grid visualization.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md("""# Gap-intervention test (18S cut)

**Step 0** of the boundary-prior initiative. Before we build the R-side gradient pipeline, validate that cutting a thin dim strip into the **18S** channel reliably forces cpsam to call multiple cells.

Two phases in one notebook:
- **Phase A — intervention calibration.** Synthetic single-ellipse merged blob (1-channel 18S, no DAPI). The only synthetic geometry that cpsam reliably *merges* under our max-recall settings — two-disk geometries already split at baseline because of the gaussian-blur saddle, so they don't test the merge→split direction. Sweep `(width × intensity_factor)`; produce a full pixel-visible mask grid plus a summary heatmap.
- **Phase B — real merged doublet.** Find a cpsam mask in ROI1 that contains ≥2 10X-reference cells, crop tightly to its bbox, **zero out everything outside the cpsam mask** so only the doublet is in frame, then sweep the same `(width × factor)` grid with the cut placed perpendicular to the line connecting the two 10X centroids.

cpsam config throughout = the validated **max-recall** settings: `cellprob_threshold=-5, flow_threshold=0.0, augment=True, niter=200`. We test against the production setting.

**Decision gate at the end:** if Phase A produces a workable operating window AND Phase B's correctly-positioned cut splits a real doublet at parameters within that window, proceed to notebooks 1–3. Otherwise, rethink.
"""))

# ── 0. Setup ──────────────────────────────────────────────────────────────
cells.append(md("## 0. Setup"))

cells.append(code("""import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import json, time
import numpy as np, pandas as pd, tifffile, torch
import matplotlib.pyplot as plt
from skimage.draw import disk, ellipse
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt
from cellpose import models as cp_modern

m_sam = cp_modern.CellposeModel(gpu=True, pretrained_model="cpsam")
print("cpsam loaded on:", m_sam.device)

# ─── Production config (validated max-recall, see segmentation_pipeline memory) ──
CPSAM_KW = dict(diameter=None, niter=200,
                cellprob_threshold=-5.0, flow_threshold=0.0, augment=True)

def run_cpsam(img):
    \"\"\"Run cpsam. Pass a (H, W) array for 1-channel or (H, W, C) for multi-channel.\"\"\"
    kw = dict(CPSAM_KW)
    if img.ndim == 3:
        kw["channel_axis"] = -1
    masks, _, _ = m_sam.eval(img.astype(np.float32), **kw)
    return masks, int(masks.max())

# ─── Visualization helpers ────────────────────────────────────────────────
def shuffle_labels(masks, seed=0):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return masks
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n+1))])
    return perm[masks]

def label_overlay(masks, seed=0, alpha_fill=0.35):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    shuf = shuffle_labels(masks, seed=seed)
    rgba = plt.get_cmap("nipy_spectral")(shuf / max(n, 1))
    rgba[..., 3] = alpha_fill * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba

def normalize_for_display(img, low=1, high=99):
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

def blue_yellow_composite(dapi, s18):
    \"\"\"DAPI → blue channel, 18S → yellow (R+G). Inputs in [0, 1].\"\"\"
    rgb = np.zeros((*dapi.shape, 3), dtype=np.float32)
    rgb[..., 0] = s18; rgb[..., 1] = s18; rgb[..., 2] = dapi
    return np.clip(rgb, 0, 1)

def outcome_color(n, target=2):
    if n == target: return "#bce8b3"   # light green
    if n < target:  return "#f4b6b2"   # light red (merged)
    return "#f6e7a3"                   # light yellow (over-fragmented)
"""))

# ── 1. Phase A ────────────────────────────────────────────────────────────
cells.append(md("""## 1. Phase A — intervention calibration on a single-ellipse merged blob

A single elongated ellipse with no DAPI signal and no shape saddle. cpsam (max-recall) merges this into one mask at baseline. The question Phase A answers: what shape and depth of dim-cut in 18S reliably produces **n=2**, and where does the test break (merging, or over-fragmenting into n>2)?

We use a **Gaussian-shaped dim profile** (smooth, no plateau) rather than a rectangular strip. The rectangular strip created two failure modes (observed empirically): (i) a wide partial-erase strip becomes its own "third cell" because cpsam recognizes a uniform-intensity region as cell-like, and (ii) intermediate widths produce sharp edges that trigger spurious flow basins. Gaussian profile avoids both: there's no uniform plateau, and edges are smooth.

Profile: `factor(x) = 1 - (1-f) · exp(-(x-x_c)² / (2σ²))`. At the center, the multiplier is `f` (same as the rectangular case); at the edges, it's `1` (no change). σ controls how wide the dim region is.
"""))

cells.append(md("### 1a. Single-ellipse generator + baseline (no cut)"))

cells.append(code("""def synthetic_merged_blob(I_18s=0.7, ry=13, rx=25,
                          canvas=96, sigma=1.0, noise=0.03, seed=0):
    \"\"\"Single elongated ellipse, 1-channel 18S. Returns 2D float32 array.\"\"\"
    rng = np.random.default_rng(seed)
    H = W = canvas
    s18 = np.zeros((H, W), dtype=np.float32)
    rr, cc = ellipse(canvas // 2, canvas // 2, ry, rx, shape=(H, W))
    s18[rr, cc] = I_18s
    s18 = gaussian(s18, sigma=sigma)
    s18 = np.clip(s18 + noise * rng.standard_normal((H, W)).astype(np.float32), 0, 1)
    return s18


def apply_gaussian_dim(s18, x_center, sigma_px, intensity_factor):
    \"\"\"Gaussian-shaped dim along x (vertical strip): factor at center = intensity_factor,
    factor at |x - x_center| → ∞ = 1. Smooth, no plateau, no sharp edges.\"\"\"
    if sigma_px <= 0: return s18.copy()
    W = s18.shape[1]
    xx = np.arange(W)
    profile = 1.0 - (1.0 - intensity_factor) * np.exp(-((xx - x_center) ** 2) / (2.0 * sigma_px ** 2))
    return s18 * profile[None, :]


def apply_gaussian_strip_along_line(s18, p1, p2, sigma_px, intensity_factor):
    \"\"\"2-D Gaussian dim perpendicular to the straight line p1->p2 (kept as a utility,
    not used in Phase B by default — see apply_gaussian_along_natural_boundary).\"\"\"
    if sigma_px <= 0: return s18.copy()
    H, W = s18.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy = 0.5 * (p1[0] + p2[0]); cx = 0.5 * (p1[1] + p2[1])
    dy, dx = p2[0] - p1[0], p2[1] - p1[1]
    L = np.hypot(dy, dx)
    if L < 1e-6: return s18.copy()
    uy, ux = dy / L, dx / L
    along = (yy - cy) * uy + (xx - cx) * ux
    profile = 1.0 - (1.0 - intensity_factor) * np.exp(-(along ** 2) / (2.0 * sigma_px ** 2))
    return s18 * profile


def apply_gaussian_dim_noisy(s18, x_center, sigma_px, intensity_factor, noise_std, seed=0):
    \"\"\"Phase A: Gaussian dim along vertical strip at x_center, with f varying row-by-row.
    Each row gets f_row = clip(intensity_factor + ε, 0, 1), ε ~ N(0, noise_std²).
    Simulates non-uniform mRNA-gradient confidence along the boundary.\"\"\"
    if sigma_px <= 0: return s18.copy()
    H, W = s18.shape
    rng = np.random.default_rng(seed)
    eps_per_row = rng.standard_normal(H) * noise_std
    f_per_row   = np.clip(intensity_factor + eps_per_row, 0, 1)
    xx = np.arange(W)
    gauss = np.exp(-((xx - x_center) ** 2) / (2.0 * sigma_px ** 2))[None, :]   # (1, W)
    profile = 1.0 - (1.0 - f_per_row[:, None]) * gauss
    return s18 * profile


def apply_gaussian_strip_along_line_noisy(s18, p1, p2, sigma_px, intensity_factor, noise_std, seed=0):
    \"\"\"Phase B: Gaussian dim perpendicular to p1->p2, with f varying along the cut tangent
    (perpendicular to the p1->p2 axis). One noise sample per 1-px bin along the tangent.\"\"\"
    if sigma_px <= 0: return s18.copy()
    H, W = s18.shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    cy = 0.5 * (p1[0] + p2[0]); cx = 0.5 * (p1[1] + p2[1])
    dy, dx = p2[0] - p1[0], p2[1] - p1[1]
    L = np.hypot(dy, dx)
    if L < 1e-6: return s18.copy()
    uy, ux = dy / L, dx / L
    along_axis = (yy - cy) * uy + (xx - cx) * ux                  # Gaussian falloff direction
    tang_uy, tang_ux = -ux, uy                                    # rotate 90° → along the cut line
    along_tang = (yy - cy) * tang_uy + (xx - cx) * tang_ux
    # 1-px bins along the tangent — one noise sample per bin
    bin_idx = np.floor(along_tang).astype(int)
    bin_idx -= bin_idx.min()
    n_bins = int(bin_idx.max()) + 1
    eps = rng.standard_normal(n_bins) * noise_std
    f_per_pixel = np.clip(intensity_factor + eps[bin_idx], 0, 1)
    profile = 1.0 - (1.0 - f_per_pixel) * np.exp(-(along_axis ** 2) / (2.0 * sigma_px ** 2))
    return s18 * profile


def apply_gaussian_along_natural_boundary(s18, mask_A, mask_B, sigma_px, intensity_factor):
    \"\"\"(Utility, not used by default.) Gaussian dim along the natural boundary curve
    between two cell masks (equidistance locus). Kept in case we want to A/B compare.\"\"\"
    if sigma_px <= 0: return s18.copy()
    dist_A = distance_transform_edt(~mask_A)
    dist_B = distance_transform_edt(~mask_B)
    perp   = np.abs(dist_A - dist_B) / 2.0
    profile = 1.0 - (1.0 - intensity_factor) * np.exp(-(perp ** 2) / (2.0 * sigma_px ** 2))
    return s18 * profile


def cell_metrics(cell_mask):
    \"\"\"Return (area_px, aspect_ratio, (cy, cx)) for a binary cell mask.
    aspect_ratio = sqrt(major_eigval / minor_eigval) — 1.0 = circle, large = elongated.\"\"\"
    yy, xx = np.where(cell_mask)
    if len(yy) < 3: return 0, 1.0, (0.0, 0.0)
    cy, cx = float(yy.mean()), float(xx.mean())
    coords = np.stack([yy - cy, xx - cx], axis=1)
    cov = np.cov(coords.T)
    eigvals = np.linalg.eigvalsh(cov)
    aspect = float(np.sqrt(max(eigvals[-1], 1e-6) / max(eigvals[0], 1e-6)))
    return int(len(yy)), aspect, (cy, cx)


def cell_axis_endpoints(cell_mask, length_factor=2.0):
    \"\"\"Return (centroid, p1, p2) along the cell's principal axis (PCA on mask coords).
    p1, p2 are at ±length_factor*std from the centroid along the major axis.\"\"\"
    yy, xx = np.where(cell_mask)
    if len(yy) < 3: return None, None, None
    cy, cx = float(yy.mean()), float(xx.mean())
    coords = np.stack([yy - cy, xx - cx], axis=1)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)         # ascending eigenvalues
    major = eigvecs[:, -1]                         # principal direction (largest variance)
    extent = float(np.sqrt(max(eigvals[-1], 1e-6)) * length_factor)
    p1 = (cy - major[0] * extent, cx - major[1] * extent)
    p2 = (cy + major[0] * extent, cx + major[1] * extent)
    return (cy, cx), p1, p2


def apply_gaussian_perpendicular_to_extent(s18, cell_mask, sigma_px, intensity_factor):
    \"\"\"(Older variant; kept for reference.) Cut perpendicular to the cell's PCA major axis
    at its mass-weighted centroid. For asymmetric doublets the centroid can be biased
    toward the bigger cell — the cut lands inside its body rather than in the cytoplasm
    bridge — so prefer apply_gaussian_cut_clipped with 10X centroids in real use.\"\"\"
    if sigma_px <= 0: return s18.copy()
    centroid, p1, p2 = cell_axis_endpoints(cell_mask)
    if p1 is None: return s18.copy()
    return apply_gaussian_strip_along_line(s18, p1, p2, sigma_px, intensity_factor)


def apply_gaussian_cut_clipped(s18, p1, p2, cell_mask, sigma_px, intensity_factor):
    \"\"\"(Kept as a utility, not used by default.) Same Gaussian dim along p1→p2 line,
    but masked to only modify pixels inside cell_mask. This *hides* the neighbor-bleed
    effect — pieces of the split cell can no longer merge with neighbors because the
    neighbors see their original signal. Useful for A/B comparison only; the default
    Phase B sweep below uses the unclipped variant so the full impact is visible.\"\"\"
    if sigma_px <= 0: return s18.copy()
    cut = apply_gaussian_strip_along_line(s18, p1, p2, sigma_px, intensity_factor)
    return np.where(cell_mask, cut, s18)


# Baseline: no intervention
s18 = synthetic_merged_blob()
x_center = s18.shape[1] // 2
masks_base, n_base = run_cpsam(s18)
print(f"baseline (no cut):  cpsam → {n_base} cell(s)")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
ax[0].imshow(s18, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[0].set_title("18S baseline (single ellipse, no DAPI)"); ax[0].axis("off")
ax[1].imshow(s18, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[1].imshow(label_overlay(masks_base, seed=1), interpolation="nearest")
ax[1].set_title(f"cpsam masks (n={n_base})"); ax[1].axis("off")
ax[2].imshow(masks_base, cmap="nipy_spectral", interpolation="nearest")
ax[2].set_title("mask label image"); ax[2].axis("off")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""### 1b. Demo — single Gaussian dim

Show what one specific Gaussian dim looks like (1-D profile + 2-D image + cpsam result)."""))

cells.append(code("""SIG_DEMO, F_DEMO = 2.0, 0.0   # σ=2 px, center fully erased

s18_gauss = apply_gaussian_dim(s18, x_center, SIG_DEMO, F_DEMO)
masks_g, n_g = run_cpsam(s18_gauss)

# 1-D profile so the smooth shape is unambiguous
xx = np.arange(s18.shape[1])
prof_g = 1.0 - (1.0 - F_DEMO) * np.exp(-((xx - x_center) ** 2) / (2.0 * SIG_DEMO ** 2))

fig = plt.figure(figsize=(13, 3.8))
gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1, 1, 1])
axp = fig.add_subplot(gs[0, 0])
axp.plot(xx, prof_g, color="C0", lw=1.5)
axp.set_xlim(x_center-15, x_center+15); axp.set_ylim(-0.05, 1.1)
axp.set_xlabel("x (px)"); axp.set_ylabel("intensity multiplier")
axp.set_title(f"Gaussian dim profile (σ={SIG_DEMO}, f={F_DEMO})"); axp.grid(alpha=0.3)
ax = [fig.add_subplot(gs[0, k]) for k in range(1, 4)]
ax[0].imshow(s18, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[0].set_title("18S baseline"); ax[0].axis("off")
ax[1].imshow(s18_gauss, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[1].set_title(f"18S after Gaussian dim\\n(σ={SIG_DEMO}, f={F_DEMO})"); ax[1].axis("off")
ax[2].imshow(s18_gauss, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[2].imshow(label_overlay(masks_g, seed=11), interpolation="nearest")
ax[2].set_title(f"cpsam (n={n_g})", backgroundcolor=outcome_color(n_g)); ax[2].axis("off")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""### 1c. Sweep σ × intensity-factor (Gaussian dim)

For each (σ, factor) apply the Gaussian dim profile, run cpsam, record the cell count and masks. Target = exactly 2 cells."""))

cells.append(code("""sigmas   = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
factors  = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]

n_grid = np.zeros((len(factors), len(sigmas)), dtype=np.int32)
runs   = {}

t0 = time.time()
for i, f in enumerate(factors):
    for j, sg in enumerate(sigmas):
        s18_cut = apply_gaussian_dim(s18, x_center, sg, f)
        masks, n = run_cpsam(s18_cut)
        n_grid[i, j] = n
        runs[(i, j)] = masks
print(f"sweep done in {time.time()-t0:.1f}s   ({len(sigmas)*len(factors)} cpsam runs)")

print("\\nCells detected (rows = factor, cols = σ):")
print("        " + "  ".join(f"σ={sg:>4.1f}" for sg in sigmas))
for i, f in enumerate(factors):
    print(f"  f={f:.1f}  " + "  ".join(f"{n_grid[i,j]:>6d}" for j in range(len(sigmas))))
"""))

cells.append(md("""### 1d. Summary heatmap"""))

cells.append(code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

im0 = ax[0].imshow(n_grid, cmap="viridis", aspect="auto",
                    vmin=0, vmax=max(3, n_grid.max()))
ax[0].set_xticks(range(len(sigmas)));  ax[0].set_xticklabels([f"{sg:g}" for sg in sigmas])
ax[0].set_yticks(range(len(factors))); ax[0].set_yticklabels([f"{f:.1f}" for f in factors])
ax[0].set_xlabel("Gaussian σ (px)")
ax[0].set_ylabel("intensity factor at center")
ax[0].set_title("cpsam cell count")
for i in range(len(factors)):
    for j in range(len(sigmas)):
        ax[0].text(j, i, str(n_grid[i, j]), ha="center", va="center",
                   color="white" if n_grid[i,j] < 2 else "black", fontsize=9)
plt.colorbar(im0, ax=ax[0], fraction=0.046)

out_map = np.zeros((*n_grid.shape, 3), dtype=np.float32)
for i in range(len(factors)):
    for j in range(len(sigmas)):
        c = outcome_color(int(n_grid[i, j]))
        out_map[i, j] = plt.matplotlib.colors.to_rgb(c)
ax[1].imshow(out_map, aspect="auto")
ax[1].set_xticks(range(len(sigmas)));  ax[1].set_xticklabels([f"{sg:g}" for sg in sigmas])
ax[1].set_yticks(range(len(factors))); ax[1].set_yticklabels([f"{f:.1f}" for f in factors])
ax[1].set_xlabel("Gaussian σ (px)")
ax[1].set_ylabel("intensity factor at center")
ax[1].set_title("outcome  (green: n=2, red: merged, yellow: over-frag)")
for i in range(len(factors)):
    for j in range(len(sigmas)):
        n = int(n_grid[i, j])
        sym = "✓" if n == 2 else ("✗ merged" if n == 1 else f"✗ n={n}")
        ax[1].text(j, i, sym, ha="center", va="center", fontsize=9)
plt.tight_layout(); plt.show()

print(f"\\n# Phase A (Gaussian) summary across {n_grid.size} configs:")
print(f"  correct (n=2):       {int((n_grid == 2).sum()):3d}   ← operating window")
print(f"  merged (n=1):        {int((n_grid == 1).sum()):3d}")
print(f"  over-fragmented (>2):{int((n_grid >  2).sum()):3d}")
"""))

cells.append(md("""### 1e. Full mask grid — every (σ × factor) combination

Pixel-visible. Each panel shows 18S after Gaussian dim with cpsam masks overlaid. Title color: green = n=2, red = merged, yellow = over-frag."""))

cells.append(code("""fig, ax = plt.subplots(len(factors), len(sigmas),
                       figsize=(1.6*len(sigmas), 1.85*len(factors)),
                       squeeze=False)
for i, f in enumerate(factors):
    for j, sg in enumerate(sigmas):
        s18_cut = apply_gaussian_dim(s18, x_center, sg, f)
        masks   = runs[(i, j)]
        n       = int(n_grid[i, j])
        ax[i, j].imshow(s18_cut, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[i, j].imshow(label_overlay(masks, seed=100 + i*len(sigmas) + j),
                         interpolation="nearest")
        ax[i, j].set_title(f"f={f:.1f}  σ={sg:g}\\nn={n}",
                           fontsize=8, backgroundcolor=outcome_color(n))
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
plt.suptitle("Phase A — Gaussian dim sweep (rows: intensity factor, cols: σ)",
             fontsize=11, y=1.005)
plt.tight_layout(); plt.show()
"""))

# ── 1.5 angle sweep — does cut direction matter? ─────────────────────────
cells.append(md("""### 1f. Angle sweep — does the cut direction matter?

Phase A's vertical cut happens to be perpendicular to the ellipse's major axis (the natural cleavage direction). What about cuts at other angles? At 90° (along the major axis) the cut bisects the cell into two thin half-ellipses — does cpsam still call them as cells, or is the resulting shape too unfamiliar?

Sweep angles at fixed σ=3, f=0 (a setting in the Phase A1 operating window for vertical cuts). Angle is measured from vertical (so 0°=vertical, 90°=horizontal, along the long axis)."""))

cells.append(code("""import math

angles_deg = [0, 30, 45, 60, 90]
SIGMA_A    = 3.0
FACTOR_A   = 0.0
cy_a = cx_a = s18.shape[0] // 2

fig, ax = plt.subplots(2, len(angles_deg),
                       figsize=(2.4*len(angles_deg), 4.6), squeeze=False)
for j, ang in enumerate(angles_deg):
    # Endpoints along direction perpendicular to the cut band, so the band sits at angle `ang` from vertical
    perp = math.radians(ang + 90)
    L = 10
    p1 = (cy_a - L * math.cos(perp), cx_a - L * math.sin(perp))
    p2 = (cy_a + L * math.cos(perp), cx_a + L * math.sin(perp))

    s18_cut = apply_gaussian_strip_along_line(s18, p1, p2, SIGMA_A, FACTOR_A)
    masks, n = run_cpsam(s18_cut)

    # Top: cut image with the cut line overlaid (cyan)
    ax[0, j].imshow(s18_cut, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    # Draw the cut band orientation as a line through the center
    band_dy = math.cos(math.radians(ang))
    band_dx = math.sin(math.radians(ang))
    line_L  = 25
    ax[0, j].plot([cx_a - line_L*band_dx, cx_a + line_L*band_dx],
                  [cy_a - line_L*band_dy, cy_a + line_L*band_dy],
                  "-", color="cyan", lw=0.8)
    ax[0, j].set_title(f"angle = {ang}° from vertical", fontsize=9)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])

    # Bottom: cpsam mask overlay
    ax[1, j].imshow(s18_cut, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[1, j].imshow(label_overlay(masks, seed=400 + j), interpolation="nearest")
    ax[1, j].set_title(f"n = {n}", fontsize=10, backgroundcolor=outcome_color(n))
    ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
plt.suptitle(f"Phase A — angle sweep (σ={SIGMA_A:g}, f={FACTOR_A:g})  "
             f"— top: 18S after cut + cut band (cyan); bottom: cpsam masks",
             fontsize=11, y=1.005)
plt.tight_layout(); plt.show()
"""))

# ── 1g. Noise tolerance ──────────────────────────────────────────────────
cells.append(md("""### 1g. Noise tolerance — what if f varies along the cut?

In the upstream pipeline (notebooks 1–2 in R), we won't know f perfectly — the mRNA gradient confidence will vary along the boundary, so f should vary pixel-to-pixel along the cut. This section sweeps `noise_std` to find how much per-row variability cpsam tolerates before splits stop firing.

Setup: at each row, draw `f_row = clip(f_mean + ε, 0, 1)` with ε ~ N(0, noise_std²). Sweep noise_std at a fixed (σ, f_mean) point that gave a clean split in 1c. Top row shows the noisy 18S profile; bottom row shows the cpsam masks."""))

cells.append(code("""SIGMA_NOISE_A   = 2.0     # narrow cut (operating regime: deep + narrow → clean split)
F_MEAN_NOISE_A  = 0.0     # deep
NOISE_LEVELS_A  = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
NOISE_SEED      = 0

results_noise_A = []
for ns in NOISE_LEVELS_A:
    s18_cut = apply_gaussian_dim_noisy(s18, x_center, SIGMA_NOISE_A, F_MEAN_NOISE_A,
                                        ns, seed=NOISE_SEED)
    masks, n = run_cpsam(s18_cut)
    results_noise_A.append((ns, s18_cut, masks, n))

fig, ax = plt.subplots(2, len(NOISE_LEVELS_A),
                       figsize=(2.0*len(NOISE_LEVELS_A), 4.0), squeeze=False)
for j, (ns, s18_cut, masks, n) in enumerate(results_noise_A):
    ax[0, j].imshow(s18_cut, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[0, j].set_title(f"noise_std = {ns:g}", fontsize=9)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    ax[1, j].imshow(s18_cut, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[1, j].imshow(label_overlay(masks, seed=500 + j), interpolation="nearest")
    ax[1, j].set_title(f"n = {n}", fontsize=10, backgroundcolor=outcome_color(n))
    ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
plt.suptitle(f"Phase A — noise robustness (σ={SIGMA_NOISE_A:g}, f_mean={F_MEAN_NOISE_A:g}; "
             f"top: 18S; bottom: cpsam masks)",
             fontsize=11, y=1.005)
plt.tight_layout(); plt.show()

# Print breakdown threshold
print("\\nNoise vs cpsam outcome:")
last_clean = None
for ns, _, _, n in results_noise_A:
    flag = "split ✓" if n == 2 else ("merged" if n < 2 else f"n={n}")
    if n == 2: last_clean = ns
    print(f"  noise_std={ns:>4.2f}  →  n={n}  ({flag})")
if last_clean is not None:
    print(f"\\n  highest noise_std with clean split: {last_clean:.2f}")
"""))

# ── 2. Phase B ────────────────────────────────────────────────────────────
cells.append(md("""## 2. Phase B — real ROI1 merged doublet in tissue context

Find a cpsam mask in ROI1 that contains ≥2 distinct 10X-reference cells (a real merged-doublet failure). Crop a fixed-size tissue-context window centered on the doublet — surrounding cells stay visible, no neighbor masking — then carve a Gaussian dim along the perpendicular bisector of the two 10X centroids and sweep `(σ × factor)`.

**Channel note:** Phase B uses real DAPI + 18S (2-channel cpsam) since both are present in the morphology stack. Phase A used 1-channel only because the synthetic doesn't include a fake DAPI.

**Trade-off note (the cost vs the justification):** The Gaussian dim is *not* clipped to the merged-cell mask. That means at aggressive σ the dim band extends into adjacent cells, and pieces of the split cell can attach to neighbors instead of staying separate — a real failure mode that will be visible in the masks. The flip side: when the cut is placed where mRNA gradients indicate a true cell boundary (the upstream pipeline notebooks 1–2 in R), we're injecting evidence cpsam didn't have access to from morphology alone — breaking ties cpsam couldn't break on its own. Some attachment errors are an expected part of the trade-off; downstream filtering (Baysor's transcript-density voting, or a post-cut cell-quality check) can recover when the misattached pieces don't carry coherent transcripts.
"""))

cells.append(md("### 2a. Load ROI1 and run baseline cpsam (full ROI)"))

cells.append(code("""DATA = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/data/Xenium_Prime_Human_Skin_FFPE_xe_outs")
ROI_DIR = DATA / "ROI1"

with open(ROI_DIR / "metadata.json") as f:
    meta = json.load(f)
print(f"ROI1: {meta['shape_yx']}, n_cells_in_mask={meta['n_cells_in_mask']}")

dapi_full = tifffile.imread(ROI_DIR / "morphology_DAPI.tif")
s18_full  = tifffile.imread(ROI_DIR / "morphology_18S.tif")
masks_10x = tifffile.imread(ROI_DIR / "cells_10x_masks.tif")
print(f"DAPI: {dapi_full.shape}  18S: {s18_full.shape}  10X masks: {masks_10x.shape}")

dapi_n = normalize_for_display(dapi_full)
s18_n  = normalize_for_display(s18_full)

img_full = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
t0 = time.time()
masks_full, n_full = run_cpsam(img_full)
print(f"cpsam baseline on full ROI1: {n_full} cells in {time.time()-t0:.1f}s")
"""))

cells.append(md("""### 2b. Find merged-doublet candidates

For each cpsam mask, count distinct 10X-mask labels overlapping it by at least `MIN_10X_OVERLAP_PX` pixels. cpsam masks with ≥2 such labels are merged-doublet candidates."""))

cells.append(code("""MIN_10X_OVERLAP_PX = 25
MIN_CELL_AREA_PX   = 400   # filter out tiny merged regions — real cells are usually >500 px

# Per-(cpsam, 10X) pair pixel counts
nz = masks_full > 0
flat_cps = masks_full[nz]
flat_10x = masks_10x[nz]
pairs = np.stack([flat_cps, flat_10x], axis=1)
uniq, counts = np.unique(pairs, axis=0, return_counts=True)

cps_to_10x = {}
for (c, t), k in zip(uniq, counts):
    if c == 0 or t == 0: continue
    cps_to_10x.setdefault(int(c), {})[int(t)] = int(k)

# Compute area + aspect ratio for each candidate cpsam mask
doublet_candidates = []
for c, d in cps_to_10x.items():
    big = [t for t, k in d.items() if k >= MIN_10X_OVERLAP_PX]
    if len(big) < 2: continue
    area, aspect, centroid = cell_metrics(masks_full == c)
    if area < MIN_CELL_AREA_PX: continue
    doublet_candidates.append({
        "cpsam": int(c), "members": big, "overlaps": d,
        "area": area, "aspect": aspect, "centroid": centroid,
    })

print(f"merged-doublet candidates after area filter (>{MIN_CELL_AREA_PX} px): {len(doublet_candidates)}")
"""))

cells.append(md("""### 2c. Pick a candidate (rank by elongation × size, with visual preview)

Rank by `aspect_ratio × √area` so big elongated cells (peanut-shaped merged doublets) come first. The previous balance-only ranking favored small cells with two evenly-overlapping 10X labels — not what we want.

Two ways to pick:
- `DOUBLET_RANK = 0, 1, 2, ...` — pick by rank in the auto-sorted list.
- `MANUAL_CPSAM_LABEL = <int>` — pick a specific cpsam mask by its label (overrides rank).

Top-N preview grid below shows what each rank looks like so you can pick visually."""))

cells.append(code("""def balance_score(d, big):
    sizes = sorted([d[t] for t in big], reverse=True)[:2]
    return min(sizes) / max(sizes) if len(sizes) == 2 else 0.0

def rank_score(area, aspect):
    return aspect * np.sqrt(area)

ranked = sorted(doublet_candidates,
                key=lambda x: rank_score(x["area"], x["aspect"]),
                reverse=True)
if not ranked:
    raise RuntimeError("No merged-doublet candidates found in ROI1 with the current filters.")

# ─── pick which candidate to use ──────────────────────────────────────
DOUBLET_RANK        = 0     # rank in the auto-sorted list (by elongation × size)
MANUAL_CPSAM_LABEL  = None  # override: set to a specific cpsam mask label (e.g., 47) to use that one
CROP_SIZE           = 150   # tissue-context window size (px)
TOP_N_PREVIEW       = 8     # how many candidates to visualize in the preview grid below
# ──────────────────────────────────────────────────────────────────────

print(f"top {TOP_N_PREVIEW} doublet candidates (ranked by elongation × √area):")
print(f"  {'rank':>4s}  {'cpsam':>6s}  {'area':>5s}  {'aspect':>6s}  {'balance':>7s}  {'centroid (y,x) px':<22s}  {'10X cells':<20s}")
for r, cand in enumerate(ranked[:TOP_N_PREVIEW]):
    cy, cx = cand["centroid"]
    bal = balance_score(cand["overlaps"], cand["members"])
    print(f"  {r:>4d}  #{cand['cpsam']:<5d}  {cand['area']:>5d}  {cand['aspect']:>6.2f}  {bal:>7.2f}  ({cy:7.1f}, {cx:7.1f})    {str(cand['members'][:2]):<20s}")

# Resolve manual override
if MANUAL_CPSAM_LABEL is not None:
    matches = [c for c in doublet_candidates if c["cpsam"] == MANUAL_CPSAM_LABEL]
    if not matches:
        raise RuntimeError(f"MANUAL_CPSAM_LABEL={MANUAL_CPSAM_LABEL} not in doublet candidates.")
    chosen = matches[0]
    print(f"\\nusing MANUAL_CPSAM_LABEL = {MANUAL_CPSAM_LABEL}")
else:
    chosen = ranked[DOUBLET_RANK]
    print(f"\\nusing rank {DOUBLET_RANK}: cpsam #{chosen['cpsam']}")

chosen_cps      = chosen["cpsam"]
chosen_10x_list = chosen["members"]
chosen_d        = chosen["overlaps"]
top2 = sorted(chosen_10x_list, key=lambda t: chosen_d[t], reverse=True)[:2]
print(f"  10X cells {top2}, overlaps {chosen_d[top2[0]]}/{chosen_d[top2[1]]} px, "
      f"area={chosen['area']}, aspect={chosen['aspect']:.2f}")

# ─── visual preview grid of top-N candidates ─────────────────────────
fig, ax = plt.subplots(1, TOP_N_PREVIEW, figsize=(2.0*TOP_N_PREVIEW, 2.4), squeeze=False)
H_full, W_full = masks_full.shape
preview_half = 60
for r, cand in enumerate(ranked[:TOP_N_PREVIEW]):
    cy_p, cx_p = int(cand["centroid"][0]), int(cand["centroid"][1])
    py0 = max(0, cy_p - preview_half); py1 = min(H_full, py0 + 2*preview_half); py0 = max(0, py1 - 2*preview_half)
    px0 = max(0, cx_p - preview_half); px1 = min(W_full, px0 + 2*preview_half); px0 = max(0, px1 - 2*preview_half)
    comp = blue_yellow_composite(dapi_n[py0:py1, px0:px1], s18_n[py0:py1, px0:px1])
    cps_outline = find_boundaries(masks_full[py0:py1, px0:px1] == cand["cpsam"], mode="outer")
    ax[0, r].imshow(comp, interpolation="nearest")
    ey, ex = np.where(cps_outline)
    ax[0, r].plot(ex, ey, ".", color="cyan", ms=1.5)
    bg = "#bce8b3" if r == DOUBLET_RANK and MANUAL_CPSAM_LABEL is None else "white"
    ax[0, r].set_title(f"r={r} #{cand['cpsam']}\\nA={cand['area']} ar={cand['aspect']:.1f}",
                        fontsize=8, backgroundcolor=bg)
    ax[0, r].set_xticks([]); ax[0, r].set_yticks([])
plt.suptitle("Top doublet candidates (ranked by elongation × √area; cyan = cpsam mask outline)", fontsize=10, y=1.02)
plt.tight_layout(); plt.show()

def centroid_of(label_img, lab):
    yy, xx = np.where(label_img == lab)
    return float(yy.mean()), float(xx.mean())

c1 = centroid_of(masks_10x, top2[0])
c2 = centroid_of(masks_10x, top2[1])

# Fixed-size tissue-context crop centered on the midpoint of the two centroids
H_full, W_full = masks_full.shape
center_y = int(round((c1[0] + c2[0]) / 2))
center_x = int(round((c1[1] + c2[1]) / 2))
y0 = max(0, center_y - CROP_SIZE // 2);  y1 = min(H_full, y0 + CROP_SIZE)
x0 = max(0, center_x - CROP_SIZE // 2);  x1 = min(W_full, x0 + CROP_SIZE)
y0 = max(0, y1 - CROP_SIZE)              # re-clamp if we hit the right/bottom edge
x0 = max(0, x1 - CROP_SIZE)
print(f"crop window: y={y0}..{y1}  x={x0}..{x1}  → {y1-y0}×{x1-x0} px (centered on doublet midpoint)")

# Natural tissue context — no masking
dapi_c = dapi_n[y0:y1, x0:x1].astype(np.float32)
s18_c  = s18_n [y0:y1, x0:x1].astype(np.float32)
m10x_c          = masks_10x [y0:y1, x0:x1]
mcps_c_baseline = masks_full[y0:y1, x0:x1]
c1_c = (c1[0] - y0, c1[1] - x0)
c2_c = (c2[0] - y0, c2[1] - x0)
composite_c = blue_yellow_composite(dapi_c, s18_c)

# 10X mask labels for the two doublet members (kept for reference / centroid comparison)
mask_A_c = (m10x_c == top2[0])
mask_B_c = (m10x_c == top2[1])

# Merged cpsam mask of the chosen doublet — this is the cell we will try to split
mask_M_c = (mcps_c_baseline == chosen_cps)
# Principal axis of the merged blob (PCA), and the perpendicular cleavage line
centroid_M, axis_p1, axis_p2 = cell_axis_endpoints(mask_M_c, length_factor=2.0)
# For visualization, also compute a longer perpendicular (the cut line itself)
if axis_p1 is not None:
    yy_M, xx_M = np.where(mask_M_c)
    coords_M = np.stack([yy_M - centroid_M[0], xx_M - centroid_M[1]], axis=1)
    cov_M = np.cov(coords_M.T)
    eigvals_M, eigvecs_M = np.linalg.eigh(cov_M)
    minor_M = eigvecs_M[:, 0]                   # minor axis = perpendicular to major
    minor_extent = float(np.sqrt(max(eigvals_M[0], 1e-6)) * 2.5)
    cut_p1 = (centroid_M[0] - minor_M[0] * minor_extent,
              centroid_M[1] - minor_M[1] * minor_extent)
    cut_p2 = (centroid_M[0] + minor_M[0] * minor_extent,
              centroid_M[1] + minor_M[1] * minor_extent)
else:
    cut_p1 = cut_p2 = (0, 0)

# Baseline cpsam on this crop (re-run for consistency with the sweep — cpsam can behave
# differently on a small crop than embedded in the full ROI)
img_crop_base = np.stack([dapi_c, s18_c], axis=-1)
masks_crop_base, n_crop_base = run_cpsam(img_crop_base)

# Did the doublet stay merged on this re-run?
def correct_split_check_local(masks, p1, p2):
    n = int(masks.max())
    y1_, x1_ = int(round(p1[0])), int(round(p1[1]))
    y2_, x2_ = int(round(p2[0])), int(round(p2[1]))
    H_, W_ = masks.shape
    if not (0 <= y1_ < H_ and 0 <= x1_ < W_ and 0 <= y2_ < H_ and 0 <= x2_ < W_):
        return n, False
    L1, L2 = int(masks[y1_, x1_]), int(masks[y2_, x2_])
    return n, (L1 != 0 and L2 != 0 and L1 != L2)
n_crop, ok_crop = correct_split_check_local(masks_crop_base, c1_c, c2_c)

# Visualize natural tissue context with the doublet highlighted
fig, ax = plt.subplots(1, 3, figsize=(15, 5.0))
ax[0].imshow(composite_c, interpolation="nearest")
# Highlight the chosen doublet's cpsam mask outline in cyan (so you see what cpsam called as 1 cell)
edges = find_boundaries(mcps_c_baseline == chosen_cps, mode="outer")
ey, ex = np.where(edges)
ax[0].plot(ex, ey, ".", color="cyan", ms=2.5)
ax[0].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=8, mec="black")
ax[0].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=8, mec="black")
ax[0].plot([c1_c[1], c2_c[1]], [c1_c[0], c2_c[0]], "-", color="white", lw=1)
ax[0].set_title(f"DAPI(blue)+18S(yellow); doublet's cpsam mask outline (cyan)\\n  cpsam #{chosen_cps}, 10X cells {top2}"); ax[0].axis("off")

# 10X centroid axis (yellow) + perpendicular cut at the midpoint (red), clipped to merged-cell mask
mid_c = (0.5 * (c1_c[0] + c2_c[0]), 0.5 * (c1_c[1] + c2_c[1]))
# Perpendicular cut endpoints: rotate (c2 - c1) by 90° from midpoint, length scaled by mask extent
dy_axis, dx_axis = c2_c[0] - c1_c[0], c2_c[1] - c1_c[1]
L_axis = (dy_axis**2 + dx_axis**2) ** 0.5
if L_axis > 1e-6:
    perp_dy, perp_dx = -dx_axis / L_axis, dy_axis / L_axis
    perp_len = 0.6 * L_axis
    cut_line_p1 = (mid_c[0] - perp_dy * perp_len, mid_c[1] - perp_dx * perp_len)
    cut_line_p2 = (mid_c[0] + perp_dy * perp_len, mid_c[1] + perp_dx * perp_len)
else:
    cut_line_p1 = cut_line_p2 = mid_c

ax[1].imshow(composite_c, interpolation="nearest")
edges = find_boundaries(mask_M_c, mode="outer")
ey, ex = np.where(edges)
ax[1].plot(ex, ey, ".", color="cyan", ms=2.0)
ax[1].plot([c1_c[1], c2_c[1]], [c1_c[0], c2_c[0]], "-", color="yellow", lw=1.0)
ax[1].plot([cut_line_p1[1], cut_line_p2[1]], [cut_line_p1[0], cut_line_p2[0]], "-", color="red", lw=2.0)
ax[1].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=6, mec="black")
ax[1].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=6, mec="black")
ax[1].plot([mid_c[1]], [mid_c[0]], "x", color="red", ms=10, mew=2)
ax[1].set_title("merged cell (cyan) + 10X-centroid axis (yellow) + cut at midpoint (red, ✕)\\n  cut applied across whole image (NOT clipped — neighbor bleed is visible)"); ax[1].axis("off")

# All cpsam masks on this crop (so you can verify it found the doublet correctly)
ax[2].imshow(composite_c, interpolation="nearest")
ax[2].imshow(label_overlay(masks_crop_base, seed=21, alpha_fill=0.25), interpolation="nearest")
ax[2].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=8, mec="black")
ax[2].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=8, mec="black")
ax[2].set_title(f"baseline cpsam on this crop  (n={n_crop_base}, doublet split={ok_crop})"); ax[2].axis("off")
plt.tight_layout(); plt.show()

print(f"\\nbaseline cpsam on this tissue-context crop: n={n_crop_base}, doublet correctly split = {ok_crop}")
print(f"  → {'still merged — good Phase B test case' if not ok_crop else 'already split — try DOUBLET_RANK=1, 2, ...'}")
"""))

cells.append(md("""### 2d. Define cut + apply (cut at 10X midpoint, no clipping)

Cut at **midpoint of the two 10X centroids**, perpendicular to the c1→c2 line. The midpoint sits in the cytoplasm bridge between nuclei by construction — the PCA centroid is mass-weighted and can be biased toward the larger cell in asymmetric doublets, landing the cut inside one cell's body where 18S is naturally low (because the nucleus displaces cytoplasm) and dimming has little effect.

**No clipping.** The Gaussian dim is applied to the whole image (within its falloff range), modifying only the input to CellPose — CellPose itself runs unmodified. We do *not* zero the dim outside the merged-cell mask, because that would hide the neighbor-bleed effect (pieces of the split cell merging with adjacent cells). We want the full impact visible. We dim 18S only; DAPI is unchanged."""))

cells.append(code("""SIG_B_DEMO, F_B_DEMO = 4.0, 0.0   # σ=4 px, fully erased at the centroid-midpoint

s18_cut = apply_gaussian_strip_along_line(s18_c, c1_c, c2_c, SIG_B_DEMO, F_B_DEMO)
img_cut = np.stack([dapi_c, s18_cut], axis=-1)
masks_cut, n_cut = run_cpsam(img_cut)
print(f"cut (σ={SIG_B_DEMO}, factor={F_B_DEMO}): cpsam → {n_cut} cells")

composite_cut = blue_yellow_composite(dapi_c, s18_cut)
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(composite_c, interpolation="nearest")
ax[0].set_title("composite baseline (DAPI blue + 18S yellow)"); ax[0].axis("off")
ax[1].imshow(composite_cut, interpolation="nearest")
ax[1].plot([c1_c[1], c2_c[1]], [c1_c[0], c2_c[0]], "-", color="cyan", lw=1.0)
ax[1].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=5, mec="black")
ax[1].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=5, mec="black")
ax[1].set_title(f"composite after cut (σ={SIG_B_DEMO}, f={F_B_DEMO})\\nGaussian applied across whole image"); ax[1].axis("off")
ax[2].imshow(composite_c, interpolation="nearest")
ax[2].imshow(label_overlay(masks_crop_base, seed=21), interpolation="nearest")
ax[2].set_title(f"cpsam baseline (n={n_crop_base})"); ax[2].axis("off")
ax[3].imshow(composite_cut, interpolation="nearest")
ax[3].imshow(label_overlay(masks_cut, seed=22), interpolation="nearest")
ax[3].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=5, mec="black")
ax[3].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=5, mec="black")
ax[3].set_title(f"cpsam after cut (n={n_cut})"); ax[3].axis("off")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""### 2e. Sweep on the real doublet (cut at 10X midpoint, no clipping)

`(σ × factor)` sweep with the Gaussian cut centered at the midpoint of the two 10X centroids, perpendicular to the c1→c2 line. No clipping — the dim band can extend into neighboring cells. The most aggressive setting (σ=24, f=0) sets the multiplier to 0 at the midpoint and produces a wide dim band reaching ~50 px on either side. Success criterion: the two 10X centroids land in *different* cpsam mask labels."""))

cells.append(code("""def split_quality(masks, p1, p2, mask_M):
    \"\"\"Did the cut cleanly split the original merged cpsam region (mask_M) into two
    cells corresponding to centroids p1, p2?  Returns a dict of metrics.

    Three things have to be true for a clean split:
      1. p1 and p2 must be in DIFFERENT non-zero cpsam labels (call them L1, L2).
      2. PER-SIDE COVERAGE: Voronoi-partition mask_M by closest centroid; each side
         must be ≥ MIN_COVERAGE covered by its centroid's label.
      3. BULK COVERAGE: L1+L2 together must claim ≥ MIN_BULK of the original merged
         region. If bulk is low, the merged cell got fragmented across many labels
         (only two of which happen to contain the centroids) — that's not a real
         2-way split, that's fragmentation that happens to put the centroids in
         different fragments. Catches the noise=0.5 / n=4 failure mode where the
         cell is effectively "not split" even though counts and centroids look ok.
    \"\"\"
    MIN_COVERAGE = 0.6
    MIN_BULK     = 0.7
    n = int(masks.max())
    H_, W_ = masks.shape
    y1_, x1_ = int(round(p1[0])), int(round(p1[1]))
    y2_, x2_ = int(round(p2[0])), int(round(p2[1]))
    in_bounds = (0 <= y1_ < H_ and 0 <= x1_ < W_ and 0 <= y2_ < H_ and 0 <= x2_ < W_)
    L1 = int(masks[y1_, x1_]) if in_bounds else 0
    L2 = int(masks[y2_, x2_]) if in_bounds else 0
    different = (L1 != 0 and L2 != 0 and L1 != L2)

    yy, xx = np.mgrid[0:H_, 0:W_]
    d1 = (yy - p1[0]) ** 2 + (xx - p1[1]) ** 2
    d2 = (yy - p2[0]) ** 2 + (xx - p2[1]) ** 2
    side1 = mask_M & (d1 <= d2)
    side2 = mask_M & (d2 <  d1)

    cov1 = float((side1 & (masks == L1)).sum()) / max(float(side1.sum()), 1) if L1 != 0 else 0.0
    cov2 = float((side2 & (masks == L2)).sum()) / max(float(side2.sum()), 1) if L2 != 0 else 0.0

    if L1 != 0 and L2 != 0:
        bulk = float((((masks == L1) | (masks == L2)) & mask_M).sum()) / max(float(mask_M.sum()), 1)
    else:
        bulk = 0.0

    clean = (different and
             cov1 >= MIN_COVERAGE and cov2 >= MIN_COVERAGE and
             bulk >= MIN_BULK)
    return dict(n=n, L1=L1, L2=L2, different=different,
                cov1=cov1, cov2=cov2, bulk=bulk, clean=clean)


sigmas_b   = [0, 2, 4, 6, 8, 12, 16, 20, 24]   # extended for real-cell scale (σ in pixels at 0.21 µm/px)
factors_b  = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]    # f=0 → multiplier of 0 at the boundary peak (full erase)

n_grid_b    = np.zeros((len(factors_b), len(sigmas_b)), dtype=np.int32)
ok_grid_b   = np.zeros_like(n_grid_b, dtype=bool)
cov1_grid_b = np.zeros((len(factors_b), len(sigmas_b)), dtype=np.float32)
cov2_grid_b = np.zeros_like(cov1_grid_b)
bulk_grid_b = np.zeros_like(cov1_grid_b)
runs_b      = {}

t0 = time.time()
for i, f in enumerate(factors_b):
    for j, sg in enumerate(sigmas_b):
        s18_cut = apply_gaussian_strip_along_line(s18_c, c1_c, c2_c, sg, f)
        img = np.stack([dapi_c, s18_cut], axis=-1)
        masks, _ = run_cpsam(img)
        q = split_quality(masks, c1_c, c2_c, mask_M_c)
        n_grid_b[i, j]    = q["n"]
        ok_grid_b[i, j]   = q["clean"]
        cov1_grid_b[i, j] = q["cov1"]
        cov2_grid_b[i, j] = q["cov2"]
        bulk_grid_b[i, j] = q["bulk"]
        runs_b[(i, j)] = masks
print(f"Phase-B sweep done in {time.time()-t0:.1f}s   ({len(sigmas_b)*len(factors_b)} cpsam runs)")
"""))

cells.append(md("""### 2f. Phase B summary heatmap"""))

cells.append(code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
im0 = ax[0].imshow(n_grid_b, cmap="viridis", aspect="auto",
                    vmin=0, vmax=max(3, n_grid_b.max()))
ax[0].set_xticks(range(len(sigmas_b)));  ax[0].set_xticklabels([f"{sg:g}" for sg in sigmas_b])
ax[0].set_yticks(range(len(factors_b))); ax[0].set_yticklabels([f"{f:.1f}" for f in factors_b])
ax[0].set_xlabel("Gaussian σ (px)"); ax[0].set_ylabel("intensity factor at center")
ax[0].set_title("Phase B — cpsam cell count")
for i in range(len(factors_b)):
    for j in range(len(sigmas_b)):
        ax[0].text(j, i, str(n_grid_b[i, j]), ha="center", va="center",
                   color="white" if n_grid_b[i,j] < 2 else "black", fontsize=9)
plt.colorbar(im0, ax=ax[0], fraction=0.046)

ax[1].imshow(ok_grid_b.astype(int), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax[1].set_xticks(range(len(sigmas_b)));  ax[1].set_xticklabels([f"{sg:g}" for sg in sigmas_b])
ax[1].set_yticks(range(len(factors_b))); ax[1].set_yticklabels([f"{f:.1f}" for f in factors_b])
ax[1].set_xlabel("Gaussian σ (px)"); ax[1].set_ylabel("intensity factor at center")
ax[1].set_title("correctly split (each centroid in a different cpsam mask)")
for i in range(len(factors_b)):
    for j in range(len(sigmas_b)):
        sym = "✓" if ok_grid_b[i, j] else "✗"
        ax[1].text(j, i, sym, ha="center", va="center", fontsize=12)
plt.tight_layout(); plt.show()

print(f"\\n# Phase B summary on doublet (cpsam #{chosen_cps}):")
print(f"  correct splits: {int(ok_grid_b.sum())}/{ok_grid_b.size}")
print(f"  any n>=2:       {int((n_grid_b >= 2).sum())}/{n_grid_b.size}")
"""))

cells.append(md("""### 2g. Phase B full mask grid — every (σ × factor) combination

**Two grids:**
1. **Composite (DAPI blue + cut-18S yellow), no mask overlay** — verify the cut is actually deep + wide enough across the cell. DAPI is shown so you can see where the nuclei are vs where the cut lands. If you don't see a clear dark band crossing the cytoplasm bridge between the two nuclei, the σ is too small.
2. **Composite + cpsam masks** — what cpsam called given the cut. Title color: green = correct split, red = same mask / n<2, yellow = n>2."""))

cells.append(code("""# ─── Grid 1: composite (DAPI + cut 18S), no mask overlay ─────────────────────────
fig, ax = plt.subplots(len(factors_b), len(sigmas_b),
                       figsize=(1.6*len(sigmas_b), 1.85*len(factors_b)),
                       squeeze=False)
for i, f in enumerate(factors_b):
    for j, sg in enumerate(sigmas_b):
        s18_cut = apply_gaussian_strip_along_line(s18_c, c1_c, c2_c, sg, f)
        ax[i, j].imshow(blue_yellow_composite(dapi_c, s18_cut), interpolation="nearest")
        ax[i, j].plot([c1_c[1], c2_c[1]], [c1_c[0], c2_c[0]], "-", color="cyan", lw=0.6)
        ax[i, j].plot([c1_c[1]], [c1_c[0]], "o", color="cyan", ms=3, mec="black")
        ax[i, j].plot([c2_c[1]], [c2_c[0]], "o", color="cyan", ms=3, mec="black")
        ax[i, j].set_title(f"f={f:.1f}  σ={sg:g}", fontsize=8)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
plt.suptitle("Phase B — composite (DAPI blue + cut-18S yellow), no mask overlay",
             fontsize=11, y=1.005)
plt.tight_layout(); plt.show()

# ─── Grid 2: composite + cpsam mask overlay — what cpsam called ──────────────────────
fig, ax = plt.subplots(len(factors_b), len(sigmas_b),
                       figsize=(1.6*len(sigmas_b), 1.85*len(factors_b)),
                       squeeze=False)
for i, f in enumerate(factors_b):
    for j, sg in enumerate(sigmas_b):
        s18_cut = apply_gaussian_strip_along_line(s18_c, c1_c, c2_c, sg, f)
        comp_cut = blue_yellow_composite(dapi_c, s18_cut)
        masks = runs_b[(i, j)]
        n     = int(n_grid_b[i, j])
        ok    = bool(ok_grid_b[i, j])

        ax[i, j].imshow(comp_cut, interpolation="nearest")
        ax[i, j].imshow(label_overlay(masks, seed=200 + i*len(sigmas_b) + j, alpha_fill=0.25),
                         interpolation="nearest")
        ax[i, j].plot([c1_c[1], c2_c[1]], [c1_c[0], c2_c[0]], "-", color="white", lw=0.8)
        ax[i, j].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=4, mec="black")
        ax[i, j].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=4, mec="black")

        cov1, cov2 = cov1_grid_b[i, j], cov2_grid_b[i, j]
        if ok:                bg = "#bce8b3"
        elif n > 2:           bg = "#f6e7a3"
        else:                 bg = "#f4b6b2"
        ax[i, j].set_title(f"f={f:.1f}  σ={sg:g}\\nn={n}  cov={cov1*100:.0f}/{cov2*100:.0f}%  {'✓' if ok else '✗'}",
                           fontsize=7.5, backgroundcolor=bg)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
plt.suptitle("Phase B — composite + cpsam masks  (rows: factor, cols: σ)",
             fontsize=11, y=1.005)
plt.tight_layout(); plt.show()
"""))

# ── 2h. Phase B noise tolerance ─────────────────────────────────────────
cells.append(md("""### 2h. Phase B noise tolerance

Same logic as 1g, on the real doublet. Auto-pick the most-aggressive *correct* (σ, f_mean) from 2e (smallest σ wins ties — matches the operating-regime preference for narrow cuts), then sweep `noise_std`. f varies along the cut tangent (perpendicular to the c1→c2 axis). Top row: composite (DAPI blue + noisy-cut 18S yellow); bottom row: cpsam masks.

**Evaluation:** counting cells is insufficient — at high noise, fragmentation can put p1 and p2 in different labels even when the actual cleavage between the two cells didn't happen (e.g. n=4 fragments that don't correspond to the original two cells). The richer `split_quality` metric Voronoi-partitions the original merged region by closest 10X centroid and reports per-side coverage. **clean ✓ requires both sides ≥50% covered by their centroid's label**, not just "different labels". Also reports `bulk` = fraction of the original merged region claimed by L1+L2 together (drops when fragmentation steals pixels)."""))

cells.append(code("""# Auto-pick (σ, f_mean) from the correct-split cells in the earlier sweep
correct_cells = [(i, j) for i in range(len(factors_b)) for j in range(len(sigmas_b))
                 if ok_grid_b[i, j] and sigmas_b[j] > 0]
if correct_cells:
    pick = min(correct_cells, key=lambda ij: (sigmas_b[ij[1]], factors_b[ij[0]]))
    SIG_NOISE_B, F_MEAN_NOISE_B = sigmas_b[pick[1]], factors_b[pick[0]]
    print(f"using (σ={SIG_NOISE_B:g}, f_mean={F_MEAN_NOISE_B:g}) — auto-picked from 2e correct splits")
else:
    SIG_NOISE_B, F_MEAN_NOISE_B = 4.0, 0.0
    print(f"no correct splits in 2e — fallback to (σ={SIG_NOISE_B:g}, f_mean={F_MEAN_NOISE_B:g})")

NOISE_LEVELS_B = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

results_noise_B = []
for ns in NOISE_LEVELS_B:
    s18_cut = apply_gaussian_strip_along_line_noisy(s18_c, c1_c, c2_c,
                                                     SIG_NOISE_B, F_MEAN_NOISE_B,
                                                     ns, seed=0)
    img = np.stack([dapi_c, s18_cut], axis=-1)
    masks, _ = run_cpsam(img)
    q = split_quality(masks, c1_c, c2_c, mask_M_c)
    results_noise_B.append((ns, s18_cut, masks, q))

fig, ax = plt.subplots(2, len(NOISE_LEVELS_B),
                       figsize=(2.0*len(NOISE_LEVELS_B), 4.0), squeeze=False)
for j, (ns, s18_cut, masks, q) in enumerate(results_noise_B):
    ax[0, j].imshow(blue_yellow_composite(dapi_c, s18_cut), interpolation="nearest")
    ax[0, j].set_title(f"noise_std = {ns:g}", fontsize=9)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    ax[1, j].imshow(blue_yellow_composite(dapi_c, s18_cut), interpolation="nearest")
    ax[1, j].imshow(label_overlay(masks, seed=600 + j, alpha_fill=0.25), interpolation="nearest")
    ax[1, j].plot([c1_c[1], c2_c[1]], [c1_c[0], c2_c[0]], "-", color="white", lw=0.7)
    ax[1, j].plot([c1_c[1]], [c1_c[0]], "o", color="white", ms=4, mec="black")
    ax[1, j].plot([c2_c[1]], [c2_c[0]], "o", color="white", ms=4, mec="black")
    bg = "#bce8b3" if q["clean"] else ("#f6e7a3" if q["n"] > 2 else "#f4b6b2")
    ax[1, j].set_title(f"n={q['n']}  cov={q['cov1']*100:.0f}/{q['cov2']*100:.0f}%\\nbulk={q['bulk']*100:.0f}%  {'✓' if q['clean'] else '✗'}",
                        fontsize=8, backgroundcolor=bg)
    ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
plt.suptitle(f"Phase B — noise robustness (σ={SIG_NOISE_B:g}, f_mean={F_MEAN_NOISE_B:g}); "
             f"cov = per-side coverage, bulk = % of original merged region claimed by L1+L2",
             fontsize=10, y=1.005)
plt.tight_layout(); plt.show()

print("\\nNoise vs cpsam outcome on the real doublet:")
print(f"  {'noise_std':>9s} {'n_cells':>7s} {'cov1':>6s} {'cov2':>6s} {'bulk':>6s}  verdict")
last_clean = None
for ns, _, _, q in results_noise_B:
    if q["clean"]:
        verdict = "✓ clean split"; last_clean = ns
    elif not q["different"]:
        if q["L1"] == q["L2"] and q["L1"] != 0:
            verdict = "✗ both centroids in same label (still merged)"
        else:
            verdict = "✗ centroid in background"
    elif q["bulk"] < 0.7:
        verdict = f"✗ fragmented — L1+L2 only cover {q['bulk']*100:.0f}% of the original merged cell"
    elif q["cov1"] < 0.6 or q["cov2"] < 0.6:
        verdict = f"✗ different labels but per-side coverage too low ({q['cov1']*100:.0f}/{q['cov2']*100:.0f}%)"
    else:
        verdict = "✗ unclassified failure"
    print(f"  {ns:>9.2f} {q['n']:>7d} {q['cov1']*100:>5.0f}% {q['cov2']*100:>5.0f}% {q['bulk']*100:>5.0f}%  {verdict}")
if last_clean is not None:
    print(f"\\n  highest noise_std with clean split: {last_clean:.2f}")
"""))

# ── 3. Findings + design implications ───────────────────────────────────
cells.append(md("""## 3. Findings + design implications for notebooks 1–3

**Step 0 outcome:** the Gaussian-dim cut on 18S is a viable boundary-prior intervention. Phase A established that cpsam responds to a smooth dim profile in the synthetic case. Phase B confirmed that a well-placed cut splits real doublets in tissue context.

**Operating regime (the key finding from Phase B):**

|              | narrow σ                     | wide σ                                  |
|--------------|------------------------------|-----------------------------------------|
| **deep (low f)**     | ✓ clean splits                | shrinks cell extent — extreme cases only |
| **shallow (high f)** | no-op (fail-safe)             | confuses cpsam (worst outcome)          |

- *Deep + narrow* is the recommended default for confident boundaries.
- *Shallow + narrow* is the **fail-safe**: if upstream confidence is wrong, the cut just doesn't fire. We don't damage cells when our gradient model is uncertain.
- *Wide* σ should be avoided in normal operation — the dim band encroaches on cell bodies, shrinking apparent extent (deep) or producing odd merges with neighbors (shallow).

**Design rule for the upstream pipeline (notebooks 1–3):**
1. Hold σ small and fixed (a single hyperparameter, not learned per-pixel).
2. Map the per-pixel mRNA-gradient confidence to the **depth f** (1 = uncertain → no-op, 0 = confident boundary → full erase).
3. Output is a per-pixel depth map; combine with the fixed-σ Gaussian to get the actual 18S multiplier.
4. The fail-safe property comes free: low-confidence regions automatically become no-ops.

**Verdict:** proceed to notebooks 1–3. Step 0 is green.

**Stretch test (optional):** re-run 2c–2g with `DOUBLET_RANK=1, 2, ...` to confirm the operating regime generalizes across multiple doublets, not just the rank-0 pick.
"""))

# ─── Write notebook ───────────────────────────────────────────────────────
nb.cells = cells

out = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/workflow/04_gap_intervention_test.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote:", out)
