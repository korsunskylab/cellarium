"""V0 assumption audit. Test every basic claim about data loading and the
math at each pipeline step. Run before trusting any V0 output.

Each check prints PASS / FAIL / WARN with the concrete evidence.

Categories:
  A. Source tiffs really contain DAPI / 18S as labelled
  B. Pixel-coordinate math (µm → px) round-trips
  C. Bbox slicing returns the spatial neighbourhood of the focal centroid
  D. Anchor extraction from zarr respects qv, valid, lineage, and bbox
  E. Anchor pixel coordinates align with morphology brightness
     (transcripts in a Mel cell should land where 18S/DAPI are bright)
  F. Gene→lineage map is correctly indexed
  G. Posterior math: Dirichlet + grey-abstain
  H. Edge one-vs-rest gives the expected sign pattern
  I. cut_field actually attenuates 18S
  J. CP-SAM 1-99 renorm preserves the cut
  K. WSI mask read by bbox matches the focal cell location
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, tifffile, zarr
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lib

BID = "DB_top100_018"
CROP_PX = 2048
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"

def report(name, ok, detail=""):
    tag = PASS if ok == "pass" else FAIL if ok == "fail" else WARN
    print(f"[{tag}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")

print("=" * 70)
print(" V0 assumption audit")
print("=" * 70)

# --------------------------------------------------------------------
# A. Source tiffs really contain DAPI vs 18S
# --------------------------------------------------------------------
print("\n[A] Source-tiff identity (DAPI vs 18S)")

# A1: OME XML channel-name claim
with tifffile.TiffFile(lib.DAPI_TIF) as tf:
    import re
    names_d = re.findall(r'Channel[^>]*Name="([^"]+)"', tf.ome_metadata or "")
with tifffile.TiffFile(lib.S18_TIF) as tf:
    names_s = re.findall(r'Channel[^>]*Name="([^"]+)"', tf.ome_metadata or "")
report("A1 morphology_focus_0000 channel names",
       "pass" if names_d and names_d[0] == "DAPI" else "fail",
       f"_0000 channels: {names_d}")
report("A1 morphology_focus_0002 channel names (page 0 should be 18S)",
       "pass" if names_s and "18S" in (names_s + [""])[2] else "warn",
       f"_0002 channels: {names_s}\n"
       f"Note: OME XML lists ALL panel channels per file but actually each file holds ONE channel (page 0). The page-0 content of _0002 should be 18S.")

# A2: DAPI vs 18S whole-page statistics differ
# (cheap full-page percentiles)
def page_quick_stats(path):
    arr = tifffile.imread(path, key=0)
    rng = np.random.default_rng(0)
    n = arr.size
    idx = rng.integers(0, n, 200_000)
    sample = arr.ravel()[idx]
    return float(np.percentile(sample, 50)), float(np.percentile(sample, 99))

d50, d99 = page_quick_stats(lib.DAPI_TIF)
s50, s99 = page_quick_stats(lib.S18_TIF)
report("A2 DAPI vs 18S page-0 distributions differ",
       "pass" if abs(d99 - s99) > 30 or abs(d50 - s50) > 30 else "fail",
       f"DAPI:  median={d50}, p99={d99}\n"
       f"18S :  median={s50}, p99={s99}")

# A3: tifffile API used in lib.load_morphology returns different arrays
# (this is the bug we just fixed; re-verify)
dapi_crop_lib, s18_crop_lib = lib.load_morphology(5000, 5050, 41000, 41050)
report("A3 lib.load_morphology returns different DAPI vs 18S",
       "pass" if not np.array_equal(dapi_crop_lib, s18_crop_lib) else "fail",
       f"50x50 sample crop: DAPI mean={dapi_crop_lib.mean():.0f}, 18S mean={s18_crop_lib.mean():.0f}\n"
       f"pixels equal: {(dapi_crop_lib == s18_crop_lib).sum()}/{dapi_crop_lib.size}")

# --------------------------------------------------------------------
# B. Pixel-coordinate math round-trip
# --------------------------------------------------------------------
print("\n[B] Pixel-coordinate math")
ux, uy = 8934.5, 1200.1
px = round(ux / lib.PIXEL_UM)
py = round(uy / lib.PIXEL_UM)
back_x = px * lib.PIXEL_UM
back_y = py * lib.PIXEL_UM
report("B1 µm ↔ px round-trip within 1 px",
       "pass" if abs(back_x - ux) < lib.PIXEL_UM and abs(back_y - uy) < lib.PIXEL_UM else "fail",
       f"(8934.5, 1200.1)µm -> ({px}, {py})px -> ({back_x:.3f}, {back_y:.3f})µm")

# --------------------------------------------------------------------
# C. Bbox clamping + size
# --------------------------------------------------------------------
print("\n[C] Bbox clamping")
with tifffile.TiffFile(lib.DAPI_TIF) as tf:
    wsi_H, wsi_W = tf.series[0].shape[-2:]
y0, y1, x0, x1 = lib.make_roi_bbox_centred(8934.5, 1200.1, CROP_PX, wsi_H, wsi_W)
report("C1 bbox size exactly CROP_PX",
       "pass" if (y1-y0 == CROP_PX) and (x1-x0 == CROP_PX) else "fail",
       f"y=[{y0},{y1}] x=[{x0},{x1}]  size: {y1-y0}×{x1-x0}")
cps_x_px = round(8934.5/lib.PIXEL_UM); cps_y_px = round(1200.1/lib.PIXEL_UM)
report("C2 focal centroid lies inside the bbox",
       "pass" if (x0 <= cps_x_px < x1 and y0 <= cps_y_px < y1) else "fail",
       f"centroid px ({cps_x_px}, {cps_y_px})  bbox x[{x0},{x1}] y[{y0},{y1}]")

# --------------------------------------------------------------------
# D. Anchor extraction filters
# --------------------------------------------------------------------
print("\n[D] Anchor extraction filters")
z = zarr.open(lib.TX_ZARR, mode="r")
g0 = z["grids"]["0"]
# Inspect one tile that should overlap the focal: 8934.5 / 250 = 35.7, 1200.1 / 250 = 4.8 → tile "35,4"
key = "35,4"
assert key in g0, key
tile = g0[key]
total_in_tile = tile["location"].shape[0]
qv_all = tile["quality_score"][:].squeeze(-1)
val_all = tile["valid"][:].squeeze(-1)
gid_all = tile["gene_identity"][:].squeeze(-1).astype(np.int64)
gene_lin = lib.load_lineage_label_map()
lin_all = gene_lin[gid_all]
n_qv = (qv_all >= 20).sum()
n_qv_valid = ((qv_all >= 20) & (val_all == 1)).sum()
n_qv_valid_lin = ((qv_all >= 20) & (val_all == 1) & (lin_all >= 0)).sum()
report("D1 qv≥20 keeps a sane fraction (>10%)",
       "pass" if n_qv > total_in_tile * 0.1 else "warn",
       f"tile {key}: {total_in_tile} raw → {n_qv} after qv≥20 ({n_qv/total_in_tile*100:.1f}%)")
report("D2 valid==1 keeps near-all of qv-passed (>90%)",
       "pass" if n_qv_valid > n_qv * 0.9 else "warn",
       f"qv≥20: {n_qv} → qv & valid: {n_qv_valid} ({n_qv_valid/n_qv*100:.1f}%)")
report("D3 lineage filter retains some fraction (>5%)",
       "pass" if n_qv_valid_lin > n_qv_valid * 0.05 else "warn",
       f"qv & valid: {n_qv_valid} → lineage-mapped: {n_qv_valid_lin} ({n_qv_valid_lin/max(n_qv_valid,1)*100:.1f}%)")

# --------------------------------------------------------------------
# E. Anchor pixel coords align with morphology brightness
# --------------------------------------------------------------------
print("\n[E] Anchor coords align with morphology brightness")
py_a, px_a, li_a = lib.load_anchors_in_bbox(y0, y1, x0, x1, gene_lin)
dapi_full, s18_full = lib.load_morphology(y0, y1, x0, x1)
# Brightness sampled AT anchor pixels vs random pixels
np.random.seed(0)
n_anc = len(py_a)
random_y = np.random.randint(0, dapi_full.shape[0], n_anc)
random_x = np.random.randint(0, dapi_full.shape[1], n_anc)
dapi_at_anc = dapi_full[py_a, px_a]
dapi_at_rand = dapi_full[random_y, random_x]
report("E1 DAPI is brighter at anchor pixels than random pixels (~3× expected)",
       "pass" if dapi_at_anc.mean() > 1.5 * dapi_at_rand.mean() else "fail",
       f"anchors n={n_anc}\n"
       f"mean DAPI at anchor: {dapi_at_anc.mean():.0f}\n"
       f"mean DAPI at random: {dapi_at_rand.mean():.0f}\n"
       f"ratio: {dapi_at_anc.mean()/max(dapi_at_rand.mean(),1):.2f}x")
s18_at_anc = s18_full[py_a, px_a]
s18_at_rand = s18_full[random_y, random_x]
report("E2 18S is brighter at anchor pixels than random pixels (~3× expected)",
       "pass" if s18_at_anc.mean() > 1.5 * s18_at_rand.mean() else "fail",
       f"mean 18S at anchor: {s18_at_anc.mean():.0f}\n"
       f"mean 18S at random: {s18_at_rand.mean():.0f}\n"
       f"ratio: {s18_at_anc.mean()/max(s18_at_rand.mean(),1):.2f}x")

# --------------------------------------------------------------------
# F. Gene → lineage map correctness
# --------------------------------------------------------------------
print("\n[F] Gene → lineage map")
gene_names = list(z.attrs["gene_names"])
gl = pd.read_parquet(lib.GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
# Spot-check known marker genes
checks = [("MITF", "Melanoma"), ("CD3E", "Tcell"), ("KRT5", "Keratinocyte"),
          ("COL1A1", "Fibroblast"), ("CD79A", "Plasma"), ("CD14", "Myeloid"),
          ("PECAM1", "Endothelial")]
gene_to_idx = {g: i for i, g in enumerate(gene_names)}
ok_count = 0; total = 0; details = []
for g, expected_lin in checks:
    if g not in gene_to_idx:
        details.append(f"  {g}: not in panel (skip)")
        continue
    total += 1
    actual_k = int(gene_lin[gene_to_idx[g]])
    actual_lin = lib.LINEAGES[actual_k] if actual_k >= 0 else f"ambiguous (label={gl.get(g, 'NA')})"
    ok = actual_lin == expected_lin
    details.append(f"  {g}: expected={expected_lin}, got={actual_lin}  {'✓' if ok else '✗'}")
    if ok: ok_count += 1
report("F1 known marker genes map to the expected lineage",
       "pass" if ok_count == total else "warn",
       "\n".join(details))

# --------------------------------------------------------------------
# G. Posterior math
# --------------------------------------------------------------------
print("\n[G] Posterior math")
# Synthetic test: single lineage with N_eff = 19 at one pixel, others 0
N_eff_test = np.zeros((lib.K, 3, 3), dtype=np.float32)
N_eff_test[0, 1, 1] = 19.0
N_total = N_eff_test.sum(0)
pi_post = (N_eff_test + lib.ALPHA) / (N_total + lib.K * lib.ALPHA + 1e-9)[None]
confidence = N_total / (N_total + lib.N_MIN + 1e-9)
pi_abst = confidence[None] * pi_post + (1.0 - confidence[None]) * (1.0/lib.K)
top = pi_abst[0, 1, 1]
# Expected: post = (19+10)/(19+70) = 29/89 = 0.326. conf = 19/22 = 0.864.
# pi_abst = 0.864 * 0.326 + 0.136 * 1/7 = 0.282 + 0.019 = 0.301
expected = (0.864 * 29/89) + (0.136 * 1/7)
report("G1 Posterior matches the math by hand (single-lineage pixel)",
       "pass" if abs(top - expected) < 0.01 else "fail",
       f"19 anchors single-lineage pixel:\n"
       f"  expected pi_abst[dom] = {expected:.3f}\n"
       f"  got                   = {top:.3f}")

# G2: with α=10 and 19 effective anchors, the maximum achievable pi_abst is ~0.30 — surface this
report("G2 max-attainable π_abst[dominant] @ α=10, N_eff=19 is only ~0.30",
       "warn",
       "This is by design (Dirichlet α=10 is a strong prior).\n"
       "It means s_k = π_top1 − π_top2 is bounded above by ~0.3 even in pure regions,\n"
       "so tanh(K · s_k) is in [tanh(0) = 0, tanh(8 · 0.3) = 0.985], not [-1, 1].\n"
       "Check whether α=10 was tuned for smaller crops/different anchor densities.")

# --------------------------------------------------------------------
# H. Edge one-vs-rest sign pattern
# --------------------------------------------------------------------
print("\n[H] Edge one-vs-rest sign pattern")
# Synthetic: 1D strip with lineage A dominant on left, B on right
W2 = 100
pi_test = np.zeros((lib.K, 1, W2), dtype=np.float32)
pi_test[0] = np.where(np.arange(W2) < 50, 0.5, 0.1)  # lineage 0 = top1 on left, fall on right
pi_test[1] = np.where(np.arange(W2) < 50, 0.1, 0.5)  # lineage 1 = top1 on right
# Fill the rest evenly with the remaining mass
remaining = 1.0 - pi_test[:2].sum(0)
for k in range(2, lib.K):
    pi_test[k] = remaining / (lib.K - 2)
edge, top_idx, margin = lib.edge_global_field(pi_test)
peak_col = int(np.argmax(edge[0]))
report("H1 edge peaks AT the lineage transition (col 50)",
       "pass" if abs(peak_col - 50) <= 2 else "fail",
       f"transition at col 50; edge peak at col {peak_col}, value {edge[0, peak_col]:.3f}")
left_label = int(top_idx[0, 25])
right_label = int(top_idx[0, 75])
report("H2 argmax label switches across the transition",
       "pass" if left_label == 0 and right_label == 1 else "fail",
       f"argmax left half = {left_label} ({lib.LINEAGES[left_label]}); right half = {right_label} ({lib.LINEAGES[right_label]})")

# --------------------------------------------------------------------
# I. apply_cut actually attenuates 18S
# --------------------------------------------------------------------
print("\n[I] apply_cut attenuates 18S correctly")
s18_test = np.ones((20, 20), dtype=np.uint16) * 1000
cut_test = np.zeros((20, 20), dtype=np.float32)
cut_test[10, 10] = 0.99
out = lib.apply_cut(s18_test, cut_test)
report("I1 strongest cut pixel becomes 1% of original",
       "pass" if 5 <= out[10, 10] <= 15 else "fail",
       f"input 1000, cut 0.99 → output {out[10, 10]} (expect ~10)")
report("I2 no-cut pixels stay at original value",
       "pass" if out[0, 0] == 1000 else "fail",
       f"input 1000, cut 0.0 → output {out[0, 0]}")

# --------------------------------------------------------------------
# J. CP-SAM 1-99 renorm preserves the cut
# --------------------------------------------------------------------
print("\n[J] Per-channel 1-99 renorm preserves cut depth")
# Use the actual ROI data
pi_abst_roi, conf_roi, _ = lib.lineage_posterior(py_a, px_a, li_a, dapi_full.shape[0], dapi_full.shape[1])
edge_roi, _, _ = lib.edge_global_field(pi_abst_roi)
percentiles = lib.load_wsi_percentiles()
evidence = lib.cell_evidence(dapi_full, s18_full, percentiles)
cut, _ = lib.cut_field_from_edge(edge_roi, conf_roi, evidence)
s_cut = lib.apply_cut(s18_full, cut)
# Renormalisation
def fresh_clip(arr):
    rng = np.random.default_rng(0)
    n = arr.size
    sample = arr.ravel() if n <= 10_000_000 else arr.ravel()[rng.integers(0, n, 10_000_000)]
    q_lo, q_hi = np.percentile(sample, [1.0, 99.0])
    return float(q_lo), float(q_hi)

s18_lo, s18_hi = fresh_clip(s18_full)
scut_lo, scut_hi = fresh_clip(s_cut)

# At the strongest cut pixel
ymx, xmx = np.unravel_index(cut.argmax(), cut.shape)
v_raw = s18_full[ymx, xmx]
v_cut = s_cut[ymx, xmx]
v_raw_norm = (v_raw - s18_lo) / max(s18_hi - s18_lo, 1)
v_cut_norm = (v_cut - scut_lo) / max(scut_hi - scut_lo, 1)
report("J1 cut attenuation visible to CP-SAM after fresh 1-99 renorm",
       "pass" if v_raw_norm - v_cut_norm > 0.3 else "warn",
       f"strongest cut pixel: raw 18S={v_raw}, s_cut={v_cut}\n"
       f"after norm: raw_norm={v_raw_norm:.3f}, cut_norm={v_cut_norm:.3f}, Δ={v_raw_norm-v_cut_norm:.3f}\n"
       f"18S 1-99: [{s18_lo:.0f}, {s18_hi:.0f}]; s_cut 1-99: [{scut_lo:.0f}, {scut_hi:.0f}]\n"
       f"WARN if Δ<0.3 means the renorm partly un-does the cut")

# --------------------------------------------------------------------
# K. WSI mask read by bbox matches focal cell
# --------------------------------------------------------------------
print("\n[K] WSI mask bbox extraction")
cps_focal = 32095
wsi_mask = tifffile.imread(lib.DATA / "cpsam_whole_slide" / "masks.tif")
report("K1 WSI mask exists and is uint32",
       "pass" if wsi_mask.dtype == np.uint32 else "warn",
       f"shape={wsi_mask.shape} dtype={wsi_mask.dtype}")
report("K2 focal cell id present in WSI mask",
       "pass" if (wsi_mask == cps_focal).any() else "fail",
       f"focal {cps_focal} pixel count in full WSI: {(wsi_mask == cps_focal).sum()}")
crop_mask = wsi_mask[y0:y1, x0:x1]
focal_pixels = (crop_mask == cps_focal).sum()
report("K3 focal cell pixel count in 2048 crop > 0 (i.e., bbox actually contains the cell)",
       "pass" if focal_pixels > 0 else "fail",
       f"focal cell {cps_focal} pixels in our 2048×2048 bbox: {focal_pixels}")

print("\n" + "=" * 70)
print(" END OF AUDIT")
print("=" * 70)
