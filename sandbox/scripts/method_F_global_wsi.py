"""Method F-global on the whole-slide image (segmentation-agnostic).

Pipeline (all inputs are raw — no CP-SAM masks, no doublet table):
  1. Load DAPI + 18S WSI (one z-plane each from morphology_focus).
  2. Load all WSI transcripts (qv ≥ 20, valid==1, lineage-mapped).
  3. Build per-lineage N_eff via gaussian-smoothed anchor density.
  4. Compute pi_abst (Dirichlet posterior + grey-abstain).
  5. Track per-pixel top1 / top2 of pi_abst across K lineages.
  6. edge_global = Σ_k |∇ tanh(K_sharp · s_k)| × conf² × cell_evidence
  7. cut_field = normalize(gaussian_filter(edge_global, σ_cut)) × depth.
  8. s_cut = s18_raw × (1 − cut_field).
  9. Save cut_field.tif and s_cut.tif (checkpoint before CP-SAM).
 10. Run CP-SAM on (DAPI, s_cut) with the validated max-recall settings.
 11. Save masks.tif (uint32).

Control = existing cpsam_whole_slide/masks.tif (DAPI + raw 18S). No re-run.

Defaults match per-doublet bench:
  ALPHA=10, N_MIN=3, K_SHARP=8, σ_cut=5 px, depth=0.99
  σ_diffuse_px = SIGMA_PX from edges_three_methods (≈ 9.4 px for 2 µm).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import gaussian_filter, sobel

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import SIGMA_PX as SIGMA_DIFFUSE_PX
from tint_cytoplasm_diffusion import LINEAGES

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
DAPI_TIF = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
S18_TIF  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
TX_ZARR  = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = DATA / "method_F_global_wsi"
OUT_DIR.mkdir(exist_ok=True)

PIXEL_SIZE_UM = 0.2125
TILE_SIZE_UM  = 250.0
ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
SIGMA_CUT_PX = 5.0; DEPTH_MAX = 0.99
QV_MIN = 20.0

LOG_PATH = OUT_DIR / "run.log"


def log(msg: str):
    print(msg, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(msg + "\n")


def percentile_clip_to_uint8(arr_u16, q_lo, q_hi, out_dtype=np.float32):
    """Memory-conscious 1-99 percentile clip to [0, 1] (matches existing WSI ctrl)."""
    rng = max(q_hi - q_lo, 1e-6)
    out = np.empty(arr_u16.shape, dtype=out_dtype)
    rows_per_chunk = max(1, 256_000_000 // max(arr_u16.shape[1], 1))
    for r in range(0, arr_u16.shape[0], rows_per_chunk):
        block = arr_u16[r:r+rows_per_chunk].astype(out_dtype)
        block -= q_lo
        block /= rng
        np.clip(block, 0.0, 1.0, out=block)
        out[r:r+rows_per_chunk] = block
    return out


def compute_or_load_percentiles(arr_u16, label):
    if WSI_PCT_JSON.exists():
        d = json.loads(WSI_PCT_JSON.read_text())
        if label in d:
            return float(d[label]["q_lo"]), float(d[label]["q_hi"])
    rng = np.random.default_rng(0)
    if arr_u16.size > 10_000_000:
        s = rng.integers(0, arr_u16.size, size=10_000_000)
        sample = arr_u16.ravel()[s]
    else:
        sample = arr_u16.ravel()
    q_lo, q_hi = np.percentile(sample, [1.0, 99.0])
    return float(q_lo), float(q_hi)


def load_wsi_anchors(H: int, W: int):
    """Iterate transcripts.zarr.zip grid tiles; return (py, px, li_arr) for qv>=20,
    valid==1, lineage-mapped anchors. li_arr is int8 over LINEAGES."""
    log("  loading transcripts.zarr.zip…")
    z = zarr.open(TX_ZARR, mode="r")
    gene_names = list(z.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
    # Build int8 lineage map across all genes (−1 = not in LINEAGES)
    lin_to_idx = {L: i for i, L in enumerate(LINEAGES)}
    gene_lin = np.full(len(gene_names), -1, dtype=np.int8)
    for gi, g in enumerate(gene_names):
        lab = gl.get(g)
        if lab in lin_to_idx:
            gene_lin[gi] = lin_to_idx[lab]
    log(f"  K={len(LINEAGES)} lineages; lineage-mapped genes: {(gene_lin >= 0).sum()}/{len(gene_names)}")

    g0 = z["grids"]["0"]
    keys = list(g0.keys())
    log(f"  transcript grid: {len(keys)} tiles")

    all_py, all_px, all_li = [], [], []
    t0 = time.time()
    n_total = 0; n_kept = 0
    for k_idx, key in enumerate(keys):
        tile = g0[key]
        loc = tile["location"][:]            # (n, 3) float32 x_um, y_um, z_um
        gid = tile["gene_identity"][:].squeeze(-1).astype(np.int64)  # (n,)
        qv  = tile["quality_score"][:].squeeze(-1)
        val = tile["valid"][:].squeeze(-1)
        n_total += loc.shape[0]

        # Filter qv, valid, lineage
        lin = gene_lin[gid]
        ok  = (qv >= QV_MIN) & (val == 1) & (lin >= 0)
        if not ok.any(): continue

        x_um = loc[ok, 0]; y_um = loc[ok, 1]
        px = np.rint(x_um / PIXEL_SIZE_UM).astype(np.int32)
        py = np.rint(y_um / PIXEL_SIZE_UM).astype(np.int32)
        keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not keep.any(): continue
        all_px.append(px[keep])
        all_py.append(py[keep])
        all_li.append(lin[ok][keep])
        n_kept += keep.sum()
        if (k_idx + 1) % 100 == 0:
            log(f"    tile {k_idx+1}/{len(keys)}  kept_so_far={n_kept:,}  ({time.time()-t0:.0f}s)")

    py = np.concatenate(all_py); px = np.concatenate(all_px); li = np.concatenate(all_li)
    log(f"  transcripts: scanned {n_total:,} → kept {n_kept:,} lineage-mapped anchors  ({time.time()-t0:.0f}s)")
    return py, px, li


def main():
    LOG_PATH.unlink(missing_ok=True)
    t_all = time.time()
    log(f"=== Method F-global WSI run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"OUT_DIR: {OUT_DIR}")
    log(f"params: ALPHA={ALPHA} N_MIN={N_MIN} K_SHARP={K_SHARP} σ_diff={SIGMA_DIFFUSE_PX:.2f}px "
        f"σ_cut={SIGMA_CUT_PX}px depth={DEPTH_MAX}")

    # ── 1) Load WSI DAPI + 18S ──────────────────────────────────────────────
    log("\n[1/9] Loading WSI morphology…")
    t = time.time()
    dapi_u16 = tifffile.imread(DAPI_TIF, key=0)
    s18_u16  = tifffile.imread(S18_TIF,  key=0)
    H, W = dapi_u16.shape
    log(f"  DAPI {dapi_u16.shape} {dapi_u16.dtype} | 18S {s18_u16.shape} {s18_u16.dtype}  ({time.time()-t:.0f}s)")

    # ── 2) Normalize for cell_evidence (use WSI percentile cache if present) ──
    log("\n[2/9] Normalising DAPI/18S for cell_evidence (using WSI percentile cache)…")
    dapi_q_lo, dapi_q_hi = compute_or_load_percentiles(dapi_u16, "DAPI")
    s18_q_lo,  s18_q_hi  = compute_or_load_percentiles(s18_u16,  "18S")
    log(f"  DAPI percentiles: [{dapi_q_lo:.0f}, {dapi_q_hi:.0f}]")
    log(f"  18S  percentiles: [{s18_q_lo:.0f}, {s18_q_hi:.0f}]")
    dapi_n = percentile_clip_to_uint8(dapi_u16, dapi_q_lo, dapi_q_hi)
    s18_n  = percentile_clip_to_uint8(s18_u16,  s18_q_lo,  s18_q_hi)
    cell_evidence = np.maximum(dapi_n, s18_n)
    del dapi_n  # not needed beyond cell_evidence
    log(f"  cell_evidence built ({cell_evidence.nbytes/1e9:.1f} GB)")

    # ── 3) Load anchors ─────────────────────────────────────────────────────
    log("\n[3/9] Loading transcript anchors…")
    py, px, li = load_wsi_anchors(H, W)
    K = len(LINEAGES)

    # ── 4) Build per-lineage N_eff and N_total ──────────────────────────────
    log(f"\n[4/9] Building per-lineage N_eff with σ_diffuse={SIGMA_DIFFUSE_PX:.2f}px…")
    N_total = np.zeros((H, W), dtype=np.float32)
    N_eff = []   # 7 arrays of float32 (H, W) ≈ 24.5 GB
    for k in range(K):
        t = time.time()
        rho_k = np.zeros((H, W), dtype=np.float32)
        mask_k = (li == k)
        if mask_k.any():
            np.add.at(rho_k, (py[mask_k], px[mask_k]), 1.0)
        n_k = int(mask_k.sum())
        rho_k = gaussian_filter(rho_k, sigma=SIGMA_DIFFUSE_PX, mode="reflect")
        rho_k *= 2.0 * np.pi * SIGMA_DIFFUSE_PX ** 2
        N_eff.append(rho_k)
        N_total += rho_k
        log(f"  k={k} ({LINEAGES[k]}): n_anchors={n_k:,}  ({time.time()-t:.0f}s)")
    del py, px, li
    confidence = N_total / (N_total + N_MIN + 1e-9)

    # ── 5) Compute pi_abst (in-place over N_eff) and track top1/top2 ────────
    log("\n[5/9] Computing pi_abst and tracking top1/top2…")
    t = time.time()
    pi_abst = []  # K float32 arrays
    inv_K = np.float32(1.0 / K)
    one_minus_conf = (1.0 - confidence).astype(np.float32)
    denom = (N_total + K * ALPHA + 1e-9).astype(np.float32)
    for k in range(K):
        pi_post_k = (N_eff[k] + ALPHA) / denom
        pi_abst_k = (confidence * pi_post_k + one_minus_conf * inv_K).astype(np.float32)
        pi_abst.append(pi_abst_k)
    del N_eff, denom  # free 24+ GB
    log(f"  pi_abst built ({time.time()-t:.0f}s)")

    t = time.time()
    top1_val = pi_abst[0].copy()
    top1_idx = np.zeros((H, W), dtype=np.int8)
    top2_val = np.full((H, W), -np.inf, dtype=np.float32)
    top2_idx = np.full((H, W), -1, dtype=np.int8)
    for k in range(1, K):
        v = pi_abst[k]
        is_new_top1 = v > top1_val
        # save old top1 into top2 where displaced; otherwise check v vs top2
        new_top2_val = np.where(is_new_top1, top1_val,
                                 np.where(v > top2_val, v, top2_val)).astype(np.float32)
        new_top2_idx = np.where(is_new_top1, top1_idx,
                                 np.where(v > top2_val, k, top2_idx)).astype(np.int8)
        new_top1_val = np.where(is_new_top1, v, top1_val).astype(np.float32)
        new_top1_idx = np.where(is_new_top1, k, top1_idx).astype(np.int8)
        top1_val, top1_idx = new_top1_val, new_top1_idx
        top2_val, top2_idx = new_top2_val, new_top2_idx
    log(f"  top1/top2 tracked ({time.time()-t:.0f}s)")

    # ── 6) Global edge field (one-vs-rest) ──────────────────────────────────
    log("\n[6/9] Computing global edge field (Σ_k |∇ tanh(K·s_k)|)…")
    t = time.time()
    edge_total = np.zeros((H, W), dtype=np.float32)
    for k in range(K):
        others_max = np.where(top1_idx == k, top2_val, top1_val).astype(np.float32)
        s_k = pi_abst[k] - others_max
        t_k = np.tanh(K_SHARP * s_k).astype(np.float32)
        gy = sobel(t_k, axis=0); gx = sobel(t_k, axis=1)
        edge_total += np.hypot(gx, gy).astype(np.float32)
        log(f"    k={k} done")
    del pi_abst, top1_val, top1_idx, top2_val, top2_idx, confidence
    edge_global = edge_total * (np.maximum(N_total / (N_total + N_MIN + 1e-9), 0) ** 2) * cell_evidence
    del N_total, edge_total
    log(f"  edge_global built ({time.time()-t:.0f}s)")

    # ── 7) Smooth + normalize + clip to depth ───────────────────────────────
    log(f"\n[7/9] Smoothing + normalizing (σ_cut={SIGMA_CUT_PX}px, depth={DEPTH_MAX})…")
    t = time.time()
    smoothed = gaussian_filter(edge_global, sigma=SIGMA_CUT_PX, mode="reflect")
    del edge_global
    s_max = float(smoothed.max())
    if s_max > 0:
        smoothed *= (DEPTH_MAX / s_max)
    cut_field = smoothed  # in [0, depth_max]
    log(f"  cut_field: max={cut_field.max():.4f}  mean={cut_field.mean():.4f}  ({time.time()-t:.0f}s)")

    log("  saving cut_field.tif (uint16 scaled)…")
    cf_u16 = np.rint(cut_field / max(cut_field.max(), 1e-6) * 65535).astype(np.uint16)
    tifffile.imwrite(OUT_DIR / "cut_field.tif", cf_u16, bigtiff=True)
    (OUT_DIR / "cut_field_scaling.json").write_text(json.dumps(
        {"max": float(cut_field.max()), "scale_to_unit": float(cut_field.max() / 65535.0)},
        indent=2))
    del cf_u16

    # ── 8) Apply cut to raw 18S; save s_cut.tif (uint16) ────────────────────
    log("\n[8/9] Building s_cut = 18S × (1 − cut_field); saving uint16…")
    t = time.time()
    s_cut_f = s18_u16.astype(np.float32) * (1.0 - cut_field)
    del cut_field
    s_cut_u16 = np.rint(np.clip(s_cut_f, 0, 65535)).astype(np.uint16)
    tifffile.imwrite(OUT_DIR / "s_cut.tif", s_cut_u16, bigtiff=True)
    log(f"  s_cut.tif saved  ({time.time()-t:.0f}s)")

    # ── 9) CP-SAM on (DAPI, s_cut) ──────────────────────────────────────────
    log("\n[9/9] Running CP-SAM on (DAPI, s_cut) — max-recall settings…")
    log("  (matches existing cpsam_whole_slide control: cprob=-5, flow=0, augment=True, niter=200)")
    # Normalize like the control: 1-99 percentile clip on each channel.
    log("  normalising DAPI and s_cut (1-99 percentile clip → [0, 1])…")
    t = time.time()
    dapi_q_lo_w, dapi_q_hi_w = compute_or_load_percentiles(dapi_u16, "DAPI")
    s_cut_q_lo,  s_cut_q_hi  = np.percentile(
        s_cut_u16.ravel()[np.random.default_rng(0).integers(0, s_cut_u16.size, size=10_000_000)],
        [1.0, 99.0])
    log(f"  DAPI  1-99 pct: [{dapi_q_lo_w:.0f}, {dapi_q_hi_w:.0f}]")
    log(f"  s_cut 1-99 pct: [{s_cut_q_lo:.0f}, {s_cut_q_hi:.0f}]")
    dapi_norm = percentile_clip_to_uint8(dapi_u16, dapi_q_lo_w, dapi_q_hi_w)
    del dapi_u16, s18_u16, s_cut_f
    s_cut_norm = percentile_clip_to_uint8(s_cut_u16, float(s_cut_q_lo), float(s_cut_q_hi))
    del s_cut_u16, cell_evidence  # free everything we don't need

    img = np.stack([dapi_norm, s_cut_norm], axis=-1)
    del dapi_norm, s_cut_norm
    log(f"  CP-SAM input stack: {img.shape} dtype={img.dtype} ({img.nbytes/1e9:.1f} GB)  ({time.time()-t:.0f}s)")

    from cellpose import models
    t = time.time()
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    log(f"  model init: {time.time()-t:.0f}s")
    t = time.time()
    masks, flows, _ = m_sam.eval(
        img, channel_axis=-1,
        diameter=None, niter=200,
        cellprob_threshold=-5.0, flow_threshold=0.0, augment=True,
    )
    eval_secs = time.time() - t
    n_cells = int(masks.max())
    log(f"  CP-SAM: {n_cells:,} cells in {eval_secs:.0f}s ({eval_secs/3600:.2f}h)")
    del img, flows

    log("  saving masks.tif (uint32)…")
    if n_cells >= 2**31:
        raise RuntimeError(f"n_cells={n_cells} exceeds int32; widen dtype")
    tifffile.imwrite(OUT_DIR / "masks.tif", masks.astype(np.uint32), bigtiff=True)

    (OUT_DIR / "scaling.json").write_text(json.dumps({
        "n_cells": n_cells,
        "shape": [H, W],
        "config": {
            "ALPHA": ALPHA, "N_MIN": N_MIN, "K_SHARP": K_SHARP,
            "SIGMA_DIFFUSE_PX": float(SIGMA_DIFFUSE_PX),
            "SIGMA_CUT_PX": SIGMA_CUT_PX, "DEPTH_MAX": DEPTH_MAX,
            "cellprob_threshold": -5.0, "flow_threshold": 0.0,
            "augment": True, "niter": 200, "normalization": "1-99 percentile clip"
        },
        "timing_secs": {"cpsam_eval": eval_secs, "total": time.time() - t_all},
    }, indent=2))

    log(f"\nAll done in {(time.time()-t_all)/3600:.2f}h. n_cells={n_cells:,}")
    log(f"Outputs in {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        log(f"  {f.name}  ({f.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
