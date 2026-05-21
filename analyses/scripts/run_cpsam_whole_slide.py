"""Run cpsam (max-recall) on whole-slide DAPI + 18S from morphology_focus/.

One-shot version of `run_cpsam_on_dir.py` adapted for whole-slide input:
reads channels straight from the OME-TIFFs, 1st-99th-percentile normalizes
each, then runs cpsam with the validated max-recall settings.

Channel mapping (from OME XML in morphology_focus/*.ome.tif):
    morphology_focus_0000.ome.tif  →  DAPI
    morphology_focus_0001.ome.tif  →  ATP1A1/CD45/E-Cadherin
    morphology_focus_0002.ome.tif  →  18S
    morphology_focus_0003.ome.tif  →  alphaSMA/Vimentin

Outputs (in out_dir):
    masks.tif      uint32  label image (0 = background, 1..N = cells)  — uint32 because
                           whole-slide cell counts exceed uint16 (~100-150k cells)
    cellprob.tif   uint16  scaled (see scaling.json)
    flow_mag.tif   uint16  scaled
    flow_dy.tif    uint16  scaled
    flow_dx.tif    uint16  scaled
    scaling.json   per-array min/max/scale/offset, + timing, + config

cpsam config (matches per-ROI):
    cellprob_threshold = -5.0   flow_threshold = 0.0   augment = True   niter = 200
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from cellpose import models

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
DAPI_TIF  = DATA_ROOT / "morphology_focus" / "morphology_focus_0000.ome.tif"
S18_TIF   = DATA_ROOT / "morphology_focus" / "morphology_focus_0002.ome.tif"
DEFAULT_OUT = DATA_ROOT / "cpsam_whole_slide"


def norm01_uint16(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile-clip + scale to float32 [0, 1]. Done in chunks to bound peak memory."""
    # np.percentile on 868M uint16 is fine (~15s), but it copies internally.
    # Sample to ~10M pixels to cap percentile-compute memory.
    rng = np.random.default_rng(0)
    if arr.size > 10_000_000:
        idx = rng.integers(0, arr.size, size=10_000_000)
        sample = arr.ravel()[idx]
    else:
        sample = arr.ravel()
    q_lo, q_hi = np.percentile(sample, [lo, hi]).astype(np.float32)
    scale = max(float(q_hi - q_lo), 1e-6)
    print(f"  percentiles {lo}-{hi}: [{q_lo:.0f}, {q_hi:.0f}]  scale={scale:.1f}")
    out = np.empty(arr.shape, dtype=np.float32)
    # Chunk along rows to avoid allocating a second full-size float32 buffer at once.
    rows_per_chunk = max(1, 256_000_000 // max(arr.shape[1], 1))   # ~256 MB float32 per chunk
    for r in range(0, arr.shape[0], rows_per_chunk):
        block = arr[r:r + rows_per_chunk].astype(np.float32)
        block -= q_lo
        block /= scale
        np.clip(block, 0.0, 1.0, out=block)
        out[r:r + rows_per_chunk] = block
    return out


def save_scaled_uint16(arr: np.ndarray, path: Path) -> dict:
    """Linear-scale a float array to uint16 [0, 65535]; return scaling params."""
    a_min = float(arr.min())
    a_max = float(arr.max())
    rng = max(a_max - a_min, 1e-6)
    # Avoid materialising another full-precision copy
    scaled = ((arr - a_min) * (65535.0 / rng)).astype(np.uint16)
    tifffile.imwrite(path, scaled, bigtiff=True)
    return {
        "min":    a_min,
        "max":    a_max,
        "scale":  rng / 65535.0,
        "offset": a_min,
    }


def main(out_dir: Path = DEFAULT_OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    t_start = time.time()

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"=== cpsam whole-slide run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"DAPI : {DAPI_TIF}")
    log(f"18S  : {S18_TIF}")
    log(f"OUT  : {out_dir}")

    # ── Load channels ─────────────────────────────────────────────────────
    log("\n[1/4] Loading channels...")
    t = time.time()
    dapi_u16 = tifffile.imread(DAPI_TIF, key=0)   # (H, W) uint16
    log(f"  DAPI loaded: {dapi_u16.shape} dtype={dapi_u16.dtype}  {time.time()-t:.1f}s")
    t = time.time()
    s18_u16  = tifffile.imread(S18_TIF, key=0)
    log(f"  18S  loaded: {s18_u16.shape}  dtype={s18_u16.dtype}  {time.time()-t:.1f}s")
    assert dapi_u16.shape == s18_u16.shape, "DAPI and 18S must have the same shape"
    H, W = dapi_u16.shape
    log(f"  whole-slide pixels: {H*W:,}")

    # ── Normalise ─────────────────────────────────────────────────────────
    log("\n[2/4] Normalising to float32 [0, 1] via 1-99 percentile clip...")
    t = time.time()
    dapi = norm01_uint16(dapi_u16)
    del dapi_u16
    log(f"  DAPI normalised  {time.time()-t:.1f}s")
    t = time.time()
    s18 = norm01_uint16(s18_u16)
    del s18_u16
    log(f"  18S  normalised  {time.time()-t:.1f}s")

    img = np.stack([dapi, s18], axis=-1)   # (H, W, 2) float32
    del dapi, s18
    log(f"  stacked image: {img.shape} dtype={img.dtype}  ({img.nbytes/1e9:.1f} GB)")

    # ── Run cpsam ─────────────────────────────────────────────────────────
    log("\n[3/4] Running cpsam (max-recall settings)...")
    t = time.time()
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    log(f"  model init: {time.time()-t:.1f}s")
    t = time.time()
    masks, flows, _ = m_sam.eval(
        img,
        channel_axis=-1,
        diameter=None,
        niter=200,
        cellprob_threshold=-5.0,
        flow_threshold=0.0,
        augment=True,
    )
    eval_secs = time.time() - t
    n_cells = int(masks.max())
    log(f"  cpsam: {n_cells} cells in {eval_secs:.1f}s ({eval_secs/3600:.2f}h)")
    del img

    # ── Save outputs ──────────────────────────────────────────────────────
    log("\n[4/4] Saving outputs...")
    cellprob = flows[2].astype(np.float32)
    flow_yx  = flows[1].astype(np.float32)
    flow_dy  = flow_yx[0]
    flow_dx  = flow_yx[1]
    flow_mag = np.linalg.norm(flow_yx, axis=0).astype(np.float32)
    del flows, flow_yx

    # masks: uint32 to handle >65535 cells
    if n_cells >= 2**31:
        raise RuntimeError(f"n_cells={n_cells} exceeds int32; widen dtype")
    t = time.time()
    tifffile.imwrite(out_dir / "masks.tif", masks.astype(np.uint32), bigtiff=True)
    log(f"  masks.tif written  {time.time()-t:.1f}s  (uint32 to fit {n_cells} cells)")
    del masks

    scaling = {
        "n_cells": n_cells,
        "shape":   [H, W],
        "cellprob": save_scaled_uint16(cellprob, out_dir / "cellprob.tif"),
        "flow_mag": save_scaled_uint16(flow_mag, out_dir / "flow_mag.tif"),
        "flow_dy":  save_scaled_uint16(flow_dy,  out_dir / "flow_dy.tif"),
        "flow_dx":  save_scaled_uint16(flow_dx,  out_dir / "flow_dx.tif"),
        "config": {
            "cellprob_threshold": -5.0,
            "flow_threshold":      0.0,
            "augment":             True,
            "niter":               200,
            "normalization":       "1-99 percentile clip → [0, 1]",
        },
        "timing": {
            "eval_secs":  eval_secs,
            "total_secs": time.time() - t_start,
        },
    }
    with (out_dir / "scaling.json").open("w") as f:
        json.dump(scaling, f, indent=2)
    log(f"\nDone in {(time.time()-t_start)/3600:.2f}h. n_cells={n_cells}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    main(out)
