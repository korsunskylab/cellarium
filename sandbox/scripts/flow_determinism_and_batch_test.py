"""Two controls for the unexpected 1024-vs-2048 flow difference at cell #6:

1. DETERMINISM: re-run 2048 with identical parameters. Compare flow at
   focal cell across two independent runs. If they agree, MPS isn't injecting
   the noise and the size-gap is real architecture-level behavior.

2. BATCHING: re-run 2048 with batch_size=1 (cellpose default 8). If flows
   then match 1024's flows, the across-batch state in the network is the
   mechanism. If they still match 2048's default-batch flows, batching is
   not the cause.

Re-uses flows_1024.npz and flows_2048.npz from figs/compare_flows/ (the
original compare_flows_at_focal.py run) as references.

Outputs:
  figs/compare_flows/flows_2048_rerun.npz       (determinism run)
  figs/compare_flows/flows_2048_bs1.npz         (batch=1 run)
  figs/compare_flows/comparison_table.csv
  figs/compare_flows/comparison_figure.png
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"

OUT_DIR = ROOT / "figs" / "compare_flows"

PIXEL_SIZE_UM = 0.2125
CX_PX, CY_PX = 41817, 6957
COMPARE_HALF = 50
CPSAM_KW_BASE = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                     flow_threshold=0.0, augment=False, normalize=False,
                     tile_overlap=0.05)


def crop_centered(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    a = arr.astype(np.float32)
    return ((a - q_lo) / max(q_hi - q_lo, 1e-6)).astype(np.float32)


def run_2048(m_sam, dapi_full, s18_full, wsi_pct, batch_size, label):
    print(f"\n=== 2048, {label} (batch_size={batch_size}) ===")
    dapi_c, _ = crop_centered(dapi_full, CX_PX, CY_PX, 2048)
    s18_c,  _ = crop_centered(s18_full,  CX_PX, CY_PX, 2048)
    dapi_n = norm_with(dapi_c, **wsi_pct["DAPI"])
    s18_n  = norm_with(s18_c,  **wsi_pct["18S"])
    img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
    kw = dict(CPSAM_KW_BASE); kw["batch_size"] = batch_size
    t0 = time.time()
    masks, flows, _ = m_sam.eval(img, channel_axis=-1, **kw)
    dt = time.time() - t0
    dP = flows[1].astype(np.float32)
    cellprob = flows[2].astype(np.float32)
    print(f"  n_total={int(masks.max())}, runtime={dt:.1f}s")
    print(f"  cellprob range: [{cellprob.min():.2f}, {cellprob.max():.2f}]")
    return dP, cellprob, int(masks.max())


def focal_diff_stats(a, b, half=COMPARE_HALF):
    """Per-pixel absolute & signed diff stats in the 100-px window around the
    crop center (where focal cell sits)."""
    H = a.shape[-1]
    c = H // 2
    af = a[..., c - half:c + half, c - half:c + half]
    bf = b[..., c - half:c + half, c - half:c + half]
    diff = af - bf
    return dict(max_abs=float(np.abs(diff).max()),
                mean_abs=float(np.abs(diff).mean()),
                max_signed=float(diff.max()), min_signed=float(diff.min()))


def main():
    t_total = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # Reference 2048 from compare_flows run
    ref_2048 = np.load(OUT_DIR / "flows_2048.npz")
    dP_2048_ref = ref_2048["dP"].astype(np.float32)
    cp_2048_ref = ref_2048["cellprob"].astype(np.float32)

    # Reference 1024 from compare_flows run
    ref_1024 = np.load(OUT_DIR / "flows_1024.npz")
    dP_1024_ref = ref_1024["dP"].astype(np.float32)
    cp_1024_ref = ref_1024["cellprob"].astype(np.float32)

    # 1. determinism re-run at 2048 with same params
    dP_2048_rerun, cp_2048_rerun, _ = run_2048(
        m_sam, dapi_full, s18_full, wsi_pct, batch_size=8, label="DEFAULT batch_size=8 (re-run)")
    np.savez_compressed(OUT_DIR / "flows_2048_rerun.npz",
                         dP=dP_2048_rerun, cellprob=cp_2048_rerun)

    # 2. batch=1 at 2048
    dP_2048_bs1, cp_2048_bs1, _ = run_2048(
        m_sam, dapi_full, s18_full, wsi_pct, batch_size=1, label="batch_size=1")
    np.savez_compressed(OUT_DIR / "flows_2048_bs1.npz",
                         dP=dP_2048_bs1, cellprob=cp_2048_bs1)

    # Comparisons
    import pandas as pd
    pairs = [
        ("2048 default vs 2048 rerun  (determinism)", dP_2048_ref, dP_2048_rerun, cp_2048_ref, cp_2048_rerun),
        ("2048 default vs 2048 bs=1   (batching)",     dP_2048_ref, dP_2048_bs1,   cp_2048_ref, cp_2048_bs1),
        ("1024 default vs 2048 bs=1   (size?)",        dP_1024_ref, dP_2048_bs1,   cp_1024_ref, cp_2048_bs1),
        ("1024 default vs 2048 default (original gap)",dP_1024_ref, dP_2048_ref,   cp_1024_ref, cp_2048_ref),
    ]
    rows = []
    print(f"\n=== focal-cell pixel-wise diff stats ({2*COMPARE_HALF}×{2*COMPARE_HALF}-px window) ===")
    for label, dPa, dPb, cpa, cpb in pairs:
        # Note: 1024 vs 2048 must be compared at the same physical window;
        # both crops are centered on the focal cell, so the center of each
        # array is the focal cell — focal_diff_stats does this.
        dY_stats = focal_diff_stats(dPa[0], dPb[0])
        dX_stats = focal_diff_stats(dPa[1], dPb[1])
        cp_stats = focal_diff_stats(cpa, cpb)
        rows.append(dict(
            comparison=label,
            dY_max_abs=dY_stats["max_abs"],   dY_mean_abs=dY_stats["mean_abs"],
            dX_max_abs=dX_stats["max_abs"],   dX_mean_abs=dX_stats["mean_abs"],
            cp_max_abs=cp_stats["max_abs"],   cp_mean_abs=cp_stats["mean_abs"],
        ))
        print(f"  {label}")
        print(f"    dY: max|Δ|={dY_stats['max_abs']:.4f}  mean|Δ|={dY_stats['mean_abs']:.5f}")
        print(f"    dX: max|Δ|={dX_stats['max_abs']:.4f}  mean|Δ|={dX_stats['mean_abs']:.5f}")
        print(f"    cp: max|Δ|={cp_stats['max_abs']:.4f}  mean|Δ|={cp_stats['mean_abs']:.5f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print(f"\n=== summary ===\n{df.to_string(index=False)}")
    print(f"\ntotal: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
