"""Batch the diffusion-vs-KDE comparison across all approved benchmark doublets.

For each of the 68 approved doublets, renders a 3×3 figure with:
  Row 0:  reference (18S+anchors) | Gaussian σ=1 | Gaussian σ=2
  Row 1:  c_linear conductance     | diff·c_linear σ=1 | diff·c_linear σ=2
  Row 2:  c_log conductance        | diff·c_log σ=1    | diff·c_log σ=2

Output: figs/tinted_cytoplasm_diffusion/<benchmark_id>_diffusion_vs_kde.png
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# Reuse the per-cell renderer
from tint_cytoplasm_diffusion import main as render_one

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
BENCHMARK_PARQUET = DATA / "benchmark_doublets.parquet"


def main():
    bm = pd.read_parquet(BENCHMARK_PARQUET)
    approved = bm[bm["approved"]].sort_values("rank").reset_index(drop=True)
    print(f"rendering diffusion-vs-KDE comparison for {len(approved)} approved doublets...\n")

    t0 = time.time()
    failures = []
    # σ sweep: under-, sweet-, borderline-, over-smoothed for each cell
    SIGMA_LIST = [0.5, 1.0, 2.0, 4.0]
    for i, bid in enumerate(approved["benchmark_id"], 1):
        t_cell = time.time()
        try:
            render_one(bid, sigma_um_list=SIGMA_LIST)
            print(f"  [{i:>3d}/{len(approved)}]  {bid}  ({time.time()-t_cell:.1f}s)")
        except Exception as e:
            print(f"  [{i:>3d}/{len(approved)}]  {bid}  FAILED: {e}")
            failures.append((bid, str(e)))

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. {len(failures)} failures.")
    if failures:
        for bid, err in failures:
            print(f"  {bid}: {err}")


if __name__ == "__main__":
    main()
