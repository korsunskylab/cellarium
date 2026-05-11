"""Run cpsam (max-recall) on DAPI + 18S TIFFs in a directory, write outputs alongside.

Designed to be called from the R notebook via system2() so the rapid-iteration
loop is R-side: build modified 18S in R → write TIFFs → invoke this script →
read TIFFs back into R → visualize.

Usage:
    python run_cpsam_on_dir.py <in_dir> <out_dir>

Reads:  <in_dir>/dapi.tif        uint16  [0, 65535]  (normalized DAPI)
        <in_dir>/s18.tif         uint16  [0, 65535]  (normalized + modified 18S)

Writes: <out_dir>/masks.tif      uint16  label image (0 = background, 1..N = cells)
        <out_dir>/cellprob.tif   uint16  scaled (see scaling.json)
        <out_dir>/flow_mag.tif   uint16  scaled (see scaling.json)
        <out_dir>/scaling.json   per-array min/max/scale/offset for cellprob & flow_mag

Why uint16 + sidecar JSON: R's `tiff` package can only read integer TIFFs reliably
in this conda env (no float32 support), so float arrays (cellprob = log-odds with
negative values; flow_mag = positive floats) are stored as uint16 with explicit
linear scaling. R recovers via:  array_u16 * scale + offset.

cpsam config = the validated max-recall settings:
    cellprob_threshold=-5.0  flow_threshold=0.0  augment=True  niter=200
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from cellpose import models


def save_scaled_uint16(arr, path):
    """Linear-scale a float array to uint16 [0, 65535]; return scaling params."""
    a_min = float(arr.min())
    a_max = float(arr.max())
    rng = max(a_max - a_min, 1e-6)
    scaled = ((arr - a_min) / rng * 65535).astype(np.uint16)
    tifffile.imwrite(path, scaled)
    return {
        "min":    a_min,
        "max":    a_max,
        "scale":  rng / 65535.0,   # in R: array_u16 * scale + offset
        "offset": a_min,
    }


def main(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load — R wrote uint16 in [0, 65535]; cellpose wants float in [0, 1]
    dapi = tifffile.imread(in_dir / "dapi.tif").astype(np.float32) / 65535.0
    s18  = tifffile.imread(in_dir / "s18.tif" ).astype(np.float32) / 65535.0
    print(f"DAPI: {dapi.shape}  range=[{dapi.min():.3f}, {dapi.max():.3f}]")
    print(f"18S : {s18.shape}   range=[{s18.min():.3f}, {s18.max():.3f}]")
    assert dapi.shape == s18.shape, "DAPI and 18S must have the same shape"

    # cpsam expects (H, W, C) with channel_axis=-1
    img = np.stack([dapi, s18], axis=-1)

    t0 = time.time()
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    masks, flows, _ = m_sam.eval(
        img.astype(np.float32),
        channel_axis=-1,
        diameter=None,
        niter=200,
        cellprob_threshold=-5.0,
        flow_threshold=0.0,
        augment=True,
    )
    n_cells = int(masks.max())
    print(f"cpsam: {n_cells} cells in {time.time() - t0:.1f}s")

    cellprob = flows[2].astype(np.float32)
    flow_yx  = flows[1]
    flow_mag = np.linalg.norm(flow_yx, axis=0).astype(np.float32)

    if n_cells >= 65535:
        # Extremely unlikely on an ROI, but guard against uint16 overflow on the label image
        raise RuntimeError(f"n_cells={n_cells} exceeds uint16; widen mask dtype.")
    tifffile.imwrite(out_dir / "masks.tif", masks.astype(np.uint16))

    scaling = {
        "n_cells":  n_cells,
        "cellprob": save_scaled_uint16(cellprob, out_dir / "cellprob.tif"),
        "flow_mag": save_scaled_uint16(flow_mag, out_dir / "flow_mag.tif"),
        "config": {
            "cellprob_threshold": -5.0,
            "flow_threshold":      0.0,
            "augment":             True,
            "niter":               200,
        },
    }
    with open(out_dir / "scaling.json", "w") as f:
        json.dump(scaling, f, indent=2)

    print(f"wrote: masks.tif (n={n_cells}), cellprob.tif, flow_mag.tif, scaling.json")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
