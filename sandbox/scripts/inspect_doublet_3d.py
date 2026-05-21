"""Interactive 3D scatter of anchored transcripts inside (and around) one
benchmark mini-ROI. Output is a standalone HTML file you can open in any
browser to rotate, zoom, hover-inspect, and toggle individual lineages.

Output: figs/zstack_inspection/<benchmark_id>_3d.html

Usage:
    python inspect_doublet_3d.py [benchmark_id]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tifffile
import zarr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
TX_ZARR     = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"
ROIS_DIR    = DATA / "benchmark_doublet_rois"
FIGS_DIR    = ROOT / "figs" / "zstack_inspection"

PIXEL_SIZE_UM = 0.2125
LINEAGES = ["Melanoma", "Myeloid", "Tcell", "Plasma", "Fibroblast", "Endothelial", "Keratinocyte"]
LINEAGE_COLOURS = {
    "Melanoma":     "#E41A1C",
    "Myeloid":      "#FF7F00",
    "Tcell":        "#377EB8",
    "Plasma":       "#984EA3",
    "Fibroblast":   "#4DAF4A",
    "Endothelial":  "#00CED1",
    "Keratinocyte": "#F781BF",
}


def main(benchmark_id):
    roi = ROIS_DIR / benchmark_id
    assert roi.exists(), f"missing ROI: {roi}"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    s18  = tifffile.imread(roi / "morphology_18S.tif")
    cpsam = tifffile.imread(roi / "cells_cpsam_masks.tif")
    tx   = pd.read_parquet(roi / "transcripts.parquet")
    meta = json.loads((roi / "metadata.json").read_text())
    H, W = s18.shape
    cx0 = meta["bbox_crop_um"]["x0"]; cy0 = meta["bbox_crop_um"]["y0"]
    cps_focal = meta["cps_id_at_bookmark"]

    # Map gene_id → lineage
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    tx["gene"] = [gene_names[g] for g in tx["gene_id"].values]
    tx["lineage"] = tx["gene"].map(gl)
    anch = tx[tx["lineage"].isin(LINEAGES)].copy()
    anch["px"] = np.rint((anch["x_um"] - cx0) / PIXEL_SIZE_UM).astype(int)
    anch["py"] = np.rint((anch["y_um"] - cy0) / PIXEL_SIZE_UM).astype(int)
    in_crop = ((anch["px"] >= 0) & (anch["px"] < W) &
               (anch["py"] >= 0) & (anch["py"] < H))
    anch = anch[in_crop].copy()
    anch["in_focal"] = cpsam[anch["py"].to_numpy(), anch["px"].to_numpy()] == cps_focal
    print(f"[{benchmark_id}] anchored tx in crop: {len(anch)}  "
          f"in focal: {anch['in_focal'].sum()}  "
          f"z range: [{anch['z_um'].min():.2f}, {anch['z_um'].max():.2f}] µm")

    # Build per-lineage traces. Each lineage gets two scatter traces:
    #   "<L> (focal)"  — bigger, outlined, only inside the focal cell
    #   "<L> (context)" — smaller, semi-transparent, outside the focal cell
    # Browser legend lets you click traces to toggle visibility.
    fig = go.Figure()
    for L in LINEAGES:
        col = LINEAGE_COLOURS[L]
        for is_focal, name_suffix, size, opacity, outline_w in (
            (True,  "(focal)",   5.5, 0.95, 1),
            (False, "(context)", 2.6, 0.40, 0),
        ):
            sub = anch[(anch["lineage"] == L) & (anch["in_focal"] == is_focal)]
            if len(sub) == 0:
                continue
            hover = [
                f"{r.gene}<br>lineage: {L}<br>"
                f"x={r.x_um:.2f}  y={r.y_um:.2f}  z={r.z_um:.2f} µm<br>"
                f"qv={r.qv:.1f}<br>"
                f"{'in focal cell' if is_focal else 'context'}"
                for r in sub.itertuples()
            ]
            fig.add_trace(go.Scatter3d(
                x=sub["x_um"], y=sub["y_um"], z=sub["z_um"],
                mode="markers",
                marker=dict(size=size, color=col,
                            opacity=opacity,
                            line=dict(width=outline_w, color="black")),
                name=f"{L} {name_suffix} (n={len(sub)})",
                text=hover, hoverinfo="text",
                legendgroup=L,
                showlegend=True,
            ))

    # Optional: outline the focal cell's bbox as a wireframe box for spatial anchor.
    bb = meta["bbox_cell_um"]
    z_min = float(anch["z_um"].min()); z_max = float(anch["z_um"].max())
    edges = [
        ((bb["x0"], bb["x0"]), (bb["y0"], bb["y1"]), (z_min, z_min)),
        ((bb["x0"], bb["x1"]), (bb["y0"], bb["y0"]), (z_min, z_min)),
        ((bb["x0"], bb["x1"]), (bb["y1"], bb["y1"]), (z_min, z_min)),
        ((bb["x1"], bb["x1"]), (bb["y0"], bb["y1"]), (z_min, z_min)),
        ((bb["x0"], bb["x0"]), (bb["y0"], bb["y1"]), (z_max, z_max)),
        ((bb["x0"], bb["x1"]), (bb["y0"], bb["y0"]), (z_max, z_max)),
        ((bb["x0"], bb["x1"]), (bb["y1"], bb["y1"]), (z_max, z_max)),
        ((bb["x1"], bb["x1"]), (bb["y0"], bb["y1"]), (z_max, z_max)),
        ((bb["x0"], bb["x0"]), (bb["y0"], bb["y0"]), (z_min, z_max)),
        ((bb["x0"], bb["x0"]), (bb["y1"], bb["y1"]), (z_min, z_max)),
        ((bb["x1"], bb["x1"]), (bb["y0"], bb["y0"]), (z_min, z_max)),
        ((bb["x1"], bb["x1"]), (bb["y1"], bb["y1"]), (z_min, z_max)),
    ]
    box_x, box_y, box_z = [], [], []
    for ex, ey, ez in edges:
        box_x += [ex[0], ex[1], None]
        box_y += [ey[0], ey[1], None]
        box_z += [ez[0], ez[1], None]
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z, mode="lines",
        line=dict(color="cyan", width=3), name="focal cell bbox",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=(f"{benchmark_id} — interactive 3D anchored transcripts<br>"
               f"<sup>rank {meta['rank']}, {meta['lineage_pair']}, "
               f"cps_id={cps_focal}, "
               f"top1={meta['top1_n']} top2={meta['top2_n']}</sup>"),
        scene=dict(
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis_title="z (µm)",
            aspectmode="data",   # preserve real µm aspect across axes
        ),
        legend=dict(itemsizing="constant", font=dict(size=10)),
        margin=dict(l=0, r=0, t=60, b=0),
        height=800,
    )
    out_html = FIGS_DIR / f"{benchmark_id}_3d.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"wrote {out_html}  ({out_html.stat().st_size/1024:.0f} KB)")
    print("open it in any browser — click lineage names in the legend to toggle.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_id", nargs="?", default="DB_top100_009")
    args = ap.parse_args()
    main(args.benchmark_id)
