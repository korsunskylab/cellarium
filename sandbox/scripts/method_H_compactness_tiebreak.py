"""Method H — compactness tie-breaker between alternative cut sides.

User insight on cell 098: when the two lineage territories form a clear
"island-in-host" geometry, there are TWO candidate cut locations:
  • cut on the host side (just outside the island) → preserves the island
  • cut on the island side (just inside the island) → fragments the island

We need a criterion to pick. Without DAPI logic (out of scope), use SHAPE
COMPACTNESS:
  • most compact (highest 4πA/P²) connected lineage-component in the focal
    is the "cell to preserve"
  • cut is placed just OUTSIDE its perimeter, in the less-compact lineage's
    body

This is the "red→white" side the user preferred.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, label as cc_label
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
sys.path.insert(0, str(ROOT / "scripts"))
from edges_three_methods import make_pi, SIGMA_PX as SIGMA_DIFFUSE_PX
from tint_cytoplasm_diffusion import LINEAGES, LINEAGE_COLOURS, norm01

DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
ROIS = DATA / "benchmark_doublet_rois"
TX_ZARR = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
BENCH = pd.read_parquet(DATA / "benchmark_doublets.parquet")
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"
OUT_DIR = ROOT / "figs" / "method_H_compactness_tiebreak"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_UM = 0.2125
ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
SIGMA_CUT_PX = 2.0
DEPTH_MAX = 0.9
DILATE_PX = 2          # how far outside the preserved component to place the cut
CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=True, normalize=False)


def posterior_with_alpha(anch, li_arr, sigma_diffuse_px, H, W, K, alpha, n_min):
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li_arr, anch["py"].to_numpy(), anch["px"].to_numpy()), 1.0)
    rho_smooth = np.stack([gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px) for k in range(K)], axis=0)
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    pi_post = (N_eff + alpha) / (N_total + K * alpha + 1e-9)[None]
    confidence = N_total / (N_total + n_min)
    pi_abst = confidence[None] * pi_post + (1.0 - confidence[None]) * (1.0 / K)
    return pi_abst, confidence


def compactness(mask_bool):
    """4πA / P² — 1.0 for perfect circle, lower for elongated/irregular shapes."""
    if mask_bool.sum() == 0: return 0.0
    rp = regionprops(mask_bool.astype(np.uint8))
    if not rp: return 0.0
    A = float(rp[0].area); P = float(rp[0].perimeter)
    if P <= 0: return 0.0
    return 4.0 * np.pi * A / (P * P)


def label_overlay(mask, alpha=0.45):
    u = np.unique(mask); u = u[u != 0]
    cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], alpha)
    return out


def main(BID="DB_top100_098"):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
    print(f"{BID}  ({lin_a} × {lin_b})")
    rdir = ROIS / BID
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
    d = make_pi(rdir, gene_names, gl)
    s18, dapi = d["s18"], d["dapi"]; s18_n, dapi_n = d["s18_n"], d["dapi_n"]
    anch = d["anch"]; li_arr = d["li_arr"]
    masks = tifffile.imread(rdir / "cells_cpsam_masks.tif").astype(np.int32)
    focal_region = (masks == cps_focal)
    H, W = s18.shape; K = d["pi"].shape[0]

    pi_abst, confidence = posterior_with_alpha(
        anch, li_arr, SIGMA_DIFFUSE_PX, H, W, K, alpha=ALPHA, n_min=N_MIN)
    pi_argmax = pi_abst.argmax(axis=0)

    # ----- Tie-break: pick the LEAST-compact (host) connected component within focal -----
    # The host wraps around the to-preserve cell. We cut on the host's STRONG side,
    # where π_host > threshold (the red→white boundary). This leaves the preserved
    # cell + its transition zone intact, and places the cut deep in the host's body.
    components = []
    for k, lineage in [(ka, lin_a), (kb, lin_b)]:
        terr = (pi_argmax == k) & focal_region
        lab, n = cc_label(terr)
        for i in range(1, n + 1):
            comp = (lab == i)
            if comp.sum() < 50: continue  # ignore tiny noise speckles
            comp_compact = compactness(comp)
            components.append(dict(lineage=lineage, k=k, area=int(comp.sum()),
                                    compactness=comp_compact, mask=comp))
    components.sort(key=lambda c: -c["compactness"])
    print(f"  components by compactness (highest = most circular):")
    for c in components:
        print(f"    {c['lineage']:>12s}  area={c['area']:>5d}  "
              f"compactness={c['compactness']:.3f}")
    # Among the pair (lin_a/lin_b), the MORE compact one is preserved, the LESS compact is the host.
    pair_only = [c for c in components if c["lineage"] in (lin_a, lin_b)]
    # Pick the LARGEST component of each lineage in the pair (ignore tiny noise)
    largest_a = max([c for c in pair_only if c["lineage"] == lin_a],
                     key=lambda c: c["area"], default=None)
    largest_b = max([c for c in pair_only if c["lineage"] == lin_b],
                     key=lambda c: c["area"], default=None)
    if largest_a is None or largest_b is None:
        print("  missing one of the pair components; abort"); return
    if largest_a["compactness"] > largest_b["compactness"]:
        preserve, host = largest_a, largest_b
    else:
        preserve, host = largest_b, largest_a
    print(f"  → PRESERVE: {preserve['lineage']} (compactness={preserve['compactness']:.3f})")
    print(f"  → HOST:     {host['lineage']} (compactness={host['compactness']:.3f})")

    # The cut is placed at the RED→WHITE boundary: the boundary of the region
    # where π_host > HOST_STRONG_THRESHOLD, restricted to the focal cell.
    HOST_STRONG_THRESHOLD = 0.30
    host_strong = (pi_abst[host["k"]] > HOST_STRONG_THRESHOLD) & focal_region
    # The OUTER edge of host_strong (the edge facing the preserve) is what we want.
    # Use the inner boundary of host_strong as the cut band.
    cut_band = find_boundaries(host_strong, mode="inner") & focal_region
    # Optionally dilate this band a tiny bit for a slightly thicker cut
    cut_band = binary_dilation(cut_band, iterations=DILATE_PX) & focal_region
    print(f"  HOST_STRONG_THRESHOLD = {HOST_STRONG_THRESHOLD}  "
          f"(cut at boundary of {{π_{host['lineage']} > {HOST_STRONG_THRESHOLD}}})")
    print(f"  cut band: {int(cut_band.sum())} pixels — red→white boundary")

    # Build a continuous edge field from the cut band, smoothed
    edge_H = cut_band.astype(np.float32)
    smoothed = gaussian_filter(edge_H, sigma=SIGMA_CUT_PX)
    if smoothed.max() > 0: cut_field = smoothed / smoothed.max() * DEPTH_MAX
    else: cut_field = smoothed
    s18_cut = s18.astype(np.float32) * (1.0 - cut_field)

    # Reference: π_diff sign change for visualization
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    sign_field = (pi_diff > 0).astype(np.uint8)
    sign_boundary_focal = find_boundaries(sign_field, mode="inner") & focal_region

    print("\nloading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_wsi(arr, key):
        return ((arr.astype(np.float32) - wsi_pct[key]["q_lo"]) /
                max(wsi_pct[key]["q_hi"] - wsi_pct[key]["q_lo"], 1e-6)).astype(np.float32)
    dapi_norm = norm_wsi(dapi, "DAPI")
    def run_cp(s_img):
        img = np.stack([dapi_norm, norm_wsi(s_img, "18S")], axis=-1).astype(np.float32)
        m, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m.astype(np.int32)
    def count_focal(m):
        a = focal_region.sum()
        if a == 0: return 0, []
        lbls = np.unique(m[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted([(int(L), int(((m==L)&focal_region).sum())/a) for L in lbls], key=lambda kv: -kv[1])
        return sum(1 for _, c in covers if c >= 0.05), covers[:5]
    m_ctrl = run_cp(s18); n_c, cov_c = count_focal(m_ctrl)
    m_H = run_cp(s18_cut); n_H, cov_H = count_focal(m_H)
    print(f"  control: n={n_c} {[f'{c*100:.0f}%' for _,c in cov_c]}")
    print(f"  method H: n={n_H} {[f'{c*100:.0f}%' for _,c in cov_H]}")

    # Render
    ys, xs = np.where(focal_region); pad = 30
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]
    preserve_bound_local = find_boundaries(preserve["mask"], mode="outer")[crop]
    cut_band_local = cut_band[crop]
    def overlays(ax):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.5)
        ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=1.5, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    sub = anch[(anch["px"]>=x0)&(anch["px"]<x1)&(anch["py"]>=y0)&(anch["py"]<y1)]
    def scatter_anchors(ax):
        for L in [lin_a, lin_b]:
            s = sub[sub["lineage"]==L]
            if len(s): ax.scatter(s["px"]-x0, s["py"]-y0, s=4, c=LINEAGE_COLOURS[L], alpha=0.9, edgecolors="none")

    # (0,0) 18S + anchors + preserve component
    axes[0,0].imshow(s18_n[crop], cmap="gray"); scatter_anchors(axes[0,0])
    axes[0,0].contour(preserve["mask"][crop].astype(int), levels=[0.5], colors="cyan", linewidths=1.8)
    axes[0,0].contour(cut_band_local.astype(int), levels=[0.5], colors="red", linewidths=1.2)
    overlays(axes[0,0])
    axes[0,0].set_title(f"REF: 18S+anchors\ncyan=preserve ({preserve['lineage']}, compact={preserve['compactness']:.2f})\nred=cut band (dilated by {DILATE_PX}px)")

    # (0,1) argmax-π
    cmap = plt.get_cmap("tab10")
    axes[0,1].imshow(pi_argmax[crop], cmap=cmap, vmin=0, vmax=len(LINEAGES)-1)
    axes[0,1].contour(preserve["mask"][crop].astype(int), levels=[0.5], colors="cyan", linewidths=1.8)
    overlays(axes[0,1])
    axes[0,1].set_title("argmax-π (per-pixel lineage)")

    # (0,2) cut field
    im = axes[0,2].imshow(cut_field[crop], cmap="hot", vmin=0, vmax=DEPTH_MAX)
    overlays(axes[0,2])
    axes[0,2].set_title(f"Method H cut field\n(only on the OUTSIDE of preserved component)")
    fig.colorbar(im, ax=axes[0,2], fraction=0.04)

    # (0,3) cut 18S
    axes[0,3].imshow(norm01(s18_cut[crop]), cmap="gray")
    overlays(axes[0,3])
    axes[0,3].set_title("18S after cut")

    # Row 1: CP-SAM
    axes[1,0].imshow(s18_n[crop], cmap="gray")
    axes[1,0].imshow(label_overlay(m_ctrl[crop]))
    scatter_anchors(axes[1,0])
    overlays(axes[1,0])
    cov_str_c = " ".join(f"{c*100:.0f}%" for _, c in cov_c[:3])
    axes[1,0].set_title(f"CONTROL  n={n_c}\n{cov_str_c}")

    axes[1,1].imshow(norm01(s18_cut[crop]), cmap="gray")
    axes[1,1].imshow(label_overlay(m_H[crop]))
    scatter_anchors(axes[1,1])
    axes[1,1].contour(preserve["mask"][crop].astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
    in_focal_bound = find_boundaries(m_H, mode="inner") & focal_region
    axes[1,1].contour(in_focal_bound[crop].astype(int), levels=[0.5], colors="white", linewidths=1.2)
    overlays(axes[1,1])
    cov_str_H = " ".join(f"{c*100:.0f}%" for _, c in cov_H[:3])
    axes[1,1].set_title(f"Method H  n={n_H}  (Δ={n_H-n_c})\n{cov_str_H}\nwhite=CP-SAM split, cyan=preserve")

    # (1,2) compactness table
    axes[1,2].axis("off")
    comp_txt = "Compactness ranking:\n"
    for c in components:
        marker = "★ PRESERVE" if c["lineage"] == preserve["lineage"] else "    "
        comp_txt += f"  {marker}  {c['lineage']:>12s}  area={c['area']:>5d}  compactness={c['compactness']:.3f}\n"
    comp_txt += f"\nDecision rule:\n  preserve = most-compact component\n  cut placed in band of {DILATE_PX}px outside it\n"
    axes[1,2].text(0.02, 0.95, comp_txt, transform=axes[1,2].transAxes,
                    fontsize=10, family="monospace", va="top")

    axes[1,3].axis("off")
    summary = (
        f"{BID}  ({lin_a} × {lin_b})\n\n"
        f"Method H — compactness-based tie-break\n\n"
        f"  preserve: {preserve['lineage']} (compactness={preserve['compactness']:.3f})\n"
        f"  cut band: {int(cut_band.sum())}px around preserved component\n"
        f"  σ_cut = {SIGMA_CUT_PX}px, depth_max = {DEPTH_MAX}\n\n"
        f"CP-SAM:\n"
        f"  Control: n_focal={n_c}\n"
        f"  Method H: n_focal={n_H} (Δ={n_H-n_c})\n"
    )
    axes[1,3].text(0.02, 0.98, summary, transform=axes[1,3].transAxes,
                    fontsize=10, family="monospace", va="top")

    plt.suptitle(f"Method H — compactness tie-break on {BID}", fontsize=12, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}_method_H.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("benchmark_id", nargs="?", default="DB_top100_098")
    a = p.parse_args()
    main(a.benchmark_id)
