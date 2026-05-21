"""Frozen test harness for the onion-merge experiment on DB_top100_019.

Crops a 1000-px region around the doublet from the WSI, saves everything
needed by the 3-agent loop, and provides an `evaluate()` function that scores
any candidate mask post-processing relative to the baseline WSI mask.

NEVER re-runs cellpose. Operates purely on the existing WSI mask.

Usage:
    # Setup (run once)
    python scripts/onion_merge_harness.py setup

    # In any candidate algorithm script:
    from scripts.onion_merge_harness import load_region, evaluate, render_diff
    region = load_region()
    candidate_masks = my_algorithm(region['masks_wsi'], region)
    metrics = evaluate(region, candidate_masks)
    render_diff(region, candidate_masks, out_png='reports/onion_merge/iterN.png')
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tifffile
import zarr
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS= DATA / "cpsam_whole_slide" / "masks.tif"
BENCH    = DATA / "benchmark_doublets.parquet"
TX_ZARR  = DATA / "transcripts.zarr.zip"
GENE_LABELS = DATA / "gene_labels_lineage.parquet"

REGION_NPZ = ROOT / "figs" / "onion_merge" / "region.npz"
REGION_NPZ.parent.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
TARGET_BENCHMARK_ID = "DB_top100_019"
WINDOW_PX = 1000              # 212 µm on each side
DAPI_PEAK_MIN_DIST_PX = 12    # ~2.5 µm — typical nuclear spacing
DAPI_PEAK_SIGMA_PX = 3        # smooth before peak detection
DAPI_PEAK_THRESH_REL = 0.10   # relative threshold to global max
MIN_COVER_FRAC = 0.05         # ≥5% to count as "covering" the focal WSI cell


def setup(verbose: bool = True):
    """Crop the test region and save everything the harness will need."""
    if verbose: print(f"loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_masks = tifffile.imread(WSI_MASKS)

    bench = pd.read_parquet(BENCH)
    row = bench[bench["benchmark_id"] == TARGET_BENCHMARK_ID].iloc[0]
    cx_px = int(round(row["x_um"] / PIXEL_SIZE_UM))
    cy_px = int(round(row["y_um"] / PIXEL_SIZE_UM))
    cps_id = int(row["cps_id_at_bookmark"])
    if verbose: print(f"  focal: {TARGET_BENCHMARK_ID}, cps={cps_id}, WSI px=({cx_px},{cy_px})")

    half = WINDOW_PX // 2
    H, W = wsi_masks.shape
    y0 = max(cy_px - half, 0); y1 = min(y0 + WINDOW_PX, H); y0 = y1 - WINDOW_PX
    x0 = max(cx_px - half, 0); x1 = min(x0 + WINDOW_PX, W); x0 = x1 - WINDOW_PX
    bbox_wsi = (x0, y0, x1, y1)
    if verbose: print(f"  bbox in WSI px: {bbox_wsi}")

    dapi_c = dapi_full[y0:y1, x0:x1]
    s18_c  = s18_full[y0:y1, x0:x1]
    masks_wsi = wsi_masks[y0:y1, x0:x1].copy().astype(np.int32)
    focal_region = (masks_wsi == cps_id)
    focal_area = int(focal_region.sum())
    if verbose: print(f"  focal WSI cell area in crop: {focal_area} px")

    if verbose: print(f"computing DAPI peaks (local maxima)…")
    dapi_smooth = gaussian(dapi_c.astype(np.float32), sigma=DAPI_PEAK_SIGMA_PX,
                            preserve_range=True)
    peaks = peak_local_max(dapi_smooth, min_distance=DAPI_PEAK_MIN_DIST_PX,
                            threshold_rel=DAPI_PEAK_THRESH_REL)
    if verbose: print(f"  found {len(peaks)} DAPI peaks in the region")

    if verbose: print(f"loading anchored transcripts in region…")
    zroot = zarr.open(TX_ZARR, mode="r")
    gene_names = list(zroot.attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS).set_index("gene")["label"]
    gname_to_lin = gl.to_dict()
    x0_um = x0 * PIXEL_SIZE_UM; x1_um = x1 * PIXEL_SIZE_UM
    y0_um = y0 * PIXEL_SIZE_UM; y1_um = y1 * PIXEL_SIZE_UM
    TILE = 250.0
    i0 = int(x0_um // TILE); i1 = int(x1_um // TILE)
    j0 = int(y0_um // TILE); j1 = int(y1_um // TILE)
    grid0 = zroot["grids"]["0"]
    xs, ys, lins = [], [], []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            key = f"{i},{j}"
            if key not in grid0: continue
            tile = grid0[key]
            loc = tile["location"][:]; gid = tile["gene_identity"][:].squeeze(-1)
            qv  = tile["quality_score"][:].squeeze(-1); valid = tile["valid"][:].squeeze(-1)
            keep = (valid > 0) & (qv >= 20)
            if not keep.any(): continue
            xc = loc[keep, 0]; yc = loc[keep, 1]; gi = gid[keep]
            in_crop = (xc >= x0_um) & (xc < x1_um) & (yc >= y0_um) & (yc < y1_um)
            for xi, yi, g in zip(xc[in_crop], yc[in_crop], gi[in_crop]):
                L = gname_to_lin.get(gene_names[g])
                if L and L != "ambiguous":
                    xs.append(xi); ys.append(yi); lins.append(L)
    anch = pd.DataFrame({"x_um": xs, "y_um": ys, "lineage": lins})
    # Convert to local px
    if len(anch):
        anch["x_local_px"] = np.rint((anch["x_um"] - x0_um) / PIXEL_SIZE_UM).astype(int)
        anch["y_local_px"] = np.rint((anch["y_um"] - y0_um) / PIXEL_SIZE_UM).astype(int)
    if verbose: print(f"  {len(anch)} anchored transcripts in region")

    np.savez_compressed(
        REGION_NPZ,
        dapi=dapi_c, s18=s18_c, masks_wsi=masks_wsi,
        bbox_wsi=np.array(bbox_wsi),
        cps_focal=cps_id, focal_local_xy=np.array([cx_px - x0, cy_px - y0]),
        dapi_peaks=peaks,  # (N, 2) as (y, x) in local px
        anchor_x_local=np.array(anch["x_local_px"].values if len(anch) else []),
        anchor_y_local=np.array(anch["y_local_px"].values if len(anch) else []),
        anchor_lineage=np.array(anch["lineage"].values if len(anch) else [], dtype=object),
    )
    if verbose: print(f"  saved {REGION_NPZ}")
    return REGION_NPZ


def load_region() -> dict:
    if not REGION_NPZ.exists():
        setup(verbose=False)
    d = np.load(REGION_NPZ, allow_pickle=True)
    return dict(
        dapi=d["dapi"], s18=d["s18"], masks_wsi=d["masks_wsi"],
        bbox_wsi=tuple(d["bbox_wsi"].tolist()),
        cps_focal=int(d["cps_focal"]),
        focal_local_xy=tuple(d["focal_local_xy"].tolist()),
        dapi_peaks=d["dapi_peaks"],
        anchor_x_local=d["anchor_x_local"],
        anchor_y_local=d["anchor_y_local"],
        anchor_lineage=d["anchor_lineage"],
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _mask_features(masks: np.ndarray) -> pd.DataFrame:
    """Per-mask features: area, aspect ratio (major/minor), solidity, centroid."""
    rows = []
    for r in regionprops(masks):
        ar = r.major_axis_length / max(r.minor_axis_length, 1e-3)
        rows.append({
            "label": r.label, "area": r.area, "centroid_y": r.centroid[0],
            "centroid_x": r.centroid[1], "aspect_ratio": ar,
            "solidity": r.solidity, "eccentricity": r.eccentricity,
        })
    return pd.DataFrame(rows)


def _ring_score(masks: np.ndarray, area_max=200, ar_min=2.5, neighbor_radius=15) -> dict:
    """Count clusters of small+thin masks (likely ring slivers).
    Returns counts and the set of ring-suspect labels.
    """
    feat = _mask_features(masks)
    if len(feat) == 0:
        return dict(n_ring_suspects=0, n_ring_clusters=0, suspect_labels=[])
    susp = feat[(feat["area"] <= area_max) & (feat["aspect_ratio"] >= ar_min)].copy()
    if len(susp) == 0:
        return dict(n_ring_suspects=0, n_ring_clusters=0, suspect_labels=[])
    # Cluster by neighbor proximity: are nearby small+thin masks together?
    from scipy.spatial import cKDTree
    pts = susp[["centroid_y", "centroid_x"]].values
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=neighbor_radius)
    # Connected components
    parent = list(range(len(susp)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for a, b in pairs:
        union(a, b)
    roots = set(find(i) for i in range(len(susp)))
    # A "cluster" is one with ≥3 members (a single sliver isn't a ring)
    cluster_sizes = {}
    for i in range(len(susp)):
        r = find(i)
        cluster_sizes[r] = cluster_sizes.get(r, 0) + 1
    big_clusters = [r for r, sz in cluster_sizes.items() if sz >= 3]
    suspect_labels = sorted(susp["label"].astype(int).tolist())
    return dict(
        n_ring_suspects=len(susp),
        n_ring_clusters=len(big_clusters),
        suspect_labels=suspect_labels,
    )


def _focal_preservation(masks_baseline, masks_candidate, focal_region) -> dict:
    """Does the focal WSI cell still get covered by a single candidate mask
    that covers ≥80% of the original focal area?"""
    focal_area = int(focal_region.sum())
    if focal_area == 0:
        return dict(preserved=False, n_focal_masks=0, top_cover_frac=0.0)
    lbls = np.unique(masks_candidate[focal_region])
    lbls = lbls[lbls != 0]
    if len(lbls) == 0:
        return dict(preserved=False, n_focal_masks=0, top_cover_frac=0.0)
    covers = sorted(
        [(int(L), int(((masks_candidate == L) & focal_region).sum()) / focal_area)
         for L in lbls],
        key=lambda kv: -kv[1])
    counted = [c for _, c in covers if c >= MIN_COVER_FRAC]
    top = covers[0][1]
    return dict(
        preserved=(len(counted) == 1 and top >= 0.80),
        n_focal_masks=len(counted),
        top_cover_frac=top,
        covers=covers[:5],
    )


def _new_groups(masks_baseline, masks_candidate, min_overlap_frac=0.4):
    """For each candidate label, find baseline labels it 'absorbed' (≥ min_overlap_frac
    of the baseline mask falls inside the candidate). A candidate that absorbed
    >1 baseline label is a 'merged group'.
    Returns a list of dicts per merged group with: candidate_label, baseline_labels_absorbed,
    pixels."""
    out = []
    for cl in np.unique(masks_candidate):
        if cl == 0: continue
        cand_mask = (masks_candidate == cl)
        # Which baseline labels overlap this candidate?
        baseline_labels_in_cand = np.unique(masks_baseline[cand_mask])
        baseline_labels_in_cand = baseline_labels_in_cand[baseline_labels_in_cand != 0]
        absorbed = []
        for bl in baseline_labels_in_cand:
            bl_mask = (masks_baseline == bl)
            overlap = (bl_mask & cand_mask).sum() / max(bl_mask.sum(), 1)
            if overlap >= min_overlap_frac:
                absorbed.append(int(bl))
        if len(absorbed) > 1:
            out.append(dict(
                candidate_label=int(cl),
                baseline_labels_absorbed=absorbed,
                n_absorbed=len(absorbed),
                pixels=int(cand_mask.sum()),
            ))
    return out


def _peaks_in_mask(mask_bool, peaks_yx) -> int:
    if len(peaks_yx) == 0: return 0
    ys = peaks_yx[:, 0].astype(int); xs = peaks_yx[:, 1].astype(int)
    H, W = mask_bool.shape
    in_bounds = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys = ys[in_bounds]; xs = xs[in_bounds]
    return int(mask_bool[ys, xs].sum())


def _anchor_purity(mask_bool, region) -> dict:
    if len(region["anchor_x_local"]) == 0:
        return dict(purity=float("nan"), n_anchors=0, n_lineages=0)
    ys = region["anchor_y_local"].astype(int)
    xs = region["anchor_x_local"].astype(int)
    H, W = mask_bool.shape
    in_b = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys = ys[in_b]; xs = xs[in_b]
    inside = mask_bool[ys, xs]
    lins = np.array(region["anchor_lineage"])[in_b][inside]
    if len(lins) == 0:
        return dict(purity=float("nan"), n_anchors=0, n_lineages=0)
    counts = pd.Series(lins).value_counts()
    return dict(
        purity=float(counts.iloc[0] / counts.sum()),
        n_anchors=int(counts.sum()),
        n_lineages=int(len(counts)),
        dominant=str(counts.index[0]),
    )


def evaluate(region: dict, candidate_masks: np.ndarray) -> dict:
    """Compute all metrics for a candidate post-processing output."""
    baseline = region["masks_wsi"]
    focal_region = (baseline == region["cps_focal"])

    base_feat = _mask_features(baseline)
    cand_feat = _mask_features(candidate_masks)

    ring_base = _ring_score(baseline)
    ring_cand = _ring_score(candidate_masks)
    focal_eval = _focal_preservation(baseline, candidate_masks, focal_region)
    merges = _new_groups(baseline, candidate_masks)

    # For each merged group, evaluate "did we merge real cells together?"
    merge_evals = []
    for m in merges:
        cl = m["candidate_label"]
        cand_mask = (candidate_masks == cl)
        n_peaks = _peaks_in_mask(cand_mask, region["dapi_peaks"])
        pur = _anchor_purity(cand_mask, region)
        m_eval = dict(m)
        m_eval["dapi_peaks_inside"] = n_peaks
        m_eval["anchor_purity"] = pur["purity"]
        m_eval["n_anchors"] = pur["n_anchors"]
        m_eval["dominant_lineage"] = pur.get("dominant", "")
        m_eval["likely_real_merge"] = (n_peaks <= 1 and (pur["purity"] >= 0.6 or
                                                          np.isnan(pur["purity"])))
        merge_evals.append(m_eval)

    n_good_merges = sum(1 for m in merge_evals if m["likely_real_merge"])
    n_bad_merges  = sum(1 for m in merge_evals if not m["likely_real_merge"])

    return dict(
        baseline_n_cells=int(baseline.max()),
        candidate_n_cells=int(candidate_masks.max()),
        delta_n_cells=int(candidate_masks.max()) - int(baseline.max()),
        ring_baseline=ring_base,
        ring_candidate=ring_cand,
        ring_reduction=ring_base["n_ring_suspects"] - ring_cand["n_ring_suspects"],
        focal=focal_eval,
        n_merges=len(merge_evals),
        n_good_merges=n_good_merges,
        n_bad_merges=n_bad_merges,
        merges=merge_evals,
        # Optional: per-mask features for the agents
        n_baseline_features=len(base_feat),
        n_candidate_features=len(cand_feat),
    )


def _label_overlay(masks, alpha=0.55, seed=3):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(seed).permutation(np.arange(1, n + 1))])
    shuf = perm[masks]
    rgba = plt.get_cmap("tab20")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def _normalize(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def render_diff(region: dict, candidate_masks: np.ndarray, out_png: Path,
                 metrics: Optional[dict] = None, title: str = ""):
    """Render baseline vs candidate side-by-side with DAPI background +
    DAPI peaks + WSI doublet contour + mask overlay."""
    baseline = region["masks_wsi"]
    focal = (baseline == region["cps_focal"])
    dapi = _normalize(region["dapi"])
    s18  = _normalize(region["s18"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    # Top row: baseline
    axes[0,0].imshow(s18, cmap="gray")
    axes[0,0].imshow(_label_overlay(baseline)); axes[0,0].set_title("Baseline 18S + masks")
    axes[0,0].contour(focal.astype(int), colors="red", linewidths=1.5)
    axes[0,1].imshow(dapi, cmap="gray")
    axes[0,1].imshow(_label_overlay(baseline)); axes[0,1].set_title("Baseline DAPI + masks")
    axes[0,1].scatter(region["dapi_peaks"][:,1], region["dapi_peaks"][:,0],
                       s=8, c="yellow", marker="*", edgecolors="black", linewidth=0.3)
    axes[0,1].contour(focal.astype(int), colors="red", linewidths=1.5)
    # ring suspects highlight
    ring = _ring_score(baseline)
    suspect_mask = np.isin(baseline, ring["suspect_labels"])
    axes[0,2].imshow(dapi, cmap="gray")
    axes[0,2].imshow(_label_overlay(baseline), alpha=0.35)
    axes[0,2].contour(suspect_mask.astype(int), colors="orange", linewidths=0.8)
    axes[0,2].set_title(f"Baseline: ring suspects (orange)\n"
                         f"{ring['n_ring_suspects']} suspects in "
                         f"{ring['n_ring_clusters']} clusters")
    # Bottom row: candidate
    axes[1,0].imshow(s18, cmap="gray")
    axes[1,0].imshow(_label_overlay(candidate_masks))
    axes[1,0].set_title("Candidate 18S + masks")
    axes[1,0].contour(focal.astype(int), colors="red", linewidths=1.5)
    axes[1,1].imshow(dapi, cmap="gray")
    axes[1,1].imshow(_label_overlay(candidate_masks))
    axes[1,1].set_title("Candidate DAPI + masks")
    axes[1,1].scatter(region["dapi_peaks"][:,1], region["dapi_peaks"][:,0],
                       s=8, c="yellow", marker="*", edgecolors="black", linewidth=0.3)
    axes[1,1].contour(focal.astype(int), colors="red", linewidths=1.5)
    ring_c = _ring_score(candidate_masks)
    suspect_mask_c = np.isin(candidate_masks, ring_c["suspect_labels"])
    axes[1,2].imshow(dapi, cmap="gray")
    axes[1,2].imshow(_label_overlay(candidate_masks), alpha=0.35)
    axes[1,2].contour(suspect_mask_c.astype(int), colors="orange", linewidths=0.8)
    axes[1,2].set_title(f"Candidate: ring suspects (orange)\n"
                         f"{ring_c['n_ring_suspects']} suspects in "
                         f"{ring_c['n_ring_clusters']} clusters")
    subt = title if title else "onion-merge candidate"
    if metrics:
        subt += (f"\nbase n_cells={metrics['baseline_n_cells']} → "
                  f"cand n_cells={metrics['candidate_n_cells']}"
                  f"  |  ring suspects {metrics['ring_baseline']['n_ring_suspects']}→"
                  f"{metrics['ring_candidate']['n_ring_suspects']}"
                  f"  |  merges: {metrics['n_good_merges']} good / "
                  f"{metrics['n_bad_merges']} bad"
                  f"  |  focal preserved: {metrics['focal']['preserved']}"
                  f" (top cover {100*metrics['focal']['top_cover_frac']:.0f}%)")
    plt.suptitle(subt, fontsize=10, y=1.005)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup()
    else:
        region = load_region()
        print(f"loaded region: dapi shape {region['dapi'].shape}, "
              f"masks_wsi max {region['masks_wsi'].max()}, "
              f"{len(region['dapi_peaks'])} DAPI peaks, "
              f"{len(region['anchor_x_local'])} anchored transcripts")
        # Identity candidate (no change) → eval should show zero rings reduction
        metrics = evaluate(region, region["masks_wsi"])
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("merges",)}, indent=2, default=str))
        out = ROOT / "figs" / "onion_merge" / "baseline_render.png"
        render_diff(region, region["masks_wsi"], out, metrics=metrics,
                     title="Identity candidate (no change) = baseline")
        print(f"saved baseline render: {out}")
