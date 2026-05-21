"""Per-cluster debug visualization for iter 9 (SDT extrusion + perimeter uniformity
+ centroid-inside + transitive merge).

Re-runs iter 9's algorithm against the baseline region and emits a multi-page PDF
where each (parent, ring-cluster) candidate gets a dedicated page showing
the intermediate evidence and the accept/reject decision.

Page layout (2x3 per cluster):
  (0,0) 18S + cells: parent green, ring candidates red, others gray
  (0,1) DAPI + peaks: yellow stars at DAPI local maxima; ⨯ at parent centroid
  (0,2) Anchors: transcripts in the cluster colored by lineage
  (1,0) Parent SDT field (color-coded by signed distance), ring outlines in red
  (1,1) Histogram of SDT samples inside each ring (Path 1 evidence)
        + boundary-CV bar chart for each ring (Criterion A evidence)
  (1,2) Decision text + validator scores (DAPI peaks, purity, area, etc.)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.measure import regionprops, find_contours
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import matplotlib.cm as cm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onion_merge_harness import load_region

OUT_PDF = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/reports/onion_merge_log/iter09_per_cluster_debug.pdf")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# Iter-9 parameters
MIN_PARENT_AREA   = 200
MIN_AREA_RATIO    = 0.7
SDT_BOUNDARY_FRAC = 0.97
SDT_OFFSET_MIN    = 1.0
SDT_OFFSET_MAX    = 25.0
SDT_CV_MAX        = 0.35      # interior pixel CV
CRITA_BOUNDARY_CV = 0.20      # perimeter CV
CENTROID_PARENT_AREA_MIN = 300
CENTROID_AREA_RATIO_MAX  = 0.5
CENTROID_DILATE          = 5
ENV_DILATE       = 30
MAX_PEAKS        = 2
MIN_PURITY       = 0.6

LINEAGE_COLORS = {
    "Melanoma": "#d62728", "Fibroblast": "#2ca02c", "Tcell": "#1f77b4",
    "Myeloid": "#ff7f0e", "Plasma": "#9467bd", "Endothelial": "#e377c2",
    "Keratinocyte": "#bcbd22",
}


# ----------------------------------------------------------------------
# Algorithm (Path 1 + Path 2) — instrumented to capture intermediate evidence
# ----------------------------------------------------------------------
def parent_sdt(mask_bool):
    """SDT positive outside the mask, negative inside."""
    return distance_transform_edt(~mask_bool) - distance_transform_edt(mask_bool)


def candidate_pairs(masks, focal_label, dapi_peaks):
    """Find candidate (parent, ring) pairs via Path 1 (SDT) + Path 2 (centroid inside).
    Returns list of dicts with all intermediate scores so we can visualize each."""
    props = {r.label: r for r in regionprops(masks)}
    labels = [L for L in props if L != focal_label and L != 0]
    label_to_idx = {L: i for i, L in enumerate(labels)}

    # Build parent candidates (large enough)
    parents = [L for L in labels if props[L].area >= MIN_PARENT_AREA]

    pairs = []
    for p_label in parents:
        P_mask = (masks == p_label)
        # SDT_P in a local envelope around the parent
        bbox = props[p_label].bbox  # (min_row, min_col, max_row, max_col)
        y0 = max(bbox[0] - ENV_DILATE, 0); y1 = min(bbox[2] + ENV_DILATE, masks.shape[0])
        x0 = max(bbox[1] - ENV_DILATE, 0); x1 = min(bbox[3] + ENV_DILATE, masks.shape[1])
        P_local = P_mask[y0:y1, x0:x1]
        sdt_local = parent_sdt(P_local)
        # Candidate rings: smaller labels inside envelope
        cand_in_env = np.unique(masks[y0:y1, x0:x1])
        cand_in_env = [L for L in cand_in_env if L != 0 and L != p_label and L != focal_label]
        for r_label in cand_in_env:
            r_area = props[r_label].area
            p_area = props[p_label].area
            if r_area >= p_area: continue
            R_mask = (masks == r_label)
            # restrict to envelope
            r_local = R_mask[y0:y1, x0:x1]
            if not r_local.any(): continue
            # ---- Path 1: SDT extrusion ----
            sdt_in_ring = sdt_local[r_local]
            n = sdt_in_ring.size
            if n == 0: continue
            n_pos = int((sdt_in_ring > 0).sum())
            frac_pos = n_pos / n
            mean_off = float(sdt_in_ring.mean())
            std_off = float(sdt_in_ring.std())
            cv_int = std_off / max(abs(mean_off), 1e-6)
            sdt_pass = (
                frac_pos >= SDT_BOUNDARY_FRAC and
                SDT_OFFSET_MIN <= mean_off <= SDT_OFFSET_MAX and
                cv_int <= SDT_CV_MAX and
                (r_area / p_area) <= MIN_AREA_RATIO
            )
            # ---- Criterion A: perimeter uniformity (only meaningful if SDT passed) ----
            crita_cv = float("nan")
            crita_pass = False
            if sdt_pass:
                # boundary samples of R: use find_contours on R_local
                try:
                    conts = find_contours(r_local.astype(np.float32), 0.5)
                except Exception:
                    conts = []
                if conts:
                    # all contour points
                    pts = np.concatenate(conts, axis=0)
                    yy = np.clip(pts[:, 0].astype(int), 0, sdt_local.shape[0]-1)
                    xx = np.clip(pts[:, 1].astype(int), 0, sdt_local.shape[1]-1)
                    samp = sdt_local[yy, xx]
                    if samp.size > 0 and samp.mean() > 0:
                        crita_cv = float(samp.std() / max(abs(samp.mean()), 1e-6))
                        crita_pass = crita_cv <= CRITA_BOUNDARY_CV
            # ---- Path 2: centroid-inside ----
            r_y, r_x = props[r_label].centroid
            r_y_loc, r_x_loc = int(r_y) - y0, int(r_x) - x0
            inside_dilated = False
            if (
                p_area >= CENTROID_PARENT_AREA_MIN and
                (r_area / p_area) <= CENTROID_AREA_RATIO_MAX
            ):
                P_dil = binary_dilation(P_local, iterations=CENTROID_DILATE)
                if 0 <= r_y_loc < P_dil.shape[0] and 0 <= r_x_loc < P_dil.shape[1]:
                    inside_dilated = bool(P_dil[r_y_loc, r_x_loc])
            path2_pass = inside_dilated
            # Only record pair if some path passes
            if not ((sdt_pass and crita_pass) or path2_pass): continue
            pairs.append(dict(
                parent=int(p_label), ring=int(r_label),
                parent_area=int(p_area), ring_area=int(r_area),
                area_ratio=float(r_area / p_area),
                sdt_n=int(n), sdt_frac_pos=float(frac_pos),
                sdt_mean=float(mean_off), sdt_std=float(std_off), sdt_cv_interior=float(cv_int),
                sdt_pass=bool(sdt_pass),
                crita_cv=float(crita_cv) if not np.isnan(crita_cv) else None,
                crita_pass=bool(crita_pass),
                path2_centroid_inside=bool(inside_dilated),
                path2_pass=bool(path2_pass),
                local_bbox=(int(y0), int(x0), int(y1), int(x1)),
            ))
    return pairs


def cluster_pairs(pairs):
    """Connected components on the (parent → ring) graph. Returns list of clusters,
    each as dict(parent, rings_set, pairs_inside_cluster)."""
    # Build adjacency
    nodes = set()
    edges = []
    for p in pairs:
        nodes.add(p["parent"]); nodes.add(p["ring"])
        edges.append((p["parent"], p["ring"]))
    # Union-find
    parent_map = {n: n for n in nodes}
    def find(x):
        while parent_map[x] != x:
            parent_map[x] = parent_map[parent_map[x]]; x = parent_map[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent_map[ra] = rb
    for a, b in edges: union(a, b)
    comps = {}
    for n in nodes:
        comps.setdefault(find(n), set()).add(n)
    out = []
    for root, member_set in comps.items():
        # Root parent = largest by area within the cluster (per iter 9 logic)
        out.append(dict(members=member_set,
                         pairs=[p for p in pairs if p["parent"] in member_set and p["ring"] in member_set]))
    return out


def validate_cluster(cluster, masks, region):
    """Apply iter 9 validators: focal untouchable, ≤MAX_PEAKS DAPI peaks, purity ≥ MIN_PURITY."""
    member_labels = sorted(cluster["members"])
    # Largest is the "parent"; others are rings absorbed into it
    sizes = {L: int((masks == L).sum()) for L in member_labels}
    parent = max(sizes, key=sizes.get)
    rings = [L for L in member_labels if L != parent]
    if region["cps_focal"] in member_labels:
        return dict(parent=int(parent), rings=rings, accepted=False, reason="focal_in_cluster",
                     peaks=None, purity=None, area=None, dominant=None)
    union = np.isin(masks, member_labels)
    area = int(union.sum())
    # DAPI peaks inside
    peaks = region["dapi_peaks"]
    if len(peaks) == 0:
        n_peaks = 0
    else:
        ys = peaks[:, 0].astype(int); xs = peaks[:, 1].astype(int)
        in_b = (ys >= 0) & (ys < masks.shape[0]) & (xs >= 0) & (xs < masks.shape[1])
        n_peaks = int(union[ys[in_b], xs[in_b]].sum())
    # Anchor purity
    if len(region["anchor_x_local"]) == 0:
        purity = float("nan"); dominant = None; n_anch = 0
    else:
        ys = region["anchor_y_local"].astype(int); xs = region["anchor_x_local"].astype(int)
        in_b = (ys >= 0) & (ys < masks.shape[0]) & (xs >= 0) & (xs < masks.shape[1])
        ys, xs = ys[in_b], xs[in_b]
        lins = np.array(region["anchor_lineage"])[in_b]
        inside = union[ys, xs]
        in_lins = lins[inside]
        n_anch = int(in_lins.size)
        if n_anch == 0:
            purity = float("nan"); dominant = None
        else:
            counts = pd.Series(in_lins).value_counts()
            purity = float(counts.iloc[0] / counts.sum())
            dominant = str(counts.index[0])
    # Decision
    failed = []
    if n_peaks > MAX_PEAKS: failed.append(f"peaks={n_peaks}>{MAX_PEAKS}")
    if not np.isnan(purity) and purity < MIN_PURITY: failed.append(f"purity={purity:.2f}<{MIN_PURITY}")
    accepted = len(failed) == 0
    reason = "; ".join(failed) if failed else "accepted"
    return dict(parent=int(parent), rings=rings, accepted=accepted, reason=reason,
                 peaks=int(n_peaks), purity=float(purity) if not np.isnan(purity) else None,
                 area=int(area), n_anchors=int(n_anch), dominant=dominant)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def norm_local(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def render_cluster_page(pdf, region, masks, cluster, validation, cluster_idx, total):
    member_labels = sorted(cluster["members"])
    sizes = {L: int((masks == L).sum()) for L in member_labels}
    parent_label = validation["parent"]
    ring_labels = validation["rings"]
    # Bounding box of cluster + margin
    union = np.isin(masks, member_labels)
    ys, xs = np.where(union)
    pad = 40
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, masks.shape[0])
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, masks.shape[1])
    crop = (slice(y0, y1), slice(x0, x1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    s18 = norm_local(region["s18"])[crop]
    dapi = norm_local(region["dapi"])[crop]
    m_crop = masks[crop]

    # ---- (0,0) 18S + cells ----
    axes[0, 0].imshow(s18, cmap="gray")
    overlay = np.zeros((*m_crop.shape, 4), dtype=np.float32)
    for L in member_labels:
        c = (0.2, 0.85, 0.2, 0.55) if L == parent_label else (0.95, 0.15, 0.15, 0.55)
        overlay[m_crop == L] = c
    axes[0, 0].imshow(overlay)
    edges = find_boundaries(m_crop, mode="outer")
    axes[0, 0].contour(edges.astype(int), levels=[0.5], colors="white", linewidths=0.4)
    axes[0, 0].set_title(f"18S + cells (green=parent {parent_label}, red=rings)")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    # ---- (0,1) DAPI + peaks ----
    axes[0, 1].imshow(dapi, cmap="gray")
    axes[0, 1].imshow(overlay, alpha=0.4)
    peaks = region["dapi_peaks"]
    p_in = ((peaks[:, 0] >= y0) & (peaks[:, 0] < y1) &
             (peaks[:, 1] >= x0) & (peaks[:, 1] < x1))
    p_local = peaks[p_in] - [y0, x0]
    if len(p_local):
        axes[0, 1].scatter(p_local[:, 1], p_local[:, 0], s=80, c="yellow",
                            marker="*", edgecolors="black", linewidth=0.8)
    # parent centroid
    p_props = regionprops((masks == parent_label).astype(np.int32))
    if p_props:
        cy, cx = p_props[0].centroid
        axes[0, 1].plot(cx - x0, cy - y0, "x", color="cyan", markersize=18, markeredgewidth=2)
    inside_peaks = validation["peaks"] if validation["peaks"] is not None else "?"
    axes[0, 1].set_title(f"DAPI + peaks (★).  Peaks inside merged: {inside_peaks}")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    # ---- (0,2) Anchors ----
    axes[0, 2].imshow(s18, cmap="gray", alpha=0.4)
    axes[0, 2].imshow(overlay, alpha=0.5)
    ax_y = region["anchor_y_local"].astype(int); ax_x = region["anchor_x_local"].astype(int)
    in_b = (ax_y >= y0) & (ax_y < y1) & (ax_x >= x0) & (ax_x < x1)
    ax_y_p = ax_y[in_b] - y0; ax_x_p = ax_x[in_b] - x0
    ax_l = np.array(region["anchor_lineage"])[in_b]
    # Color by lineage
    lin_to_color = {L: c for L, c in LINEAGE_COLORS.items()}
    for L in np.unique(ax_l):
        sel = ax_l == L
        color = lin_to_color.get(str(L), "#888888")
        axes[0, 2].scatter(ax_x_p[sel], ax_y_p[sel], s=4, c=color, alpha=0.7, label=str(L))
    if len(np.unique(ax_l)) > 0:
        axes[0, 2].legend(fontsize=7, loc="upper right", markerscale=2)
    pur = validation["purity"]
    dom = validation["dominant"]
    pur_str = f"{pur:.2f}" if pur is not None else "—"
    axes[0, 2].set_title(f"Anchors. Purity={pur_str} ({dom}). N={validation.get('n_anchors',0)}")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    # ---- (1,0) Parent SDT field ----
    P_mask = (masks == parent_label)[crop]
    sdt = distance_transform_edt(~P_mask) - distance_transform_edt(P_mask)
    sdt_clipped = np.clip(sdt, -30, 30)
    im = axes[1, 0].imshow(sdt_clipped, cmap="RdBu_r", vmin=-30, vmax=30)
    # outline rings
    for L in ring_labels:
        ring_local = (m_crop == L)
        axes[1, 0].contour(ring_local.astype(int), levels=[0.5], colors="black", linewidths=1.2)
    # parent boundary
    p_bound = find_boundaries(P_mask, mode="outer")
    axes[1, 0].contour(p_bound.astype(int), levels=[0.5], colors="lime", linewidths=1.0)
    axes[1, 0].set_title(f"Parent SDT (px). green=parent, black outlines=rings")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    fig.colorbar(im, ax=axes[1, 0], fraction=0.04, pad=0.02)

    # ---- (1,1) SDT histograms per ring ----
    axes[1, 1].axhline(0, color="gray", linewidth=0.5)
    for i, L in enumerate(ring_labels):
        ring_local = (m_crop == L)
        samp = sdt[ring_local]
        if samp.size == 0: continue
        axes[1, 1].hist(samp, bins=30, alpha=0.6, label=f"ring {L} (n={samp.size})")
    axes[1, 1].axvspan(SDT_OFFSET_MIN, SDT_OFFSET_MAX, alpha=0.10, color="green", label=f"valid offset [{SDT_OFFSET_MIN:.0f},{SDT_OFFSET_MAX:.0f}]")
    axes[1, 1].set_xlabel("signed distance to parent boundary (px)")
    axes[1, 1].set_ylabel("ring pixel count")
    axes[1, 1].legend(fontsize=7, loc="upper right")
    axes[1, 1].set_title("Path 1 SDT signal: ring pixels' distance to parent")

    # ---- (1,2) Decision text ----
    axes[1, 2].axis("off")
    decision_color = "green" if validation["accepted"] else "red"
    text = (f"CLUSTER {cluster_idx+1}/{total}\n"
            f"parent label: {parent_label}  (area={sizes[parent_label]} px)\n"
            f"ring labels: {ring_labels}\n"
            f"ring areas: {[sizes[L] for L in ring_labels]} px\n"
            f"\n"
            f"VALIDATORS:\n"
            f"  DAPI peaks in merged: {validation['peaks']} (max {MAX_PEAKS}) → "
                f"{'PASS' if validation['peaks'] is not None and validation['peaks']<=MAX_PEAKS else 'FAIL'}\n"
            f"  anchor purity: {pur_str} (min {MIN_PURITY}, dominant={dom}) → "
                f"{'PASS' if pur is None or pur>=MIN_PURITY else 'FAIL'}\n"
            f"  N anchors: {validation['n_anchors']}\n"
            f"  area: {validation['area']} px\n"
            f"\n"
            f"DECISION: {'ACCEPTED' if validation['accepted'] else 'REJECTED'}\n"
            f"reason: {validation['reason']}\n"
            f"\n"
            f"PATH-LEVEL EVIDENCE (per pair):\n")
    for p in cluster["pairs"]:
        crita_str = f"{p['crita_cv']:.2f}" if p['crita_cv'] is not None else "n/a"
        text += (f"  parent={p['parent']} → ring={p['ring']}\n"
                  f"    SDT: frac_pos={p['sdt_frac_pos']:.2f}, mean={p['sdt_mean']:.1f}px, "
                  f"cv_int={p['sdt_cv_interior']:.2f} → {'pass' if p['sdt_pass'] else 'fail'}\n"
                  f"    CritA boundary-CV: {crita_str} → "
                  f"{'pass' if p['crita_pass'] else 'fail'}\n"
                  f"    Path2 centroid-inside: {p['path2_centroid_inside']} → "
                  f"{'pass' if p['path2_pass'] else 'fail'}\n")
    axes[1, 2].text(0.02, 0.98, text, transform=axes[1, 2].transAxes,
                     fontsize=8, family="monospace", verticalalignment="top",
                     color=decision_color if validation["accepted"] else "darkred")

    title = (f"Cluster {cluster_idx+1}/{total}  —  "
              f"{'ACCEPTED' if validation['accepted'] else 'REJECTED'} ({validation['reason']})")
    plt.suptitle(title, fontsize=12, color="green" if validation["accepted"] else "darkred", y=1.005)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def main():
    print("loading region…")
    region = load_region()
    masks = region["masks_wsi"].copy()
    focal = region["cps_focal"]

    print("finding candidate (parent, ring) pairs (Path 1 + Path 2)…")
    pairs = candidate_pairs(masks, focal, region["dapi_peaks"])
    print(f"  {len(pairs)} candidate pairs")

    print("clustering pairs into components…")
    clusters = cluster_pairs(pairs)
    print(f"  {len(clusters)} clusters")

    print("validating each cluster…")
    validations = [validate_cluster(c, masks, region) for c in clusters]
    n_acc = sum(1 for v in validations if v["accepted"])
    print(f"  {n_acc} accepted, {len(validations)-n_acc} rejected")

    # Sort: accepted first, then rejected, both by area (largest first) so the user
    # sees the biggest decisions first.
    indexed = list(enumerate(zip(clusters, validations)))
    indexed.sort(key=lambda iv: (not iv[1][1]["accepted"], -iv[1][1]["area"]))

    print(f"writing PDF: {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        cover = (
            f"Iter 9 — per-cluster debug\n\n"
            f"  Algorithm: SDT extrusion + perimeter-uniformity + centroid-inside, transitive multi-round.\n"
            f"  Region: DB_top100_019 1000×1000 px, 426 non-focal masks, 193 DAPI peaks, "
            f"{len(region['anchor_x_local'])} anchored transcripts.\n\n"
            f"  Total candidate pairs (Path1 ∪ Path2): {len(pairs)}\n"
            f"  Connected components (≥2 masks): {len(clusters)}\n"
            f"  Accepted: {n_acc}  |  Rejected: {len(clusters)-n_acc}\n\n"
            f"  Validators: focal cell {focal} untouched, ≤{MAX_PEAKS} DAPI peaks in merged region, "
            f"anchor purity ≥{MIN_PURITY}.\n\n"
            f"  Pages are sorted: accepted clusters first (largest area first), then rejected.\n"
        )
        ax.text(0.05, 0.95, cover, fontsize=12, family="monospace", va="top")
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        for cluster_idx, (orig_idx, (cluster, val)) in enumerate(indexed):
            render_cluster_page(pdf, region, masks, cluster, val, cluster_idx, len(clusters))
            if (cluster_idx + 1) % 5 == 0:
                print(f"  rendered {cluster_idx+1}/{len(clusters)}")

    print(f"done. {OUT_PDF.stat().st_size//1024} KB, {len(clusters)+1} pages")


if __name__ == "__main__":
    main()
