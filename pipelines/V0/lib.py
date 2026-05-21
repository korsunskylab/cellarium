"""V0 boundary-prior pipeline — helper functions.

Reference implementation of the segmentation-agnostic 18S cropping that runs
*before* Cellpose-SAM. Designed to be readable end-to-end from the companion
notebook 00_run_pipeline.ipynb.

All constants are defined here in one place so any pipeline tweak is a single-
line edit. Anything outside this file should call these functions, not
re-implement the math.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import gaussian_filter, sobel

# ── Paths (relative to repo root) ────────────────────────────────────────────
REPO   = Path(__file__).resolve().parents[2]
DATA   = REPO / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
DAPI_TIF = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
S18_TIF  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
TX_ZARR  = DATA / "transcripts.zarr.zip"
GENE_LABELS_PARQ = DATA / "gene_labels_lineage.parquet"
WSI_PCT_JSON     = DATA / "wsi_morphology_percentiles.json"

# ── V0 lineage panel ────────────────────────────────────────────────────────
LINEAGES = ["Melanoma", "Myeloid", "Tcell", "Plasma",
            "Fibroblast", "Endothelial", "Keratinocyte"]
K = len(LINEAGES)
LIN_TO_IDX = {L: i for i, L in enumerate(LINEAGES)}

# ── V0 parameters (frozen; change here to retune) ──────────────────────────
PIXEL_UM      = 0.2125
SIGMA_UM      = 2.0
SIGMA_DIFFUSE_PX = SIGMA_UM / PIXEL_UM          # 9.4118 px — anchor smoothing
ALPHA         = 10.0                            # Dirichlet concentration
N_MIN         = 3.0                             # grey-abstain anchor threshold
K_SHARP       = 8.0                             # tanh sharpening on margin
SIGMA_CUT_PX  = 5.0                             # edge-field smoothing (≈1.06 µm)
DEPTH_MAX     = 0.99                            # max 18S attenuation
QV_MIN        = 20.0                            # transcript quality floor

CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=True)


# ── 1) ROI definition + morphology load ─────────────────────────────────────

def make_roi_bbox_centred(cx_um: float, cy_um: float, size_px: int,
                          wsi_H: int, wsi_W: int):
    """Return integer pixel bbox (y0, y1, x0, x1) for a size×size crop centred
    on (cx_um, cy_um) and clamped inside the WSI."""
    half = size_px // 2
    cx_px = int(round(cx_um / PIXEL_UM))
    cy_px = int(round(cy_um / PIXEL_UM))
    x0 = max(0, cx_px - half); x1 = x0 + size_px
    if x1 > wsi_W: x1 = wsi_W; x0 = x1 - size_px
    y0 = max(0, cy_px - half); y1 = y0 + size_px
    if y1 > wsi_H: y1 = wsi_H; y0 = y1 - size_px
    return int(y0), int(y1), int(x0), int(x1)


def load_morphology(y0: int, y1: int, x0: int, x1: int):
    """Load DAPI and 18S WSI crops for the given pixel bbox. Returns uint16."""
    with tifffile.TiffFile(DAPI_TIF) as tf:
        dapi = tf.series[0].asarray(key=0)[y0:y1, x0:x1].copy()
    with tifffile.TiffFile(S18_TIF) as tf:
        s18 = tf.series[0].asarray(key=0)[y0:y1, x0:x1].copy()
    return dapi.astype(np.uint16), s18.astype(np.uint16)


# ── 2) Gene → lineage label (V0: fixed input) ───────────────────────────────

def load_lineage_label_map() -> np.ndarray:
    """Return int8 array of length n_genes (Xenium panel) mapping each gene
    to its lineage index in LINEAGES, or -1 if 'ambiguous' / not labelled."""
    gene_names = list(zarr.open(TX_ZARR, mode="r").attrs["gene_names"])
    gl = pd.read_parquet(GENE_LABELS_PARQ).set_index("gene")["label"].to_dict()
    out = np.full(len(gene_names), -1, dtype=np.int8)
    for i, g in enumerate(gene_names):
        lab = gl.get(g)
        if lab in LIN_TO_IDX:
            out[i] = LIN_TO_IDX[lab]
    return out


# ── 3) Anchor extraction (transcripts → lineage-mapped pixel locations) ─────

def load_anchors_in_bbox(y0: int, y1: int, x0: int, x1: int,
                         gene_lin: np.ndarray, qv_min: float = QV_MIN):
    """Iterate transcripts.zarr.zip 250-µm grid tiles overlapping the ROI bbox,
    apply QC (qv ≥ qv_min, valid==1) and lineage filter, return arrays:

        py_local (int32) : y pixel in the *crop* frame  (0..H-1)
        px_local (int32) : x pixel in the *crop* frame  (0..W-1)
        li       (int8)  : lineage index (0..K-1)
    """
    z = zarr.open(TX_ZARR, mode="r")
    g0 = z["grids"]["0"]
    TILE_UM = 250.0
    H = y1 - y0; W = x1 - x0
    x0_um = x0 * PIXEL_UM; x1_um = x1 * PIXEL_UM
    y0_um = y0 * PIXEL_UM; y1_um = y1 * PIXEL_UM
    i0 = int(x0_um // TILE_UM); i1 = int(x1_um // TILE_UM)
    j0 = int(y0_um // TILE_UM); j1 = int(y1_um // TILE_UM)

    py_acc, px_acc, li_acc = [], [], []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            key = f"{i},{j}"
            if key not in g0: continue
            tile = g0[key]
            loc = tile["location"][:]
            gid = tile["gene_identity"][:].squeeze(-1).astype(np.int64)
            qv  = tile["quality_score"][:].squeeze(-1)
            val = tile["valid"][:].squeeze(-1)
            lin = gene_lin[gid]
            ok = (qv >= qv_min) & (val == 1) & (lin >= 0) \
                 & (loc[:, 0] >= x0_um) & (loc[:, 0] < x1_um) \
                 & (loc[:, 1] >= y0_um) & (loc[:, 1] < y1_um)
            if not ok.any(): continue
            px = np.rint(loc[ok, 0] / PIXEL_UM).astype(np.int32) - x0
            py = np.rint(loc[ok, 1] / PIXEL_UM).astype(np.int32) - y0
            keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if not keep.any(): continue
            py_acc.append(py[keep])
            px_acc.append(px[keep])
            li_acc.append(lin[ok][keep])
    if not py_acc:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int8))
    return (np.concatenate(py_acc), np.concatenate(px_acc), np.concatenate(li_acc))


# ── 4) Per-lineage posterior (Dirichlet + grey-abstain) ─────────────────────

def lineage_posterior(py: np.ndarray, px: np.ndarray, li: np.ndarray,
                      H: int, W: int,
                      sigma_diffuse_px: float = SIGMA_DIFFUSE_PX,
                      alpha: float = ALPHA,
                      n_min: float = N_MIN):
    """Compute per-pixel lineage posterior π and confidence c.

    Returns:
        pi_abst (K, H, W) float32 — grey-abstain-blended posterior
        confidence (H, W) float32 — N_total / (N_total + n_min)
        N_eff    (K, H, W) float32 — effective per-pixel anchor count
    """
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li, py, px), 1.0)
    rho_smooth = np.stack(
        [gaussian_filter(rho_pts[k], sigma=sigma_diffuse_px, mode="reflect")
         for k in range(K)],
        axis=0,
    )
    N_eff = rho_smooth * 2.0 * np.pi * sigma_diffuse_px ** 2
    N_total = N_eff.sum(axis=0)
    pi_post = (N_eff + alpha) / (N_total + K * alpha + 1e-9)[None]
    confidence = N_total / (N_total + n_min + 1e-9)
    pi_abst = confidence[None] * pi_post + (1.0 - confidence[None]) * (1.0 / K)
    return pi_abst.astype(np.float32), confidence.astype(np.float32), N_eff


# ── 5) Edge field (one-vs-rest) ─────────────────────────────────────────────

def edge_global_field(pi_abst: np.ndarray, k_sharp: float = K_SHARP):
    """One-vs-rest edge field. For each lineage k:
        s_k = π_k − max_{j≠k} π_j
        e_k = ‖∇ tanh(k_sharp · s_k)‖₂
    Returns sum_k e_k (H, W) float32. Plus the argmax label map (int8) for QC.
    """
    K_, H, W = pi_abst.shape
    top_idx = np.argmax(pi_abst, axis=0).astype(np.int8)
    top1_val = np.take_along_axis(pi_abst, top_idx[None].astype(np.int64), axis=0)[0]
    masked = np.where(
        np.arange(K_, dtype=np.int8)[:, None, None] == top_idx[None],
        -np.inf, pi_abst)
    top2_val = np.max(masked, axis=0)

    edge_total = np.zeros((H, W), dtype=np.float32)
    for k in range(K_):
        others_max = np.where(top_idx == k, top2_val, top1_val)
        s_k = pi_abst[k] - others_max
        t_k = np.tanh(k_sharp * s_k).astype(np.float32)
        gy = sobel(t_k, axis=0); gx = sobel(t_k, axis=1)
        edge_total += np.hypot(gx, gy).astype(np.float32)
    return edge_total, top_idx, (top1_val - top2_val).astype(np.float32)


# ── 6) Cell-evidence gate (morphology only, segmentation-agnostic) ──────────

def load_wsi_percentiles():
    """Cached 1-99 percentiles of the WSI for DAPI and 18S."""
    return json.loads(WSI_PCT_JSON.read_text())


def clip01(arr_u16: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    a = arr_u16.astype(np.float32)
    rng = max(q_hi - q_lo, 1e-6)
    a = (a - q_lo) / rng
    np.clip(a, 0.0, 1.0, out=a)
    return a


def cell_evidence(dapi_u16, s18_u16, percentiles):
    d = clip01(dapi_u16, percentiles["DAPI"]["q_lo"], percentiles["DAPI"]["q_hi"])
    s = clip01(s18_u16,  percentiles["18S" ]["q_lo"], percentiles["18S" ]["q_hi"])
    return np.maximum(d, s)


# ── 7) Cut field ────────────────────────────────────────────────────────────

def cut_field_from_edge(edge_total: np.ndarray,
                        confidence: np.ndarray,
                        evidence: np.ndarray,
                        sigma_cut_px: float = SIGMA_CUT_PX,
                        depth_max: float = DEPTH_MAX):
    """Compose the weighted edge field, smooth, normalise to depth_max."""
    weighted = edge_total * (confidence ** 2) * evidence
    smoothed = gaussian_filter(weighted, sigma=sigma_cut_px, mode="reflect")
    mx = float(smoothed.max())
    if mx > 0:
        smoothed = smoothed * (depth_max / mx)
    return smoothed.astype(np.float32), weighted.astype(np.float32)


# ── 8) Apply cut → modified 18S ─────────────────────────────────────────────

def apply_cut(s18_u16: np.ndarray, cut_field: np.ndarray) -> np.ndarray:
    """s_cut = round(S18 · (1 − cut_field)) as uint16 (matches WSI pipeline)."""
    out = s18_u16.astype(np.float32) * (1.0 - cut_field)
    return np.rint(np.clip(out, 0, 65535)).astype(np.uint16)


# ── 9) CP-SAM wrapper ───────────────────────────────────────────────────────

def run_cpsam(dapi_u16: np.ndarray, s18_u16_for_input: np.ndarray):
    """Run Cellpose-SAM on (DAPI, channel-2). Both inputs 1-99 percentile clipped
    to [0, 1] on their own data (matches existing WSI control pipeline)."""
    from cellpose import models

    def fresh_clip(arr_u16):
        rng = np.random.default_rng(0)
        n = arr_u16.size
        if n > 10_000_000:
            sample = arr_u16.ravel()[rng.integers(0, n, size=10_000_000)]
        else:
            sample = arr_u16.ravel()
        q_lo, q_hi = np.percentile(sample, [1.0, 99.0])
        return clip01(arr_u16, float(q_lo), float(q_hi)), float(q_lo), float(q_hi)

    d_norm, d_lo, d_hi = fresh_clip(dapi_u16)
    s_norm, s_lo, s_hi = fresh_clip(s18_u16_for_input)
    img = np.stack([d_norm, s_norm], axis=-1).astype(np.float32)

    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    masks, flows, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
    return masks.astype(np.int32), dict(
        n_cells=int(masks.max()),
        dapi_percentiles=[d_lo, d_hi],
        s_percentiles=[s_lo, s_hi],
    )
