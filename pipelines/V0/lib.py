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
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt
from skimage.segmentation import find_boundaries

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

# Lineage colors designed for max separability — Fib is yellow (not purple) so the
# Mel(blue) ↔ Fib(yellow) blend renders as green-ish (distinct from both parents)
# instead of an ambiguous blue-purple gradation. Use this everywhere via lib.V0_LINEAGE_COLORS.
V0_LINEAGE_COLORS = [
    (0.12, 0.47, 0.71),   # Melanoma:    blue
    (1.00, 0.50, 0.05),   # Myeloid:     orange
    (0.17, 0.63, 0.17),   # Tcell:       green
    (0.84, 0.15, 0.16),   # Plasma:      red
    (0.98, 0.82, 0.10),   # Fibroblast:  YELLOW   (was purple — was visually too close to Mel blue)
    (0.55, 0.34, 0.29),   # Endothelial: brown
    (0.89, 0.47, 0.76),   # Keratinocyte: pink
]

# ── V0 parameters (frozen; change here to retune) ──────────────────────────
PIXEL_UM      = 0.2125
SIGMA_UM      = 2.0
SIGMA_DIFFUSE_PX = SIGMA_UM / PIXEL_UM          # 9.4118 px — anchor smoothing (isotropic)
ALPHA         = 10.0                            # Dirichlet concentration
N_MIN         = 3.0                             # grey-abstain anchor threshold
K_SHARP       = 8.0                             # tanh sharpening on margin
SIGMA_CUT_PX  = 5.0                             # edge-field smoothing (≈1.06 µm)
DEPTH_MAX     = 0.99                            # max 18S attenuation
QV_MIN        = 20.0                            # transcript quality floor
# Anisotropic-diffusion knobs (alternative to isotropic Gaussian).
# Key fact: σ_eff_px = sqrt(n_iter · dt · c²) where c² is the LOCAL conductance squared.
# With linear-clipped 18S, typical cell-interior c ≈ 0.4–0.5 → c² ≈ 0.2 → diffusion runs
# ~5× slower than the homogeneous-c=1 case. If we calibrate n_iter purely off σ_target²/dt,
# anchors barely spread inside cells (the user's "punctate points" complaint).
# Calibration: ANISO_N_ITER_FACTOR compensates for typical c² inside cells.
DT_ANISO              = 0.2                     # explicit-Euler step size (<1/4 for 4-connected stability)
ANISO_TYPICAL_C2      = 0.25                    # typical (cell-interior c)²; tune if cells look under-spread
ANISO_N_ITER_FACTOR   = 1.0 / ANISO_TYPICAL_C2   # = 4: multiplier so σ_eff ≈ σ_target in TYPICAL cells
N_ITER_ANISO          = int(round(ANISO_N_ITER_FACTOR * SIGMA_DIFFUSE_PX ** 2 / DT_ANISO))  # ≈1772 for σ_target=2 µm

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


# Module-level cache for the full-WSI morphology pages.  Loading each is ~1.7 GB
# and ~10 s; we want to amortise that across many ROI crops in a batch.
_FULL_DAPI = None
_FULL_S18  = None


def _load_full_wsi_pages():
    """Read the full DAPI and 18S WSI pages once and cache them in memory.

    IMPORTANT: We must use `tifffile.imread(path, key=0)` here, NOT
    `TiffFile(path).series[0].asarray(key=0)`. The latter hits the "OME
    series cannot read multi-file pyramids" path inside tifffile and
    silently returns the *same* IFD for both DAPI and 18S files — a
    devastating bug that makes the boundary-prior cut a no-op.
    """
    global _FULL_DAPI, _FULL_S18
    if _FULL_DAPI is None:
        _FULL_DAPI = tifffile.imread(DAPI_TIF, key=0)
    if _FULL_S18 is None:
        _FULL_S18 = tifffile.imread(S18_TIF, key=0)
    return _FULL_DAPI, _FULL_S18


def load_morphology(y0: int, y1: int, x0: int, x1: int):
    """Load DAPI and 18S WSI crops for the given pixel bbox. Returns uint16.

    Uses the module-level full-WSI cache so that batch runs amortise the
    ~3.4 GB of I/O. Single-ROI use also works — the cache is lazy.
    """
    full_dapi, full_s18 = _load_full_wsi_pages()
    dapi = full_dapi[y0:y1, x0:x1].copy().astype(np.uint16)
    s18  = full_s18[y0:y1, x0:x1].copy().astype(np.uint16)
    return dapi, s18


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

def make_conductance(s18_raw, mode="linear"):
    """18S brightness → per-pixel conductance ∈ [0, 1] for anisotropic diffusion.

    mode='linear': 1-99 percentile clip then linear rescale to [0, 1].
                   Edge weight c(p)·c(q) is then product of linear values.
    mode='log'   : log1p(s18) then 1-99 percentile clip + rescale (use sparingly;
                   user prefers linear for analysis).
    """
    if mode == "linear":
        arr = s18_raw.astype(np.float32)
    elif mode == "log":
        arr = np.log1p(s18_raw.astype(np.float32))
    else:
        raise ValueError(f"unknown conductance mode: {mode}")
    lo, hi = np.percentile(arr, [1, 99])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def diffuse_anisotropic(rho, conductance, n_iter, dt=DT_ANISO):
    """4-neighbour anisotropic diffusion with per-edge conductance c(p)·c(q).

    Anchor information flows along bright cytoplasm (high 18S → high conductance)
    and is blocked at dim inter-cell gaps. This is exactly the structural
    property the isotropic Gaussian lacks — Gaussian leaks anchors across cell
    gaps into empty stroma, producing phantom argmax boundaries there.

    rho: (K, H, W) float32 — per-lineage density to smooth
    conductance: (H, W) float32 in [0, 1]
    n_iter: diffusion steps; effective σ_px ≈ sqrt(n_iter · dt) in homogeneous-
            conductance regions. In low-conductance regions, σ_eff is much
            smaller — exactly the desired anisotropy.
    Zero-flux Neumann boundaries; mass-conserving (Σ rho preserved exactly).
    """
    c = conductance
    c_right = c * np.roll(c, -1, axis=-1); c_right[..., :, -1] = 0.0
    c_left  = c * np.roll(c,  1, axis=-1); c_left[...,  :,  0] = 0.0
    c_down  = c * np.roll(c, -1, axis=-2); c_down[...,  -1, :] = 0.0
    c_up    = c * np.roll(c,  1, axis=-2); c_up[...,    0,  :] = 0.0
    for _ in range(n_iter):
        flux  = (np.roll(rho, -1, axis=-1) - rho) * c_right
        flux += (np.roll(rho,  1, axis=-1) - rho) * c_left
        flux += (np.roll(rho, -1, axis=-2) - rho) * c_down
        flux += (np.roll(rho,  1, axis=-2) - rho) * c_up
        rho = rho + dt * flux
    return rho


def lineage_posterior_anisotropic(py: np.ndarray, px: np.ndarray, li: np.ndarray,
                                   H: int, W: int,
                                   s18_raw: np.ndarray,
                                   alpha: float = ALPHA,
                                   n_min: float = N_MIN,
                                   n_iter: int = N_ITER_ANISO,
                                   dt: float = DT_ANISO,
                                   conductance_mode: str = "linear"):
    """Same outputs as `lineage_posterior`, but smooths the per-lineage anchor
    density with **anisotropic diffusion guided by 18S** instead of isotropic
    Gaussian. Uses the same Dirichlet posterior + grey-abstain blend afterwards.

    Validated approach from the tinted-cytoplasm pipeline (2026-05-14). Slower
    than isotropic Gaussian (~10-30s on a 2048×2048 ROI at n_iter=443) but
    structurally correct: anchor information stays inside cells.
    """
    K = len(LINEAGES)
    rho_pts = np.zeros((K, H, W), dtype=np.float32)
    np.add.at(rho_pts, (li, py, px), 1.0)
    cond = make_conductance(s18_raw, mode=conductance_mode)
    rho_smooth = diffuse_anisotropic(rho_pts, cond, n_iter=n_iter, dt=dt)
    sigma_eff_px = (n_iter * dt) ** 0.5
    N_eff = rho_smooth * 2.0 * np.pi * sigma_eff_px ** 2
    N_total = N_eff.sum(axis=0)
    pi_post = (N_eff + alpha) / (N_total + K * alpha + 1e-9)[None]
    confidence = N_total / (N_total + n_min + 1e-9)
    pi_abst = confidence[None] * pi_post + (1.0 - confidence[None]) * (1.0 / K)
    return pi_abst.astype(np.float32), confidence.astype(np.float32), N_eff


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
    """LEGACY V0 cut: smooth weighted_edge, normalize by GLOBAL MAX, clip to depth.

    Known issue: at 2048 px crops with multiple lineage boundaries in the frame,
    the global-max normalization gives the depth_max budget to the strongest
    inter-cell boundary, leaving intra-cell heterotypic boundaries with only a
    fraction of the depth (~0.25 on cell 018). Kept for reference; use
    `cut_field_perp_gauss` instead.
    """
    weighted = edge_total * (confidence ** 2) * evidence
    smoothed = gaussian_filter(weighted, sigma=sigma_cut_px, mode="reflect")
    mx = float(smoothed.max())
    if mx > 0:
        smoothed = smoothed * (depth_max / mx)
    return smoothed.astype(np.float32), weighted.astype(np.float32)


def cut_field_perp_gauss(pi_abst: np.ndarray,
                          sigma_perp_px: float = 2.0,
                          depth_max: float = DEPTH_MAX):
    """Perpendicular-Gaussian deep cut at argmax(π) lineage boundaries.

    At every pixel where argmax(π) differs from a neighbour, the cut depth is
    exactly depth_max. Perpendicular to that 1-pixel-wide boundary curve, the
    cut falls off as a Gaussian with sigma_perp_px. Mathematically:

        boundary(x) = 1 if argmax(π) differs from any 8-connected neighbour
        d(x)        = Euclidean distance from x to nearest boundary pixel
        cut(x)      = depth_max · exp(−d(x)² / (2 · σ_perp²))

    Confirmed on cell 018 (2026-05-21 sanity check): σ=2 gives a clean 2-way
    split, σ=5 over-splits to 3. Does NOT weight by confidence or evidence —
    all heterotypic-label transitions get full depth, including inter-cell
    boundaries between two pure-but-different lineages (off-target).

    Returns: (cut, boundary_mask).
    """
    top_idx = np.argmax(pi_abst, axis=0).astype(np.int32)
    boundary = find_boundaries(top_idx, mode="thick").astype(bool)
    dist = distance_transform_edt(~boundary).astype(np.float32)
    cut = (depth_max * np.exp(-(dist ** 2) / (2.0 * sigma_perp_px ** 2))).astype(np.float32)
    return cut, boundary


def cut_field_perp_gauss_conf(pi_abst: np.ndarray,
                                confidence: np.ndarray,
                                sigma_perp_px: float = 2.0,
                                depth_max: float = DEPTH_MAX):
    """Perpendicular-Gaussian cut at argmax(π) boundaries, modulated by confidence².

    Same boundary geometry and perpendicular Gaussian profile as
    `cut_field_perp_gauss`, but the depth assigned to each boundary pixel is
    `depth_max · confidence(x)²` instead of a uniform `depth_max`.

    Why this discriminates intra-cell from inter-cell boundaries: anchor
    density (and therefore confidence²) is high inside cell bodies where
    heterotypic doublets live, and lower at inter-cell gaps. The same
    confidence² used as a multiplier in V0's legacy `weighted_edge` is
    repurposed here as a per-boundary-pixel depth gate.

        boundary(x)    = 1 if argmax(π) differs from any 8-connected neighbour
        bd_depth(x)    = depth_max · confidence(x)²  on the boundary, 0 elsewhere
        d(x), nn(x)    = Euclidean distance + nearest-boundary-pixel index
        cut(x)         = bd_depth(nn(x)) · exp(−d(x)² / (2 · σ_perp²))

    Returns: (cut, boundary_mask).
    """
    top_idx = np.argmax(pi_abst, axis=0).astype(np.int32)
    boundary = find_boundaries(top_idx, mode="thick").astype(bool)
    dist, nearest_yx = distance_transform_edt(~boundary, return_indices=True)
    nearest_y, nearest_x = nearest_yx[0], nearest_yx[1]
    bd_depth = np.where(boundary, depth_max * (confidence ** 2), 0.0).astype(np.float32)
    on_nearest = bd_depth[nearest_y, nearest_x]
    perp_gauss = np.exp(-(dist.astype(np.float32) ** 2) / (2.0 * sigma_perp_px ** 2))
    cut = (on_nearest * perp_gauss).astype(np.float32)
    return cut, boundary


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
