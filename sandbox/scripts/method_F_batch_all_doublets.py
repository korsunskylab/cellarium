"""Method F applied to ALL 68 approved doublets — σ sweep at depth=0.99.

Sweep σ_cut ∈ {2, 5, 10} px to find the universal default. User flagged σ=10
as too wide for typical cases; need to see per-doublet outcome at each σ.

Per-doublet figure (4 rows × 3 cols):
  Row 0:  18S+anchors  |  edge_F (σ-independent)  |  Control CP-SAM
  Row 1 (σ=2):   cut_field  |  s_cut  |  CP-SAM split
  Row 2 (σ=5):   cut_field  |  s_cut  |  CP-SAM split
  Row 3 (σ=10):  cut_field  |  s_cut  |  CP-SAM split

Each panel carries yellow=focal CP-SAM contour, lime=π sign-change line.

Outputs:
  figs/method_F_batch_all_doublets/<benchmark_id>.png
  figs/method_F_batch_all_doublets/_summary.csv
"""
from __future__ import annotations
import sys, json, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from skimage.segmentation import find_boundaries
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
OUT_DIR = ROOT / "figs" / "method_F_batch_all_doublets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 10.0; N_MIN = 3.0; K_SHARP = 8.0
SIGMAS_CUT_PX = [2.0, 5.0, 10.0]
DEPTH_MAX = 0.99
ANCHOR_DOT_SIZE = 18
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


def label_overlay(mask, alpha=0.45):
    u = np.unique(mask); u = u[u != 0]
    cm = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(u), 1)))
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, L in enumerate(u): out[mask == L] = (*cm[i % len(cm)][:3], alpha)
    return out


def count_focal(m, focal_region):
    a = focal_region.sum()
    if a == 0: return 0, []
    lbls = np.unique(m[focal_region]); lbls = lbls[lbls != 0]
    covers = sorted(
        [(int(L), int(((m == L) & focal_region).sum()) / a) for L in lbls],
        key=lambda kv: -kv[1]
    )
    n = sum(1 for _, c in covers if c >= 0.05)
    return n, covers[:5]


def process_one(BID, m_sam, norm_dapi, norm_18s):
    row = BENCH[BENCH["benchmark_id"] == BID].iloc[0]
    cps_focal = int(row["cps_id_at_bookmark"])
    pair = row["lineage_pair"].split(" × ")
    lin_a, lin_b = pair[0], pair[1]
    ka, kb = LINEAGES.index(lin_a), LINEAGES.index(lin_b)
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
    pi_diff = (pi_abst[ka] - pi_abst[kb]).astype(np.float32)
    t = np.tanh(K_SHARP * pi_diff)
    gy = sobel(t, axis=0); gx = sobel(t, axis=1)
    g_raw = np.hypot(gx, gy).astype(np.float32)
    cell_ev = np.maximum(dapi_n, s18_n).astype(np.float32)
    edge_F = g_raw * (confidence ** 2) * cell_ev
    sign_field = (pi_diff > 0).astype(np.uint8)
    sign_boundary_focal = find_boundaries(sign_field, mode="inner") & focal_region

    def run_cp(s_img):
        img = np.stack([norm_dapi(dapi), norm_18s(s_img)], axis=-1).astype(np.float32)
        m, _, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        return m.astype(np.int32)

    m_ctrl = run_cp(s18); n_c, cov_c = count_focal(m_ctrl, focal_region)

    per_sigma = {}
    for sigma in SIGMAS_CUT_PX:
        smoothed = gaussian_filter(edge_F, sigma=sigma)
        if smoothed.max() > 0:
            cut_field = (smoothed / smoothed.max()) * DEPTH_MAX
        else:
            cut_field = smoothed
        s_cut = s18.astype(np.float32) * (1.0 - cut_field)
        m_cut = run_cp(s_cut); n_x, cov_x = count_focal(m_cut, focal_region)
        per_sigma[sigma] = dict(cut_field=cut_field, s_cut=s_cut,
                                 m_cut=m_cut, n=n_x, cov=cov_x)

    # crop centred on focal
    ys, xs = np.where(focal_region); pad = 35
    if len(ys) == 0:
        raise RuntimeError(f"{BID}: focal_region empty")
    y0 = max(ys.min() - pad, 0); y1 = min(ys.max() + pad, H)
    x0 = max(xs.min() - pad, 0); x1 = min(xs.max() + pad, W)
    crop = (slice(y0, y1), slice(x0, x1))
    focal_local = focal_region[crop]
    focal_bound = find_boundaries(focal_local, mode="outer")
    sign_local = sign_boundary_focal[crop]
    in_focal_ctrl = find_boundaries(m_ctrl, mode="inner") & focal_region

    in_crop = ((anch["py"].to_numpy() >= y0) & (anch["py"].to_numpy() < y1) &
               (anch["px"].to_numpy() >= x0) & (anch["px"].to_numpy() < x1))
    a_py = anch["py"].to_numpy()[in_crop] - y0
    a_px = anch["px"].to_numpy()[in_crop] - x0
    a_li = li_arr[in_crop]

    def overlays(ax, show_anchors=False):
        ax.contour(focal_bound.astype(int), levels=[0.5], colors="yellow", linewidths=1.3)
        ax.contour(sign_local.astype(int), levels=[0.5], colors="lime", linewidths=1.7)
        if show_anchors:
            for k_lin, lin in enumerate(LINEAGES):
                m_lin = a_li == k_lin
                if not m_lin.any(): continue
                ax.scatter(a_px[m_lin], a_py[m_lin],
                           s=ANCHOR_DOT_SIZE, color=LINEAGE_COLOURS[lin],
                           edgecolors="black", linewidths=0.4)
        ax.set_xticks([]); ax.set_yticks([])

    nrows = 1 + len(SIGMAS_CUT_PX)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.6 * nrows))

    # Row 0 — inputs / σ-independent intermediates
    ax = axes[0, 0]
    ax.imshow(norm01(s18[crop]), cmap="gray")
    overlays(ax, show_anchors=True)
    ax.set_title("18S + anchors\n(yellow=focal CP-SAM, lime=π sign-change)")

    ax = axes[0, 1]
    ev_pct = np.percentile(edge_F[focal_region], 99) if focal_region.any() else 1.0
    ax.imshow(edge_F[crop], cmap="hot", vmin=0, vmax=max(ev_pct, 1e-6))
    overlays(ax)
    ax.set_title("edge_F = |∇tanh| × conf² × cell_ev\n(σ-independent)")

    ax = axes[0, 2]
    ax.imshow(norm01(s18[crop]), cmap="gray")
    ax.imshow(label_overlay(m_ctrl[crop]))
    ax.contour(in_focal_ctrl[crop].astype(int), levels=[0.5], colors="white", linewidths=1.4)
    overlays(ax)
    cov_c_str = " ".join(f"{c*100:.0f}%" for _, c in cov_c[:3])
    ax.set_title(f"Control CP-SAM (no cut)\nn_focal={n_c}  {cov_c_str}")

    # σ rows
    for i, sigma in enumerate(SIGMAS_CUT_PX, start=1):
        d_s = per_sigma[sigma]
        cf = d_s["cut_field"]
        s_cut = d_s["s_cut"]
        m_cut = d_s["m_cut"]
        in_focal_cut = find_boundaries(m_cut, mode="inner") & focal_region

        ax = axes[i, 0]
        ax.imshow(cf[crop], cmap="hot", vmin=0, vmax=1)
        overlays(ax)
        ax.set_title(f"σ={sigma:.0f}px  cut_field (depth={DEPTH_MAX})")

        ax = axes[i, 1]
        ax.imshow(norm01(s_cut[crop]), cmap="gray")
        overlays(ax)
        ax.set_title(f"σ={sigma:.0f}px  s_cut = 18S × (1 − cut)\nwhat CP-SAM sees")

        ax = axes[i, 2]
        ax.imshow(norm01(s_cut[crop]), cmap="gray")
        ax.imshow(label_overlay(m_cut[crop]))
        ax.contour(in_focal_cut[crop].astype(int), levels=[0.5], colors="white", linewidths=1.5)
        overlays(ax)
        cov_x_str = " ".join(f"{c*100:.0f}%" for _, c in d_s["cov"][:3])
        delta = "+" if d_s["n"] > n_c else ("=" if d_s["n"] == n_c else "−")
        ax.set_title(f"σ={sigma:.0f}px  CP-SAM on s_cut  [{delta}]\n"
                      f"n_focal={d_s['n']}  {cov_x_str}")

    plt.suptitle(
        f"{BID} ({lin_a} × {lin_b}) — Method F  α={ALPHA}, K_sharp={K_SHARP}, depth={DEPTH_MAX}",
        fontsize=12, y=1.003)
    plt.tight_layout()
    out = OUT_DIR / f"{BID}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

    rec = {
        "benchmark_id": BID,
        "lineage_pair": row["lineage_pair"],
        "n_anchors": int(row["n_anchors"]),
        "focal_px": int(focal_region.sum()),
        "n_control": int(n_c),
        "top1_control_pct": float(cov_c[0][1] * 100) if cov_c else 0.0,
    }
    for sigma in SIGMAS_CUT_PX:
        rec[f"n_sigma{int(sigma)}"] = int(per_sigma[sigma]["n"])
        rec[f"top1_sigma{int(sigma)}_pct"] = (
            float(per_sigma[sigma]["cov"][0][1] * 100) if per_sigma[sigma]["cov"] else 0.0)
    return rec


def main():
    bench_ok = BENCH[BENCH["approved"].fillna(False)].copy() if "approved" in BENCH.columns else BENCH
    present_ids = sorted([p.name for p in ROIS.iterdir() if p.is_dir()])
    bench_present = bench_ok[bench_ok["benchmark_id"].isin(present_ids)].copy()
    bench_present = bench_present.sort_values("benchmark_id").reset_index(drop=True)
    print(f"processing {len(bench_present)} doublets, σ sweep={SIGMAS_CUT_PX}, depth={DEPTH_MAX}")

    wsi_pct = json.loads(WSI_PCT_JSON.read_text())
    def norm_dapi(arr):
        return ((arr.astype(np.float32) - wsi_pct["DAPI"]["q_lo"]) /
                max(wsi_pct["DAPI"]["q_hi"] - wsi_pct["DAPI"]["q_lo"], 1e-6)).astype(np.float32)
    def norm_18s(arr):
        return ((arr.astype(np.float32) - wsi_pct["18S"]["q_lo"]) /
                max(wsi_pct["18S"]["q_hi"] - wsi_pct["18S"]["q_lo"], 1e-6)).astype(np.float32)

    print("loading cellpose…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    rows = []
    failures = []
    for i, r in bench_present.iterrows():
        BID = r["benchmark_id"]
        try:
            res = process_one(BID, m_sam, norm_dapi, norm_18s)
            rows.append(res)
            ns = "  ".join(f"σ{int(s)}={res[f'n_sigma{int(s)}']}" for s in SIGMAS_CUT_PX)
            print(f"  [{i+1:2d}/{len(bench_present)}] {BID}: control={res['n_control']}  {ns}")
            # write summary after each successful doublet (resilient to crash)
            pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
        except Exception as e:
            failures.append((BID, repr(e)))
            print(f"  [{i+1:2d}/{len(bench_present)}] {BID}: FAILED — {e}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    csv = OUT_DIR / "_summary.csv"
    df.to_csv(csv, index=False)
    print(f"\nsaved {csv}  ({len(df)} rows)")
    if not df.empty:
        n_split_c = (df["n_control"] >= 2).sum()
        print(f"split-rate control: {n_split_c}/{len(df)}")
        for sigma in SIGMAS_CUT_PX:
            col = f"n_sigma{int(sigma)}"
            n_split = (df[col] >= 2).sum()
            n_up = (df[col] > df["n_control"]).sum()
            n_dn = (df[col] < df["n_control"]).sum()
            print(f"  σ={int(sigma)}: split-rate {n_split}/{len(df)}  "
                  f"(↑{n_up}  ↓{n_dn}  ={len(df)-n_up-n_dn})")
    if failures:
        print(f"\n{len(failures)} failures:")
        for bid, err in failures: print(f"  {bid}: {err}")


if __name__ == "__main__":
    main()
