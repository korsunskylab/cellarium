"""Onion ring — flow-field analysis at the ring location.

Cell at the ring location IS correctly 1 mask at size 256 (preliminary run
confirmed labels_at_ring=1). It explodes to 5 at 512, more at larger sizes,
and 17 at 4096. Goal: see the flow fields at each scale to find the mechanism,
then test fixes.

PHASE 1 — size sweep at ring location:
    sizes 256/512/1024/2048/4096, augment=True, normalize=True (cellpose's
    default local %ile normalization — same as what made 256 work)

PHASE 2 — additional knobs at size=4096:
    A. bsize=512 (fewer tile boundaries — test if it's tile-stitching)
    B. niter=500 (let flows converge more — test if it's incomplete convergence)
    C. norm_from_256: use the SMALL-ROI's local percentiles to normalize the
       4096 image, then run with normalize=False. If this makes 4096 give 1
       label, normalization is the cause — and the fix is to always normalize
       with the small-ROI-style %iles.

Each run saves masks + flow_y + flow_x + cellprob, so the master figure can
visualize the actual flow field at the ring location across all conditions.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.segmentation import find_boundaries

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
OUT_DIR = ROOT / "figs" / "onion_ring_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_UM = 0.2125
RING_WSI_X, RING_WSI_Y = 41680, 6996
SIZES_PX = [256, 512, 1024, 2048, 4096]
RING_HALF_WIN = 20

CPSAM_KW_BASE = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                     flow_threshold=0.0, augment=True, normalize=True)


def crop_centered_wsi_px(arr, cx_px, cy_px, size_px):
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = max(cy_px - half, 0); y1 = min(y0 + size_px, H); y0 = y1 - size_px
    x0 = max(cx_px - half, 0); x1 = min(x0 + size_px, W); x0 = x1 - size_px
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_local_pct(arr, lo=1, hi=99):
    a = arr.astype(np.float32)
    qlo, qhi = np.percentile(a, [lo, hi])
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32), float(qlo), float(qhi)


def norm_with(arr, qlo, qhi):
    a = arr.astype(np.float32)
    return np.clip((a - qlo) / max(qhi - qlo, 1e-6), 0, 1).astype(np.float32)


def count_labels_at(masks, y, x, half=RING_HALF_WIN):
    H, W = masks.shape
    y0 = max(y - half, 0); y1 = min(y + half, H)
    x0 = max(x - half, 0); x1 = min(x + half, W)
    if y1 <= y0 or x1 <= x0: return 0
    patch = masks[y0:y1, x0:x1]
    return len(np.unique(patch)) - (1 if 0 in patch else 0)


def flow_coherence_at(flow_y, flow_x, y, x, half=RING_HALF_WIN):
    H, W = flow_x.shape
    y0 = max(y - half, 0); y1 = min(y + half, H)
    x0 = max(x - half, 0); x1 = min(x + half, W)
    if y1 <= y0 or x1 <= x0: return 0.0
    fx = flow_x[y0:y1, x0:x1].astype(np.float32)
    fy = flow_y[y0:y1, x0:x1].astype(np.float32)
    mag = np.sqrt(fx * fx + fy * fy)
    if mag.max() < 1e-6: return 0.0
    ux = fx / (mag + 1e-6); uy = fy / (mag + 1e-6)
    return float(np.sqrt(ux.mean() ** 2 + uy.mean() ** 2))


def label_overlay(masks, alpha=0.55):
    masks = masks.astype(np.int64)
    n = int(masks.max())
    if n == 0: return np.zeros((*masks.shape, 4), dtype=np.float32)
    perm = np.concatenate([[0], np.random.default_rng(0).permutation(np.arange(1, n + 1))])
    shuf = perm[masks]
    rgba = plt.get_cmap("nipy_spectral")(shuf / max(n, 1))
    rgba[..., 3] = alpha * (masks > 0)
    edges = find_boundaries(masks, mode="outer")
    rgba[edges] = (1, 1, 1, 1.0)
    return rgba


def main():
    t_total = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)

    print("loading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(dapi_in, s18_in, **kw):
        k = dict(CPSAM_KW_BASE); k.update(kw)
        img = np.stack([dapi_in, s18_in], axis=-1).astype(np.float32)
        masks, flows, _ = m_sam.eval(img, channel_axis=-1, **k)
        return masks.astype(np.int32), flows

    # =========== PHASE 1: size sweep at ring location ===========
    results = []
    pct_by_size = {}  # save local percentiles per size, used by Phase 2C
    for sp in SIZES_PX:
        print(f"\n--- PHASE 1: size={sp}px, augment=True, normalize=True (local %iles) ---")
        dapi_c, bbox = crop_centered_wsi_px(dapi_full, RING_WSI_X, RING_WSI_Y, sp)
        s18_c,  _    = crop_centered_wsi_px(s18_full,  RING_WSI_X, RING_WSI_Y, sp)
        # Record the LOCAL percentiles (so Phase 2C can transfer them)
        _, dlo, dhi = norm_local_pct(dapi_c); _, slo, shi = norm_local_pct(s18_c)
        pct_by_size[sp] = {"dapi_qlo": dlo, "dapi_qhi": dhi, "s18_qlo": slo, "s18_qhi": shi}
        # Cellpose's built-in normalize=True will do its own per-image normalization
        # — pass RAW so it matches "what made 256 work" exactly
        ring_local_x = RING_WSI_X - bbox[0]; ring_local_y = RING_WSI_Y - bbox[1]
        t0 = time.time()
        masks, flows = run(dapi_c.astype(np.float32), s18_c.astype(np.float32))
        dt = time.time() - t0
        flow_xy = flows[1].astype(np.float32)
        cellprob = flows[2].astype(np.float32)
        flow_y, flow_x = flow_xy[0], flow_xy[1]
        n_lab = count_labels_at(masks, ring_local_y, ring_local_x)
        coh = flow_coherence_at(flow_y, flow_x, ring_local_y, ring_local_x)
        cp_mean = float(cellprob[max(ring_local_y - RING_HALF_WIN, 0):ring_local_y + RING_HALF_WIN,
                                  max(ring_local_x - RING_HALF_WIN, 0):ring_local_x + RING_HALF_WIN].mean())
        rec = dict(phase="size_sweep", size_px=sp, condition=f"default_aug_localnorm",
                   n_total=int(masks.max()), n_labels_at_ring=n_lab,
                   flow_coherence=coh, cellprob_mean=cp_mean, runtime_s=round(dt, 1))
        results.append(rec)
        # Save arrays + raw image for visualization
        np.savez_compressed(OUT_DIR / f"phase1_size{sp}.npz",
                            masks=masks, flow_y=flow_y, flow_x=flow_x,
                            cellprob=cellprob, dapi_raw=dapi_c, s18_raw=s18_c,
                            bbox=np.array(bbox),
                            ring_local=np.array([ring_local_x, ring_local_y]),
                            local_pct=np.array([dlo, dhi, slo, shi]))
        print(f"  n_total={masks.max()}, labels_at_ring={n_lab}, "
              f"flow_coherence={coh:.3f}, cellprob_mean={cp_mean:.3f}, t={dt:.1f}s")

    # =========== PHASE 2: extra knobs at size 4096 ===========
    sp = 4096
    dapi_c, bbox = crop_centered_wsi_px(dapi_full, RING_WSI_X, RING_WSI_Y, sp)
    s18_c,  _    = crop_centered_wsi_px(s18_full,  RING_WSI_X, RING_WSI_Y, sp)
    ring_local_x = RING_WSI_X - bbox[0]; ring_local_y = RING_WSI_Y - bbox[1]

    phase2_runs = []
    # 2A — bsize=512 SKIPPED: cellpose-SAM has fixed ViT positional embedding,
    #      bsize must stay at 256 (RuntimeError: tensor a (64) vs b (32)).
    # 2B — niter=500
    phase2_runs.append(("phase2B_niter500", dict(niter=500), dapi_c, s18_c))
    # 2C — transfer 256-ROI local %iles to 4096 image, run with normalize=False
    pct256 = pct_by_size[256]
    dapi_norm256 = norm_with(dapi_c, pct256["dapi_qlo"], pct256["dapi_qhi"])
    s18_norm256  = norm_with(s18_c,  pct256["s18_qlo"],  pct256["s18_qhi"])
    phase2_runs.append(("phase2C_norm_from_256", dict(normalize=False),
                        dapi_norm256, s18_norm256))

    for name, extra_kw, dapi_in, s18_in in phase2_runs:
        print(f"\n--- PHASE 2: {name} at size=4096 ---")
        t0 = time.time()
        masks, flows = run(dapi_in, s18_in, **extra_kw)
        dt = time.time() - t0
        flow_xy = flows[1].astype(np.float32)
        cellprob = flows[2].astype(np.float32)
        flow_y, flow_x = flow_xy[0], flow_xy[1]
        n_lab = count_labels_at(masks, ring_local_y, ring_local_x)
        coh = flow_coherence_at(flow_y, flow_x, ring_local_y, ring_local_x)
        cp_mean = float(cellprob[max(ring_local_y - RING_HALF_WIN, 0):ring_local_y + RING_HALF_WIN,
                                  max(ring_local_x - RING_HALF_WIN, 0):ring_local_x + RING_HALF_WIN].mean())
        rec = dict(phase="phase2_4096", size_px=sp, condition=name,
                   n_total=int(masks.max()), n_labels_at_ring=n_lab,
                   flow_coherence=coh, cellprob_mean=cp_mean, runtime_s=round(dt, 1))
        results.append(rec)
        np.savez_compressed(OUT_DIR / f"{name}.npz",
                            masks=masks, flow_y=flow_y, flow_x=flow_x,
                            cellprob=cellprob,
                            dapi_raw=dapi_c if "norm_from_256" not in name else dapi_in,
                            s18_raw=s18_c if "norm_from_256" not in name else s18_in,
                            bbox=np.array(bbox),
                            ring_local=np.array([ring_local_x, ring_local_y]))
        print(f"  n_total={masks.max()}, labels_at_ring={n_lab}, "
              f"flow_coherence={coh:.3f}, cellprob_mean={cp_mean:.3f}, t={dt:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    print("\n=== summary ===")
    print(df.to_string(index=False))

    # =========== FIGURE 1: size-sweep master grid (Phase 1) ===========
    # Rows = sizes, cols = [DAPI, 18S, masks, flow_y, flow_x, cellprob]
    ZOOM_PX = 100; half_z = ZOOM_PX // 2
    cols = ["DAPI (raw)", "18S (raw)", "masks", "flow_y", "flow_x", "cellprob"]
    fig, axes = plt.subplots(len(SIZES_PX), len(cols),
                              figsize=(3 * len(cols), 3 * len(SIZES_PX)))
    for i, sp in enumerate(SIZES_PX):
        arr = np.load(OUT_DIR / f"phase1_size{sp}.npz")
        masks = arr["masks"]; flow_y = arr["flow_y"]; flow_x = arr["flow_x"]
        cellprob = arr["cellprob"]; dapi_r = arr["dapi_raw"]; s18_r = arr["s18_raw"]
        lx, ly = arr["ring_local"]
        H, W = masks.shape
        zy0 = max(ly - half_z, 0); zy1 = min(ly + half_z, H)
        zx0 = max(lx - half_z, 0); zx1 = min(lx + half_z, W)
        n_lab = count_labels_at(masks, ly, lx)
        # 0: DAPI raw — auto-scale each panel since absolute values differ widely
        axes[i, 0].imshow(dapi_r[zy0:zy1, zx0:zx1], cmap="gray")
        axes[i, 0].set_ylabel(f"size={sp}px\nlabels at ring = {n_lab}",
                              fontsize=10, rotation=0, ha="right", va="center")
        # 1: 18S raw
        axes[i, 1].imshow(s18_r[zy0:zy1, zx0:zx1], cmap="gray")
        # 2: mask overlay
        axes[i, 2].imshow(s18_r[zy0:zy1, zx0:zx1], cmap="gray")
        axes[i, 2].imshow(label_overlay(masks[zy0:zy1, zx0:zx1]))
        # 3: flow_y
        vmax = max(abs(flow_y[zy0:zy1, zx0:zx1]).max(), 1e-3)
        axes[i, 3].imshow(flow_y[zy0:zy1, zx0:zx1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        # 4: flow_x
        vmax = max(abs(flow_x[zy0:zy1, zx0:zx1]).max(), 1e-3)
        axes[i, 4].imshow(flow_x[zy0:zy1, zx0:zx1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        # 5: cellprob
        axes[i, 5].imshow(cellprob[zy0:zy1, zx0:zx1], cmap="viridis")
        for j in range(len(cols)):
            axes[i, j].plot(lx - zx0, ly - zy0, "+", color="red", markersize=14, markeredgewidth=2)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        if i == 0:
            for j, c in enumerate(cols):
                axes[i, j].set_title(c, fontsize=11)
    plt.suptitle(f"PHASE 1 — same 100-px window at WSI ring ({RING_WSI_X}, {RING_WSI_Y}) "
                 f"across ROI sizes; red + = ring center\n"
                 "default settings (augment=True, normalize=True, niter=200, bsize=256)",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_size_sweep_flows.png", dpi=130, bbox_inches="tight")
    plt.close()

    # =========== FIGURE 2: PHASE 2 — additional knobs at 4096 vs baseline 4096 ===========
    p2_files = [
        ("4096 BASELINE (Phase 1)", OUT_DIR / "phase1_size4096.npz"),
        ("4096 + bsize=512",        OUT_DIR / "phase2A_bsize512.npz"),
        ("4096 + niter=500",        OUT_DIR / "phase2B_niter500.npz"),
        ("4096 + norm-from-256",    OUT_DIR / "phase2C_norm_from_256.npz"),
        ("256 (reference, Phase 1)", OUT_DIR / "phase1_size256.npz"),
    ]
    fig, axes = plt.subplots(len(p2_files), len(cols),
                              figsize=(3 * len(cols), 3 * len(p2_files)))
    for i, (title, f) in enumerate(p2_files):
        arr = np.load(f)
        masks = arr["masks"]; flow_y = arr["flow_y"]; flow_x = arr["flow_x"]
        cellprob = arr["cellprob"]; dapi_r = arr["dapi_raw"]; s18_r = arr["s18_raw"]
        lx, ly = arr["ring_local"]
        H, W = masks.shape
        zy0 = max(ly - half_z, 0); zy1 = min(ly + half_z, H)
        zx0 = max(lx - half_z, 0); zx1 = min(lx + half_z, W)
        n_lab = count_labels_at(masks, ly, lx)
        axes[i, 0].imshow(dapi_r[zy0:zy1, zx0:zx1], cmap="gray")
        axes[i, 0].set_ylabel(f"{title}\nlabels = {n_lab}",
                              fontsize=9, rotation=0, ha="right", va="center")
        axes[i, 1].imshow(s18_r[zy0:zy1, zx0:zx1], cmap="gray")
        axes[i, 2].imshow(s18_r[zy0:zy1, zx0:zx1], cmap="gray")
        axes[i, 2].imshow(label_overlay(masks[zy0:zy1, zx0:zx1]))
        vmax = max(abs(flow_y[zy0:zy1, zx0:zx1]).max(), 1e-3)
        axes[i, 3].imshow(flow_y[zy0:zy1, zx0:zx1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        vmax = max(abs(flow_x[zy0:zy1, zx0:zx1]).max(), 1e-3)
        axes[i, 4].imshow(flow_x[zy0:zy1, zx0:zx1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i, 5].imshow(cellprob[zy0:zy1, zx0:zx1], cmap="viridis")
        for j in range(len(cols)):
            axes[i, j].plot(lx - zx0, ly - zy0, "+", color="red", markersize=14, markeredgewidth=2)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        if i == 0:
            for j, c in enumerate(cols):
                axes[i, j].set_title(c, fontsize=11)
    plt.suptitle("PHASE 2 — extra knobs at size=4096 vs baseline 4096 vs reference 256\n"
                 f"100-px window at WSI ring ({RING_WSI_X}, {RING_WSI_Y}); red + = ring center",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_phase2_knobs.png", dpi=130, bbox_inches="tight")
    plt.close()

    # =========== FIGURE 3: metric trends vs size ===========
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sub = df[df["phase"] == "size_sweep"].sort_values("size_px")
    axes[0].plot(sub["size_px"], sub["n_labels_at_ring"], "o-", color="#d62728",
                 linewidth=2, markersize=10)
    axes[0].set_xscale("log"); axes[0].set_xlabel("ROI size (px)")
    axes[0].set_ylabel("# labels at ring (40-px window)")
    axes[0].axhline(1, color="grey", linestyle=":", alpha=0.5, label="goal: 1")
    axes[0].set_title("Ring severity vs ROI size (Phase 1)"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(sub["size_px"], sub["flow_coherence"], "o-", color="#2ca02c",
                 linewidth=2, markersize=10)
    axes[1].set_xscale("log"); axes[1].set_xlabel("ROI size (px)")
    axes[1].set_ylabel("flow coherence (0..1)")
    axes[1].set_title("Flow vector alignment at ring"); axes[1].grid(alpha=0.3)
    axes[2].plot(sub["size_px"], sub["cellprob_mean"], "o-", color="#1f77b4",
                 linewidth=2, markersize=10)
    axes[2].set_xscale("log"); axes[2].set_xlabel("ROI size (px)")
    axes[2].set_ylabel("mean cellprob at ring")
    axes[2].set_title("Cellprob at ring"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_metrics_vs_size.png", dpi=130, bbox_inches="tight")
    plt.close()

    (OUT_DIR / "results.json").write_text(json.dumps({
        "ring_wsi_px": [RING_WSI_X, RING_WSI_Y],
        "phase1_sizes": SIZES_PX,
        "phase2_runs": [name for name, _, _, _ in phase2_runs],
        "pct_by_size": pct_by_size,
        "rows": results,
    }, indent=2))
    print(f"\ntotal runtime: {(time.time() - t_total)/60:.1f} min")
    print(f"outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
