"""Re-do the 1024-vs-2048 flow comparison using an INTERIOR doublet that has
margin ≥ 2048 px from all WSI edges, so crop_centered cannot clamp at either
ROI size. Includes a MANDATORY sanity check: the 100×100 18S window at the
crop center must be bit-identical between 1024 and 2048 crops — if it isn't,
we abort before drawing any conclusions from the flow diff.

Focal: DB_top100_003 (rank 3, Fibroblast × Melanoma, cps_id=12293)
       at WSI px (33129, 3356) — 9619 px from right edge, 16938 px from bottom.
       Well outside the 1024-px margin needed at both ROI sizes.

Outputs:
  figs/compare_flows_interior/flows_{1024,2048}.npz
  figs/compare_flows_interior/sanity_check.txt   — patches identical or not
  figs/compare_flows_interior/compare_flows.png  — same layout as before
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models

ROOT = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium")
DATA = ROOT / "data" / "Xenium_Prime_Human_Skin_FFPE_xe_outs"
WSI_DAPI = DATA / "morphology_focus" / "morphology_focus_0000.ome.tif"
WSI_18S  = DATA / "morphology_focus" / "morphology_focus_0002.ome.tif"
WSI_MASKS= DATA / "cpsam_whole_slide" / "masks.tif"
WSI_PCT_JSON = DATA / "wsi_morphology_percentiles.json"

OUT_DIR = ROOT / "figs" / "compare_flows_interior"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DB_top100_003 — interior doublet, no edge clamping at 2048
CX_PX, CY_PX = 33129, 3356
CPS_FOCAL    = 12293
COMPARE_HALF = 50
ROI_SIZES    = [1024, 2048]
CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=False, normalize=False,
                tile_overlap=0.05)


def crop_centered_strict(arr, cx_px, cy_px, size_px):
    """No clamping — assert the crop fits. Returns crop + bbox."""
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = cy_px - half; y1 = y0 + size_px
    x0 = cx_px - half; x1 = x0 + size_px
    assert 0 <= y0 and y1 <= H, f"y0={y0}, y1={y1}, H={H} — crop would clamp!"
    assert 0 <= x0 and x1 <= W, f"x0={x0}, x1={x1}, W={W} — crop would clamp!"
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def norm_with(arr, q_lo, q_hi):
    a = arr.astype(np.float32)
    return ((a - q_lo) / max(q_hi - q_lo, 1e-6)).astype(np.float32)


def main():
    t_total = time.time()
    print("loading WSI…")
    dapi_full = tifffile.imread(WSI_DAPI, key=0)
    s18_full  = tifffile.imread(WSI_18S, key=0)
    wsi_masks = tifffile.imread(WSI_MASKS)
    wsi_pct   = json.loads(WSI_PCT_JSON.read_text())
    print(f"  WSI shape: {dapi_full.shape}")
    print(f"  focal cell WSI px: ({CX_PX}, {CY_PX})")
    print(f"  margin to nearest edge: x={min(CX_PX, dapi_full.shape[1]-CX_PX)}, "
          f"y={min(CY_PX, dapi_full.shape[0]-CY_PX)}")

    # ============ MANDATORY SANITY CHECK ============
    crops = {}
    for sz in ROI_SIZES:
        dapi_c, bbox_d = crop_centered_strict(dapi_full, CX_PX, CY_PX, sz)
        s18_c,  bbox_s = crop_centered_strict(s18_full,  CX_PX, CY_PX, sz)
        crops[sz] = dict(dapi=dapi_c, s18=s18_c, bbox=bbox_d)
        print(f"  ROI={sz}: bbox WSI px = {bbox_d}")
        cx_local = CX_PX - bbox_d[0]; cy_local = CY_PX - bbox_d[1]
        print(f"    focal cell at local ({cx_local}, {cy_local})  "
              f"(should be ({sz//2}, {sz//2}))")
        assert (cx_local, cy_local) == (sz // 2, sz // 2), "Focal not centered!"

    # Compare the 100×100 raw 18S window between the two crops
    h = COMPARE_HALF
    s18_1024_win = crops[1024]["s18"][crops[1024]["s18"].shape[0]//2 - h:crops[1024]["s18"].shape[0]//2 + h,
                                       crops[1024]["s18"].shape[1]//2 - h:crops[1024]["s18"].shape[1]//2 + h]
    s18_2048_win = crops[2048]["s18"][crops[2048]["s18"].shape[0]//2 - h:crops[2048]["s18"].shape[0]//2 + h,
                                       crops[2048]["s18"].shape[1]//2 - h:crops[2048]["s18"].shape[1]//2 + h]
    patch_diff = np.abs(s18_1024_win.astype(np.int64) - s18_2048_win.astype(np.int64))
    max_diff = int(patch_diff.max())
    sanity_msg = (f"100x100 raw 18S window centered on focal cell:\n"
                  f"  1024 crop: shape={s18_1024_win.shape}, min={s18_1024_win.min()}, "
                  f"max={s18_1024_win.max()}, mean={s18_1024_win.mean():.2f}\n"
                  f"  2048 crop: shape={s18_2048_win.shape}, min={s18_2048_win.min()}, "
                  f"max={s18_2048_win.max()}, mean={s18_2048_win.mean():.2f}\n"
                  f"  max |diff| = {max_diff} (MUST be 0)")
    print("\n=== SANITY CHECK ===\n" + sanity_msg)
    (OUT_DIR / "sanity_check.txt").write_text(sanity_msg)
    if max_diff != 0:
        print("\n*** SANITY CHECK FAILED — aborting before cellpose runs ***")
        sys.exit(1)
    print("\nSanity check passed — patches bit-identical. Proceeding to cellpose.")

    # ============ run cellpose at both ROIs ============
    print("\nloading cellpose-SAM…")
    m_sam = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    saved = {}
    for sz in ROI_SIZES:
        print(f"\n--- ROI = {sz} ---")
        dapi_n = norm_with(crops[sz]["dapi"], **wsi_pct["DAPI"])
        s18_n  = norm_with(crops[sz]["s18"],  **wsi_pct["18S"])
        img = np.stack([dapi_n, s18_n], axis=-1).astype(np.float32)
        t0 = time.time()
        masks, flows, _ = m_sam.eval(img, channel_axis=-1, **CPSAM_KW)
        dt = time.time() - t0
        dP = flows[1].astype(np.float32)
        cellprob = flows[2].astype(np.float32)
        # count n_focal masks (cover ≥5% of WSI doublet)
        wsi_c, _ = crop_centered_strict(wsi_masks, CX_PX, CY_PX, sz)
        focal_region = (wsi_c == CPS_FOCAL)
        focal_area = int(focal_region.sum())
        lbls = np.unique(masks[focal_region]); lbls = lbls[lbls != 0]
        covers = sorted(
            [(int(L), int(((masks==L)&focal_region).sum())/focal_area) for L in lbls],
            key=lambda kv: -kv[1])
        n_focal = sum(1 for _, c in covers if c >= 0.05)
        print(f"  n_total={int(masks.max())}, n_focal={n_focal}, "
              f"covers=[{', '.join(f'{c*100:.0f}%' for _, c in covers[:5])}], runtime={dt:.1f}s")
        saved[sz] = dict(dP=dP, cellprob=cellprob, masks=masks,
                          s18_n=s18_n, focal_region=focal_region, n_focal=n_focal)
        np.savez_compressed(OUT_DIR / f"flows_{sz}.npz",
                             dP=dP, cellprob=cellprob, masks=masks)

    # ============ pixel-wise diff in focal window ============
    cy1 = saved[1024]["dP"].shape[1] // 2; cx1 = saved[1024]["dP"].shape[2] // 2
    cy2 = saved[2048]["dP"].shape[1] // 2; cx2 = saved[2048]["dP"].shape[2] // 2
    dP_1024_win = saved[1024]["dP"][:, cy1-h:cy1+h, cx1-h:cx1+h]
    dP_2048_win = saved[2048]["dP"][:, cy2-h:cy2+h, cx2-h:cx2+h]
    cp_1024_win = saved[1024]["cellprob"][cy1-h:cy1+h, cx1-h:cx1+h]
    cp_2048_win = saved[2048]["cellprob"][cy2-h:cy2+h, cx2-h:cx2+h]
    dY_d = (dP_2048_win[0] - dP_1024_win[0])
    dX_d = (dP_2048_win[1] - dP_1024_win[1])
    cp_d = (cp_2048_win - cp_1024_win)

    print(f"\n=== pixel-wise diff in 100x100 focal window (INTERIOR doublet, no clamping) ===")
    print(f"  dY: max|Δ|={float(np.abs(dY_d).max()):.4f}  mean|Δ|={float(np.abs(dY_d).mean()):.5f}")
    print(f"  dX: max|Δ|={float(np.abs(dX_d).max()):.4f}  mean|Δ|={float(np.abs(dX_d).mean()):.5f}")
    print(f"  cp: max|Δ|={float(np.abs(cp_d).max()):.4f}  mean|Δ|={float(np.abs(cp_d).mean()):.5f}")
    print(f"  cp range 1024: [{cp_1024_win.min():.2f}, {cp_1024_win.max():.2f}]")
    print(f"  cp range 2048: [{cp_2048_win.min():.2f}, {cp_2048_win.max():.2f}]")
    print(f"  n_focal 1024={saved[1024]['n_focal']}, n_focal 2048={saved[2048]['n_focal']}")

    # ============ figure with the sanity check + flow diff ============
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    s18_1024_n = saved[1024]["s18_n"]; s18_2048_n = saved[2048]["s18_n"]
    s18_1024_disp = s18_1024_n[cy1-h:cy1+h, cx1-h:cx1+h]
    s18_2048_disp = s18_2048_n[cy2-h:cy2+h, cx2-h:cx2+h]
    s18_diff = s18_2048_disp - s18_1024_disp

    # row 0: raw 18S + masks contours, plus identical-image proof
    axes[0,0].imshow(s18_1024_disp, cmap="gray")
    axes[0,0].contour(saved[1024]["masks"][cy1-h:cy1+h, cx1-h:cx1+h].astype(int),
                      colors="white", linewidths=0.5)
    axes[0,0].set_title("ROI 1024: 18S + mask contours")
    axes[0,1].imshow(s18_2048_disp, cmap="gray")
    axes[0,1].contour(saved[2048]["masks"][cy2-h:cy2+h, cx2-h:cx2+h].astype(int),
                      colors="white", linewidths=0.5)
    axes[0,1].set_title("ROI 2048: 18S + mask contours")
    v = max(abs(s18_diff).max(), 1e-6)
    axes[0,2].imshow(np.abs(s18_diff), cmap="hot", vmin=0, vmax=v)
    axes[0,2].set_title(f"|18S Δ|  (max={abs(s18_diff).max():.4f} — should be 0)")
    axes[0,3].axis("off")

    # rows 1,2: dY, dX comparison
    for r, (lbl, a, b, d) in enumerate([
        ("dY", dP_1024_win[0], dP_2048_win[0], dY_d),
        ("dX", dP_1024_win[1], dP_2048_win[1], dX_d),
    ]):
        v_ab = max(abs(a).max(), abs(b).max(), 1e-3)
        axes[r+1,0].imshow(a, cmap="RdBu_r", vmin=-v_ab, vmax=v_ab)
        axes[r+1,0].set_title(f"ROI 1024: {lbl}")
        axes[r+1,1].imshow(b, cmap="RdBu_r", vmin=-v_ab, vmax=v_ab)
        axes[r+1,1].set_title(f"ROI 2048: {lbl}")
        v_d = max(abs(d).max(), 1e-3)
        axes[r+1,2].imshow(np.abs(d), cmap="hot")
        axes[r+1,2].set_title(f"|Δ{lbl}| max={float(np.abs(d).max()):.3f} mean={float(np.abs(d).mean()):.4f}")
        axes[r+1,3].imshow(d, cmap="RdBu_r", vmin=-v_d, vmax=v_d)
        axes[r+1,3].set_title(f"Δ{lbl} (signed)")

    # row 3: cellprob
    v_ab = max(abs(cp_1024_win).max(), abs(cp_2048_win).max(), 1e-3)
    axes[3,0].imshow(cp_1024_win, cmap="viridis", vmin=-v_ab, vmax=v_ab)
    axes[3,0].set_title(f"ROI 1024: cellprob")
    axes[3,1].imshow(cp_2048_win, cmap="viridis", vmin=-v_ab, vmax=v_ab)
    axes[3,1].set_title(f"ROI 2048: cellprob")
    v_d = max(abs(cp_d).max(), 1e-3)
    axes[3,2].imshow(np.abs(cp_d), cmap="hot")
    axes[3,2].set_title(f"|Δcp| max={float(np.abs(cp_d).max()):.3f} mean={float(np.abs(cp_d).mean()):.4f}")
    axes[3,3].imshow(cp_d, cmap="RdBu_r", vmin=-v_d, vmax=v_d)
    axes[3,3].set_title("Δcellprob (signed)")

    for r in range(4):
        for c in range(4):
            axes[r,c].set_xticks([]); axes[r,c].set_yticks([])

    plt.suptitle(f"INTERIOR doublet DB_top100_003 (rank 3, Fib×Mel, WSI px ({CX_PX},{CY_PX})) — "
                 f"100×100 focal window\n"
                 f"NO crop-clamping at either ROI; 18S patches verified bit-identical; "
                 f"only ROI size differs.  n_focal: 1024={saved[1024]['n_focal']}, 2048={saved[2048]['n_focal']}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "compare_flows.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {OUT_DIR/'compare_flows.png'}")
    print(f"total: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
