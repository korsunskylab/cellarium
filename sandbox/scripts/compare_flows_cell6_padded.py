"""Re-do the 1024-vs-2048 flow comparison for cell #6 specifically, with
reflection padding on the WSI right edge so the 2048 crop can be properly
centered (no clamping). The focal cell sits in the LEFT half of the padded
2048 crop, comfortably inside a single tile; the reflection-pad pixels sit
in the RIGHT-side tiles, far from the focal cell.

Includes the mandatory sanity check: the 100x100 raw 18S window at the
crop center MUST be bit-identical between 1024 and padded-2048 crops. If
it is identical AND the flows at the focal cell still match, then cell #6's
"per-tile prediction depends on ROI size" finding from before was entirely
a clamping artifact.

Focal: WSI px (41817, 6957) — cell #6 (cps_id=45463).
Padding: WSI right edge extended by 1100 px (reflection) so 2048 crop fits
         centered on cell #6 with 169 px to spare.
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

OUT_DIR = ROOT / "figs" / "compare_flows_cell6_padded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CX_PX, CY_PX = 41817, 6957
CPS_FOCAL    = 45463
PAD_X_RIGHT  = 1100              # right-edge reflection pad
COMPARE_HALF = 50
ROI_SIZES    = [1024, 2048]
CPSAM_KW = dict(diameter=None, niter=200, cellprob_threshold=-5.0,
                flow_threshold=0.0, augment=False, normalize=False,
                tile_overlap=0.05)


def crop_centered_strict(arr, cx_px, cy_px, size_px):
    """No clamping — assert the crop fits."""
    half = size_px // 2
    H, W = arr.shape[-2:]
    y0 = cy_px - half; y1 = y0 + size_px
    x0 = cx_px - half; x1 = x0 + size_px
    assert 0 <= y0 and y1 <= H, f"y clamp: {y0=}, {y1=}, {H=}"
    assert 0 <= x0 and x1 <= W, f"x clamp: {x0=}, {x1=}, {W=}"
    return arr[..., y0:y1, x0:x1], (x0, y0, x1, y1)


def reflect_pad_right(arr, pad_x):
    """Pad the last axis (x) on the right side by pad_x pixels with reflection."""
    # arr shape is (H, W) — reflect pad on x only (right side)
    if arr.ndim == 2:
        return np.pad(arr, ((0, 0), (0, pad_x)), mode="reflect")
    elif arr.ndim == 3:
        return np.pad(arr, ((0, 0), (0, 0), (0, pad_x)), mode="reflect")
    raise ValueError("expected 2D or 3D")


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
    H, W = dapi_full.shape
    print(f"  WSI shape: {dapi_full.shape}")
    print(f"  focal cell at ({CX_PX}, {CY_PX});  margin to right edge: {W - CX_PX}")
    print(f"  applying reflection pad of {PAD_X_RIGHT} px to the right")

    dapi_p = reflect_pad_right(dapi_full, PAD_X_RIGHT)
    s18_p  = reflect_pad_right(s18_full,  PAD_X_RIGHT)
    masks_p = reflect_pad_right(wsi_masks, PAD_X_RIGHT)
    print(f"  padded WSI shape: {dapi_p.shape}")

    # ============ MANDATORY SANITY CHECK ============
    crops = {}
    for sz in ROI_SIZES:
        d_c, bbox = crop_centered_strict(dapi_p, CX_PX, CY_PX, sz)
        s_c, _    = crop_centered_strict(s18_p,  CX_PX, CY_PX, sz)
        crops[sz] = dict(dapi=d_c, s18=s_c, bbox=bbox)
        cx_loc = CX_PX - bbox[0]; cy_loc = CY_PX - bbox[1]
        print(f"  ROI={sz}: bbox={bbox}, focal local=({cx_loc},{cy_loc})  "
              f"(want ({sz//2},{sz//2}))")
        assert (cx_loc, cy_loc) == (sz//2, sz//2)

    h = COMPARE_HALF
    s18_1024_w = crops[1024]["s18"][512-h:512+h, 512-h:512+h]
    s18_2048_w = crops[2048]["s18"][1024-h:1024+h, 1024-h:1024+h]
    diff = np.abs(s18_1024_w.astype(np.int64) - s18_2048_w.astype(np.int64))
    max_diff = int(diff.max())
    print(f"\n=== SANITY CHECK: 100x100 raw 18S at focal cell ===")
    print(f"  1024 window: min={s18_1024_w.min()}, max={s18_1024_w.max()}, mean={s18_1024_w.mean():.2f}")
    print(f"  2048 window: min={s18_2048_w.min()}, max={s18_2048_w.max()}, mean={s18_2048_w.mean():.2f}")
    print(f"  max |diff| = {max_diff}  (MUST be 0)")
    if max_diff != 0:
        print("\n*** SANITY CHECK FAILED ***")
        sys.exit(1)
    print("  ✓ patches bit-identical")

    # ============ cellpose ============
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
        wsi_c, _ = crop_centered_strict(masks_p, CX_PX, CY_PX, sz)
        focal_region = (wsi_c == CPS_FOCAL)
        focal_area = int(focal_region.sum())
        lbls = np.unique(masks[focal_region]); lbls = lbls[lbls!=0]
        covers = sorted([(int(L), int(((masks==L)&focal_region).sum())/focal_area) for L in lbls],
                         key=lambda kv: -kv[1])
        n_focal = sum(1 for _, c in covers if c >= 0.05)
        print(f"  n_total={int(masks.max())}, n_focal={n_focal}, "
              f"covers=[{', '.join(f'{c*100:.0f}%' for _, c in covers[:5])}], "
              f"runtime={dt:.1f}s")
        saved[sz] = dict(dP=dP, cellprob=cellprob, masks=masks,
                          s18_n=s18_n, n_focal=n_focal)
        np.savez_compressed(OUT_DIR / f"flows_{sz}.npz",
                             dP=dP, cellprob=cellprob, masks=masks)

    cy1 = saved[1024]["dP"].shape[1]//2; cx1 = saved[1024]["dP"].shape[2]//2
    cy2 = saved[2048]["dP"].shape[1]//2; cx2 = saved[2048]["dP"].shape[2]//2
    dP1 = saved[1024]["dP"][:, cy1-h:cy1+h, cx1-h:cx1+h]
    dP2 = saved[2048]["dP"][:, cy2-h:cy2+h, cx2-h:cx2+h]
    cp1 = saved[1024]["cellprob"][cy1-h:cy1+h, cx1-h:cx1+h]
    cp2 = saved[2048]["cellprob"][cy2-h:cy2+h, cx2-h:cx2+h]
    dY_d = dP2[0] - dP1[0]; dX_d = dP2[1] - dP1[1]; cp_d = cp2 - cp1

    print(f"\n=== flow diff at cell #6 (100x100 focal window, properly centered) ===")
    print(f"  dY:        max|Δ|={float(np.abs(dY_d).max()):.4f}  mean|Δ|={float(np.abs(dY_d).mean()):.5f}")
    print(f"  dX:        max|Δ|={float(np.abs(dX_d).max()):.4f}  mean|Δ|={float(np.abs(dX_d).mean()):.5f}")
    print(f"  cellprob:  max|Δ|={float(np.abs(cp_d).max()):.4f}  mean|Δ|={float(np.abs(cp_d).mean()):.5f}")
    print(f"  cp range 1024: [{cp1.min():.2f}, {cp1.max():.2f}]")
    print(f"  cp range 2048: [{cp2.min():.2f}, {cp2.max():.2f}]")
    print(f"  n_focal: 1024={saved[1024]['n_focal']}, 2048={saved[2048]['n_focal']}")

    # figure
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    s18_1024_disp = saved[1024]["s18_n"][cy1-h:cy1+h, cx1-h:cx1+h]
    s18_2048_disp = saved[2048]["s18_n"][cy2-h:cy2+h, cx2-h:cx2+h]
    s18_diff = s18_2048_disp - s18_1024_disp
    axes[0,0].imshow(s18_1024_disp, cmap="gray")
    axes[0,0].contour(saved[1024]["masks"][cy1-h:cy1+h, cx1-h:cx1+h].astype(int), colors="white", linewidths=0.5)
    axes[0,0].set_title("ROI 1024: 18S + mask contours")
    axes[0,1].imshow(s18_2048_disp, cmap="gray")
    axes[0,1].contour(saved[2048]["masks"][cy2-h:cy2+h, cx2-h:cx2+h].astype(int), colors="white", linewidths=0.5)
    axes[0,1].set_title("ROI 2048 (right-padded): 18S + mask contours")
    axes[0,2].imshow(np.abs(s18_diff), cmap="hot")
    axes[0,2].set_title(f"|18S Δ|  (max={float(np.abs(s18_diff).max()):.4f} — should be 0)")
    axes[0,3].axis("off")
    for r, (lbl, a, b, d) in enumerate([
        ("dY", dP1[0], dP2[0], dY_d),
        ("dX", dP1[1], dP2[1], dX_d),
    ]):
        v_ab = max(abs(a).max(), abs(b).max(), 1e-3)
        axes[r+1,0].imshow(a, cmap="RdBu_r", vmin=-v_ab, vmax=v_ab)
        axes[r+1,0].set_title(f"ROI 1024: {lbl}")
        axes[r+1,1].imshow(b, cmap="RdBu_r", vmin=-v_ab, vmax=v_ab)
        axes[r+1,1].set_title(f"ROI 2048 (padded): {lbl}")
        v_d = max(abs(d).max(), 1e-3)
        axes[r+1,2].imshow(np.abs(d), cmap="hot")
        axes[r+1,2].set_title(f"|Δ{lbl}|  max={float(np.abs(d).max()):.3f}  mean={float(np.abs(d).mean()):.4f}")
        axes[r+1,3].imshow(d, cmap="RdBu_r", vmin=-v_d, vmax=v_d)
        axes[r+1,3].set_title(f"Δ{lbl} (signed)")
    v_ab = max(abs(cp1).max(), abs(cp2).max(), 1e-3)
    axes[3,0].imshow(cp1, cmap="viridis", vmin=-v_ab, vmax=v_ab); axes[3,0].set_title("ROI 1024: cellprob")
    axes[3,1].imshow(cp2, cmap="viridis", vmin=-v_ab, vmax=v_ab); axes[3,1].set_title("ROI 2048 (padded): cellprob")
    v_d = max(abs(cp_d).max(), 1e-3)
    axes[3,2].imshow(np.abs(cp_d), cmap="hot"); axes[3,2].set_title(f"|Δcellprob|  max={float(np.abs(cp_d).max()):.3f}  mean={float(np.abs(cp_d).mean()):.4f}")
    axes[3,3].imshow(cp_d, cmap="RdBu_r", vmin=-v_d, vmax=v_d); axes[3,3].set_title("Δcellprob (signed)")
    for r in range(4):
        for c in range(4): axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
    plt.suptitle(f"Cell #6 (WSI px (41817, 6957)) — 1024 vs 2048 ROI with WSI right edge "
                 f"reflection-padded by {PAD_X_RIGHT} px so 2048 crop centers properly on cell.\n"
                 f"Sanity-checked bit-identical 18S patches. "
                 f"n_focal: 1024={saved[1024]['n_focal']}, 2048={saved[2048]['n_focal']}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "compare_flows.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {OUT_DIR / 'compare_flows.png'}")
    print(f"total: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
