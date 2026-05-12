# refs/

Reading list for the PC-gradient-field / boundary-detection MVP in `sandbox/pc_gradient_field.ipynb`.

## Core (MVP foundations)

| File | Citation | Read | Why |
|---|---|---|---|
| `di_zenzo_1986.pdf` | Di Zenzo 1986, *A note on the gradient of a multi-image* (CVGIP) | All (2 pp) | Multi-channel gradient → 2×2 structure tensor → `\|\|J\|\|_F` (§1-2 of notebook) |
| `canny_1986.pdf` | Canny 1986, *A computational approach to edge detection* (TPAMI) | Sections on the 3-stage pipeline (skim the optimality proofs) | Smooth → gradient magnitude + direction → NMS + hysteresis. The framework everything else slots into. |
| `martin_fowlkes_malik_2004.pdf` | Martin, Fowlkes, Malik 2004, *Learning to detect natural image boundaries* (TPAMI) | **Section 2 only** | Pb-style half-disk χ² gradient — the MVP in §3 of the notebook. Skip the classifier-training part of the paper. |

## Primer (read first, if you only read one)

| Source | What |
|---|---|
| Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed) — **Chapter 7** | One-stop primer that pulls together everything above. Free PDF available at <https://szeliski.org/Book/> (requires a quick form). ~30 pages. |

Not included in this folder because Szeliski's website asks for an email before serving the PDF — fill the form once, save it locally if you want it offline.

## Extensions (one rabbit hole per question)

| File | Citation | Question it answers |
|---|---|---|
| `lindeberg_1998.pdf` | Lindeberg 1998, *Edge detection and ridge detection with automatic scale selection* (IJCV) | How do I pick the radius `r` automatically per pixel? |
| `arbelaez_2011.pdf` | Arbeláez, Maire, Fowlkes, Malik 2011, *Contour Detection and Hierarchical Image Segmentation* (TPAMI) | What does the classical state-of-the-art (gPb-OWT-UCM) look like? **Read Sections 3-4; skip the OWT-UCM hierarchical part on first pass.** |
| `xie_tu_2015_hed.pdf` | Xie & Tu 2015, *Holistically-Nested Edge Detection* (ICCV) | What's the deep-learning equivalent? Short, accessible, deep version of the same pipeline. |

## Suggested sequence

| Day | Read |
|---|---|
| 1 | Szeliski Ch. 7 (primer) |
| 2 | `di_zenzo_1986.pdf` + `canny_1986.pdf` skim |
| 3 | `martin_fowlkes_malik_2004.pdf` §2 |
| 4 (optional) | One of the extensions, depending on your next question |
