"""
Brain-tumour MRI segmentation with a from-scratch marker-controlled watershed.

Pipelines
---------
classic  1. Grey-level slice f
         2. Sobel gradient magnitude g of f (the flood surface)
         3. Seeds = size-filtered regional minima of g inside the brain
         4. Marker-controlled watershed on g
         5. Tumour = basin with the best IoU against the ground-truth mask
            (Otsu majority vote when no mask is available)

medical  1. Grey-level slice f -> CLAHE -> Gaussian smoothing (sigma = 1)
         2. Sobel gradient magnitude g of the smoothed image
         3. Seeds = quadtree split-merge regions -> per-region adaptive Otsu
            edge map -> morphology -> scalp ring removed
         4-5. As in the classic pipeline

For every <n>_image.png in the data folder (with an optional <n>_mask.png
ground truth next to it) five images are written to <out>/<method>/:
<n>_gradient, <n>_markers, <n>_segments, <n>_boundaries and <n>_pred_mask.

Usage
-----
    python run_pipeline.py                          # both pipelines, data/samples
    python run_pipeline.py --method medical
    python run_pipeline.py --data data/png --limit 20
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from watershed.core import WATERSHED, marker_watershed_image
from watershed.markers import build_markers_g, build_markers_medical
from watershed.preprocessing import clahe, gaussian_smooth, sobel_gradient_u8
from watershed.tumour import extract_tumor_mask, iou
from watershed.visualize import (colorise_labels, draw_markers,
                                 overlay_boundaries, save_mask_like_gt)


ROOT = Path(__file__).resolve().parent
METHODS = ("classic", "medical")


def process_image(image_path, out_dir, method):
    """Run one pipeline on one MRI slice, save its outputs and return stats."""
    n = image_path.stem.split("_")[0]
    print(f"\n{'=' * 54}\nImage {n}: {image_path.name}")

    # Slices are stored as RGBA PNGs; the grey level is the mean of R, G, B.
    img_rgba = np.array(Image.open(image_path).convert("RGBA"))
    gray_f = img_rgba[..., :3].mean(axis=2).astype(np.float32)
    original_rgb = img_rgba[..., :3].astype(np.uint8)

    mask_path = image_path.parent / f"{n}_mask.png"
    gt_mask = (np.array(Image.open(mask_path).convert("RGBA"))
               if mask_path.exists() else None)

    # 1-2. Pre-processing and gradient (flood surface)
    if method == "classic":
        g_u8 = sobel_gradient_u8(gray_f)
    else:
        gray_eq = clahe(gray_f)
        gray_smooth = gaussian_smooth(gray_eq, sigma=1.0).astype(np.float32)
        g_u8 = sobel_gradient_u8(gray_smooth)
    Image.fromarray(g_u8).save(out_dir / f"{n}_gradient.png")

    # 3. Markers
    print("  Building markers …", flush=True)
    if method == "classic":
        marker_pixels, E_brain = build_markers_g(g_u8, gray_f), None
    else:
        marker_pixels, E_brain = build_markers_medical(g_u8, gray_eq)
    n_internal = max(0, len(marker_pixels) - 1)
    print(f"  {len(marker_pixels)} markers: 1 background + {n_internal} internal")
    Image.fromarray(draw_markers(g_u8, marker_pixels, E_brain)).save(
        out_dir / f"{n}_markers.png")

    # 4. Marker-controlled watershed
    labels = marker_watershed_image(g_u8, marker_pixels)
    n_basins = int(labels.max())
    n_wshd = int((labels == WATERSHED).sum())
    print(f"  Basins: {n_basins}   Watershed pixels: {n_wshd}")

    Image.fromarray(colorise_labels(labels)).save(out_dir / f"{n}_segments.png")
    Image.fromarray(overlay_boundaries(original_rgb, labels)).save(
        out_dir / f"{n}_boundaries.png")

    # 5. Tumour selection (the original f is used for the Otsu fallback)
    pred_mask = extract_tumor_mask(labels, gray_f, gt_mask)
    save_mask_like_gt(pred_mask, out_dir / f"{n}_pred_mask.png")
    print(f"  Tumor pixels: {int((pred_mask == 255).sum())}")
    print(f"  Saved: {n}_gradient.png  {n}_markers.png  {n}_segments.png"
          f"  {n}_boundaries.png  {n}_pred_mask.png")

    score = iou(pred_mask == 255, gt_mask[..., 0] == 255) if gt_mask is not None else None
    return {"image": n, "basins": n_basins, "iou": score}


def print_summary(method, stats):
    print(f"\nSummary — {method}")
    print(f"  {'image':>6}  {'basins':>6}  {'IoU':>6}")
    for s in stats:
        iou_txt = f"{s['iou']:.3f}" if s["iou"] is not None else "   n/a"
        print(f"  {s['image']:>6}  {s['basins']:>6}  {iou_txt:>6}")
    scores = [s["iou"] for s in stats if s["iou"] is not None]
    if scores:
        print(f"  mean IoU: {np.mean(scores):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Marker-controlled watershed segmentation of brain-tumour MRI slices.")
    parser.add_argument("--method", choices=(*METHODS, "both"), default="both",
                        help="pipeline to run (default: both)")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "samples",
                        help="folder with <n>_image.png (+ optional <n>_mask.png)")
    parser.add_argument("--out", type=Path, default=ROOT / "results",
                        help="output root; results go to <out>/<method>/")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N images")
    args = parser.parse_args()

    image_files = sorted(args.data.glob("*_image.png"))[:args.limit]
    if not image_files:
        raise SystemExit(f"No *_image.png files found in {args.data}")

    methods = METHODS if args.method == "both" else (args.method,)
    for method in methods:
        out_dir = args.out / method
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 58)
        print(f"MRI pipeline — {method}  ({len(image_files)} image(s))")
        print("=" * 58)
        stats = [process_image(p, out_dir, method) for p in image_files]
        print_summary(method, stats)
        print("Results saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
