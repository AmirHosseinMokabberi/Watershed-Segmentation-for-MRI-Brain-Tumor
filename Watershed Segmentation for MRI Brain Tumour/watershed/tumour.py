"""
Selecting the tumour among the watershed basins.
"""

import numpy as np
from skimage.filters import threshold_otsu


def iou(a, b):
    """Intersection-over-union of two boolean masks (0 if both are empty)."""
    union = (a | b).sum()
    return (a & b).sum() / union if union > 0 else 0.0


def extract_tumor_mask(labels, gray_f, gt_mask=None):
    """
    Build a binary tumour mask (0 / 255) from a watershed label map.

    Basin ID 1 is the background marker and is never selected.

    Priority 1 — ground truth available: select the single basin with the
        highest IoU against the ground-truth mask. This is an oracle choice
        that measures how well the best basin can match the tumour.
    Priority 2 — no ground truth (or no overlapping basin): mark every basin
        in which at least 50 % of the pixels are brighter than the global
        Otsu threshold of the image.
    """
    max_id = int(labels.max())
    pred = np.zeros(labels.shape, dtype=np.uint8)

    if gt_mask is not None:
        gt = gt_mask[..., 0] == 255
        best_iou, best_bid = 0.0, -1
        for bid in range(2, max_id + 1):
            basin = labels == bid
            if not basin.any():
                continue
            score = iou(basin, gt)
            if score > best_iou:
                best_iou, best_bid = score, bid
        if best_bid > 0 and best_iou > 0.0:
            pred[labels == best_bid] = 255
            print(f"  Tumor basin ID={best_bid}  IoU={best_iou:.3f}")
            return pred

    thresh = threshold_otsu(gray_f)
    above = gray_f > thresh
    for bid in range(2, max_id + 1):
        bm = labels == bid
        if bm.sum() > 0 and above[bm].mean() >= 0.50:
            pred[bm] = 255
    print(f"  Otsu threshold={thresh:.1f}  (no GT mask — majority vote)")
    return pred
