"""
Visualisation and saving helpers for the MRI pipeline outputs.
"""

import numpy as np
from PIL import Image

from .core import WATERSHED


def colorise_labels(labels):
    """
    Give every basin a random (seeded, reproducible) colour.

    Watershed lines (0) and unlabelled pixels (-1) are drawn black.
    """
    rng = np.random.default_rng(seed=42)
    max_id = int(labels.max())
    pal = rng.integers(60, 230, size=(max_id + 1, 3), dtype=np.uint8)
    pal[0] = [0, 0, 0]
    return pal[np.maximum(labels, 0)]


def draw_markers(g_u8, markers, E_brain=None):
    """
    Draw the seeds on the gradient image.

    Markers are green 5x5 squares; the optional boundary-ring mask E_brain
    (medical pipeline) is drawn in red.
    """
    rgb = np.stack([g_u8] * 3, axis=2).copy()
    if E_brain is not None:
        rgb[E_brain] = [220, 30, 30]
    rows, cols = g_u8.shape
    for r, c in markers:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    rgb[nr, nc] = [0, 220, 0]
    return rgb


def overlay_boundaries(original_rgb, labels, colour=(220, 30, 30)):
    """Paint the watershed lines on top of the original image."""
    out = original_rgb.copy()
    out[labels == WATERSHED] = colour
    return out


def save_mask_like_gt(pred, path):
    """Save a 0/255 mask as an opaque RGBA PNG — the same format as the GT masks."""
    rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    rgba[..., :3] = pred[..., np.newaxis]
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(path)   # (H, W, 4) uint8 is inferred as RGBA
