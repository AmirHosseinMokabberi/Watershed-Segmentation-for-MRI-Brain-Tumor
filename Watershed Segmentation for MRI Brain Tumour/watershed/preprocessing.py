"""
Image pre-processing: Gaussian smoothing, contrast enhancement and the
gradient-magnitude image that the watershed floods.
"""

import numpy as np
from scipy.ndimage import sobel as nd_sobel
from skimage.exposure import equalize_adapthist


# ── Gaussian smoothing (from scratch) ────────────────────────────────────────

def gaussian_kernel_1d(sigma):
    """1-D Gaussian kernel truncated at ceil(3·sigma) per side, L1-normalised."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_smooth(image, sigma):
    """
    Separable 2-D Gaussian blur: filter the rows, then the columns.

    Borders are reflect-padded so the output has the same shape as the input.
    """
    k = gaussian_kernel_1d(sigma)
    radius = len(k) // 2
    rows, cols = image.shape
    img = image.astype(float)

    # Horizontal pass
    padded = np.pad(img, ((0, 0), (radius, radius)), mode='reflect')
    tmp = np.array([
        [float(np.dot(padded[r, c:c + len(k)], k)) for c in range(cols)]
        for r in range(rows)
    ])

    # Vertical pass
    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode='reflect')
    return np.array([
        [float(np.dot(padded[r:r + len(k), c], k)) for c in range(cols)]
        for r in range(rows)
    ])


# ── Contrast enhancement ─────────────────────────────────────────────────────

def clahe(gray_f):
    """
    CLAHE (contrast-limited adaptive histogram equalisation) on a 0-255 image.

    Local equalisation strengthens weak tumour boundaries without saturating
    bright regions. Returns float32 in the 0-255 range.
    """
    return (equalize_adapthist(gray_f / 255.0) * 255.0).astype(np.float32)


# ── Gradient magnitude (the flood surface) ───────────────────────────────────

def sobel_gradient_u8(gray):
    """
    Sobel gradient magnitude g = sqrt(gx² + gy²), min-max scaled to uint8.

    High values mark edges (ridges of the relief); homogeneous tissue gives
    low values (valleys). A constant image yields an all-zero gradient.
    """
    gx = nd_sobel(gray, axis=1)
    gy = nd_sobel(gray, axis=0)
    gradient = np.hypot(gx, gy)
    g_min, g_max = gradient.min(), gradient.max()
    if g_max > g_min:
        return ((gradient - g_min) / (g_max - g_min) * 255).astype(np.uint8)
    return np.zeros_like(gradient, dtype=np.uint8)
