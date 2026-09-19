"""
watershed — a from-scratch watershed segmentation package.

    core           regional minima, flooding, minima imposition
    preprocessing  Gaussian smoothing, CLAHE, Sobel gradient
    markers        seed extraction (classic g-minima, medical split-merge + adaptive Otsu)
    tumour         choosing the tumour basin
    visualize      colouring, overlays and mask export
"""

from .core import (UNLABELED, WATERSHED, find_regional_maxima,
                   find_regional_minima, impose_minima,
                   marker_controlled_watershed, marker_watershed_image,
                   watershed_from_scratch)
from .markers import (build_markers_g, build_markers_medical,
                      build_markers_morph, split_merge_regions)
from .preprocessing import clahe, gaussian_smooth, sobel_gradient_u8
from .tumour import extract_tumor_mask, iou

__all__ = [
    "UNLABELED", "WATERSHED",
    "find_regional_minima", "find_regional_maxima", "watershed_from_scratch",
    "impose_minima", "marker_controlled_watershed", "marker_watershed_image",
    "build_markers_g", "build_markers_medical", "build_markers_morph",
    "split_merge_regions",
    "clahe", "gaussian_smooth", "sobel_gradient_u8",
    "extract_tumor_mask", "iou",
]
