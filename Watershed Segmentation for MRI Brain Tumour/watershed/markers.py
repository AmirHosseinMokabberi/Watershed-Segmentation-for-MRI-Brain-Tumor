"""
Marker (seed) extraction for the marker-controlled watershed.

Every builder returns `marker_pixels`, a list of (row, col) seeds where
marker_pixels[0] is always the background marker (basin ID 1). Each further
entry becomes its own basin.

Builders:
    build_markers_g        — classic pipeline: size-filtered regional minima
                             of the gradient inside the brain.
    build_markers_medical  — medical pipeline: CLAHE-equalised image,
                             quadtree split-merge regions, per-region adaptive
                             Otsu edge map, then morphology.
    build_markers_morph    — alternative: global Otsu edge map + morphology
                             (not used by the default pipelines).
"""

import numpy as np
from scipy.ndimage import (binary_closing, binary_erosion, label as nd_label,
                           zoom)
from skimage.filters import threshold_otsu

from .core import find_regional_minima
from .preprocessing import gaussian_smooth


MIN_REGION_SIZE: int = 5    # minimum pixel count for a gradient minimum to become a seed
BRAIN_INTENSITY: int = 10   # grey level above which a pixel counts as brain (not background)


# ── Classic: seeds from gradient minima ──────────────────────────────────────

def build_markers_g(g_u8, gray_f, min_size=MIN_REGION_SIZE):
    """
    Seed one marker per significant regional minimum of the gradient.

    Filtering rules (no external information needed):
        - Size filter  : skip minima smaller than `min_size` pixels (noise).
        - Brain filter : skip minima with fewer than half of their pixels
                         inside the brain (gray_f > BRAIN_INTENSITY), i.e.
                         the black background outside the skull.

    `min_size` trades over- against under-segmentation: very small values
    keep many noise seeds, while very large values (> 200 px) risk merging
    the tumour minimum into its surroundings.

    Returns:
        marker_pixels — [(0, 0) background corner] + one representative
        pixel per accepted minimum.
    """
    brain_mask = gray_f > BRAIN_INTENSITY
    minima = find_regional_minima(g_u8)
    marker_pixels = [(0, 0)]   # image corner = background seed

    for region in minima:
        px = list(region)
        if len(px) < min_size:
            continue
        if sum(brain_mask[r, c] for r, c in px) < len(px) * 0.5:
            continue
        rep = px[len(px) // 2]   # any pixel of the plateau is a valid seed
        marker_pixels.append((rep[0], rep[1]))

    return marker_pixels


# ── Shared: edge map -> enclosed regions -> markers ──────────────────────────

def _markers_from_edge_map(B):
    """
    Turn a binary edge map into watershed markers.

    1. Close small gaps to connect broken edge fragments.
    2. Heavy closing fills blobs; subtracting the erosion leaves 1-px rings.
    3. Remove the largest ring (the outer scalp boundary).
    4. Label the regions enclosed by the remaining rings. The largest one is
       the background; every other region gets one internal marker.

    Returns:
        (marker_pixels, E_brain) — E_brain is the boundary-ring mask, used for
        visualisation. marker_pixels is empty if no regions were enclosed.
    """
    struct3 = np.ones((3, 3), dtype=bool)
    struct15 = np.ones((15, 15), dtype=bool)

    B_conn = binary_closing(B, structure=struct3, iterations=2)
    B_filled = binary_closing(B_conn, structure=struct15)
    E_thin = B_filled & ~binary_erosion(B_filled, structure=struct3)

    ring_lbl, n_rings = nd_label(E_thin, structure=struct3)
    if n_rings == 0:
        return [], E_thin
    ring_sizes = np.bincount(ring_lbl.ravel())
    ring_sizes[0] = 0
    scalp_id = int(np.argmax(ring_sizes))
    E_brain = E_thin.copy()
    E_brain[ring_lbl == scalp_id] = False

    int_lbl, n_int = nd_label(~E_brain)
    if n_int <= 1:
        return [], E_brain

    int_sizes = np.bincount(int_lbl.ravel())
    int_sizes[0] = 0
    bg_id = int(np.argmax(int_sizes))

    bg_px = np.argwhere(int_lbl == bg_id)
    marker_pixels = [tuple(bg_px[len(bg_px) // 2])]

    for lbl_id in range(1, n_int + 1):
        if lbl_id == bg_id:
            continue
        px = np.argwhere(int_lbl == lbl_id)
        if len(px) > 0:
            marker_pixels.append(tuple(px[len(px) // 2]))

    return marker_pixels, E_brain


def build_markers_morph(g_u8):
    """
    Alternative marker builder: global Otsu edge map followed by morphology.

    Returns (marker_pixels, E_brain); see `_markers_from_edge_map`.
    """
    B = g_u8 > threshold_otsu(g_u8)
    return _markers_from_edge_map(B)


# ── Block-based adaptive Otsu (alternative) ──────────────────────────────────

def adaptive_otsu_threshold(g_u8, block_size=64):
    """
    Block-wise adaptive Otsu edge map (not used by the default pipelines).

    Computes one Otsu threshold per block_size x block_size tile, bilinearly
    interpolates the thresholds to full resolution and returns g_u8 > T_map.
    Compared with a global Otsu, weak edges in dim regions get a lower local
    threshold and are still detected.
    """
    h, w = g_u8.shape
    bh = int(np.ceil(h / block_size))
    bw = int(np.ceil(w / block_size))

    T_small = np.zeros((bh, bw), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            r0, r1 = i * block_size, min((i + 1) * block_size, h)
            c0, c1 = j * block_size, min((j + 1) * block_size, w)
            block = g_u8[r0:r1, c0:c1]
            # A flat block has no bimodal histogram; use its value as threshold.
            T_small[i, j] = (threshold_otsu(block) if block.max() > block.min()
                             else float(block.max()))

    T_map = zoom(T_small, (h / bh, w / bw), order=1)[:h, :w]
    return g_u8 > T_map


# ── Quadtree split-and-merge ─────────────────────────────────────────────────

def _quadtree_split(img, r0, c0, r1, c1, min_size, std_thresh, leaves):
    """Recursively split [r0:r1, c0:c1] into quadrants until homogeneous."""
    h, w = r1 - r0, c1 - c0
    if h <= min_size or w <= min_size:
        leaves.append((r0, c0, r1, c1))
        return
    if float(img[r0:r1, c0:c1].std()) <= std_thresh:
        leaves.append((r0, c0, r1, c1))
        return
    rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
    _quadtree_split(img, r0, c0, rm, cm, min_size, std_thresh, leaves)
    _quadtree_split(img, r0, cm, rm, c1, min_size, std_thresh, leaves)
    _quadtree_split(img, rm, c0, r1, cm, min_size, std_thresh, leaves)
    _quadtree_split(img, rm, cm, r1, c1, min_size, std_thresh, leaves)


def split_merge_regions(image, min_size=8, std_thresh=10.0):
    """
    Quadtree split-and-merge segmentation.

    Homogeneity predicate: Q(R) = TRUE  iff  std(R) <= std_thresh.

    Split — recursively divide a region into four quadrants while Q(R) is
            FALSE and the block is larger than `min_size`. Leaves are small
            in textured areas and large in uniform areas.
    Merge — for every pair of 4-adjacent leaves, merge them if
            Q(Ri ∪ Rj) is TRUE. A union-find keeps running (count, sum,
            sum of squares) per root, so each merge test is O(1) instead of
            re-scanning pixels.

    Returns:
        int32 label map; pixels of the same merged region share an ID.
    """
    rows, cols = image.shape
    img = image.astype(float)

    # ── Split phase ──────────────────────────────────────────────────────────
    leaves = []
    _quadtree_split(img, 0, 0, rows, cols, min_size, std_thresh, leaves)

    label_map = np.zeros((rows, cols), dtype=np.int32)
    for lid, (r0, c0, r1, c1) in enumerate(leaves, start=1):
        label_map[r0:r1, c0:c1] = lid
    n = len(leaves)

    # ── Merge phase: union-find with running statistics ──────────────────────
    parent = list(range(n + 1))
    counts = [0] * (n + 1)
    sums = [0.0] * (n + 1)
    sum_sqs = [0.0] * (n + 1)
    for lid, (r0, c0, r1, c1) in enumerate(leaves, start=1):
        block = img[r0:r1, c0:c1].ravel()
        counts[lid] = len(block)
        sums[lid] = float(block.sum())
        sum_sqs[lid] = float((block ** 2).sum())

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def try_merge(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        nc = counts[ra] + counts[rb]
        s = sums[ra] + sums[rb]
        sq = sum_sqs[ra] + sum_sqs[rb]
        var = max(0.0, sq / nc - (s / nc) ** 2)   # Var = E[x²] - E[x]²
        if var ** 0.5 <= std_thresh:
            parent[rb] = ra
            counts[ra] = nc
            sums[ra] = s
            sum_sqs[ra] = sq

    # Visit every vertical, then every horizontal, pair of adjacent pixels
    # that lie in different leaves.
    for r in range(rows - 1):
        for c in range(cols):
            a, b = int(label_map[r, c]), int(label_map[r + 1, c])
            if a != b:
                try_merge(a, b)
    for r in range(rows):
        for c in range(cols - 1):
            a, b = int(label_map[r, c]), int(label_map[r, c + 1])
            if a != b:
                try_merge(a, b)

    root_of = np.array([find(i) for i in range(n + 1)], dtype=np.int32)
    return root_of[label_map]


def split_merge_adaptive_otsu(g_u8, region_labels):
    """
    Per-region adaptive Otsu edge map.

    Each split-merge region gets its own Otsu threshold, computed only from
    the gradient values inside that region; pixels above it are edges. Pixels
    are grouped by sorting on the region label (O(n log n)), which avoids
    allocating one boolean mask per region.

    Returns:
        B — bool edge mask (True = edge pixel).
    """
    flat_g = g_u8.ravel().astype(np.float32)
    flat_r = region_labels.ravel()
    T_flat = np.empty_like(flat_g)

    order = np.argsort(flat_r, kind='stable')
    s_g = flat_g[order]
    s_r = flat_r[order]
    # Start/end index of each run of equal labels in the sorted array
    bounds = np.concatenate([[0], np.where(np.diff(s_r))[0] + 1, [len(s_r)]])

    for i in range(len(bounds) - 1):
        sl = slice(int(bounds[i]), int(bounds[i + 1]))
        idx = order[sl]
        vals = s_g[sl]
        t = (float(threshold_otsu(vals)) if int(vals.max()) > int(vals.min())
             else float(vals.max()))
        T_flat[idx] = t

    return g_u8 > T_flat.reshape(g_u8.shape)


# ── Medical: split-merge + adaptive Otsu + morphology ────────────────────────

def build_markers_medical(g_u8, gray_f, sigma=2.0, min_size=8, std_thresh=10.0):
    """
    Marker extraction for the medical pipeline.

    1. Gaussian-smooth g (sigma) to merge nearby spurious edge fragments.
    2. Quadtree split-merge on gray_f -> content-adaptive homogeneous regions.
    3. Per-region adaptive Otsu on the smoothed g -> binary edge map B.
    4-7. Morphology, scalp removal and region labelling
         (see `_markers_from_edge_map`).

    Returns:
        (marker_pixels, E_brain); marker_pixels[0] is the background marker.
    """
    # Step 1: smooth g
    g_smooth = gaussian_smooth(g_u8.astype(float), sigma)
    g_for_otsu = np.clip(g_smooth, 0, 255).astype(np.uint8)

    # Step 2: content-adaptive regions
    print("    split-merge …", flush=True)
    regions = split_merge_regions(gray_f, min_size=min_size, std_thresh=std_thresh)

    # Step 3: per-region Otsu edge map
    B = split_merge_adaptive_otsu(g_for_otsu, regions)

    # Steps 4-7
    return _markers_from_edge_map(B)
