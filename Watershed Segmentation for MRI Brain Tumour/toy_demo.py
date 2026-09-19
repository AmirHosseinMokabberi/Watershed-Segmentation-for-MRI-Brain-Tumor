"""
Step-by-step console demo of the watershed on tiny hand-made surfaces.

Part 1 — Raw watershed (every regional minimum is a seed)
    1. Find regional minima (including flat plateaus).
    2. Give each minimum its own basin ID.
    3. Flood outward with a min-priority queue (lowest value first).
    4. A pixel touched by one basin joins it; by two or more it becomes a
       watershed pixel (W).

Part 2 — Marker-controlled watershed
    A. Raw watershed on a noisy gradient -> over-segmentation.
    B. Gaussian-smooth the original image; its regional maxima (one per true
       object) become the internal markers.
    C. Impose minima on the gradient at the marker locations.
    D. Flood the modified gradient from the markers only.

Usage:
    python toy_demo.py
"""

import numpy as np

from watershed.core import (UNLABELED, WATERSHED, find_regional_maxima,
                            find_regional_minima, impose_minima,
                            marker_controlled_watershed, watershed_from_scratch)
from watershed.preprocessing import gaussian_smooth


# ── Toy surfaces ─────────────────────────────────────────────────────────────

# Two pits (0) enclosed by a ridge (9), on a flat outer ring (1).
SURFACE = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 9, 9, 9, 9, 9, 1],
    [1, 9, 0, 9, 0, 9, 1],
    [1, 9, 9, 9, 9, 9, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.int32)

# Gradient of two objects: true minima (0) plus two noise minima (2).
NOISY_SURFACE = np.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 9, 9, 9, 9, 9, 9, 9, 5],
    [5, 9, 0, 9, 9, 9, 0, 9, 5],
    [5, 9, 9, 9, 9, 9, 9, 9, 5],
    [5, 9, 2, 9, 9, 9, 2, 9, 5],
    [5, 9, 9, 9, 9, 9, 9, 9, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=np.int32)

# Intensity image behind NOISY_SURFACE: two bright objects (peak 9) each with
# a weaker noise bump (6), separated by a dark column.
ORIGINAL_IMAGE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 1, 3, 3, 3, 0],
    [0, 3, 9, 3, 1, 3, 9, 3, 0],
    [0, 3, 3, 3, 1, 3, 3, 3, 0],
    [0, 3, 6, 3, 1, 3, 6, 3, 0],
    [0, 3, 3, 3, 1, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float64)


# ── Text rendering ───────────────────────────────────────────────────────────

def render(surface, labels, basin_seeds):
    """Print the surface, the label grid (A, B, … / W) and per-basin counts."""
    rows, cols = surface.shape
    bid_to_letter = {bid: chr(ord('A') + i) for i, bid in enumerate(basin_seeds)}

    def cell(r, c):
        v = labels[r, c]
        if v == UNLABELED:
            return "?"
        if v == WATERSHED:
            return "W"
        return bid_to_letter.get(v, str(v))

    header = "  " + " ".join(str(c) for c in range(cols))
    print("\nFlood surface g:")
    print(header)
    for r in range(rows):
        print(f"{r} " + " ".join(f"{surface[r, c]}" for c in range(cols)))
    print("\nSegmentation result:")
    print(header)
    for r in range(rows):
        print(f"{r} " + " ".join(cell(r, c) for c in range(cols)))
    print()
    for bid, region in basin_seeds.items():
        letter = bid_to_letter[bid]
        count = int(np.sum(labels == bid))
        seed = sorted(region)[0]
        print(f"  Basin {letter}  (ID={bid}, seed={seed}): {count} pixel(s)")
    wc = int(np.sum(labels == WATERSHED))
    uc = int(np.sum(labels == UNLABELED))
    print(f"  Watershed W : {wc} pixel(s)")
    if uc:
        print(f"  Unlabeled   : {uc} pixel(s)  <- check surface connectivity")
    print(f"  Total       : {rows * cols}")


def render_float(image, title):
    rows, cols = image.shape
    print(f"\n{title}:")
    print("  " + " ".join(f"{c:5}" for c in range(cols)))
    for r in range(rows):
        print(f"{r} " + " ".join(f"{image[r, c]:5.2f}" for c in range(cols)))


def render_int(surface, title):
    rows, cols = surface.shape
    print(f"\n{title}:")
    print("  " + " ".join(str(c) for c in range(cols)))
    for r in range(rows):
        print(f"{r} " + " ".join(f"{surface[r, c]}" for c in range(cols)))


def touches_border(region, shape):
    rows, cols = shape
    return any(r in (0, rows - 1) or c in (0, cols - 1) for r, c in region)


# ── Demo ─────────────────────────────────────────────────────────────────────

def part1_raw_watershed():
    print("=" * 58)
    print("Part 1 -- Raw Watershed (all regional minima as seeds)")
    print("=" * 58)

    # Sorted like watershed_from_scratch does, so the letters match the grid.
    print("\nStep 1 - Regional minima (plateau-aware):")
    minima = sorted(find_regional_minima(SURFACE),
                    key=lambda region: SURFACE[next(iter(region))])
    for i, region in enumerate(minima, 1):
        v = SURFACE[next(iter(region))]
        print(f"  Basin {chr(ord('A') + i - 1)}  value={v}  size={len(region)} pixel(s)  "
              f"e.g. {sorted(region)[0]}")

    print("\nStep 2-4 - Flooding ...")
    labels, basin_seeds = watershed_from_scratch(SURFACE)
    render(SURFACE, labels, basin_seeds)


def part2_marker_controlled():
    print("\n" + "=" * 58)
    print("Part 2 -- Marker-Controlled Watershed")
    print("=" * 58)

    # Step A: every minimum, including noise, becomes a basin.
    print("\n-- Step A: Raw watershed on noisy gradient (over-segmentation) --")
    noisy_minima = find_regional_minima(NOISY_SURFACE)
    print(f"Regional minima found: {len(noisy_minima)}  "
          f"(2 true objects + 2 noise + 1 background ring)")
    for i, region in enumerate(
            sorted(noisy_minima, key=lambda r: NOISY_SURFACE[next(iter(r))]), 1):
        v = NOISY_SURFACE[next(iter(region))]
        seed = sorted(region)[0]
        if touches_border(region, NOISY_SURFACE.shape):
            tag = "background"
        else:
            tag = "true" if v == 0 else "spurious noise"
        print(f"  Basin {chr(ord('A') + i - 1)}  value={v} at {seed}  [{tag}]")
    raw_labels, raw_seeds = watershed_from_scratch(NOISY_SURFACE)
    render(NOISY_SURFACE, raw_labels, raw_seeds)

    # Step B: smoothing removes the noise bumps, leaving one maximum per object.
    print("\n-- Step B: Gaussian smoothing -> internal markers --")
    sigma = 1.0
    smoothed = gaussian_smooth(ORIGINAL_IMAGE, sigma)
    render_float(ORIGINAL_IMAGE, f"Original image f (sigma={sigma} will be applied)")
    render_float(smoothed, f"Smoothed image f_smooth (sigma={sigma})")
    all_maxima = find_regional_maxima(smoothed)
    print(f"\nRegional maxima of smoothed image: {len(all_maxima)}"
          f"  (expected 1 per true object)")
    markers = []
    for i, region in enumerate(
            sorted(all_maxima, key=lambda r: -smoothed[next(iter(r))]), 1):
        rep = sorted(region)[0]
        print(f"  Maximum {i}: representative pixel {rep}"
              f"  smoothed_value={smoothed[rep]:.3f}")
        markers.append(rep)

    # Step C: only the markers remain minima.
    print("\n-- Step C: Minima imposition on gradient --")
    print("Markers:", markers)
    g_mod = impose_minima(NOISY_SURFACE, markers)
    render_int(NOISY_SURFACE, "Original gradient g")
    render_int(g_mod, "Modified gradient g_modified (spurious minima raised)")
    print("\nKey change: spurious minima at (4,2) and (4,6) raised from 2 to 9")

    # Step D: one basin per marker. Note: here g_modified is a flat plateau of
    # 9s outside the markers, so where the two basins meet is decided by the
    # heap's (value, row, col) tie-breaking rather than by the geometry.
    print("\n-- Step D: Marker-controlled segmentation --")
    mc_labels, mc_seeds, _ = marker_controlled_watershed(NOISY_SURFACE, markers)
    render(g_mod, mc_labels, mc_seeds)
    print("Result: exactly 2 basins — one per true object, no over-segmentation.")


if __name__ == "__main__":
    part1_raw_watershed()
    part2_marker_controlled()
