"""
From-scratch watershed transform.

The input surface (usually a gradient magnitude image) is treated as a
topographic relief. Every regional minimum is a lake that is flooded from
below; the water level rises one grey level at a time. A pixel reached by a
single lake joins that lake's basin, while a pixel where two or more lakes
meet becomes a dam — the watershed line that separates objects.

Label convention used throughout the package:
    >= 1  basin ID
       0  watershed (dam) pixel      -> WATERSHED
      -1  not reached / unlabelled   -> UNLABELED

Two priority-queue implementations are provided:
    * binary heap (heapq)  — any integer surface; used by the toy examples.
    * 256-level bucket queue — O(1) push/pop for uint8 images; used for the
      512 x 512 MRI slices.
"""

import heapq
from collections import deque

import numpy as np


UNLABELED = -1
WATERSHED = 0   # dam pixel


# ── Neighbourhood ────────────────────────────────────────────────────────────

def neighbors_8(r, c, rows, cols):
    """Yield every valid 8-connected (row, col) neighbour of pixel (r, c)."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


# ── Regional minima / maxima (plateau-aware) ─────────────────────────────────

def find_regional_minima(surface):
    """
    Find all regional minima of `surface`.

    A regional minimum is a connected plateau of equal value with no strictly
    lower 8-connected neighbour. Each plateau is collected with a BFS, so flat
    minima wider than one pixel are handled correctly.

    Returns:
        list of frozensets, one per minimum, each holding its (row, col) pixels.
    """
    rows, cols = surface.shape
    visited = np.zeros((rows, cols), dtype=bool)
    basins = []

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr, sc]:
                continue
            v = surface[sr, sc]
            plateau = {(sr, sc)}
            queue = deque([(sr, sc)])
            visited[sr, sc] = True
            is_minimum = True

            # Grow the plateau; any strictly lower neighbour disqualifies it.
            while queue:
                r, c = queue.popleft()
                for nr, nc in neighbors_8(r, c, rows, cols):
                    nv = surface[nr, nc]
                    if nv < v:
                        is_minimum = False
                    elif nv == v and not visited[nr, nc]:
                        visited[nr, nc] = True
                        plateau.add((nr, nc))
                        queue.append((nr, nc))

            if is_minimum:
                basins.append(frozenset(plateau))

    return basins


def find_regional_maxima(image):
    """Find all regional maxima (the regional minima of the negated image)."""
    return find_regional_minima(-np.asarray(image, dtype=float))


# ── Flooding engines ─────────────────────────────────────────────────────────
#
# Both engines expand an already-seeded label map in place. A popped pixel is
# absorbed when exactly one basin touches it, otherwise it becomes a dam.

def _flood_heap(surface, labels, seeds):
    """Priority-flood `labels` over an integer `surface` using a binary heap."""
    rows, cols = surface.shape
    heap = []
    in_queue = np.zeros((rows, cols), dtype=bool)

    def push_neighbours(r, c):
        for nr, nc in neighbors_8(r, c, rows, cols):
            if labels[nr, nc] == UNLABELED and not in_queue[nr, nc]:
                heapq.heappush(heap, (int(surface[nr, nc]), nr, nc))
                in_queue[nr, nc] = True

    for r, c in seeds:
        push_neighbours(r, c)

    while heap:
        _, r, c = heapq.heappop(heap)
        if labels[r, c] != UNLABELED:
            continue
        touching = {
            labels[nr, nc]
            for nr, nc in neighbors_8(r, c, rows, cols)
            if labels[nr, nc] not in (UNLABELED, WATERSHED)
        }
        if len(touching) == 1:
            labels[r, c] = touching.pop()
            push_neighbours(r, c)
        else:
            labels[r, c] = WATERSHED

    return labels


def _flood_buckets(surface_u8, labels, seeds):
    """
    Priority-flood `labels` over a uint8 surface using 256 FIFO buckets.

    Buckets are drained in increasing grey level. Seeds are pushed in the
    order given, so the result is deterministic for a fixed seed order.
    """
    rows, cols = surface_u8.shape
    buckets = [deque() for _ in range(256)]
    in_queue = np.zeros((rows, cols), dtype=bool)

    def push_neighbours(r, c):
        for nr, nc in neighbors_8(r, c, rows, cols):
            if labels[nr, nc] == UNLABELED and not in_queue[nr, nc]:
                buckets[int(surface_u8[nr, nc])].append((nr, nc))
                in_queue[nr, nc] = True

    for r, c in seeds:
        push_neighbours(r, c)

    start_level = next((i for i in range(256) if buckets[i]), 0)
    for level in range(start_level, 256):
        while buckets[level]:
            r, c = buckets[level].popleft()
            if labels[r, c] != UNLABELED:
                continue
            touching = {
                labels[nr, nc]
                for nr, nc in neighbors_8(r, c, rows, cols)
                if labels[nr, nc] not in (UNLABELED, WATERSHED)
            }
            if len(touching) == 1:
                labels[r, c] = touching.pop()
                push_neighbours(r, c)
            elif len(touching) > 1:
                labels[r, c] = WATERSHED

    return labels


def _labelled_pixels(labels):
    """All labelled pixels in raster order, as (row, col) int tuples."""
    return [(int(r), int(c)) for r, c in np.argwhere(labels != UNLABELED)]


# ── Part 1: raw watershed (every regional minimum is a seed) ─────────────────

def watershed_from_scratch(surface):
    """
    Flood `surface` from all of its regional minima simultaneously.

    Steps:
        1. Find regional minima (plateau-aware).
        2. Give each minimum its own basin ID (lowest minimum -> ID 1).
        3. Flood outward with a min-priority queue (lowest value first).
        4. A pixel touched by one basin joins it; by two or more -> WATERSHED.

    Returns:
        labels      : int32 label map (see module docstring for the convention).
        basin_seeds : dict {basin ID: frozenset of seed pixels}.
    """
    labels = np.full(surface.shape, UNLABELED, dtype=np.int32)

    minima = find_regional_minima(surface)
    minima.sort(key=lambda region: surface[next(iter(region))])
    basin_seeds = {}
    for bid, region in enumerate(minima, start=1):
        for r, c in region:
            labels[r, c] = bid
        basin_seeds[bid] = region

    _flood_heap(surface, labels, _labelled_pixels(labels))
    return labels, basin_seeds


# ── Part 2: marker-controlled watershed ──────────────────────────────────────

def impose_minima(gradient, marker_pixels):
    """
    Modify `gradient` so that its only regional minima sit at `marker_pixels`.

    Markers are set to 0 and a priority flood propagates outward, giving every
    reached pixel p the value max(propagated_value, gradient[p]). Gradient
    ridges are therefore preserved, while valleys that contain no marker are
    filled up and can no longer start a basin of their own.

    Returns:
        g_modified (int32, same shape as `gradient`).
    """
    rows, cols = gradient.shape
    MAX_VAL = int(gradient.max()) + 1   # sentinel: "not reached yet"

    result = np.full((rows, cols), MAX_VAL, dtype=np.int32)
    in_heap = np.zeros((rows, cols), dtype=bool)
    heap = []

    for r, c in marker_pixels:
        result[r, c] = 0
        for nr, nc in neighbors_8(r, c, rows, cols):
            if not in_heap[nr, nc]:
                heapq.heappush(heap, (0, nr, nc))
                in_heap[nr, nc] = True

    while heap:
        prev, r, c = heapq.heappop(heap)
        if result[r, c] != MAX_VAL:
            continue
        new_val = max(prev, int(gradient[r, c]))
        result[r, c] = new_val
        for nr, nc in neighbors_8(r, c, rows, cols):
            if result[nr, nc] == MAX_VAL and not in_heap[nr, nc]:
                heapq.heappush(heap, (new_val, nr, nc))
                in_heap[nr, nc] = True

    return result


def marker_controlled_watershed(gradient, marker_pixels):
    """
    Marker-controlled watershed on a small integer surface (heap-based).

    impose_minima -> one basin per marker -> flood the modified gradient.

    Returns:
        labels, basin_seeds, g_modified
    """
    g_mod = impose_minima(gradient, marker_pixels)
    labels = np.full(gradient.shape, UNLABELED, dtype=np.int32)

    basin_seeds = {}
    for bid, (r, c) in enumerate(marker_pixels, start=1):
        labels[r, c] = bid
        basin_seeds[bid] = frozenset([(r, c)])

    _flood_heap(g_mod, labels, marker_pixels)
    return labels, basin_seeds, g_mod


# ── Image-scale versions (uint8 gradient, bucket queue) ──────────────────────

def marker_watershed_image(g_u8, marker_pixels):
    """
    Marker-controlled watershed for a full uint8 gradient image.

    1. Impose minima at the marker pixels -> g_modified.
    2. Bucket-queue flood of g_modified from all markers at once.
       Basin ID k belongs to marker_pixels[k - 1].

    Falls back to the unconstrained watershed if no markers are supplied.
    """
    if not marker_pixels:
        print("  No markers — falling back to classic watershed.", flush=True)
        return _classic_watershed(g_u8)

    print(f"  Imposing minima at {len(marker_pixels)} marker(s) …", flush=True)
    g_mod = impose_minima(g_u8, marker_pixels)
    g_mod_u8 = np.clip(g_mod, 0, 255).astype(np.uint8)

    labels = np.full(g_u8.shape, UNLABELED, dtype=np.int32)
    for bid, (r, c) in enumerate(marker_pixels, start=1):
        labels[r, c] = bid

    print("  Flooding …", flush=True)
    return _flood_buckets(g_mod_u8, labels, marker_pixels)


def _classic_watershed(surface_u8):
    """Bucket-queue watershed seeded at every regional minimum (fallback)."""
    labels = np.full(surface_u8.shape, UNLABELED, dtype=np.int32)

    print("  Finding regional minima …", flush=True)
    minima = find_regional_minima(surface_u8)
    minima.sort(key=lambda r: int(surface_u8[next(iter(r))]))
    for bid, region in enumerate(minima, start=1):
        for r, c in region:
            labels[r, c] = bid
    print(f"  {len(minima)} basins seeded", flush=True)

    print("  Flooding …", flush=True)
    return _flood_buckets(surface_u8, labels, _labelled_pixels(labels))
