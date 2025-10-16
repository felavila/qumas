from typing import List, Tuple, Optional, Dict, Any
import numpy as np



def _bilinear_sample(arr: np.ndarray, rows_f: np.ndarray, cols_f: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear interpolation over a 2D array.
    rows_f, cols_f are float indices (0..H-1, 0..W-1).
    """
    H, W = arr.shape
    r0 = np.clip(np.floor(rows_f).astype(int), 0, H - 1)
    c0 = np.clip(np.floor(cols_f).astype(int), 0, W - 1)
    r1 = np.clip(r0 + 1, 0, H - 1)
    c1 = np.clip(c0 + 1, 0, W - 1)

    dr = rows_f - r0
    dc = cols_f - c0

    v00 = arr[r0, c0]
    v10 = arr[r1, c0]
    v01 = arr[r0, c1]
    v11 = arr[r1, c1]

    return ((1 - dr) * (1 - dc) * v00 +
            dr       * (1 - dc) * v10 +
            (1 - dr) * dc       * v01 +
            dr       * dc       * v11)

def _sample_profile_along_line_bilinear_pixels(
    mag_map_2d: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    extent: Tuple[float, float, float, float],
):
    """
    Bilinear sampling with one sample per *crossed pixel*.
    Returns:
      x_pix: np.arange(Npixels)  (0..Npixels-1)
      vals:  sampled values
      Npixels: number of pixels crossed (max(|dr|,|dc|)+1)
    """
    H, W = mag_map_2d.shape
    # endpoints in pixel space
    r0, c0 = _data_to_pixel_xy(p0[0], p0[1], H, W, extent)
    r1, c1 = _data_to_pixel_xy(p1[0], p1[1], H, W, extent)

    # number of discrete pixels crossed
    Npixels = max(abs(r1 - r0), abs(c1 - c0)) + 1
    if Npixels <= 1:
        # Degenerate line → one sample at the endpoint
        x_pix = np.array([0], dtype=int)
        rows_f = np.array([r0], dtype=float)
        cols_f = np.array([c0], dtype=float)
        vals = _bilinear_sample(mag_map_2d, rows_f, cols_f)
        return x_pix, vals, Npixels

    # Parameter along the line: exactly Npixels points, including endpoints
    s = np.linspace(0.0, 1.0, Npixels)

    # Convert endpoints back to data coords for precise subpixel positions along the line
    xmin, xmax, ymin, ymax = extent
    # data coords of endpoints (we already have them as p0/p1):
    x0, y0 = p0
    x1, y1 = p1
    xs = x0 + s * (x1 - x0)
    ys = y0 + s * (y1 - y0)

    # Map data → float pixel indices for bilinear sampling
    cols_f = (xs - xmin) / (xmax - xmin) * (W - 1)
    rows_f = (ymax - ys) / (ymax - ymin) * (H - 1)

    vals = _bilinear_sample(mag_map_2d, rows_f, cols_f)
    x_pix = np.arange(Npixels, dtype=int)
    return x_pix, vals, Npixels


def _data_to_pixel_xy(x: float, y: float, H: int, W: int, extent: Tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = extent
    col = (x - xmin) / (xmax - xmin) * (W - 1)
    row = (ymax - y) / (ymax - ymin) * (H - 1)
    return int(round(row)), int(round(col))


def _pixel_to_data_xy(r: int, c: int, H: int, W: int, extent: Tuple[float, float, float, float]):
    """Inverse of _data_to_pixel_xy: pixel (row,col) -> data coords (x,y) for imshow(extent, origin='upper')."""
    xmin, xmax, ymin, ymax = extent
    x = xmin + (c / (W - 1)) * (xmax - xmin)
    y = ymax - (r / (H - 1)) * (ymax - ymin)
    return float(x), float(y)


def bresenham_line(r0, c0, r1, c1):
    """Return integer (row,col) indices for a Bresenham line."""
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = (dr if dr > dc else -dc) // 2

    rr, cc = [], []
    r, c = r0, c0
    while True:
        rr.append(r); cc.append(c)
        if r == r1 and c == c1:
            break
        e2 = err
        if e2 > -dr:
            err -= dc
            r += sr
        if e2 <  dc:
            err += dr
            c += sc
    return np.asarray(rr, int), np.asarray(cc, int)



def _sample_profile_along_line_bilinear_pixels(
    mag_map_2d: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    extent: Tuple[float, float, float, float],
):
    """
    Bilinear sampling with one sample per *crossed pixel*.
    Returns:
      x_pix: np.arange(Npixels)  (0..Npixels-1)
      vals:  sampled values
      Npixels: number of pixels crossed (max(|dr|,|dc|)+1)
    """
    H, W = mag_map_2d.shape
    # endpoints in pixel space
    r0, c0 = _data_to_pixel_xy(p0[0], p0[1], H, W, extent)
    r1, c1 = _data_to_pixel_xy(p1[0], p1[1], H, W, extent)

    # number of discrete pixels crossed
    Npixels = max(abs(r1 - r0), abs(c1 - c0)) + 1
    if Npixels <= 1:
        # Degenerate line → one sample at the endpoint
        x_pix = np.array([0], dtype=int)
        rows_f = np.array([r0], dtype=float)
        cols_f = np.array([c0], dtype=float)
        vals = _bilinear_sample(mag_map_2d, rows_f, cols_f)
        return x_pix, vals, Npixels

    # Parameter along the line: exactly Npixels points, including endpoints
    s = np.linspace(0.0, 1.0, Npixels)

    # Convert endpoints back to data coords for precise subpixel positions along the line
    xmin, xmax, ymin, ymax = extent
    # data coords of endpoints (we already have them as p0/p1):
    x0, y0 = p0
    x1, y1 = p1
    xs = x0 + s * (x1 - x0)
    ys = y0 + s * (y1 - y0)

    # Map data → float pixel indices for bilinear sampling
    cols_f = (xs - xmin) / (xmax - xmin) * (W - 1)
    rows_f = (ymax - ys) / (ymax - ymin) * (H - 1)

    vals = _bilinear_sample(mag_map_2d, rows_f, cols_f)
    x_pix = np.arange(Npixels, dtype=int)
    return x_pix, vals, Npixels

def _sample_profile_along_line_pixels(
    mag_map_2d: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    extent: Tuple[float, float, float, float]
):
    """
    Discrete (nearest) profile: exact set of pixels hit by the line (Bresenham).
    Returns:
      x_pix: np.arange(Npixels)
      values: mag_map_2d[rr, cc]
      rr, cc: pixel coordinates along the line
    """
    H, W = mag_map_2d.shape
    r0, c0 = _data_to_pixel_xy(p0[0], p0[1], H, W, extent)
    r1, c1 = _data_to_pixel_xy(p1[0], p1[1], H, W, extent)
    rr, cc = bresenham_line(r0, c0, r1, c1)
    values = mag_map_2d[rr, cc]
    x_pix = np.arange(values.size, dtype=int)
    return x_pix, values, rr, cc

def generate_random_tracks(
    shape: Tuple[int, int],
    lengths: List[int] | int,
    *,
    n_tracks: Optional[int] = None,   # required if lengths is an int (shared length for all)
    seed: Optional[int] = None,
    max_attempts: int = 5000,
    extent: Optional[Tuple[float, float, float, float]] = None,
    avoid_overlap: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate straight tracks (Bresenham lines) with exact pixel lengths.

    Parameters
    ----------
    shape : (H, W)
    lengths : list[int] or int
        If int, you must also pass n_tracks (all tracks same length).
        If list, its length is the number of tracks and each entry is the pixel length.
    n_tracks : int | None
        Number of tracks if `lengths` is an int.
    seed : int | None
        RNG seed.
    max_attempts : int
        Max attempts per track to find a valid line of exact length.
    extent : (xmin, xmax, ymin, ymax) | None
        If provided, data coords (p0, p1) will also be included in the result.
    avoid_overlap : bool
        If True, tries to avoid reusing pixels used by previous tracks.

    Returns
    -------
    tracks : list of dict
        Each dict has:
            - 'pix0': (r0, c0)
            - 'pix1': (r1, c1)
            - 'rr': array of row indices (Bresenham path)
            - 'cc': array of col indices (Bresenham path)
            - 'length': int (number of pixels)
            - 'p0': (x0, y0)  [only if extent provided]
            - 'p1': (x1, y1)  [only if extent provided]
    """
    rng = np.random.default_rng(seed)
    H, W = shape

    # Normalize lengths input
    if isinstance(lengths, int):
        if n_tracks is None:
            raise ValueError("If `lengths` is an int, you must pass `n_tracks`.")
        Ls = [lengths] * n_tracks
    else:
        Ls = list(lengths)

    used_mask = np.zeros(shape, dtype=bool) if avoid_overlap else None
    tracks: List[Dict[str, Any]] = []

    for L in Ls:
        if L < 1:
            raise ValueError("Track length must be >= 1")

        # Short length: pick single pixel
        if L == 1:
            for _ in range(max_attempts):
                r0 = rng.integers(0, H)
                c0 = rng.integers(0, W)
                if avoid_overlap and used_mask[r0, c0]:
                    continue
                rr, cc = np.array([r0]), np.array([c0])
                track = {
                    "pix0": (r0, c0),
                    "pix1": (r0, c0),
                    "rr": rr, "cc": cc,
                    "length": 1,
                    "coords": ((r0, c0),(r0, c0)),
                    "coords_pix": (rr, cc)
                }
                if extent is not None:
                    x0, y0 = _pixel_to_data_xy(r0, c0, H, W, extent)
                    track["p0"] = (x0, y0)
                    track["p1"] = (x0, y0)
                tracks.append(track)
                if avoid_overlap:
                    used_mask[rr, cc] = True
                break
            else:
                raise RuntimeError("Could not place length-1 track without overlap.")
            continue

        # For L >= 2:
        placed = False
        for _ in range(max_attempts):
            # random start pixel
            r0 = rng.integers(0, H)
            c0 = rng.integers(0, W)

            # random direction as unit vector on grid via angle
            theta = rng.random() * 2.0 * np.pi
            # Convert required *pixel count* to endpoint displacement in continuous space:
            # We want exactly L pixels on the Bresenham path,
            # so try to make the endpoint approximately (L-1) pixels away.
            dr = (L - 1) * np.sin(theta)
            dc = (L - 1) * np.cos(theta)

            r1 = int(round(r0 + dr))
            c1 = int(round(c0 + dc))

            # Keep inside bounds
            r1 = max(0, min(H - 1, r1))
            c1 = max(0, min(W - 1, c1))

            rr, cc = bresenham_line(r0, c0, r1, c1)

            if rr.size != L:
                # Not the length we need; try again
                continue

            if avoid_overlap and used_mask[rr, cc].any():
                continue

            track = {
                "pix0": (r0, c0),
                "pix1": (r1, c1),
                "rr": rr, "cc": cc,
                "length": int(L),
                "coords": ((r0, c0),(r1, c1)),
                "coords_pix": (rr, cc)
            }
            if extent is not None:
                x0, y0 = _pixel_to_data_xy(r0, c0, H, W, extent)
                x1, y1 = _pixel_to_data_xy(r1, c1, H, W, extent)
                track["p0"] = (x0, y0)
                track["p1"] = (x1, y1)

            tracks.append(track)
            if avoid_overlap:
                used_mask[rr, cc] = True
            placed = True
            break

        if not placed:
            raise RuntimeError(f"Could not place a track of length {L} after {max_attempts} attempts. "
                            "Consider reducing length, allowing overlap, or increasing max_attempts.")

    return tracks