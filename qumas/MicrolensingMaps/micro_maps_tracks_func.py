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


def _angle_deg_from_pixels(r0: int, c0: int, r1: int, c1: int, *, y_down: bool = True) -> float:
    """Angle in degrees from (r0,c0) to (r1,c1).
    y_down=True gives image-style angles; set False for Cartesian."""
    dx = c1 - c0
    dy = r1 - r0
    if not y_down:
        dy = -dy
    ang = np.degrees(np.arctan2(dy, dx))   # (-180, 180]
    return float((ang + 360) % 360)        # [0, 360)

def _angle_deg_from_data(p0: Tuple[float,float], p1: Tuple[float,float]) -> float:
    """Angle in degrees in data coords (x to the right, y up)."""
    x0, y0 = p0
    x1, y1 = p1
    ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))  # (-180, 180]
    return float((ang + 360) % 360)                 # [0, 360)
def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classic Bresenham line between (r0, c0) and (r1, c1), inclusive.
    Returns arrays rr, cc of same length.
    """
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1

    rr = []
    cc = []

    if dc > dr:
        err = dc // 2
        r = r0
        for c in range(c0, c1 + sc, sc):
            rr.append(r)
            cc.append(c)
            err -= dr
            if err < 0:
                r += sr
                err += dc
    else:
        err = dr // 2
        c = c0
        for r in range(r0, r1 + sr, sr):
            rr.append(r)
            cc.append(c)
            err -= dc
            if err < 0:
                c += sc
                err += dr

    return np.asarray(rr, dtype=int), np.asarray(cc, dtype=int)


def _angle_deg_from_pixels(r0: int, c0: int, r1: int, c1: int, *, y_down: bool = True) -> float:
    """
    Angle in degrees from (r0,c0) to (r1,c1).
    y_down=True uses image-style axes (rows increase downward).
    """
    dx = c1 - c0
    dy = r1 - r0
    if not y_down:
        dy = -dy
    ang = np.degrees(np.arctan2(dy, dx))    # (-180, 180]
    return float((ang + 360) % 360)          # [0, 360)


def _angle_deg_from_data(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    """Angle in degrees in data coords (x right, y up)."""
    x0, y0 = p0
    x1, y1 = p1
    ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))   # (-180, 180]
    return float((ang + 360) % 360)                  # [0, 360)


def _pixel_to_data_xy(r: int, c: int, H: int, W: int,
                      extent: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Map pixel center (r,c) to data coords given extent=(xmin, xmax, ymin, ymax).
    x increases with column, y increases upward (so row 0 is at ymax).
    """
    xmin, xmax, ymin, ymax = extent
    x = xmin + (c + 0.5) * (xmax - xmin) / W
    # rows go down, so invert to get y-up coordinates (center-of-pixel mapping)
    y = ymax - (r + 0.5) * (ymax - ymin) / H
    return float(x), float(y)

# ------------------------- main generator --------------------------------

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

    Angles are no longer biased: endpoints are sampled uniformly on the
    Chebyshev perimeter { (dr,dc): max(|dr|,|dc|)=L-1 }, which admits all
    lattice directions including true 45° diagonals and guarantees a
    Bresenham path of length L (unless border clipping occurs, in which
    case we retry).

    Parameters
    ----------
    shape : (H, W)
    lengths : list[int] or int
        If int, you must also pass n_tracks (all tracks same length).
        If list, its length is the number of tracks and each entry is
        the pixel length.
    n_tracks : int | None
        Number of tracks if `lengths` is an int.
    seed : int | None
        RNG seed.
    max_attempts : int
        Max attempts per track to find a valid line of exact length.
    extent : (xmin, xmax, ymin, ymax) | None
        If provided, data coords (p0, p1) and data-space angle will be included.
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
            - 'angle_deg': float (pixel-space, y-down convention, [0,360))
            - 'p0': (x0, y0)  [only if extent provided]
            - 'p1': (x1, y1)  [only if extent provided]
            - 'angle_deg_data': float [only if extent provided]
            - 'coords': ((r0,c0),(r1,c1))
            - 'coords_pix': (rr,cc)
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

        # L == 1: single-pixel "track" (angle undefined)
        if L == 1:
            for _ in range(max_attempts):
                r0 = rng.integers(0, H)
                c0 = rng.integers(0, W)
                if avoid_overlap and used_mask[r0, c0]:
                    continue
                rr, cc = np.array([r0]), np.array([c0])
                track: Dict[str, Any] = {
                    "pix0": (r0, c0),
                    "pix1": (r0, c0),
                    "rr": rr, "cc": cc,
                    "length": 1,
                    "coords": ((r0, c0), (r0, c0)),
                    "coords_pix": (rr, cc),
                    "angle_deg": np.nan,
                }
                if extent is not None:
                    x0, y0 = _pixel_to_data_xy(r0, c0, H, W, extent)
                    track["p0"] = (x0, y0)
                    track["p1"] = (x0, y0)
                    track["angle_deg_data"] = np.nan
                tracks.append(track)
                if avoid_overlap:
                    used_mask[rr, cc] = True
                break
            else:
                raise RuntimeError("Could not place length-1 track without overlap.")
            continue

        # L >= 2
        M = L - 1
        placed = False
        for _ in range(max_attempts):
            # pick start with margin when possible so the whole track fits
            if H > 2 * M and W > 2 * M:
                r0 = rng.integers(M, H - M)
                c0 = rng.integers(M, W - M)
            else:
                r0 = rng.integers(0, H)
                c0 = rng.integers(0, W)

            # --- uniform endpoint on Chebyshev perimeter ---
            side = rng.integers(0, 4)
            t = rng.integers(-M, M + 1)
            if side == 0:         # top edge: Δr = -M
                dr, dc = -M, t
            elif side == 1:       # right edge: Δc = +M
                dr, dc = t, +M
            elif side == 2:       # bottom edge: Δr = +M
                dr, dc = +M, t
            else:                 # left edge: Δc = -M
                dr, dc = t, -M

            r1 = r0 + dr
            c1 = c0 + dc

            # If we used margins, clipping should rarely happen.
            if not (0 <= r1 < H and 0 <= c1 < W):
                # would go out of bounds; try again
                continue

            rr, cc = bresenham_line(r0, c0, r1, c1)
            if rr.size != L:
                # this should basically never happen with the Chebyshev construction
                continue

            if avoid_overlap and used_mask[rr, cc].any():
                continue

            angle_deg = _angle_deg_from_pixels(r0, c0, r1, c1, y_down=True)

            track: Dict[str, Any] = {
                "pix0": (r0, c0),
                "pix1": (r1, c1),
                "rr": rr, "cc": cc,
                "length": int(L),
                "coords": ((r0, c0), (r1, c1)),
                "coords_pix": (rr, cc),
                "angle_deg": angle_deg,
            }

            if extent is not None:
                x0, y0 = _pixel_to_data_xy(r0, c0, H, W, extent)
                x1, y1 = _pixel_to_data_xy(r1, c1, H, W, extent)
                track["p0"] = (x0, y0)
                track["p1"] = (x1, y1)
                track["angle_deg_data"] = _angle_deg_from_data(track["p0"], track["p1"])

            tracks.append(track)
            if avoid_overlap:
                used_mask[rr, cc] = True
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Could not place a track of length {L} after {max_attempts} attempts. "
                "Consider reducing length, allowing overlap, or increasing max_attempts."
            )

    return tracks