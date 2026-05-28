"""
OBB label I/O for SAR ship detection ground truth.

These functions define the ONLY path between pixel-coordinate OBB annotations
(from label_chips.py) and the YOLO OBB .txt format consumed by obb_metrics.py.
All format-critical decisions live here; label_chips.py and obb_metrics.py
each import from this module so there is one authoritative implementation.

YOLO OBB .txt format (per line):
  {class_id} x1 y1 x2 y2 x3 y3 x4 y4
  - coordinates normalized to [0, 1]
  - 6 decimal places (f"{x:.6f}")
  - corner order: TL→TR→BR→BL in the rotated frame (long axis first)
  - long-edge convention: h >= w (enforced via rsdd_to_yolo._enforce_long_edge)
  - empty file = confirmed-empty chip (zero ships; distinct from missing file)
  - missing file = unreviewed chip (excluded from evaluation)

Polygon format for obb_metrics.py:
  flat list of 8 floats [x1, y1, x2, y2, x3, y3, x4, y4]
  same corner ordering as above, same normalization

These formats are identical to what src/data/rsdd_to_yolo.py writes, by
construction: both call robndbox_to_polygon() with the same arguments.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.data.rsdd_to_yolo import robndbox_to_polygon


# ---------------------------------------------------------------------------
# OBB geometry fitting
# ---------------------------------------------------------------------------

def fit_obb_from_corners(
    corners_px: list[tuple[float, float]],
) -> tuple[float, float, float, float, float]:
    """
    Fit a minimum-area oriented bounding box to 4 pixel-coordinate corners.

    Args:
        corners_px: 4 (x, y) pixel coordinates (in any order — they will be
                    projected onto principal axes, so click order does not matter).

    Returns:
        (cx, cy, h, w, angle_rad) with h >= w (long-edge convention).
        cx, cy, h, w are in pixel coordinates.
        angle is the orientation of the long axis, in radians.

    This is the same (cx, cy, h, w, angle) parameterisation used by
    rsdd_to_yolo.robndbox_to_polygon, so the output feeds directly into that
    function without further conversion.

    Raises:
        ValueError: if fewer than 2 distinct input points (degenerate input).
    """
    pts = np.array(corners_px, dtype=float)
    cx, cy = pts.mean(axis=0).tolist()
    centered = pts - [cx, cy]

    # SVD: Vt[0] is the direction of MAXIMUM variance = the long axis of the box.
    # Use full_matrices=False for the compact form.
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        raise ValueError("fit_obb_from_corners: SVD failed (degenerate input).")

    # angle_long: world-space direction of the long axis (maximum variance).
    long_axis_dir = Vt[0]
    angle_long = math.atan2(float(long_axis_dir[1]), float(long_axis_dir[0]))

    # Project corners onto the long axis and its perpendicular to get h and w.
    cos_l = math.cos(angle_long)
    sin_l = math.sin(angle_long)
    along = centered @ np.array([cos_l, sin_l])          # extent along long axis
    perp  = centered @ np.array([-sin_l, cos_l])         # extent perpendicular

    h_raw = float(along.max() - along.min())
    w_raw = float(perp.max()  - perp.min())

    # CONVENTION NOTE — robndbox_to_polygon uses `angle` to mean the direction of the
    # SHORT axis (the x-axis of the rotated local frame), not the long axis.  The long
    # axis is at world angle `angle + π/2`.  PCA gives us the long axis, so we subtract
    # π/2 to convert to robndbox_to_polygon's convention.
    #
    # SVD sign is arbitrary (Vt[0] could point either way along the long axis).
    # The ±π ambiguity maps to ±π on the short axis, which robndbox_to_polygon handles
    # correctly (a π rotation produces the same physical box, just different start corner).

    if h_raw >= w_raw:
        # PCA correctly identified the long axis.
        h, w = h_raw, w_raw
        angle = angle_long - math.pi / 2   # long_axis − π/2 = short axis convention
    else:
        # The 4 corners had greater spread along what was supposed to be the short axis
        # (can happen with degenerate or nearly-square inputs).
        # angle_long is actually the short axis; long axis = angle_long + π/2.
        h, w = w_raw, h_raw
        angle = angle_long                  # angle_long IS the short axis already

    return float(cx), float(cy), h, w, angle


# ---------------------------------------------------------------------------
# Label line formatting (shared with rsdd_to_yolo.py output)
# ---------------------------------------------------------------------------

def obb_to_label_line(
    cx: float, cy: float, h: float, w: float, angle: float,
    chip_w: int, chip_h: int,
    class_id: int = 0,
) -> str:
    """
    Convert OBB parameters (pixel coords) to a YOLO OBB label line string.

    Output is character-for-character identical to what rsdd_to_yolo.py writes:
      f"{class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)

    No trailing newline — callers join lines themselves.
    Long-edge convention is enforced by robndbox_to_polygon internally.
    """
    corners = robndbox_to_polygon(cx, cy, h, w, angle, chip_w, chip_h)
    coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
    return f"{class_id} {coord_str}"


# ---------------------------------------------------------------------------
# Label file I/O
# ---------------------------------------------------------------------------

def write_obb_label(
    label_path: Path,
    annotations: list[tuple[float, float, float, float, float]],
    chip_w: int,
    chip_h: int,
) -> None:
    """
    Write a YOLO OBB label .txt file.

    Args:
        label_path:   Destination .txt path (sibling of the chip PNG).
        annotations:  List of (cx, cy, h, w, angle_rad) in pixel coords.
                      Pass [] for confirmed-empty chips — writes an empty file.
        chip_w, chip_h: Chip dimensions in pixels (used for normalization).

    File format:
        - One line per ship: "{class_id} x1 y1 x2 y2 x3 y3 x4 y4\\n"
        - Empty file (0 bytes or blank) = confirmed-empty chip.
        - Missing file = unreviewed chip (callers must not create missing files
          for "probably empty" chips — every file requires explicit human review).
    """
    lines = [
        obb_to_label_line(cx, cy, h, w, angle, chip_w, chip_h)
        for cx, cy, h, w, angle in annotations
    ]
    # Match rsdd_to_yolo.py exactly: join with \n and add trailing \n if non-empty
    content = "\n".join(lines) + ("\n" if lines else "")
    label_path.write_text(content)


def read_obb_label(label_path: Path) -> list[dict]:
    """
    Read a YOLO OBB label .txt file into the polygon format expected by obb_metrics.py.

    Returns:
        List of {"polygon": [x1, y1, x2, y2, x3, y3, x4, y4]} dicts (normalized [0,1]).
        [] for confirmed-empty chips (empty file) AND for unreviewed chips (missing file).

    Callers that need to distinguish "empty" from "missing" should check
    label_path.exists() before calling this function.
    """
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []  # confirmed empty

    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 9:
            # class_id x1 y1 x2 y2 x3 y3 x4 y4
            poly = [float(v) for v in parts[1:]]
            boxes.append({"polygon": poly})
    return boxes
