"""
Round-trip format compatibility test for the OBB labeling pipeline.

Verifies that label_io.write_obb_label produces output that is:
  1. Byte-compatible with rsdd_to_yolo.py (same 6-decimal precision,
     same corner ordering TL→TR→BR→BL, same trailing newline)
  2. Correctly parsed by read_obb_label into the polygon format obb_metrics.py expects
  3. polygon_iou(written, read_back) == 1.0 (no precision loss in round-trip)

The long-edge h >= w convention is tested explicitly: a box with w > h must be
corrected to the same output as a box with h, w swapped and angle+π/2.

Corner winding is verified: robndbox_to_polygon must produce a non-self-intersecting
quadrilateral that Shapely accepts as valid without buffer(0) repair.

This test MUST pass before any labeling session begins.  A silent mismatch in
corner ordering or number format would corrupt every IoU in the evaluation.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from src.data.rsdd_to_yolo import robndbox_to_polygon, _enforce_long_edge
from src.eval.obb_metrics import polygon_iou
from src.data.label_io import (
    fit_obb_from_corners,
    obb_to_label_line,
    write_obb_label,
    read_obb_label,
)

CHIP_W = CHIP_H = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flat(corners: list[tuple[float, float]]) -> list[float]:
    """[(x,y),...] → [x1,y1,x2,y2,x3,y3,x4,y4]"""
    return [v for xy in corners for v in xy]


# ---------------------------------------------------------------------------
# 1. Format compatibility — byte-level match with rsdd_to_yolo.py
# ---------------------------------------------------------------------------

class TestFormatCompatibility:
    """Writer output must be character-for-character identical to rsdd_to_yolo.py."""

    def test_label_line_matches_rsdd_to_yolo(self):
        """
        obb_to_label_line must produce the exact string that rsdd_to_yolo.py would
        write for the same OBB: class_id + 8 floats at 6 decimal places, space-separated.
        """
        cx, cy, h, w, angle = 256.0, 200.0, 120.0, 40.0, math.pi / 6

        corners_ref = robndbox_to_polygon(cx, cy, h, w, angle, CHIP_W, CHIP_H)
        expected = "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in corners_ref)
        actual   = obb_to_label_line(cx, cy, h, w, angle, CHIP_W, CHIP_H)

        assert actual == expected, (
            f"Label line format mismatch:\n  expected: {expected!r}\n  actual:   {actual!r}"
        )

    def test_file_trailing_newline_matches_rsdd_to_yolo(self, tmp_path):
        """
        write_obb_label must append a trailing newline when ships are present,
        matching rsdd_to_yolo.py's '\\n'.join(lines) + '\\n' pattern.
        """
        label_path = tmp_path / "chip.txt"
        write_obb_label(label_path, [(256.0, 256.0, 80.0, 30.0, 0.0)], CHIP_W, CHIP_H)
        content = label_path.read_bytes()
        assert content.endswith(b"\n"), "Label file must end with a newline (rsdd_to_yolo.py convention)"

    def test_empty_file_has_no_trailing_newline(self, tmp_path):
        """Confirmed-empty chip writes a zero-byte file (no spurious newline)."""
        label_path = tmp_path / "empty.txt"
        write_obb_label(label_path, [], CHIP_W, CHIP_H)
        assert label_path.exists()
        assert label_path.stat().st_size == 0, (
            f"Empty label file should be 0 bytes, got {label_path.stat().st_size}"
        )


# ---------------------------------------------------------------------------
# 2. Round-trip: write → read → polygon_iou == 1.0
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Written OBBs must survive a read-back with polygon_iou == 1.0."""

    @pytest.mark.parametrize("cx,cy,h,w,angle", [
        (256.0, 256.0, 100.0, 35.0, 0.0),
        (300.0, 200.0,  80.0, 25.0, math.pi / 4),
        (100.0, 400.0, 120.0, 40.0, -math.pi / 3),
        (450.0, 128.0,  60.0, 20.0, math.pi / 6),
        # Near-edge ship (corners may be close to boundary)
        ( 10.0,  10.0,  70.0, 22.0, math.pi / 8),
    ])
    def test_single_ship_roundtrip_iou(self, tmp_path, cx, cy, h, w, angle):
        label_path = tmp_path / "chip.txt"
        write_obb_label(label_path, [(cx, cy, h, w, angle)], CHIP_W, CHIP_H)
        boxes = read_obb_label(label_path)

        assert len(boxes) == 1
        poly_written = flat(robndbox_to_polygon(cx, cy, h, w, angle, CHIP_W, CHIP_H))
        poly_read    = boxes[0]["polygon"]

        # Exact float equality after 6-decimal round-trip
        assert poly_read == pytest.approx(poly_written, abs=1e-6), (
            f"Read-back polygon differs from written polygon:\n"
            f"  written: {poly_written}\n  read:    {poly_read}"
        )
        iou = polygon_iou(poly_written, poly_read)
        assert iou == pytest.approx(1.0, abs=1e-3), f"Round-trip IoU = {iou:.10f} (expected 1.0)"

    def test_multiple_ships_roundtrip(self, tmp_path):
        obbs = [
            (150.0, 150.0,  60.0, 20.0, 0.0),
            (350.0, 300.0,  90.0, 30.0, math.pi / 3),
            (256.0, 400.0,  50.0, 15.0, -math.pi / 5),
        ]
        label_path = tmp_path / "multi.txt"
        write_obb_label(label_path, obbs, CHIP_W, CHIP_H)
        boxes = read_obb_label(label_path)

        assert len(boxes) == len(obbs), f"Expected {len(obbs)} boxes, got {len(boxes)}"
        for i, (cx, cy, h, w, angle) in enumerate(obbs):
            poly_ref  = flat(robndbox_to_polygon(cx, cy, h, w, angle, CHIP_W, CHIP_H))
            poly_read = boxes[i]["polygon"]
            iou = polygon_iou(poly_ref, poly_read)
            assert iou == pytest.approx(1.0, abs=1e-3), (
                f"Ship {i}: round-trip IoU = {iou:.10f}"
            )

    def test_confirmed_empty_roundtrip(self, tmp_path):
        label_path = tmp_path / "empty.txt"
        write_obb_label(label_path, [], CHIP_W, CHIP_H)
        assert read_obb_label(label_path) == []

    def test_unreviewed_chip_returns_empty_list(self, tmp_path):
        """Missing .txt (unreviewed) → [] — callers must not treat as confirmed empty."""
        label_path = tmp_path / "nonexistent.txt"
        assert not label_path.exists()
        assert read_obb_label(label_path) == []


# ---------------------------------------------------------------------------
# 3. Long-edge convention h >= w
# ---------------------------------------------------------------------------

class TestLongEdgeConvention:
    """Every OBB written must obey h >= w, regardless of what the caller passes."""

    def test_wide_box_produces_same_output_as_corrected_box(self, tmp_path):
        """
        write_obb_label with (h=30, w=100, angle=0) must produce the same file
        content as (h=100, w=30, angle=π/2) — both are the same physical box.
        Any other result means the long-edge convention is broken.
        """
        cx, cy = 256.0, 256.0

        path_wide      = tmp_path / "wide.txt"
        path_corrected = tmp_path / "corrected.txt"

        write_obb_label(path_wide,      [(cx, cy, 30.0, 100.0, 0.0)],               CHIP_W, CHIP_H)
        write_obb_label(path_corrected, [(cx, cy, 100.0, 30.0, math.pi / 2)], CHIP_W, CHIP_H)

        assert path_wide.read_text() == path_corrected.read_text(), (
            "Wide-box (w>h) was not corrected: output differs from long-edge-convention output.\n"
            f"  wide:      {path_wide.read_text()!r}\n"
            f"  corrected: {path_corrected.read_text()!r}"
        )

    def test_square_box_valid(self, tmp_path):
        """h == w is valid (square ship silhouette); must round-trip cleanly."""
        label_path = tmp_path / "square.txt"
        write_obb_label(label_path, [(256.0, 256.0, 50.0, 50.0, 0.3)], CHIP_W, CHIP_H)
        boxes = read_obb_label(label_path)
        assert len(boxes) == 1
        poly = flat(robndbox_to_polygon(256.0, 256.0, 50.0, 50.0, 0.3, CHIP_W, CHIP_H))
        assert polygon_iou(poly, boxes[0]["polygon"]) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Corner winding — polygon must be valid in Shapely without repair
# ---------------------------------------------------------------------------

class TestCornerWinding:
    """Shapely must accept the polygon as valid without calling buffer(0)."""

    @pytest.mark.parametrize("cx,cy,h,w,angle_deg", [
        (256, 256, 100, 40,  0),
        (256, 256, 100, 40, 30),
        (256, 256, 100, 40, 45),
        (256, 256, 100, 40, 89),
        (256, 256,  50, 50,  0),   # square
    ])
    def test_polygon_valid_no_repair_needed(self, cx, cy, h, w, angle_deg):
        angle = math.radians(angle_deg)
        corners = robndbox_to_polygon(cx, cy, h, w, angle, CHIP_W, CHIP_H)
        pts = [(x, y) for x, y in corners]
        p = Polygon(pts)
        # Must be valid WITHOUT buffer(0)
        assert p.is_valid, (
            f"Polygon invalid for angle={angle_deg}°: {pts}\n"
            "Corner ordering in robndbox_to_polygon produces self-intersecting polygon."
        )
        assert p.area > 0, "Polygon has zero area"

    def test_self_iou_exactly_1(self):
        """polygon_iou(p, p) == 1.0 for any valid polygon."""
        corners = robndbox_to_polygon(300.0, 200.0, 90.0, 30.0, math.pi / 6, CHIP_W, CHIP_H)
        poly = flat(corners)
        assert polygon_iou(poly, poly) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. fit_obb_from_corners geometry
# ---------------------------------------------------------------------------

class TestFitObb:
    """fit_obb_from_corners must produce a polygon with IoU >= 0.99 vs the original."""

    @pytest.mark.parametrize("cx,cy,h,w,angle", [
        (256.0, 256.0, 120.0, 40.0, math.pi / 6),
        (200.0, 300.0,  80.0, 25.0, math.pi / 4),
        (350.0, 150.0, 100.0, 35.0, -math.pi / 3),
        (256.0, 256.0,  60.0, 60.0, 0.0),          # square
    ])
    def test_fit_roundtrip_iou(self, tmp_path, cx, cy, h, w, angle):
        """
        Known OBB → canonical corners → fit_obb_from_corners → write → read → IoU >= 0.99.
        Tolerance is 0.99 (not 1.0) because PCA fitting introduces sub-pixel numerical error.
        """
        # Canonical corners as user would click them (de-normalized to pixels)
        corners_norm = robndbox_to_polygon(cx, cy, h, w, angle, CHIP_W, CHIP_H)
        corners_px   = [(x * CHIP_W, y * CHIP_H) for x, y in corners_norm]

        cx2, cy2, h2, w2, a2 = fit_obb_from_corners(corners_px)

        label_path = tmp_path / "fit.txt"
        write_obb_label(label_path, [(cx2, cy2, h2, w2, a2)], CHIP_W, CHIP_H)
        boxes = read_obb_label(label_path)

        assert len(boxes) == 1
        poly_orig   = flat(corners_norm)
        poly_fitted = boxes[0]["polygon"]
        iou = polygon_iou(poly_orig, poly_fitted)
        assert iou >= 0.99, (
            f"fit_obb_from_corners round-trip IoU = {iou:.4f} (expected >= 0.99).\n"
            f"  Original (cx={cx}, cy={cy}, h={h}, w={w}, angle={angle:.3f})\n"
            f"  Fitted   (cx={cx2:.2f}, cy={cy2:.2f}, h={h2:.2f}, w={w2:.2f}, angle={a2:.3f})"
        )

    def test_fit_enforces_long_edge(self):
        """fit_obb_from_corners must always return h >= w."""
        # A horizontally wide box (width >> height)
        corners_px = [(200.0, 240.0), (312.0, 240.0), (312.0, 270.0), (200.0, 270.0)]
        _, _, h, w, _ = fit_obb_from_corners(corners_px)
        assert h >= w, (
            f"fit_obb_from_corners returned h={h:.2f} < w={w:.2f} (long-edge convention violated)"
        )
