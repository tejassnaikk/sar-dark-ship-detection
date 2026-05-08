"""
Tests for src/data/rsdd_to_yolo.py

Covers:
- axis-aligned box at angle=0
- rotation at angle=π/2
- long-edge enforcement (w > h → swap + add π/2)
- normalized coordinates stay in [0, 1]
- files starting with ._ are skipped during directory walking
"""
import math
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from src.data.rsdd_to_yolo import (
    _enforce_long_edge,
    robndbox_to_polygon,
    convert_split,
    parse_annotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xml(cx, cy, h, w, angle, img_w=512, img_h=512) -> str:
    return textwrap.dedent(f"""\
        <annotation>
          <folder>RSDD-SAR</folder>
          <filename>test.xml</filename>
          <size><width>{img_w}</width><height>{img_h}</height><depth>3</depth></size>
          <object>
            <type>robndbox</type>
            <name>ship</name>
            <difficult>0</difficult>
            <robndbox>
              <cx>{cx}</cx><cy>{cy}</cy>
              <h>{h}</h><w>{w}</w>
              <angle>{angle}</angle>
            </robndbox>
          </object>
        </annotation>
    """)


def _corners_approx(corners, expected, tol=1e-5):
    assert len(corners) == len(expected)
    for (x, y), (ex, ey) in zip(corners, expected):
        assert abs(x - ex) < tol, f"x={x} expected={ex}"
        assert abs(y - ey) < tol, f"y={y} expected={ey}"


# ---------------------------------------------------------------------------
# Tests: _enforce_long_edge
# ---------------------------------------------------------------------------

class TestEnforceLongEdge:
    def test_already_long_edge_unchanged(self):
        cx, cy, h, w, angle = _enforce_long_edge(10, 20, 30, 10, 0.5)
        assert h == 30 and w == 10 and angle == pytest.approx(0.5)

    def test_swap_when_w_greater_than_h(self):
        # w=30, h=10 → should become h=30, w=10, angle += π/2
        cx, cy, h, w, angle = _enforce_long_edge(10, 20, 10, 30, 0.0)
        assert h == 30
        assert w == 10
        assert angle == pytest.approx(math.pi / 2)

    def test_swap_preserves_center(self):
        cx_out, cy_out, *_ = _enforce_long_edge(100, 200, 5, 50, 1.0)
        assert cx_out == 100 and cy_out == 200

    def test_equal_h_w_no_swap(self):
        # h == w: no swap needed (square box)
        cx, cy, h, w, angle = _enforce_long_edge(0, 0, 20, 20, 1.0)
        assert h == 20 and w == 20 and angle == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: robndbox_to_polygon — axis-aligned boxes
# ---------------------------------------------------------------------------

class TestAxisAlignedBox:
    """angle=0 → long axis vertical → corners match a simple rectangle."""

    def test_angle_zero_produces_axis_aligned_polygon(self):
        """
        Box: center=(256,256), h=20, w=10, angle=0, image=512x512.
        At angle=0 the rotation matrix is identity, so corners should be:
          TL: (256-5, 256-10), TR: (256+5, 256-10),
          BR: (256+5, 256+10), BL: (256-5, 256+10)
        Normalized by 512.
        """
        corners = robndbox_to_polygon(256, 256, 20, 10, 0.0, 512, 512)
        expected = [
            (251/512, 246/512),   # TL: cx - half_w, cy - half_h
            (261/512, 246/512),   # TR
            (261/512, 266/512),   # BR
            (251/512, 266/512),   # BL
        ]
        _corners_approx(corners, expected)

    def test_angle_zero_center_at_origin(self):
        # With cx=cy=0 and angle=0, all corners have negative coordinates.
        # The function still computes them; bounds check is separate.
        corners = robndbox_to_polygon(0, 0, 10, 4, 0.0, 100, 100)
        assert len(corners) == 4


# ---------------------------------------------------------------------------
# Tests: robndbox_to_polygon — π/2 rotation
# ---------------------------------------------------------------------------

class TestHalfPiRotation:
    def test_angle_half_pi_rotates_long_axis_to_horizontal(self):
        """
        Box: center=(256,256), h=20, w=4, angle=π/2, image=512x512.

        At angle=π/2: cos=0, sin=1.
        offset (dx,dy) rotates to (dx*0 - dy*1, dx*1 + dy*0) = (-dy, dx).

        Corners (before rotation): TL=(-2,-10), TR=(2,-10), BR=(2,10), BL=(-2,10)
        After rotation:
          TL: (-(-10), -2) = (10, -2)  → cx+10, cy-2
          TR: (-(-10),  2) = (10,  2)  → cx+10, cy+2
          BR: (-(10),   2) = (-10, 2)  → cx-10, cy+2
          BL: (-(10), -2) = (-10, -2)  → cx-10, cy-2
        """
        corners = robndbox_to_polygon(256, 256, 20, 4, math.pi / 2, 512, 512)
        expected = [
            ((256 + 10) / 512, (256 - 2) / 512),
            ((256 + 10) / 512, (256 + 2) / 512),
            ((256 - 10) / 512, (256 + 2) / 512),
            ((256 - 10) / 512, (256 - 2) / 512),
        ]
        _corners_approx(corners, expected, tol=1e-5)


# ---------------------------------------------------------------------------
# Tests: long-edge enforcement in polygon function
# ---------------------------------------------------------------------------

class TestLongEdgeInPolygon:
    def test_w_greater_than_h_triggers_swap(self):
        """
        Passing w=20, h=4 should produce the same polygon as h=20, w=4, angle+=π/2.
        """
        # swapped input
        corners_swapped = robndbox_to_polygon(256, 256, 4, 20, 0.0, 512, 512)
        # equivalent canonical form
        corners_canonical = robndbox_to_polygon(256, 256, 20, 4, math.pi / 2, 512, 512)
        _corners_approx(corners_swapped, corners_canonical)


# ---------------------------------------------------------------------------
# Tests: normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_all_corners_in_unit_square(self):
        """A well-contained box should produce corners strictly in [0, 1]."""
        corners = robndbox_to_polygon(256, 256, 40, 20, 0.3, 512, 512)
        for x, y in corners:
            assert 0.0 <= x <= 1.0, f"x={x} out of range"
            assert 0.0 <= y <= 1.0, f"y={y} out of range"

    def test_normalization_scales_with_image_size(self):
        """Doubling image size halves the normalized coordinates."""
        c1 = robndbox_to_polygon(100, 100, 20, 10, 0.0, 200, 200)
        c2 = robndbox_to_polygon(100, 100, 20, 10, 0.0, 400, 400)
        for (x1, y1), (x2, y2) in zip(c1, c2):
            assert x1 == pytest.approx(x2 * 2, rel=1e-5)
            assert y1 == pytest.approx(y2 * 2, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests: ._ file skipping
# ---------------------------------------------------------------------------

class TestDotUnderscoreSkip:
    def test_dotunderscore_xml_files_skipped(self):
        """
        convert_split reads a list of stems from the split file and looks up
        <stem>.xml in annotations_dir.  A ._<stem>.xml file should never be
        parsed even if one exists alongside the real file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ann_dir = tmp / "Annotations"
            ann_dir.mkdir()
            labels_dir = tmp / "labels"

            # Write one real annotation
            good_stem = "ship001"
            (ann_dir / f"{good_stem}.xml").write_text(
                _make_xml(256, 256, 20, 10, 0.0)
            )
            # Write a macOS metadata twin that must NOT be parsed
            (ann_dir / f"._{good_stem}.xml").write_text("not valid xml <<>>")

            # Split file references only the real stem
            split_file = tmp / "train.txt"
            split_file.write_text(good_stem + "\n")

            # Should succeed — ._file is never touched
            n = convert_split(split_file, ann_dir, labels_dir)
            assert n == 1
            assert (labels_dir / f"{good_stem}.txt").exists()

    def test_stem_starting_with_dot_underscore_not_in_split(self):
        """
        If somehow a ._stem ended up in the split file it would be skipped
        because there is no ._stem.xml annotation (the annotation has a
        normal name).  Verify convert_split logs a warning and returns 0.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ann_dir = tmp / "Annotations"
            ann_dir.mkdir()
            labels_dir = tmp / "labels"

            split_file = tmp / "train.txt"
            split_file.write_text("._ghost\n")

            n = convert_split(split_file, ann_dir, labels_dir)
            assert n == 0


# ---------------------------------------------------------------------------
# Tests: parse_annotation
# ---------------------------------------------------------------------------

class TestParseAnnotation:
    def test_parses_single_object(self, tmp_path):
        xml = _make_xml(100, 200, 30, 10, 0.5)
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml)

        img_w, img_h, objs = parse_annotation(xml_path)
        assert img_w == 512 and img_h == 512
        assert len(objs) == 1
        assert objs[0]["cx"] == pytest.approx(100)
        assert objs[0]["h"] == pytest.approx(30)
        assert objs[0]["angle"] == pytest.approx(0.5)
