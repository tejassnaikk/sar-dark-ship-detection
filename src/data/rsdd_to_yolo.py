"""
Convert RSDD-SAR VOC-style rotated-box annotations to YOLOv11-OBB polygon format.

RSDD-SAR convention:
  <robndbox> cx, cy, h, w, angle (radians)
  Long-edge convention: h >= w  (the angle rotates the long axis)

YOLOv11-OBB format (per line in label .txt):
  <class_id> x1 y1 x2 y2 x3 y3 x4 y4
  All coordinates normalized to [0, 1].
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def _enforce_long_edge(cx: float, cy: float, h: float, w: float, angle: float) -> tuple[float, float, float, float, float]:
    """
    Ensure h >= w (long-edge convention).

    If the annotation was written with w > h, the stored angle is off by π/2.
    We swap h and w and add π/2 to recover the correct orientation.

    This is the #1 silent failure mode in rotated detection: if you train with
    mixed conventions the model learns inconsistent angle targets and converges
    to garbage. Every box must pass through this function before polygon conversion.
    """
    if w > h:
        h, w = w, h
        angle = angle + math.pi / 2
    return cx, cy, h, w, angle


def robndbox_to_polygon(
    cx: float, cy: float, h: float, w: float, angle: float,
    img_w: int, img_h: int,
) -> list[tuple[float, float]]:
    """
    Convert a rotated box (long-edge convention) to 4 normalized corners.

    The box is defined as:
      - center (cx, cy) in pixels
      - half-long-axis along the direction given by `angle`
      - half-short-axis perpendicular

    Returns corners in order: top-left, top-right, bottom-right, bottom-left
    (relative to the rotated frame), normalized to [0, 1].
    """
    cx, cy, h, w, angle = _enforce_long_edge(cx, cy, h, w, angle)

    half_h = h / 2.0
    half_w = w / 2.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Offsets for the four corners in the rotated frame
    offsets = [
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
    ]

    corners = []
    for dx, dy in offsets:
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        corners.append((x / img_w, y / img_h))

    return corners


def _check_bounds(corners: list[tuple[float, float]], filename: str) -> None:
    for x, y in corners:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            log.warning(
                "%s: polygon corner (%.4f, %.4f) outside [0,1] — "
                "box is near/outside image edge (expected for border ships).",
                filename, x, y,
            )
            return


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_annotation(xml_path: Path) -> tuple[int, int, list[dict]]:
    """
    Parse an RSDD-SAR VOC XML file.

    Returns (img_width, img_height, list_of_objects) where each object is a dict
    with keys: name, cx, cy, h, w, angle.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    objects = []
    for obj in root.iter("object"):
        box = obj.find("robndbox")
        if box is None:
            continue
        objects.append({
            "name": obj.findtext("name", default="ship"),
            "cx":    float(box.findtext("cx")),
            "cy":    float(box.findtext("cy")),
            "h":     float(box.findtext("h")),
            "w":     float(box.findtext("w")),
            "angle": float(box.findtext("angle")),
        })

    return img_w, img_h, objects


# ---------------------------------------------------------------------------
# Split conversion
# ---------------------------------------------------------------------------

CLASS_MAP = {"ship": 0}


def convert_split(
    split_file: Path,
    annotations_dir: Path,
    output_labels_dir: Path,
) -> int:
    """
    Convert all annotations listed in `split_file` to YOLO-OBB label files.

    Returns the number of images converted.
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    stem_list = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]

    converted = 0
    for stem in stem_list:
        xml_path = annotations_dir / f"{stem}.xml"
        if not xml_path.exists():
            log.warning("Annotation not found: %s — skipping.", xml_path)
            continue

        img_w, img_h, objects = parse_annotation(xml_path)

        lines = []
        for obj in objects:
            class_id = CLASS_MAP.get(obj["name"], 0)
            corners = robndbox_to_polygon(
                obj["cx"], obj["cy"], obj["h"], obj["w"], obj["angle"],
                img_w, img_h,
            )
            _check_bounds(corners, stem)
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            lines.append(f"{class_id} {coord_str}")

        (output_labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        converted += 1

    return converted


def convert_images_split(
    split_file: Path,
    images_dir: Path,
    output_images_dir: Path,
) -> None:
    """Symlink (or copy) images listed in split_file into output_images_dir."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    stem_list = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    for stem in stem_list:
        for ext in (".jpg", ".png", ".JPG", ".PNG"):
            src = images_dir / f"{stem}{ext}"
            if src.exists():
                dst = output_images_dir / src.name
                if not dst.exists():
                    dst.symlink_to(src.resolve())
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RSDD-SAR to YOLO-OBB format.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--output", default="data/rsdd_yolo", help="Output directory")
    parser.add_argument(
        "--copy-images", action="store_true",
        help="Symlink images into output dir (needed if trainer can't follow paths)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["rsdd_sar_root"])
    annotations_dir = root / cfg["annotations_dir"]
    images_dir = root / cfg["images_dir"]
    imagesets_dir = root / cfg["imagesets_dir"]
    output_dir = Path(args.output)
    splits = cfg["splits"]

    log.info("RSDD-SAR root: %s", root)
    log.info("Output:        %s", output_dir)

    for split in splits:
        split_file = imagesets_dir / f"{split}.txt"
        if not split_file.exists():
            log.warning("Split file not found: %s — skipping.", split_file)
            continue

        out_labels = output_dir / "labels" / split
        n = convert_split(split_file, annotations_dir, out_labels)
        log.info("%-15s → %d label files written to %s", split, n, out_labels)

        if args.copy_images:
            out_imgs = output_dir / "images" / split
            convert_images_split(split_file, images_dir, out_imgs)
            log.info("%-15s → images symlinked to %s", split, out_imgs)

    # Write a YOLO dataset.yaml for the converted data
    yolo_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/test",      # use test as val (no separate val split exists)
        "test":  "images/test",
        "names": cfg["names"],
    }
    out_yaml = output_dir / "dataset.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(yolo_yaml, f, default_flow_style=False)
    log.info("YOLO dataset.yaml written to %s", out_yaml)


if __name__ == "__main__":
    main()
