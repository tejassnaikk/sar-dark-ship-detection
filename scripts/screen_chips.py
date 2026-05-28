"""
Chip pre-screener for SAR dark-ship labeling pipeline.

PURPOSE: triage/sort only.  This script NEVER writes label files.
Output is a CSV of chips sorted by score (descending) so label_chips.py
can present high-priority chips first.

Score definition
----------------
  score = number of pixels in the chip that are:
    (a) a local maximum in a 11×11 neighborhood (scipy.ndimage.maximum_filter)
    AND
    (b) strictly greater than chip_median + 40 uint8 units

This captures individual bright scatterers (ship superstructure, corner
reflectors) while ignoring diffuse ocean clutter.  A chip with score=0
is NOT confirmed empty — it must still be reviewed by a human.

Output
------
  data/labels/{aoi}_screen.csv
  Columns: chip_path, score
  Sorted descending by score.  score=0 chips appear last (in file order —
  label_chips.py randomizes the score=0 tail on load).

Usage
-----
  python scripts/screen_chips.py <aoi>
  python scripts/screen_chips.py maasvlakte
  python scripts/screen_chips.py ijmuiden
  python scripts/screen_chips.py ots_aoi_a
  python scripts/screen_chips.py ots_aoi_b

  Run with --all to process all four AOIs at once.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

# ── Paths ───────────────────────────────────────────────────────────────────
CHIPS_ROOT  = Path("/Volumes/Tejas SSD/datasets/s1_chips")
LABELS_DIR  = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/data/labels")

KNOWN_AOIS = ["maasvlakte", "ijmuiden", "ots_aoi_a", "ots_aoi_b"]

# Screener parameters
MAX_FILTER_SIZE = 11   # neighborhood for local-maximum detection
BRIGHT_OFFSET   = 40   # uint8 units above chip median to call a pixel "bright"


def compute_score(chip_path: Path) -> int:
    """
    Return the brightness-peak score for one 512×512 uint8 chip.

    score = number of local maxima (in an 11×11 neighborhood) that are
            strictly greater than chip_median + BRIGHT_OFFSET.

    Returns 0 for any chip that cannot be opened or is not grayscale.
    """
    try:
        img = Image.open(chip_path)
        arr = np.array(img)
    except Exception:
        return 0

    if arr.ndim != 2:
        # Convert RGB → grayscale if needed (shouldn't happen with our pipeline)
        arr = np.array(img.convert("L"))

    median_val = float(np.median(arr))
    threshold  = median_val + BRIGHT_OFFSET

    # Local maxima: pixel value equals neighborhood maximum
    local_max = maximum_filter(arr, size=MAX_FILTER_SIZE)
    is_local_max = (arr == local_max)

    bright_peaks = is_local_max & (arr.astype(float) > threshold)
    return int(bright_peaks.sum())


def screen_aoi(aoi: str) -> Path:
    """
    Screen all chips in one AOI directory.

    Writes data/labels/{aoi}_screen.csv and returns the path.
    Skips ._* macOS metadata files.
    """
    chip_dir = CHIPS_ROOT / aoi
    if not chip_dir.exists():
        print(f"ERROR: chip directory not found: {chip_dir}", file=sys.stderr)
        sys.exit(1)

    chip_paths = sorted(
        p for p in chip_dir.glob("*.png")
        if not p.name.startswith("._")
    )
    if not chip_paths:
        print(f"WARNING: no chips found in {chip_dir}", file=sys.stderr)
        return

    print(f"\n── {aoi}  ({len(chip_paths)} chips) ─────────────────────────────")

    scores: list[tuple[Path, int]] = []
    for i, cp in enumerate(chip_paths, 1):
        s = compute_score(cp)
        scores.append((cp, s))
        if i % 50 == 0 or i == len(chip_paths):
            print(f"   {i}/{len(chip_paths)} scored …", end="\r", flush=True)

    print()  # newline after \r progress

    # Sort descending by score; ties keep file order (stable sort)
    scores.sort(key=lambda t: t[1], reverse=True)

    # Write CSV
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LABELS_DIR / f"{aoi}_screen.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_path", "score"])
        for cp, s in scores:
            writer.writerow([str(cp), s])

    # Distribution summary
    score_arr = np.array([s for _, s in scores])
    zeros   = int((score_arr == 0).sum())
    one     = int((score_arr == 1).sum())
    two     = int((score_arr == 2).sum())
    three_p = int((score_arr >= 3).sum())
    max_s   = int(score_arr.max())
    median_s = float(np.median(score_arr))

    print(f"   Score distribution:")
    print(f"     0        : {zeros:4d} chips  ({100*zeros/len(scores):.1f}%)")
    print(f"     1        : {one:4d} chips  ({100*one/len(scores):.1f}%)")
    print(f"     2        : {two:4d} chips  ({100*two/len(scores):.1f}%)")
    print(f"     ≥3       : {three_p:4d} chips  ({100*three_p/len(scores):.1f}%)")
    print(f"     max      : {max_s}")
    print(f"     median   : {median_s:.1f}")
    print(f"   CSV → {csv_path}")

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("aoi", nargs="?", choices=KNOWN_AOIS + ["--all"],
                        help="AOI name to screen")
    parser.add_argument("--all", dest="all_aois", action="store_true",
                        help="Screen all four AOIs")
    args = parser.parse_args()

    if args.all_aois or args.aoi is None:
        for aoi in KNOWN_AOIS:
            screen_aoi(aoi)
    else:
        screen_aoi(args.aoi)

    print("\nDone.  Run label_chips.py <aoi> to begin labeling.")


if __name__ == "__main__":
    main()
