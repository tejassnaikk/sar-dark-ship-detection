"""
Re-review already-labeled chips for land vs water targets.

Shows each labeled chip with its existing OBB box(es) drawn.

Keys:
  k   keep   — label correct, target is in water, advance
  d   delete — label wrong (land / building), removes the .txt and
               un-reviews the chip so it re-queues in label_chips.py
  q   quit   — save session state and exit

Chips are shown in the order they appear in {aoi}_session.json (i.e., the
order they were originally labeled).  Already-corrected chips are skipped
on resume so you can re-run safely.

Usage:
    python scripts/review_labels.py maasvlakte
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CHIPS_ROOT = Path("/Volumes/Tejas SSD/datasets/s1_chips")
LABELS_DIR = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/data/labels")
CHIP_W = CHIP_H = 512

# Colours
CLR_BOX   = "#00e870"   # existing box outline
CLR_KEEP  = "#00e870"
CLR_DEL   = "#ff4444"


def read_label_polygons(label_path: Path) -> list[list[float]]:
    """Return list of 8-float normalized polygons from a YOLO OBB .txt file."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    polys = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 9:
            polys.append([float(v) for v in parts[1:]])
    return polys


def draw_polygon(ax, poly_norm: list[float], color: str) -> None:
    """Draw one normalized 8-float polygon on ax (pixel coords)."""
    xs = [poly_norm[i] * CHIP_W for i in range(0, 8, 2)]
    ys = [poly_norm[i] * CHIP_H for i in range(1, 8, 2)]
    xs.append(xs[0]); ys.append(ys[0])
    ax.plot(xs, ys, "-", color=color, linewidth=2, zorder=8)
    cx = sum(xs[:-1]) / 4; cy = sum(ys[:-1]) / 4
    ax.plot(cx, cy, "+", color=color, markersize=10, zorder=9)


def review_aoi(aoi: str) -> None:
    session_path = LABELS_DIR / f"{aoi}_session.json"
    if not session_path.exists():
        print(f"No session file found for {aoi}.", file=sys.stderr)
        sys.exit(1)

    session_data = json.loads(session_path.read_text())
    reviewed: dict[str, str] = session_data["reviewed"]

    # Only chips currently marked "labeled", in session insertion order
    labeled_stems = [s for s, v in reviewed.items() if v == "labeled"]
    if not labeled_stems:
        print("No labeled chips found — nothing to review.")
        return

    aoi_label_dir = LABELS_DIR / aoi

    # Track which chips we've already confirmed/deleted this review pass
    # (stored as a separate key in session so resume works)
    confirmed: set[str] = set(session_data.get("review_confirmed", []))
    deleted:   set[str] = set(session_data.get("review_deleted",   []))
    already_done = confirmed | deleted

    to_review = [s for s in labeled_stems if s not in already_done]
    total     = len(labeled_stems)
    done_so_far = len(already_done)

    print(f"\nmaasvlakte re-review: {total} labeled chips")
    print(f"  Already confirmed : {len(confirmed)}")
    print(f"  Already deleted   : {len(deleted)}")
    print(f"  To review now     : {len(to_review)}")
    if not to_review:
        print("All chips already reviewed — re-review complete.")
        return

    result = {"action": None}

    def save_session() -> None:
        session_data["review_confirmed"] = sorted(confirmed)
        session_data["review_deleted"]   = sorted(deleted)
        session_path.write_text(json.dumps(session_data, indent=2))

    kept = 0; deleted_count = 0
    for idx, stem in enumerate(to_review):
        result["action"] = None

        # Fresh figure each chip — same fix as label_chips.py
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        chip_path  = CHIPS_ROOT / aoi / f"{stem}.png"
        label_path = aoi_label_dir / f"{stem}.txt"
        polys      = read_label_polygons(label_path)

        try:
            img = np.array(Image.open(chip_path).convert("L"))
        except Exception:
            img = np.zeros((CHIP_W, CHIP_H), dtype=np.uint8)

        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xlim(0, CHIP_W); ax.set_ylim(CHIP_H, 0)

        for poly in polys:
            draw_polygon(ax, poly, CLR_BOX)

        n_boxes = len(polys)
        ax.set_title(
            f"{stem}  |  {n_boxes} box{'es' if n_boxes != 1 else ''}  |"
            f"  {idx + 1}/{len(to_review)} (#{done_so_far + idx + 1} of {total})\n"
            "k = keep (water target)     d = delete (land target)     q = quit",
            fontsize=8, color="white",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.tick_params(colors="#666666")

        def on_key(event) -> None:
            if event.key in ("k", "d", "q"):
                result["action"] = event.key
                plt.close("all")

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=True)

        action = result["action"]
        if action == "k":
            confirmed.add(stem)
            kept += 1
            print(f"  [keep]   {stem}")
            save_session()
        elif action == "d":
            # Remove label file
            label_path = aoi_label_dir / f"{stem}.txt"
            if label_path.exists():
                label_path.unlink()
            # Remove from reviewed so it re-queues in label_chips.py
            reviewed.pop(stem, None)
            deleted.add(stem)
            deleted_count += 1
            print(f"  [DELETE] {stem}  — label removed, chip returned to queue")
            save_session()
        elif action == "q":
            print(f"\nQuit after reviewing {idx} chip(s) this run.")
            save_session()
            break
    else:
        save_session()
        print(f"\nRe-review complete: {kept} kept, {deleted_count} deleted.")
        remaining_labeled = sum(1 for v in reviewed.values() if v == "labeled")
        print(f"Labeled chips remaining: {remaining_labeled}")

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("aoi", nargs="?", default="maasvlakte",
                        choices=["maasvlakte", "ijmuiden", "ots_aoi_a", "ots_aoi_b"])
    args = parser.parse_args()
    review_aoi(args.aoi)


if __name__ == "__main__":
    main()
