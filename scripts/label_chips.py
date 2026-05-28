"""
Interactive OBB labeling tool for SAR ship detection ground truth.

Workflow
--------
Chips are presented in score-descending order (high-priority first).
Within chips that screened to score=0, order is randomized (different seed
each session start; seed stored in session JSON so resume is reproducible).

Key bindings
------------
  e   confirmed empty  — write zero-byte .txt, advance to next chip
  w   save OBB labels  — commit drawn boxes, write .txt, advance
  .   defer            — skip for now, append to {aoi}_deferred.csv on disk
                         immediately (not queued in memory)
  q   quit             — save session state; unreviewed chips get no .txt file

OBB drawing
-----------
Click 4 corners of the ship in any order.  A green OBB is fitted via PCA
(fit_obb_from_corners) and drawn after the 4th click.
Press BACKSPACE to undo the last click.
Press 'c' to clear all clicks for the current chip and redraw.
After drawing one box, click 4 more corners to add another box.
Press 'w' to commit all drawn boxes and advance.

Session persistence
-------------------
  data/labels/{aoi}_session.json  — maps chip stem → "empty" | "labeled"
      Loaded on startup; reviewed chips are skipped on resume.
  data/labels/{aoi}_deferred.csv  — appended on each '.' press (not overwritten).
      Chips in deferred.csv are re-queued at the end of the session (after
      all un-reviewed chips), in score-descending order.

Spot-check protocol
-------------------
After the first complete pass (all non-deferred chips reviewed), the tool
resurfaces 20 randomly chosen chips previously marked 'e' (confirmed empty)
for a second-look pass.  If any ship is found during spot-check, the tool
warns that the empty-threshold may be under-detecting and recommends a
re-review of the full AOI.

Label file semantics (CRITICAL)
--------------------------------
  empty .txt (0 bytes)  = confirmed-empty chip  → counted as true negative in eval
  missing .txt          = unreviewed chip        → excluded from eval entirely
  Any ship .txt         = at least one OBB label

NEVER auto-label a chip.  Every label file requires a keypress from a human reviewer.

Usage
-----
  python scripts/label_chips.py <aoi>
  python scripts/label_chips.py maasvlakte
  python scripts/label_chips.py ots_aoi_a

  Requires screen_chips.py to have been run first ({aoi}_screen.csv must exist).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so `src.*` imports work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("MacOSX")  # native macOS interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from src.data.label_io import fit_obb_from_corners, write_obb_label, read_obb_label
from src.data.rsdd_to_yolo import robndbox_to_polygon

# ── Paths ───────────────────────────────────────────────────────────────────
CHIPS_ROOT = Path("/Volumes/Tejas SSD/datasets/s1_chips")
LABELS_DIR = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/data/labels")

CHIP_W = CHIP_H = 512
SPOT_CHECK_N = 20   # number of confirmed-empty chips to resurface for spot-check


# ── Colour scheme ─────────────────────────────────────────────────────────
CLR_PENDING  = "yellow"   # click dots before 4th click
CLR_OBB      = "#00ff88"  # committed OBB outline
CLR_SPOTCK   = "#ff6600"  # spot-check mode highlight


# ── Session state ──────────────────────────────────────────────────────────

class Session:
    """Persistent session for one AOI labeling run."""

    def __init__(self, aoi: str):
        self.aoi = aoi
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        self.session_path  = LABELS_DIR / f"{aoi}_session.json"
        self.deferred_path = LABELS_DIR / f"{aoi}_deferred.csv"

        # chip stem → "empty" | "labeled"
        self.reviewed: dict[str, str] = {}
        # random seed for score=0 ordering (set once, persisted)
        self.zero_seed: Optional[int] = None

        self._load()

    def _load(self) -> None:
        if self.session_path.exists():
            try:
                data = json.loads(self.session_path.read_text())
                self.reviewed  = data.get("reviewed", {})
                self.zero_seed = data.get("zero_seed", None)
                print(f"  Resuming session: {len(self.reviewed)} chips already reviewed.")
            except Exception as e:
                print(f"  WARNING: could not load session file ({e}); starting fresh.")

    def save(self) -> None:
        data = {"reviewed": self.reviewed, "zero_seed": self.zero_seed}
        self.session_path.write_text(json.dumps(data, indent=2))

    def mark(self, chip_stem: str, status: str) -> None:
        """status: 'empty' or 'labeled'"""
        self.reviewed[chip_stem] = status
        self.save()

    def defer(self, chip_path: Path, score: int) -> None:
        """Append one deferred chip to the CSV immediately."""
        write_header = not self.deferred_path.exists()
        with open(self.deferred_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["chip_path", "score"])
            writer.writerow([str(chip_path), score])

    def load_deferred_paths(self) -> list[tuple[Path, int]]:
        """Return deferred chips not yet reviewed, sorted by score desc."""
        if not self.deferred_path.exists():
            return []
        rows = []
        with open(self.deferred_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Path(row["chip_path"])
                s = int(row["score"])
                if p.stem not in self.reviewed:
                    rows.append((p, s))
        rows.sort(key=lambda t: t[1], reverse=True)
        return rows


def load_screen_csv(aoi: str) -> list[tuple[Path, int]]:
    """Load {aoi}_screen.csv → list of (chip_path, score), original order."""
    csv_path = LABELS_DIR / f"{aoi}_screen.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.  Run screen_chips.py first.", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((Path(row["chip_path"]), int(row["score"])))
    return rows


def build_queue(
    screen_rows: list[tuple[Path, int]],
    session: Session,
    deferred_first: bool = False,
) -> list[tuple[Path, int]]:
    """
    Build the ordered presentation queue, excluding already-reviewed chips.

    Order:
      1. score > 0  chips, sorted descending (as in screen CSV)
      2. score == 0 chips, randomized with session.zero_seed
      (3. Deferred chips, appended when deferred_first=False and called after
          first pass — handled separately in main loop)

    On first call, if zero_seed is None, assigns and persists a new seed.
    """
    positives = [(p, s) for p, s in screen_rows if s > 0 and p.stem not in session.reviewed]
    zeros     = [(p, s) for p, s in screen_rows if s == 0 and p.stem not in session.reviewed]

    # Assign seed once, persist so resume is reproducible
    if session.zero_seed is None:
        session.zero_seed = random.randint(0, 2**31 - 1)
        session.save()

    rng = random.Random(session.zero_seed)
    rng.shuffle(zeros)

    return positives + zeros


# ── OBB drawing state ──────────────────────────────────────────────────────

class ChipReviewer:
    """
    Matplotlib-based single-chip reviewer.

    Lifecycle per chip:
      1. show_chip(path, score)  — display image
      2. user clicks corners / presses keys
      3. self.result is set:
           "empty"    → caller writes empty .txt
           "labeled"  → caller calls get_labels() → write_obb_label()
           "deferred" → caller appends to deferred CSV
           "quit"     → caller exits
    """

    def __init__(self):
        self.fig = None
        self.ax  = None
        self._reset_chip_state()
        self.result: Optional[str] = None
        self.current_path: Optional[Path] = None
        self.current_score: int = 0
        self.spot_check_mode: bool = False

    def _new_figure(self) -> None:
        """Create (or recreate) the figure and reconnect event handlers.

        Called at the start of every show_chip() call so that the figure is
        always fresh — plt.close(self.fig) in a key handler destroys the axes
        object, and reusing it on the next chip causes a silent no-op draw.
        """
        plt.close("all")
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

    def _reset_chip_state(self) -> None:
        """Reset per-chip drawing state (not session-level state)."""
        self.clicks: list[tuple[float, float]] = []   # pixel coords
        self.boxes:  list[tuple[float, float, float, float, float]] = []  # (cx,cy,h,w,a)
        self._click_artists:  list = []
        self._obb_artists:    list = []

    def show_chip(
        self,
        path: Path,
        score: int,
        total_remaining: int,
        spot_check: bool = False,
    ) -> None:
        """Display a chip and block until the user presses a key."""
        self.result = None
        self.current_path  = path
        self.current_score = score
        self.spot_check_mode = spot_check
        self._reset_chip_state()

        # Recreate figure — previous figure was closed by the key handler
        self._new_figure()
        try:
            img = np.array(Image.open(path))
        except Exception as e:
            print(f"  WARNING: could not open {path}: {e}")
            self.result = "deferred"
            return

        self.ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        self.ax.set_xlim(0, CHIP_W)
        self.ax.set_ylim(CHIP_H, 0)

        mode_tag = " [SPOT-CHECK]" if spot_check else ""
        self.ax.set_title(
            f"{path.name}  |  score={score}  |  {total_remaining} remaining{mode_tag}\n"
            "e=empty  w=save  .=defer  q=quit  | click 4 corners for OBB | "
            "backspace=undo click  c=clear boxes",
            fontsize=8,
            color=CLR_SPOTCK if spot_check else "white",
        )
        self.fig.patch.set_facecolor("#1a1a1a")
        self.ax.set_facecolor("#1a1a1a")
        self.ax.tick_params(colors="#666666")
        for spine in self.ax.spines.values():
            spine.set_edgecolor(CLR_SPOTCK if spot_check else "#444444")

        self._redraw()
        plt.show(block=True)

    # ── event handlers ────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        x, y = float(event.xdata), float(event.ydata)
        self.clicks.append((x, y))
        self._redraw()

        if len(self.clicks) == 4:
            self._fit_and_draw_obb()
            self.clicks = []   # reset for next box on same chip

    def _on_key(self, event) -> None:
        key = event.key
        if key == "e":
            self.result = "empty"
            plt.close(self.fig)
        elif key == "w":
            if not self.boxes and not self.clicks:
                # No boxes drawn and no pending clicks — confirm intent
                self._set_title_warn("No boxes drawn — draw boxes then press w, "
                                     "or press e for confirmed empty")
            else:
                self.result = "labeled"
                plt.close(self.fig)
        elif key == ".":
            self.result = "deferred"
            plt.close(self.fig)
        elif key == "q":
            self.result = "quit"
            plt.close(self.fig)
        elif key == "backspace":
            if self.clicks:
                self.clicks.pop()
                self._redraw()
        elif key == "c":
            self.boxes = []
            self.clicks = []
            for a in self._click_artists + self._obb_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            self._click_artists = []
            self._obb_artists = []
            self._redraw()

    # ── drawing helpers ───────────────────────────────────────────────────

    def _redraw(self) -> None:
        # Remove stale click dots
        for a in self._click_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._click_artists = []

        # Draw pending clicks
        for i, (x, y) in enumerate(self.clicks):
            dot, = self.ax.plot(x, y, "o", color=CLR_PENDING, markersize=6, zorder=10)
            lbl  = self.ax.text(x + 4, y - 4, str(i + 1), color=CLR_PENDING,
                                fontsize=7, zorder=10)
            self._click_artists.extend([dot, lbl])

        # Draw all committed OBBs
        for a in self._obb_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._obb_artists = []
        for cx, cy, h, w, angle in self.boxes:
            self._draw_obb_outline(cx, cy, h, w, angle)

        self.fig.canvas.draw_idle()

    def _fit_and_draw_obb(self) -> None:
        """Fit OBB to 4 clicks, store in self.boxes, draw outline."""
        try:
            cx, cy, h, w, angle = fit_obb_from_corners(self.clicks)
        except ValueError as e:
            self._set_title_warn(f"OBB fit failed: {e}")
            return
        self.boxes.append((cx, cy, h, w, angle))
        self._draw_obb_outline(cx, cy, h, w, angle)
        self.fig.canvas.draw_idle()

    def _draw_obb_outline(self, cx, cy, h, w, angle) -> None:
        corners_norm = robndbox_to_polygon(cx, cy, h, w, angle, CHIP_W, CHIP_H)
        corners_px = [(x * CHIP_W, y * CHIP_H) for x, y in corners_norm]
        xs = [p[0] for p in corners_px] + [corners_px[0][0]]
        ys = [p[1] for p in corners_px] + [corners_px[0][1]]
        line, = self.ax.plot(xs, ys, "-", color=CLR_OBB, linewidth=1.5, zorder=8)
        dot,  = self.ax.plot(cx, cy, "+", color=CLR_OBB, markersize=8, zorder=9)
        self._obb_artists.extend([line, dot])

    def _set_title_warn(self, msg: str) -> None:
        self.ax.set_title(msg, fontsize=9, color="#ff4444")
        self.fig.canvas.draw_idle()

    def get_labels(self) -> list[tuple[float, float, float, float, float]]:
        """Return the list of (cx, cy, h, w, angle) OBBs drawn for current chip."""
        return list(self.boxes)


# ── Main labeling loop ─────────────────────────────────────────────────────

def label_aoi(aoi: str) -> None:
    session = Session(aoi)
    screen_rows = load_screen_csv(aoi)
    print(f"\nLoaded {len(screen_rows)} chips from screen CSV.")

    # Labels directory for this AOI
    aoi_label_dir = LABELS_DIR / aoi
    aoi_label_dir.mkdir(parents=True, exist_ok=True)

    queue = build_queue(screen_rows, session)
    total_in_pass = len(queue)
    print(f"Queue: {total_in_pass} chips to review "
          f"({sum(1 for _,s in queue if s > 0)} score>0, "
          f"{sum(1 for _,s in queue if s == 0)} score=0)")
    if len(session.reviewed) > 0:
        print(f"       ({len(session.reviewed)} already reviewed — skipped)")

    reviewer = ChipReviewer()
    reviewed_this_session = 0

    def process_chip(path: Path, score: int, total_remaining: int,
                     spot_check: bool = False) -> str:
        """Present chip, handle keypress, write label, return result string."""
        nonlocal reviewed_this_session
        reviewer.show_chip(path, score, total_remaining, spot_check=spot_check)
        result = reviewer.result

        label_path = aoi_label_dir / (path.stem + ".txt")

        if result == "empty":
            write_obb_label(label_path, [], CHIP_W, CHIP_H)
            session.mark(path.stem, "empty")
            reviewed_this_session += 1
            print(f"  [empty]   {path.name}")

        elif result == "labeled":
            labels = reviewer.get_labels()
            if labels:
                write_obb_label(label_path, labels, CHIP_W, CHIP_H)
                session.mark(path.stem, "labeled")
                reviewed_this_session += 1
                print(f"  [labeled] {path.name}  ({len(labels)} box{'es' if len(labels)!=1 else ''})")
            else:
                # 'w' pressed with no boxes drawn  → treat as empty with warning
                write_obb_label(label_path, [], CHIP_W, CHIP_H)
                session.mark(path.stem, "empty")
                reviewed_this_session += 1
                print(f"  [empty*]  {path.name}  (w pressed with no boxes — saved as empty)")

        elif result == "deferred":
            session.defer(path, score)
            print(f"  [deferred] {path.name}  (score={score})")
            # NOTE: no label file written, not marked reviewed

        elif result == "quit":
            print(f"\n  Quit after {reviewed_this_session} chips this session.")
            session.save()
            return "quit"

        return result or "deferred"

    # ── First pass ─────────────────────────────────────────────────────────
    for i, (path, score) in enumerate(queue):
        remaining = len(queue) - i
        result = process_chip(path, score, remaining)
        if result == "quit":
            _print_summary(session, aoi)
            return

    # ── Deferred pass ──────────────────────────────────────────────────────
    deferred = session.load_deferred_paths()
    if deferred:
        print(f"\n── Deferred pass: {len(deferred)} chips ───────────────────────────")
        for i, (path, score) in enumerate(deferred):
            remaining = len(deferred) - i
            result = process_chip(path, score, remaining)
            if result == "quit":
                _print_summary(session, aoi)
                return

    # ── Spot-check pass (20 random confirmed-empty chips) ──────────────────
    empty_chips = [
        stem for stem, status in session.reviewed.items()
        if status == "empty"
    ]
    if len(empty_chips) >= 1:
        n_spot = min(SPOT_CHECK_N, len(empty_chips))
        spot_sample = random.sample(empty_chips, n_spot)
        print(f"\n── Spot-check pass: {n_spot} confirmed-empty chips re-surfaced ──────")

        # Build path→score lookup
        score_map = {p.stem: s for p, s in screen_rows}
        spot_found_ship = False

        for i, stem in enumerate(spot_sample):
            # Find path
            chip_path = CHIPS_ROOT / aoi / f"{stem}.png"
            if not chip_path.exists():
                continue
            score = score_map.get(stem, 0)
            remaining = n_spot - i

            result = process_chip(chip_path, score, remaining, spot_check=True)
            if result == "labeled":
                spot_found_ship = True
                print(f"  *** SPOT-CHECK HIT: {stem}.png was previously marked empty "
                      f"but now has a box! ***")
            if result == "quit":
                _print_summary(session, aoi)
                return

        if spot_found_ship:
            print("\n⚠  WARNING: At least one spot-check chip was re-labeled as containing"
                  " a ship.")
            print("   This indicates the brightness threshold may be under-detecting dim"
                  " targets.")
            print("   Recommendation: manually review all score=0 confirmed-empty chips"
                  " for this AOI before computing mAP.")
        else:
            print(f"\n✓  Spot-check passed: 0/{n_spot} empty chips contained ships.")

    _print_summary(session, aoi)
    plt.close("all")


def _print_summary(session: Session, aoi: str) -> None:
    n_empty   = sum(1 for s in session.reviewed.values() if s == "empty")
    n_labeled = sum(1 for s in session.reviewed.values() if s == "labeled")

    # Count label files on disk
    aoi_label_dir = LABELS_DIR / aoi
    n_files = sum(1 for f in aoi_label_dir.glob("*.txt")) if aoi_label_dir.exists() else 0

    print(f"\n── Session summary: {aoi} ─────────────────────────────────────────")
    print(f"   Reviewed total : {len(session.reviewed)}")
    print(f"     confirmed empty : {n_empty}")
    print(f"     with labels     : {n_labeled}")
    print(f"   Label files on disk: {n_files}")
    print(f"   Session file: {session.session_path}")
    print(f"   Deferred CSV: {session.deferred_path}")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("aoi",
                        choices=["maasvlakte", "ijmuiden", "ots_aoi_a", "ots_aoi_b"],
                        help="AOI to label")
    args = parser.parse_args()

    # Verify screen CSV exists
    csv_path = LABELS_DIR / f"{args.aoi}_screen.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        print("Run first:  python scripts/screen_chips.py", args.aoi, file=sys.stderr)
        sys.exit(1)

    label_aoi(args.aoi)


if __name__ == "__main__":
    main()
