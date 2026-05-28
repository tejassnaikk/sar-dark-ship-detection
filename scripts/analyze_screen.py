"""
Screener diagnostic, PBR re-ranking, and contact sheets.

Part 1  Pixel intensity histograms for 10 random chips per AOI.
        Shows where median+40 falls vs the actual speckle distribution.

Part 2  Add PBR (peak-to-background ratio = P99 − P50) as a second column
        in each _screen.csv and re-sort descending by PBR.
        Old 'score' column (local-maxima count) is retained for reference.

Part 3  Contact-sheet PNG per AOI: top-10 chips (highest PBR, most likely
        ship-bearing) and bottom-10 chips (lowest PBR, most likely empty),
        labelled with rank and PBR value.

Usage:
    python scripts/analyze_screen.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from PIL import Image

CHIPS_ROOT = Path("/Volumes/Tejas SSD/datasets/s1_chips")
LABELS_DIR = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/data/labels")
FIGS_DIR   = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/reports/figures")
AOIS       = ["maasvlakte", "ijmuiden", "ots_aoi_a", "ots_aoi_b"]

FIGS_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)   # reproducible random chip selection for histograms


# ── Part 1: Histograms ─────────────────────────────────────────────────────

def plot_histograms(aoi: str, n: int = 10) -> Path:
    chip_dir = CHIPS_ROOT / aoi
    chips = sorted(p for p in chip_dir.glob("*.png") if not p.name.startswith("._"))
    sample = random.sample(chips, min(n, len(chips)))

    ncols = min(5, len(sample))
    nrows = (len(sample) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"{aoi} — pixel intensity histograms (10 random chips)  |  "
        "yellow=P50  orange dashes=P50+40  red=P99",
        fontsize=9, color="white", y=1.01,
    )

    all_p50, all_p99, all_pbr = [], [], []

    for idx, chip_path in enumerate(sample):
        ax = axes[idx // ncols, idx % ncols]
        ax.set_facecolor("#111111")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        arr = np.array(Image.open(chip_path).convert("L"))
        p50 = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
        pbr = p99 - p50
        all_p50.append(p50); all_p99.append(p99); all_pbr.append(pbr)

        ax.hist(arr.ravel(), bins=128, range=(0, 255),
                color="#4488cc", alpha=0.85, log=True, edgecolor="none")

        ax.axvline(p50,       color="#ffcc00", lw=1.5, zorder=5)
        ax.axvline(p50 + 40,  color="#ff8800", lw=1.0, linestyle="--", zorder=5)
        ax.axvline(p99,       color="#ff4444", lw=1.5, zorder=5)

        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0.9)
        ax.set_title(
            f"P50={p50:.0f}  P90={p90:.0f}  P99={p99:.0f}  PBR={pbr:.0f}",
            fontsize=6.5, color="#cccccc", pad=2,
        )
        ax.tick_params(colors="#666666", labelsize=5)
        ax.set_xlabel("uint8 pixel value", fontsize=5.5, color="#888888")
        ax.set_ylabel("count (log)", fontsize=5.5, color="#888888")

    # Hide any spare axes
    for idx in range(len(sample), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    out = FIGS_DIR / f"hist_{aoi}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)

    print(f"  {aoi}:")
    print(f"    P50  range   : {min(all_p50):.0f} – {max(all_p50):.0f}")
    print(f"    P50+40 range : {min(all_p50)+40:.0f} – {max(all_p50)+40:.0f}  ← old threshold")
    print(f"    P99  range   : {min(all_p99):.0f} – {max(all_p99):.0f}")
    print(f"    PBR  range   : {min(all_pbr):.0f} – {max(all_pbr):.0f}")
    return out


# ── Part 2: Add PBR, re-sort ───────────────────────────────────────────────

def add_pbr_resort(aoi: str) -> list[dict]:
    csv_path = LABELS_DIR / f"{aoi}_screen.csv"
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"chip_path": row["chip_path"],
                         "score":     int(row["score"])})

    print(f"  {aoi}: computing PBR for {len(rows)} chips …", end=" ", flush=True)
    for row in rows:
        p = Path(row["chip_path"])
        try:
            arr = np.array(Image.open(p).convert("L"))
            pbr = float(np.percentile(arr, 99) - np.median(arr))
        except Exception:
            pbr = 0.0
        row["pbr"] = round(pbr, 1)
    print("done")

    rows.sort(key=lambda r: r["pbr"], reverse=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chip_path", "score", "pbr"])
        w.writeheader()
        w.writerows(rows)

    pbrs = np.array([r["pbr"] for r in rows])
    p = lambda q: float(np.percentile(pbrs, q))
    print(f"    PBR: min={pbrs.min():.1f}  "
          f"p25={p(25):.1f}  p50={p(50):.1f}  p75={p(75):.1f}  "
          f"p90={p(90):.1f}  p99={p(99):.1f}  max={pbrs.max():.1f}")

    # Histogram-style bins
    bins = [0, 20, 40, 60, 80, 100, 150, 200, 500]
    for lo, hi in zip(bins, bins[1:]):
        n = ((pbrs >= lo) & (pbrs < hi)).sum()
        if n > 0:
            bar = "█" * max(1, int(20 * n / len(pbrs)))
            print(f"    [{lo:>4}–{hi:<4}) : {n:4d}  {bar}")
    n_top = (pbrs >= 500).sum()
    if n_top:
        print(f"    [ 500+    ) : {n_top:4d}")

    return rows


# ── Part 3: Contact sheets ─────────────────────────────────────────────────

def make_contact_sheet(aoi: str, rows: list[dict]) -> Path:
    n = len(rows)
    top10 = rows[:10]
    bot10 = list(reversed(rows[-10:]))  # lowest PBR first

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"{aoi}  |  contact sheet  |  n={n} chips\n"
        f"Top-10 (green, highest PBR = P99−P50) · Bottom-10 (red, lowest PBR)",
        color="white", fontsize=9, y=0.98,
    )

    # Two groups with a clear gap
    outer  = GridSpec(2, 1, figure=fig, hspace=0.55, top=0.90, bottom=0.03)
    top_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[0], hspace=0.05, wspace=0.04)
    bot_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[1], hspace=0.05, wspace=0.04)

    def fill_group(chips: list[dict], gs, color: str, ranks: list[int]) -> None:
        for i, (row, rank) in enumerate(zip(chips, ranks)):
            ax = fig.add_subplot(gs[i // 5, i % 5])
            p = Path(row["chip_path"])
            try:
                img = np.array(Image.open(p).convert("L"))
            except Exception:
                img = np.zeros((512, 512), dtype=np.uint8)

            ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            ax.set_title(
                f"#{rank}  PBR={row['pbr']:.0f}",
                fontsize=6, color=color, pad=1.5,
            )

    fill_group(top10, top_gs, "#00e870",
               ranks=list(range(1, 11)))
    fill_group(bot10, bot_gs, "#ff4444",
               ranks=list(range(n, n - 10, -1)))

    # Group labels
    fig.text(0.01, 0.955, "▲ TOP-10 (most likely ship-bearing)",
             color="#00e870", fontsize=7, va="top")
    fig.text(0.01, 0.490, "▼ BOTTOM-10 (most likely empty)",
             color="#ff4444", fontsize=7, va="top")

    out = FIGS_DIR / f"contactsheet_{aoi}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Contact sheet: {out.name}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("Part 1: Pixel intensity histograms")
    print("=" * 62)
    for aoi in AOIS:
        plot_histograms(aoi)

    print("\n" + "=" * 62)
    print("Part 2: Add PBR = P99−P50, re-sort CSVs descending")
    print("=" * 62)
    all_rows: dict[str, list[dict]] = {}
    for aoi in AOIS:
        all_rows[aoi] = add_pbr_resort(aoi)

    print("\n" + "=" * 62)
    print("Part 3: Contact sheets")
    print("=" * 62)
    for aoi in AOIS:
        make_contact_sheet(aoi, all_rows[aoi])

    print("\nAll outputs written to:")
    print(f"  Figures : {FIGS_DIR}")
    print(f"  CSVs    : {LABELS_DIR}")


if __name__ == "__main__":
    main()
