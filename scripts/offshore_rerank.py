"""
Offshore re-ranking: add bright_frac and tail_snr to offshore CSVs.

Computes two additional screening signals for ots_aoi_a and ots_aoi_b only.
Does NOT touch maasvlakte or ijmuiden.

Signals added:
  bright_frac  = fraction of chip pixels >= BRIGHT_THRESH (245)
                 Captures near-saturated pixels; a coherent ship return pushes
                 a cluster of pixels to the top of the uint8 range while pure
                 speckle hits the ceiling only sparsely.

  tail_snr     = (P99 − P75) / max(1, P75 − P25)
                 IQR-normalized right-tail excess.  For pure speckle that fills
                 the range uniformly this ratio is ~1; a chip with a compact
                 bright target (ship) elevates P99 while the IQR stays anchored
                 to the speckle floor, pushing the ratio up.

  pbr          = P99 − P50  (already in CSV, retained as third signal)

CSV column order after this script:
  chip_path, score, pbr, bright_frac, tail_snr

The CSV is left sorted by pbr (the previous sort key).  After the user picks
the winning signal from the contact sheets, re-run with --apply <signal> to
resort and write label_chips.py's canonical sort order.

Contact sheets generated (4 PNG files):
  contactsheet_ots_aoi_a_bright_frac.png
  contactsheet_ots_aoi_a_tail_snr.png
  contactsheet_ots_aoi_b_bright_frac.png
  contactsheet_ots_aoi_b_tail_snr.png

Each sheet shows top-10 + bottom-10 by that signal with all three values
labelled on every thumbnail for cross-reference.

Usage:
    python scripts/offshore_rerank.py            # compute + show both sheets
    python scripts/offshore_rerank.py --apply bright_frac   # resort CSVs
    python scripts/offshore_rerank.py --apply tail_snr      # resort CSVs
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from PIL import Image

LABELS_DIR   = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/data/labels")
FIGS_DIR     = Path("/Volumes/Tejas SSD/sar-dark-ship-detection/reports/figures")
CHIPS_ROOT   = Path("/Volumes/Tejas SSD/datasets/s1_chips")
OFFSHORE_AOIS = ["ots_aoi_a", "ots_aoi_b"]

BRIGHT_THRESH = 245   # uint8; pixels >= this count toward bright_frac
CHIP_AREA     = 512 * 512

FIGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Signal computation ────────────────────────────────────────────────────

def compute_signals(arr: np.ndarray) -> tuple[float, float]:
    """
    Returns (bright_frac, tail_snr) for one uint8 chip.

    bright_frac = fraction of pixels >= BRIGHT_THRESH
    tail_snr    = (P99 − P75) / max(1, P75 − P25)
    """
    p25  = float(np.percentile(arr, 25))
    p75  = float(np.percentile(arr, 75))
    p99  = float(np.percentile(arr, 99))
    iqr  = max(1.0, p75 - p25)

    bright_frac = float((arr >= BRIGHT_THRESH).sum()) / CHIP_AREA
    tail_snr    = (p99 - p75) / iqr
    return bright_frac, tail_snr


# ── CSV update ────────────────────────────────────────────────────────────

def update_csv(aoi: str) -> list[dict]:
    """
    Read existing CSV (chip_path, score, pbr), compute + add bright_frac and
    tail_snr, write back.  Sort order is NOT changed here (stays pbr-sorted).
    """
    csv_path = LABELS_DIR / f"{aoi}_screen.csv"
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "chip_path": row["chip_path"],
                "score":     int(row["score"]),
                "pbr":       float(row["pbr"]),
            })

    print(f"  {aoi}: computing bright_frac + tail_snr for {len(rows)} chips …",
          end=" ", flush=True)
    for row in rows:
        p = Path(row["chip_path"])
        try:
            arr = np.array(Image.open(p).convert("L"))
            bf, ts = compute_signals(arr)
        except Exception:
            bf, ts = 0.0, 1.0
        row["bright_frac"] = round(bf, 6)
        row["tail_snr"]    = round(ts, 4)
    print("done")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["chip_path", "score", "pbr", "bright_frac", "tail_snr"])
        w.writeheader()
        w.writerows(rows)

    # Report distributions
    bfs = np.array([r["bright_frac"] for r in rows])
    tss = np.array([r["tail_snr"]    for r in rows])
    pct = lambda a, q: float(np.percentile(a, q))

    print(f"    bright_frac: min={bfs.min():.4f}  "
          f"p25={pct(bfs,25):.4f}  p50={pct(bfs,50):.4f}  "
          f"p75={pct(bfs,75):.4f}  p90={pct(bfs,90):.4f}  max={bfs.max():.4f}")
    print(f"    tail_snr:    min={tss.min():.3f}  "
          f"p25={pct(tss,25):.3f}  p50={pct(tss,50):.3f}  "
          f"p75={pct(tss,75):.3f}  p90={pct(tss,90):.3f}  max={tss.max():.3f}")
    return rows


# ── Contact sheet ─────────────────────────────────────────────────────────

def make_contact_sheet(
    aoi: str,
    rows: list[dict],
    sort_key: str,
) -> Path:
    """
    Generate one contact sheet sorted by sort_key (bright_frac or tail_snr).
    Each thumbnail shows all three signal values for cross-reference.
    """
    # Sort in memory — does NOT change the CSV
    ordered = sorted(rows, key=lambda r: r[sort_key], reverse=True)
    n       = len(ordered)
    top10   = ordered[:10]
    bot10   = list(reversed(ordered[-10:]))   # lowest first

    if sort_key == "bright_frac":
        signal_label = f"bright_frac (pixels ≥ {BRIGHT_THRESH})"
        sort_abbr    = "BF"
    else:
        signal_label = "tail_snr = (P99−P75)/(P75−P25)"
        sort_abbr    = "TS"

    fig = plt.figure(figsize=(15, 8.5))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"{aoi}  |  ranked by {signal_label}  |  n={n} chips\n"
        f"Top-10 ▲ (green)   Bottom-10 ▼ (red)"
        f"     thumbnail labels: {sort_abbr} · PBR · other",
        color="white", fontsize=8.5, y=0.99,
    )

    outer  = GridSpec(2, 1, figure=fig, hspace=0.60, top=0.91, bottom=0.02)
    top_gs = GridSpecFromSubplotSpec(
        2, 5, subplot_spec=outer[0], hspace=0.08, wspace=0.04)
    bot_gs = GridSpecFromSubplotSpec(
        2, 5, subplot_spec=outer[1], hspace=0.08, wspace=0.04)

    def _fmt_row(row: dict, rank: int) -> str:
        if sort_key == "bright_frac":
            primary   = f"BF={row['bright_frac']*100:.2f}%"
            secondary = f"TS={row['tail_snr']:.2f}"
        else:
            primary   = f"TS={row['tail_snr']:.2f}"
            secondary = f"BF={row['bright_frac']*100:.2f}%"
        return f"#{rank}  {primary}\n{secondary}  PBR={row['pbr']:.0f}"

    def fill_group(chips: list[dict], gs, color: str, rank_list: list[int]) -> None:
        for i, (row, rank) in enumerate(zip(chips, rank_list)):
            ax = fig.add_subplot(gs[i // 5, i % 5])
            p  = Path(row["chip_path"])
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
                _fmt_row(row, rank),
                fontsize=5.2, color=color, pad=1.5,
                linespacing=1.3,
            )

    fill_group(top10, top_gs, "#00e870", list(range(1, 11)))
    fill_group(bot10, bot_gs, "#ff4444", list(range(n, n - 10, -1)))

    fig.text(0.008, 0.955, f"▲ TOP-10  highest {sort_abbr}",
             color="#00e870", fontsize=7.5, va="top")
    fig.text(0.008, 0.490, f"▼ BOTTOM-10  lowest {sort_abbr}",
             color="#ff4444", fontsize=7.5, va="top")

    out = FIGS_DIR / f"contactsheet_{aoi}_{sort_key}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  → {out.name}")
    return out


# ── Apply sort (called with --apply <signal>) ─────────────────────────────

def apply_sort(aoi: str, sort_key: str) -> None:
    csv_path = LABELS_DIR / f"{aoi}_screen.csv"
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "chip_path":  row["chip_path"],
                "score":      int(row["score"]),
                "pbr":        float(row["pbr"]),
                "bright_frac": float(row["bright_frac"]),
                "tail_snr":   float(row["tail_snr"]),
            })
    rows.sort(key=lambda r: r[sort_key], reverse=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["chip_path", "score", "pbr", "bright_frac", "tail_snr"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {aoi}: CSV re-sorted by {sort_key} descending.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", choices=["bright_frac", "tail_snr"],
                        help="Re-sort offshore CSVs by this signal and exit.")
    args = parser.parse_args()

    if args.apply:
        print(f"Applying sort key: {args.apply}")
        for aoi in OFFSHORE_AOIS:
            apply_sort(aoi, args.apply)
        print("Done.  label_chips.py will now present chips in this order.")
        return

    # ── Step 1: Compute + write new columns ──────────────────────────────
    print("=" * 62)
    print("Step 1: Computing bright_frac and tail_snr")
    print("=" * 62)
    all_rows: dict[str, list[dict]] = {}
    for aoi in OFFSHORE_AOIS:
        all_rows[aoi] = update_csv(aoi)

    # ── Step 2: Contact sheets for each signal ────────────────────────────
    print("\n" + "=" * 62)
    print("Step 2: Generating contact sheets")
    print("=" * 62)
    for aoi in OFFSHORE_AOIS:
        print(f"\n{aoi}:")
        for sig in ["bright_frac", "tail_snr"]:
            make_contact_sheet(aoi, all_rows[aoi], sig)

    print(f"\nAll figures → {FIGS_DIR}")
    print("\nNext step: review the 4 contact sheets, then run:")
    print("  python scripts/offshore_rerank.py --apply bright_frac")
    print("  python scripts/offshore_rerank.py --apply tail_snr")
    print("to lock the winning sort order into the CSVs.")


if __name__ == "__main__":
    main()
