"""
YOLOv11n-OBB inference on all 1063 chips → predictions.parquet + ground_truth.parquet.

Outputs (reports/):
  predictions.parquet    one row per predicted box
  ground_truth.parquet   one row per GT box

Parquet schema
--------------
  predictions:   image_id str | polygon list[float8] | confidence float | split str
  ground_truth:  image_id str | polygon list[float8] | split str

  polygon = 8 normalized floats [x1,y1,x2,y2,x3,y3,x4,y4] in [0,1].
  split   = "inshore" (maasvlakte, ijmuiden) | "offshore" (ots_aoi_a, ots_aoi_b).

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --conf 0.01 --iou 0.5 --batch 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.data.label_io import read_obb_label

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_PATH  = _ROOT / "models_local" / "best.pt"
CHIPS_ROOT  = Path("/Volumes/Tejas SSD/datasets/s1_chips")
LABELS_DIR  = _ROOT / "data" / "labels"
REPORTS_DIR = _ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PRED_OUT = REPORTS_DIR / "predictions.parquet"
GT_OUT   = REPORTS_DIR / "ground_truth.parquet"

AOI_SPLIT = {
    "maasvlakte": "inshore",
    "ijmuiden":   "inshore",
    "ots_aoi_a":  "offshore",
    "ots_aoi_b":  "offshore",
}
CHIP_SIZE = 512


def chips_for_aoi(aoi: str) -> list[Path]:
    return sorted(
        p for p in (CHIPS_ROOT / aoi).glob("*.png")
        if not p.name.startswith("._")
    )


def run(conf_thresh: float, iou_thresh: float, batch_size: int) -> None:
    print(f"Model : {MODEL_PATH}")
    print(f"Conf  : {conf_thresh}   NMS IoU : {iou_thresh}   Batch : {batch_size}")

    model = YOLO(str(MODEL_PATH))

    pred_rows: list[dict] = []
    gt_rows:   list[dict] = []

    for aoi, split in AOI_SPLIT.items():
        paths = chips_for_aoi(aoi)
        print(f"\n── {aoi}  ({len(paths)} chips, split={split}) ──────────────────────")

        # ── Inference in batches ────────────────────────────────────────────
        n_pred_aoi = 0
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            results = model.predict(
                source=[str(p) for p in batch],
                imgsz=CHIP_SIZE,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
            )
            for chip_path, result in zip(batch, results):
                stem = chip_path.stem
                if result.obb is None or len(result.obb) == 0:
                    continue
                # Normalized 4-corner coords: try xyxyxyxyn first, fall back
                try:
                    corners_n = result.obb.xyxyxyxyn.cpu().numpy()  # (N, 4, 2)
                except AttributeError:
                    corners_n = result.obb.xyxyxyxy.cpu().numpy() / CHIP_SIZE
                confs = result.obb.conf.cpu().numpy()
                for j in range(len(confs)):
                    poly = corners_n[j].flatten().tolist()  # [x1,y1,...,x4,y4]
                    pred_rows.append({
                        "image_id":   stem,
                        "polygon":    poly,
                        "confidence": float(confs[j]),
                        "split":      split,
                    })
                    n_pred_aoi += 1

            end = min(start + batch_size, len(paths))
            print(f"   {end}/{len(paths)} …", end="\r", flush=True)

        print(f"   {len(paths)}/{len(paths)}   predictions: {n_pred_aoi}")

        # ── Ground truth from label files ────────────────────────────────────
        n_gt_aoi = 0
        for chip_path in paths:
            label_path = LABELS_DIR / aoi / f"{chip_path.stem}.txt"
            boxes = read_obb_label(label_path)
            for box in boxes:
                gt_rows.append({
                    "image_id": chip_path.stem,
                    "polygon":  box["polygon"],
                    "split":    split,
                })
                n_gt_aoi += 1
        print(f"   GT boxes: {n_gt_aoi}")

    # ── Write parquets ─────────────────────────────────────────────────────
    pred_df = pd.DataFrame(pred_rows)
    gt_df   = pd.DataFrame(gt_rows)

    # polygon column must be stored as object (list) — explicitly cast
    if not pred_df.empty:
        pred_df["polygon"] = pred_df["polygon"].apply(list)
    if not gt_df.empty:
        gt_df["polygon"] = gt_df["polygon"].apply(list)

    pred_df.to_parquet(PRED_OUT, index=False)
    gt_df.to_parquet(GT_OUT,   index=False)

    print(f"\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Total predictions : {len(pred_df)}")
    print(f"  Total GT boxes    : {len(gt_df)}")
    for split in ["inshore", "offshore"]:
        n_p = (pred_df["split"] == split).sum() if not pred_df.empty else 0
        n_g = (gt_df["split"]   == split).sum() if not gt_df.empty  else 0
        print(f"  {split:10s}  preds={n_p:5d}  gt={n_g:4d}")
    print(f"\n  predictions  → {PRED_OUT}")
    print(f"  ground_truth → {GT_OUT}")
    print(f"\nNext:")
    print(f"  python -m src.eval.obb_metrics \\")
    print(f"    --predictions {PRED_OUT} \\")
    print(f"    --ground-truth {GT_OUT} \\")
    print(f"    --split-column split \\")
    print(f"    --output reports/map_results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf",  type=float, default=0.01,
                        help="YOLO confidence threshold (default 0.01 — keep all for PR curve)")
    parser.add_argument("--iou",   type=float, default=0.5,
                        help="NMS IoU threshold (default 0.5)")
    parser.add_argument("--batch", type=int,   default=32,
                        help="Inference batch size (default 32)")
    args = parser.parse_args()
    run(args.conf, args.iou, args.batch)


if __name__ == "__main__":
    main()
