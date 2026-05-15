"""
Standalone rotated-box mAP evaluation harness.

Designed to be format-agnostic: takes any predictions + ground truth as plain
Python dicts (or Parquet files via CLI) and computes rotated mAP using Shapely
polygon IoU. No dependency on Ultralytics label directories.

Polygon format throughout: flat list of 8 floats [x1,y1,x2,y2,x3,y3,x4,y4].
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def polygon_iou(poly_a: list[float], poly_b: list[float]) -> float:
    """
    Compute IoU between two quadrilateral polygons.

    Each polygon is a flat list of 8 floats [x1,y1,x2,y2,x3,y3,x4,y4].
    Uses Shapely for exact intersection. Handles degenerate (zero-area or
    self-intersecting) polygons by calling .buffer(0) and returning 0.0
    if either polygon is still invalid or has zero area.
    """
    def _make_poly(coords: list[float]) -> Optional[Polygon]:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        p = Polygon(pts)
        if not p.is_valid or p.area == 0:
            p = p.buffer(0)  # attempt to fix self-intersections
        if not p.is_valid or p.area == 0:
            return None
        return p

    a = _make_poly(poly_a)
    b = _make_poly(poly_b)
    if a is None or b is None:
        return 0.0

    inter = a.intersection(b).area
    union = a.union(b).area
    if union == 0:
        return 0.0
    return float(inter / union)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_predictions(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Greedy matching of predictions to ground truth boxes for a single image.

    Predictions are sorted by confidence descending. Each prediction is matched
    to the highest-IoU unmatched GT box. If the IoU >= iou_threshold, it is a
    true positive; otherwise a false positive. Unmatched GTs are false negatives
    (captured via n_ground_truth).

    Args:
        predictions: list of {"polygon": [8 floats], "confidence": float}
        ground_truth: list of {"polygon": [8 floats]}
        iou_threshold: minimum IoU to count as a match

    Returns:
        {
            "tp_flags": list[bool],     # one entry per prediction, sorted by descending confidence
            "confidences": list[float], # matching order
            "n_ground_truth": int,
        }
    """
    if not predictions:
        return {"tp_flags": [], "confidences": [], "n_ground_truth": len(ground_truth)}

    sorted_preds = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    matched_gt = set()
    tp_flags = []
    confidences = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            iou = polygon_iou(pred["polygon"], gt["polygon"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_flags.append(True)
            matched_gt.add(best_gt_idx)
        else:
            tp_flags.append(False)
        confidences.append(pred["confidence"])

    return {
        "tp_flags": tp_flags,
        "confidences": confidences,
        "n_ground_truth": len(ground_truth),
    }


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

def compute_ap(
    tp_flags: list[bool],
    confidences: list[float],
    n_ground_truth: int,
) -> float:
    """
    Compute Average Precision using all-points (area-under-curve) interpolation.

    This is the modern PASCAL VOC / COCO method: precision is interpolated as the
    maximum precision at any recall >= the current recall point, then AP is the
    area under that monotonically non-increasing precision envelope.

    Args:
        tp_flags: ordered by descending confidence (output of match_predictions)
        confidences: matching order (used only to assert sorting; values not needed here)
        n_ground_truth: total number of GT boxes (determines max recall)

    Returns:
        AP in [0, 1]. Returns 0.0 if n_ground_truth == 0 or tp_flags is empty.
    """
    if n_ground_truth == 0 or not tp_flags:
        return 0.0

    tp_arr = np.array(tp_flags, dtype=float)
    tp_cumsum = np.cumsum(tp_arr)
    fp_cumsum = np.cumsum(1 - tp_arr)

    recalls = tp_cumsum / n_ground_truth
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Prepend sentinel (recall=0, precision=1) and append (recall=max, precision=last)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])

    # Monotone non-increasing envelope: at each recall, precision = max(precision[recall:])
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # AP = area under the envelope
    ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    return ap


# ---------------------------------------------------------------------------
# Dataset-level mAP
# ---------------------------------------------------------------------------

def compute_map(
    predictions_by_image: dict,
    gt_by_image: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Aggregate mAP across all images in a dataset split.

    All TP flags and confidences from every image are pooled together (sorted by
    confidence globally), then a single AP is computed over the full pool. This
    matches the standard detection evaluation protocol.

    Args:
        predictions_by_image: {image_id: [{"polygon": ..., "confidence": ...}, ...]}
        gt_by_image:          {image_id: [{"polygon": ...}, ...]}
        iou_threshold:        IoU threshold for a match

    Returns:
        {
            "mAP": float,
            "n_predictions": int,
            "n_ground_truth": int,
            "iou_threshold": float,
        }
    """
    all_tp_flags: list[bool] = []
    all_confidences: list[float] = []
    total_gt = 0

    # Gather predictions from images that have predictions
    all_image_ids = set(predictions_by_image) | set(gt_by_image)
    for image_id in all_image_ids:
        preds = predictions_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])
        total_gt += len(gts)

        if preds:
            result = match_predictions(preds, gts, iou_threshold)
            all_tp_flags.extend(result["tp_flags"])
            all_confidences.extend(result["confidences"])

    # Re-sort everything by confidence globally before computing AP
    if all_confidences:
        order = np.argsort(all_confidences)[::-1]
        all_tp_flags = [all_tp_flags[i] for i in order]
        all_confidences = [all_confidences[i] for i in order]

    map_score = compute_ap(all_tp_flags, all_confidences, total_gt)

    return {
        "mAP": round(map_score, 6),
        "n_predictions": len(all_tp_flags),
        "n_ground_truth": total_gt,
        "iou_threshold": iou_threshold,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(results_by_split: dict) -> str:
    """
    Format and print a markdown table of per-split mAP results.

    Args:
        results_by_split: {"split_name": {"mAP": ..., "n_predictions": ..., "n_ground_truth": ...}, ...}

    Returns:
        The markdown table string (also printed to stdout).
    """
    header = "| Split | mAP@IoU | # Predictions | # Ground Truth |"
    sep    = "|-------|---------|---------------|----------------|"
    rows   = [header, sep]

    for split, m in results_by_split.items():
        iou_str = f"@{m['iou_threshold']:.2f}" if "iou_threshold" in m else ""
        rows.append(
            f"| {split} | {m['mAP']:.4f}{iou_str} | {m['n_predictions']} | {m['n_ground_truth']} |"
        )

    table = "\n".join(rows)
    print(table)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_parquet_split(path: str, split_col: Optional[str]) -> dict:
    """Load a predictions or GT parquet and return dict keyed by (split, image_id) or image_id."""
    import pandas as pd
    df = pd.read_parquet(path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute rotated mAP from predictions and ground-truth Parquet files."
    )
    parser.add_argument("--predictions", required=True, help="Parquet file with predictions")
    parser.add_argument("--ground-truth", required=True, help="Parquet file with ground truth")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--split-column", default=None,
        help="Column name to group results by (e.g. 'split'). Omit for single overall result.",
    )
    args = parser.parse_args()

    import pandas as pd

    pred_df = pd.read_parquet(args.predictions)
    gt_df = pd.read_parquet(args.ground_truth)

    def _df_to_by_image(df: pd.DataFrame, has_confidence: bool) -> dict:
        result = {}
        for _, row in df.iterrows():
            img_id = str(row["image_id"])
            entry = {"polygon": list(row["polygon"])}
            if has_confidence:
                entry["confidence"] = float(row["confidence"])
            result.setdefault(img_id, []).append(entry)
        return result

    splits = [None]
    if args.split_column and args.split_column in pred_df.columns:
        splits = pred_df[args.split_column].unique().tolist()

    results_by_split = {}
    for split in splits:
        if split is not None:
            p_sub = pred_df[pred_df[args.split_column] == split]
            g_sub = gt_df[gt_df[args.split_column] == split] if args.split_column in gt_df.columns else gt_df
            label = str(split)
        else:
            p_sub = pred_df
            g_sub = gt_df
            label = "overall"

        pred_by_img = _df_to_by_image(p_sub, has_confidence=True)
        gt_by_img = _df_to_by_image(g_sub, has_confidence=False)
        results_by_split[label] = compute_map(pred_by_img, gt_by_img, args.iou_threshold)

    if args.split_column and len(splits) > 1:
        # Also compute an overall across all splits
        pred_by_img_all = _df_to_by_image(pred_df, has_confidence=True)
        gt_by_img_all = _df_to_by_image(gt_df, has_confidence=False)
        results_by_split["overall"] = compute_map(pred_by_img_all, gt_by_img_all, args.iou_threshold)

    table = print_metrics_table(results_by_split)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# Evaluation Results\n\n{table}\n")
    log.info("Written to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
