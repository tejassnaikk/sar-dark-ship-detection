"""
Tests for src/eval/obb_metrics.py

All tests use synthetic axis-aligned or rotated polygons as flat 8-float lists
[x1,y1,x2,y2,x3,y3,x4,y4]. No external files or model loading.
"""
import math

import pytest

from src.eval.obb_metrics import (
    polygon_iou,
    match_predictions,
    compute_ap,
    compute_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def square(x0: float, y0: float, side: float) -> list[float]:
    """Axis-aligned square: bottom-left at (x0, y0)."""
    return [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side]


def rotated_square(cx: float, cy: float, half: float, angle_deg: float) -> list[float]:
    """Square centered at (cx, cy) rotated by angle_deg degrees."""
    a = math.radians(angle_deg)
    offsets = [(-half, -half), (half, -half), (half, half), (-half, half)]
    pts = []
    for dx, dy in offsets:
        pts.append(cx + dx * math.cos(a) - dy * math.sin(a))
        pts.append(cy + dx * math.sin(a) + dy * math.cos(a))
    return pts


def pred(polygon: list[float], confidence: float) -> dict:
    return {"polygon": polygon, "confidence": confidence}


def gt(polygon: list[float]) -> dict:
    return {"polygon": polygon}


# ---------------------------------------------------------------------------
# 1. polygon_iou — identical squares
# ---------------------------------------------------------------------------

def test_polygon_iou_identical_squares():
    sq = square(0, 0, 4)
    assert polygon_iou(sq, sq) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. polygon_iou — non-overlapping
# ---------------------------------------------------------------------------

def test_polygon_iou_non_overlapping():
    a = square(0, 0, 4)
    b = square(10, 10, 4)
    assert polygon_iou(a, b) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. polygon_iou — partial overlap
# ---------------------------------------------------------------------------

def test_polygon_iou_partial_overlap():
    # Two 4×4 squares offset by 2 in x:
    #   a covers [0,4] × [0,4], b covers [2,6] × [0,4]
    #   intersection = [2,4] × [0,4] = 2×4 = 8
    #   union = 16 + 16 - 8 = 24
    #   IoU = 8/24 = 1/3
    a = square(0, 0, 4)
    b = square(2, 0, 4)
    assert polygon_iou(a, b) == pytest.approx(1 / 3, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. polygon_iou — degenerate polygon (all same point)
# ---------------------------------------------------------------------------

def test_polygon_iou_degenerate_returns_zero():
    degenerate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    normal = square(0, 0, 4)
    # Must return 0.0 without crashing
    result = polygon_iou(degenerate, normal)
    assert result == 0.0

    result2 = polygon_iou(degenerate, degenerate)
    assert result2 == 0.0


# ---------------------------------------------------------------------------
# 5. polygon_iou — 45°-rotated square matched to itself
# ---------------------------------------------------------------------------

def test_polygon_iou_rotated_square_self():
    sq = rotated_square(cx=5, cy=5, half=3, angle_deg=45)
    iou = polygon_iou(sq, sq)
    assert iou == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. match_predictions — perfect predictions → all TP
# ---------------------------------------------------------------------------

def test_match_perfect_predictions_all_tp():
    boxes = [square(0, 0, 4), square(10, 0, 4), square(20, 0, 4)]
    preds = [pred(b, 0.9 - 0.1 * i) for i, b in enumerate(boxes)]
    gts   = [gt(b) for b in boxes]

    result = match_predictions(preds, gts, iou_threshold=0.5)
    assert result["n_ground_truth"] == 3
    assert len(result["tp_flags"]) == 3
    assert all(result["tp_flags"]), "All predictions should be TPs"


# ---------------------------------------------------------------------------
# 7. match_predictions — zero predictions → all FN
# ---------------------------------------------------------------------------

def test_match_zero_predictions_all_fn():
    gts = [gt(square(0, 0, 4)) for _ in range(3)]
    result = match_predictions([], gts, iou_threshold=0.5)
    assert result["tp_flags"] == []
    assert result["confidences"] == []
    assert result["n_ground_truth"] == 3


# ---------------------------------------------------------------------------
# 8. match_predictions — zero GT → all FP
# ---------------------------------------------------------------------------

def test_match_zero_gt_all_fp():
    preds = [pred(square(i * 10, 0, 4), 0.9) for i in range(3)]
    result = match_predictions(preds, [], iou_threshold=0.5)
    assert len(result["tp_flags"]) == 3
    assert not any(result["tp_flags"]), "All predictions should be FPs (no GT)"
    assert result["n_ground_truth"] == 0


# ---------------------------------------------------------------------------
# 9. match_predictions — greedy: highest confidence wins
# ---------------------------------------------------------------------------

def test_match_greedy_highest_confidence_wins():
    # 1 GT box; 2 predictions both overlapping it.
    # Higher confidence (0.9) should take the TP; lower (0.3) becomes FP.
    gt_box = square(0, 0, 4)
    high_conf = pred(square(0, 0, 4), 0.9)   # exact match → IoU 1.0
    low_conf  = pred(square(0, 0, 4), 0.3)   # also exact match → but GT already taken

    result = match_predictions([low_conf, high_conf], [gt(gt_box)], iou_threshold=0.5)

    # Results are sorted by confidence: high_conf first
    assert result["confidences"][0] == pytest.approx(0.9)
    assert result["tp_flags"][0] is True   # high confidence → TP
    assert result["tp_flags"][1] is False  # low confidence → FP (GT taken)


# ---------------------------------------------------------------------------
# 10. compute_ap — perfect ranking yields AP = 1.0
# ---------------------------------------------------------------------------

def test_compute_ap_perfect_ranking_is_one():
    # All predictions are TPs in descending confidence order, n_gt == n_tp
    n = 5
    tp_flags = [True] * n
    confidences = [1.0 - 0.1 * i for i in range(n)]
    ap = compute_ap(tp_flags, confidences, n_ground_truth=n)
    assert ap == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 11. compute_ap — zero GT returns 0.0
# ---------------------------------------------------------------------------

def test_compute_ap_zero_gt_returns_zero():
    ap = compute_ap([True, False], [0.9, 0.5], n_ground_truth=0)
    assert ap == 0.0


# ---------------------------------------------------------------------------
# 12. compute_map — end-to-end synthetic example
# ---------------------------------------------------------------------------

def test_compute_map_end_to_end_synthetic():
    """
    2-image dataset:
      image_1: 1 GT, 1 perfect prediction (TP)
      image_2: 1 GT, 1 wrong prediction (FP, no overlap), 1 GT unmatched (FN)

    Expected:
      Total GT = 2, Total preds = 2
      TP = 1 (image_1 prediction), FP = 1 (image_2 prediction)
      Precision-recall:
        After 1st pred (conf=0.9, TP): P=1.0, R=0.5
        After 2nd pred (conf=0.8, FP): P=0.5, R=0.5
      AP with all-points interpolation:
        recall goes 0 → 0.5 → 0.5 (unchanged for FP)
        precision envelope: max right of each recall point
          At R=0.5: max(1.0, 0.5) = 1.0
        AP = (0.5 - 0) * 1.0 = 0.5
    """
    predictions_by_image = {
        "image_1": [pred(square(0, 0, 4), 0.9)],
        "image_2": [pred(square(50, 50, 4), 0.8)],  # far from GT, FP
    }
    gt_by_image = {
        "image_1": [gt(square(0, 0, 4))],
        "image_2": [gt(square(0, 0, 4))],
    }

    result = compute_map(predictions_by_image, gt_by_image, iou_threshold=0.5)

    assert result["n_ground_truth"] == 2
    assert result["n_predictions"] == 2
    assert result["iou_threshold"] == 0.5
    assert 0.0 < result["mAP"] <= 1.0
    assert result["mAP"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# 13. compute_map — no predictions returns 0.0 mAP
# ---------------------------------------------------------------------------

def test_compute_map_no_predictions_returns_zero():
    gt_by_image = {"img_a": [gt(square(0, 0, 4))], "img_b": [gt(square(10, 0, 4))]}
    result = compute_map({}, gt_by_image, iou_threshold=0.5)
    assert result["mAP"] == 0.0
    assert result["n_ground_truth"] == 2
    assert result["n_predictions"] == 0


# ---------------------------------------------------------------------------
# 14. compute_map — perfect predictions returns mAP = 1.0
# ---------------------------------------------------------------------------

def test_compute_map_perfect_returns_one():
    boxes = {"img_a": [square(0, 0, 4)], "img_b": [square(10, 0, 4)]}
    pred_by_img = {k: [pred(v[0], 0.99)] for k, v in boxes.items()}
    gt_by_img   = {k: [gt(v[0])]        for k, v in boxes.items()}
    result = compute_map(pred_by_img, gt_by_img, iou_threshold=0.5)
    assert result["mAP"] == pytest.approx(1.0, abs=1e-6)
