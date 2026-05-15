# Metrics — SAR Dark Ship Detection

Source of truth for all evaluation results. Each training run appends a row.
Numbers are from Ultralytics `.val()` at IoU threshold 0.5 unless noted.

---

## Baseline: v1_n (Week 1)

**Model:** `yolo11n-obb` (nano, ~2.7 M params)  
**Training data:** RSDD-SAR train split (5,000 images)  
**Eval data:** RSDD-SAR test split (2,000 images)  
**Weights:** [`tejassnaikk/rsdd-yolo11n-obb-v1`](https://huggingface.co/tejassnaikk/rsdd-yolo11n-obb-v1)

| Setting | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | N images |
|---------|---------|--------------|-----------|--------|----------|
| RSDD test (in-domain, overall) | **0.938** | 0.640 | 0.933 | 0.871 | 2,000 |
| RSDD test — inshore | 0.763 | 0.483 | 0.760 | 0.696 | 159 |
| RSDD test — offshore | 0.971 | 0.673 | 0.954 | 0.931 | 1,841 |

### Commentary

The overall mAP@0.5 of **0.938** is a strong in-domain result for a nano-scale model.
The more interesting story is in the inshore/offshore breakdown.

**The 21-point inshore/offshore gap (0.763 vs 0.971) is the central finding of Week 1.**

- **Offshore** (open ocean, isolated ships): near-optimal. mAP 0.971, recall 0.931.
  Ships in open water produce clean, high-contrast SAR signatures with minimal
  surrounding clutter. The model has learned this pattern reliably.

- **Inshore** (harbours, ports, coastal): substantially weaker. mAP 0.763, recall 0.696.
  Port environments present closely-spaced vessels (touching bounding boxes), high-backscatter
  infrastructure (quays, cranes, metal roofs), and ships partially occluded by structures.
  ~30% of inshore ships are missed entirely (recall 0.696 vs 0.931 offshore).

This gap is well-characterised and expected given the dataset's composition
(offshore images dominate: 1,841 vs 159). It is **not a bug** — it is the honest
performance envelope of the baseline model, and it sets up the Week 3 investigation:
when we apply this model to Sentinel-1 GRD (10 m resolution, which blurs small vessels
further), the inshore performance will almost certainly degrade more than the offshore.

Supporting plots: `reports/figures/baseline_v1/` (confusion matrix, P/R curve, loss curves).

---

## Planned future rows

| Week | Experiment | Expected |
|------|-----------|---------|
| Week 3 | Cross-domain: RSDD-SAR model → Sentinel-1 GRD | mAP drop (resolution mismatch) |
| Week 3 | Domain-adapted model (despeckling + resolution matching) | Partial recovery |
| Week 4 | AIS-fusion candidate flagging | Precision/recall on AIS-dark flags |
