---
license: agpl-3.0
tags:
  - object-detection
  - oriented-bounding-box
  - sar
  - remote-sensing
  - ship-detection
  - yolov11
library_name: ultralytics
---

# YOLOv11n-OBB — RSDD-SAR Ship Detector (v1 baseline)

Rotated bounding-box ship detector trained on RSDD-SAR. Single class: **ship**.
Part of the [SAR Dark Ship Detection](https://github.com/tejassnaikk/sar-dark-ship-detection) portfolio project.

---

## Model overview

- **Architecture:** YOLOv11n-OBB (nano variant, ~2.7 M parameters)
- **Task:** Oriented object detection — outputs rotated bounding boxes as 4-corner polygons
- **Classes:** 1 (`ship`)
- **Input size:** 512 × 512 px

---

## Intended use

This model is intended for:

- **Research** into SAR-based maritime vessel detection
- **Portfolio demonstration** of rotated-box detection on remote sensing data
- **Baseline** for cross-domain evaluation (high-res RSDD-SAR → Sentinel-1 GRD)

### Out-of-scope use

- **Do not use** to accuse individual vessels of illegal activity
- **Do not use** as sole evidence in any enforcement or legal decision
- Outputs are *candidate detections*, not verified vessel identifications
- Operational maritime surveillance requires multi-source fusion and human review

---

## Training data

**Dataset:** RSDD-SAR (Radar Satellite Dataset for Ship Detection)
- ~7,000 SAR chip images (512 × 512 px, ~3 m ground resolution)
- 10,263 annotated ship instances with rotated bounding boxes
- Annotations in long-edge convention: `cx, cy, h, w, angle` (radians, `h ≥ w`)
- Source imagery from Gaofen-3 (C-band, HH/HV polarization)

**Split used:**
- **Train:** 5,000 images (from `ImageSets/train.txt`)
- **Val / Test:** 2,000 images (from `ImageSets/test.txt` — no separate val split in dataset; test used for both)
- **Test subsets:** inshore (159 images), offshore (1,841 images)

Annotations were converted from VOC rotated-box XML to YOLOv11-OBB polygon format using
[`src/data/rsdd_to_yolo.py`](https://github.com/tejassnaikk/sar-dark-ship-detection/blob/main/src/data/rsdd_to_yolo.py).
Long-edge enforcement was applied to all boxes before conversion.

---

## Training procedure

| Hyperparameter | Value |
|----------------|-------|
| Base model | `yolo11n-obb.pt` (pretrained) |
| Epochs | 100 (early-stopped at 75, patience 20) |
| Image size | 512 × 512 |
| Batch size | 16 |
| Optimizer | AdamW (Ultralytics default) |
| Hardware | Colab Pro T4 GPU |
| Training time | ~2 hours |

---

## Evaluation results

Evaluated with Ultralytics `.val()` at IoU threshold 0.5. mAP@0.5:0.95 is the COCO-style
area under the IoU-threshold curve from 0.5 to 0.95 in 0.05 steps.

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | N images |
|-------|---------|--------------|-----------|--------|----------|
| Overall test | **0.938** | 0.640 | 0.933 | 0.871 | 2,000 |
| Inshore | 0.763 | 0.483 | 0.760 | 0.696 | 159 |
| Offshore | 0.971 | 0.673 | 0.954 | 0.931 | 1,841 |

**Key finding:** 21-point inshore/offshore gap (mAP@0.5: 0.763 vs 0.971).
The model handles isolated ships in open water well but struggles in port and
coastal environments due to clutter, infrastructure returns, and closely spaced vessels.

---

## Limitations

1. **Inshore/offshore gap:** ~21-point mAP gap. Port clutter and coastal infrastructure
   produce false positives and suppress true detections near shorelines.
2. **Resolution mismatch:** Trained on ~3 m RSDD-SAR; will degrade on Sentinel-1 GRD (~10 m).
   Cross-domain evaluation is planned for Week 3 of the project.
3. **Single class:** No vessel-type classification (tanker, cargo, fishing, etc.).
4. **Bright-object false positives:** Oil platforms, wind turbines, breakwaters, and other
   high-backscatter structures can produce spurious detections.
5. **Polarization:** Training data is primarily HH-polarized Gaofen-3. Performance on
   other polarizations (VV, dual-pol) is untested.

---

## Ethical considerations

Maritime surveillance is inherently dual-use. This model was built for research; outputs
should never be used without human review:

- Detections are **AIS-dark vessel candidates**, not confirmed illegal ships
- A vessel being absent from AIS does not mean it is acting illegally — AIS failures,
  fishing vessels under tonnage thresholds, and legitimate AIS-off transit all produce
  the same signature
- False positive rates at Sentinel-1 resolution (~10 m) are non-trivial and have not
  been characterized for the cross-domain case
- Any operational use must involve multi-source fusion, domain experts, and legal oversight

---

## How to use

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Downloads weights (~5.5 MB) and caches locally
weights = hf_hub_download(
    repo_id="tejassnaikk/rsdd-yolo11n-obb-v1",
    filename="best.pt",
)
model = YOLO(weights)

# Run on a SAR image (grayscale or 3-channel, 512×512 recommended)
results = model("path/to/sar_chip.jpg", imgsz=512, conf=0.25)
results[0].show()   # display with rotated boxes
```

Output format: `results[0].obb` contains oriented bounding boxes as `xywhr` tensors
(center x, center y, width, height, rotation in radians).

---

## Citation / Project

```
@misc{sar-dark-ship-detection,
  author = {Tejas Naik},
  title  = {SAR Dark Ship Detection},
  year   = {2025},
  url    = {https://github.com/tejassnaikk/sar-dark-ship-detection},
}
```

See the [project repo](https://github.com/tejassnaikk/sar-dark-ship-detection) for the
full pipeline including annotation conversion, evaluation harness, and Sentinel-1
cross-domain work.
