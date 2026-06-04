# SAR Dark Ship Detection

Detecting AIS-dark vessels in Sentinel-1 SAR imagery using a rotated-box detector trained on RSDD-SAR, fused with AIS broadcast data.

---

## Week 2 Results

**Cross-domain evaluation (2026-06-04):** The RSDD-SAR-trained YOLOv11n-OBB model was evaluated on two hand-labeled Sentinel-1 GRD scenes (1,063 chips, 233 ground-truth ship boxes across 4 AOIs — 2 inshore port crops, 2 offshore open-water). Performance dropped severely from in-domain (mAP@0.5 = 0.938 on RSDD-SAR test) to cross-domain (overall mAP@0.5 = 0.002, mAP@0.25 = 0.137). Two distinct failure modes were diagnosed: (1) offshore box regression systematically under-sizes predictions to ~64% of ground-truth area (model locates 53% of ships at IoU@0.25 but draws tighter boxes than hand-drawn hull labels); (2) inshore predictions fire heavily on port infrastructure (cranes, container yards, quays) that the open-water-only training set contained no examples of as negatives. See [`reports/metrics.md`](reports/metrics.md) for full methodology, ground truth statistics, audit summary, and diagnostics.

---

## Motivation

"Dark ships" are vessels that disable or spoof their Automatic Identification System (AIS) transponder — a tactic used to hide illegal fishing, sanctions violations, or ship-to-ship cargo transfers. Commercial SAR satellites (Sentinel-1, ICEYE, Capella) image the ocean regardless of AIS status, making them a powerful independent sensing layer.

This project builds an end-to-end pipeline that:
1. Detects ships in SAR imagery using a rotated bounding-box detector (YOLOv11-OBB).
2. Cross-references detections with AIS broadcasts via probabilistic Mahalanobis matching.
3. Flags **AIS-dark vessel candidates** — detections with no matching AIS record within a spatiotemporal window.

> **Important framing:** outputs are *candidates*, not confirmed violations. SAR resolution, AIS propagation delays, and matching tolerances all introduce false positives. This tool is for research and maritime awareness, not operational enforcement.

---

## Architecture

```
RSDD-SAR (~3 m, 7 k chips)
        │
        ▼
  YOLOv11-OBB training
  (rotated bbox, ship class)
        │
        ▼
Sentinel-1 GRD inference     ◄── AIS broadcast stream
  (~10 m, public data)              (MarineTraffic / OpenAIS)
        │                                    │
        └──────────── Mahalanobis fusion ────┘
                              │
                              ▼
                  AIS-dark candidate flags
                              │
                              ▼
                  Gradio + Folium demo (HF Spaces)
```

*(Architecture diagram placeholder — will be replaced with a figure once the pipeline is end-to-end.)*

---

## Stack

| Component | Tool |
|-----------|------|
| Detector | Ultralytics YOLOv11-OBB |
| SAR data | Sentinel-1 GRD (Copernicus Data Space) |
| AIS data | OpenAIS / MarineTraffic historical |
| Metrics | Shapely polygon IoU, rotated mAP |
| Storage | pandas / Parquet |
| Demo UI | Gradio + Folium |
| Model hosting | HuggingFace Hub |

---

## Status

**Week 1 — Complete ✅** · [`reports/metrics.md §Baseline`](reports/metrics.md)

- [x] Repo skeleton and environment
- [x] Annotation converter (VOC rotated → YOLOv11-OBB polygon)
- [x] Baseline `yolo11n-obb` training (Colab Pro, T4 GPU)
- [x] Rotated mAP evaluation harness (Shapely polygon IoU, 14 unit tests)
- [x] Model card and metrics table

**Week 2 — Complete ✅** · [`reports/metrics.md §Week2`](reports/metrics.md) · [`docs/labeling_protocol.md`](docs/labeling_protocol.md)

- [x] Two Sentinel-1 GRD scenes acquired and preprocessed (Rotterdam + Outer Thames)
- [x] 1,063 chips hand-labeled across 4 AOIs (233 ship boxes, 888 confirmed empty)
- [x] Cross-domain inference with best.pt (1,513 predictions)
- [x] Two-failure-mode diagnostic: offshore box regression undersizing + inshore infrastructure misfiring
- [x] Full evaluation report with audit summary and Week 3 remediation plan

**Week 3 — Planned:** Fine-tune box regression on Sentinel-1 labeled set; add port-infrastructure hard negatives

---

## Reproducibility

### 1. Clone and set up environment

```bash
git clone https://github.com/tejassnaikk/sar-dark-ship-detection.git
cd sar-dark-ship-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Obtain RSDD-SAR dataset

Download RSDD-SAR from the [official release](https://github.com/CAESAR-Radi/RSDD-SAR) and extract to a local path. Then update `configs/dataset.yaml`:

```yaml
rsdd_sar_root: /your/local/path/RSDD-SAR
```

### 3. Convert annotations

```bash
python -m src.data.rsdd_to_yolo \
    --input /path/to/RSDD-SAR \
    --output data/rsdd_yolo \
    --copy-images
```

### 4. Train (Colab)

Open `notebooks/01_train_baseline.ipynb` in Google Colab Pro. Follow the notebook instructions to mount your dataset and run training.

### 5. Evaluate

```bash
python -m src.eval.obb_metrics \
    --predictions predictions.parquet \
    --ground-truth ground_truth.parquet \
    --output reports/metrics_eval.md
```

---

## Model

**[tejassnaikk/rsdd-yolo11n-obb-v1](https://huggingface.co/tejassnaikk/rsdd-yolo11n-obb-v1)** — YOLOv11n-OBB, 5.5 MB, trained on RSDD-SAR.

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(repo_id="tejassnaikk/rsdd-yolo11n-obb-v1", filename="best.pt")
model = YOLO(weights)
results = model("path/to/sar_image.jpg", imgsz=512)
```

**[Live demo](https://huggingface.co/spaces/tejassnaikk/sar-dark-ship-demo)** — upload a SAR chip to see rotated box predictions.

---

## Results

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| Overall test (in-domain) | **0.938** | 0.640 | 0.933 | 0.871 |
| Inshore | 0.763 | 0.483 | 0.760 | 0.696 |
| Offshore | 0.971 | 0.673 | 0.954 | 0.931 |

21-point inshore/offshore gap (0.763 vs 0.971) — model handles open-water ships well,
struggles with port clutter. See [`reports/metrics.md`](reports/metrics.md) for full commentary.

---

## Ethical considerations

- Outputs are **research candidates**, not actionable enforcement decisions.
- False positive rate at Sentinel-1 resolution (~10 m) is non-trivial; all detections require human review before operational use.
- Dataset (RSDD-SAR) is publicly released for research; no vessel identities or private data are used.

---

## License

MIT for code. RSDD-SAR dataset has its own license — see the dataset repo.
