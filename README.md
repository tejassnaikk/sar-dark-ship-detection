# SAR Dark Ship Detection

Detecting AIS-dark vessels in Sentinel-1 SAR imagery using a rotated-box detector trained on RSDD-SAR, fused with AIS broadcast data.

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

**Week 1 — Baseline training on RSDD-SAR**

- [x] Repo skeleton and environment
- [ ] Annotation converter (VOC rotated → YOLOv11-OBB polygon)
- [ ] Baseline `yolo11n-obb` training (Colab Pro)
- [ ] Rotated mAP evaluation harness
- [ ] Model card and metrics table

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
python src/data/rsdd_to_yolo.py --config configs/dataset.yaml --output data/rsdd_yolo
```

### 4. Train (Colab)

Open `notebooks/01_train_baseline.ipynb` in Google Colab Pro. Follow the notebook instructions to mount your dataset and run training.

### 5. Evaluate

```bash
python src/eval/obb_metrics.py --predictions <path> --split test
```

---

## Model

Trained weights: [tejassnaikk/rsdd-yolo11n-obb-v1](https://huggingface.co/tejassnaikk/rsdd-yolo11n-obb-v1) *(coming after Week 1 training)*

---

## Results

*(Populated after baseline training — see `reports/metrics.md`)*

---

## Ethical considerations

- Outputs are **research candidates**, not actionable enforcement decisions.
- False positive rate at Sentinel-1 resolution (~10 m) is non-trivial; all detections require human review before operational use.
- Dataset (RSDD-SAR) is publicly released for research; no vessel identities or private data are used.

---

## License

MIT for code. RSDD-SAR dataset has its own license — see the dataset repo.
