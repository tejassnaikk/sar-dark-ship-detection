# Project Handoff — Week 2

## Context

SAR Dark Ship Detection portfolio project. Week 1 complete. Starting Week 2: cross-domain
evaluation of the RSDD-trained model on real Sentinel-1 GRD imagery.

## Week 1 result

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| RSDD test (overall) | 0.938 | 0.640 | 0.933 | 0.871 |
| Inshore | 0.763 | 0.483 | 0.760 | 0.696 |
| Offshore | 0.971 | 0.673 | 0.954 | 0.931 |

The 21-point inshore/offshore gap is the established portfolio narrative.
Week 2 measures what domain shift does to both halves.

## Week 2 scope (locked)

2 Sentinel-1 GRD scenes — 1 offshore, 1 inshore. Carefully hand-verified ground truth.
Two scenes is the target and is enough; do not rush labels to get more scenes.

**Locked decisions:**
- Ground truth is VISUAL hand-labeling, never AIS. AIS misses dark ships by definition.
  AIS is a cross-check only, not a label source.
- Prefer VV polarization, low-wind scenes, no sea ice.
- Expected key finding: resolution mismatch (RSDD ~2–3 m/px vs S1 GRD ~10 m/px) drives
  most degradation. Sets up Week 3 resolution matching.

## Environment

- Project root: `/Volumes/Tejas SSD/sar-dark-ship-detection`
- Venv: `.venv/` (activate: `source .venv/bin/activate`)
- Branch: `tejas_local`
- Weights: `models_local/best.pt` (YOLOv11n-OBB, single class `ship`)
- Eval harness: `src/eval/obb_metrics.py`
  - Predictions schema: `image_id` (str), `polygon` (8 flat floats), `confidence` (float), optional `split`
  - GT schema: `image_id` (str), `polygon` (8 floats), optional `split`
- Results placeholder: `reports/metrics.md`
- Copernicus Data Space account: created in Week 1

## Known environment quirk

exFAT/NTFS SSD creates `._*` macOS metadata files that break imports and XML parsing.
After any pip install or file operation: `find . -type f -name "._*" -delete`.
All file-walking code must skip filenames starting with `._`.

## Step order for Week 2

1. Save + commit this handoff ← you are here
2. Verify model loads cleanly
3. Check current Copernicus Data Space auth/download flow (search web — platform changes)
4. Pick 2 scenes (AOI, date, polarization) — show candidates, user chooses
5. Download the 2 GRD scenes
6. GRD preprocessing: calibration, chipping to model-input-sized tiles
7. Run `best.pt` inference → predictions parquet
8. Hand-label visual ground truth for both scenes → GT parquet (do not rush this)
9. Run `obb_metrics.py` → cross-domain mAP, split inshore/offshore
10. Write results into `reports/metrics.md` placeholder rows with honest commentary
11. Commit Week 2 work on `tejas_local`
