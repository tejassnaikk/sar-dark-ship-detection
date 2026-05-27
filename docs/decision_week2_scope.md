# Week 2 Scope Decision Record

Date: 2026-05-26

## Scene 1 (2025-04-30) — inshore AOIs only

Product: `S1A_IW_GRDH_1SDV_20250430T055840_20250430T055905_058982_07507C_E57F.SAFE`

Two inshore AOIs extracted from this acquisition:

| AOI | Bounds | Role |
|-----|--------|------|
| `maasvlakte` | 51.93–52.03°N, 3.98–4.18°E | Port of Rotterdam outer terminal |
| `ijmuiden`   | 52.45–52.55°N, 4.50–4.70°E | North Sea Canal entrance (Amsterdam) |

### Why no offshore AOI from this scene

A full diagnostic grid search was run across the scene.

- **Western half (2.5–4.0°E)** — genuine open water but ship-sparse.
  The originally planned offshore AOI (`snsl`, 51.70–52.10°N, 2.50–3.50°E)
  was abandoned because coordinate analysis confirmed the southern portion
  (51.70–51.87°N, 2.70–3.40°E) overlaps the Belgian offshore wind farm zone
  (Belwind/Norther). Detection confirmed: 234 local-max targets at >+30 dB
  above ocean background in a regular grid pattern — unambiguous wind turbine
  signature. Cannot be disentangled from ship returns without a dedicated
  turbine filter, which is out of scope for Week 2.

- **Eastern half (4.0–5.5°E)** — ship-dense but exclusively near-port or
  port-interior. Every ship-rich cell is the Rotterdam outer approach,
  Maasvlakte basin, or IJmuiden port complex. No genuinely open-water
  offshore cell with significant ship traffic exists in this scene at these
  longitudes. Classifying a port approach as "offshore" would invalidate the
  inshore/offshore performance gap comparison.

**Decision:** use this scene for inshore AOIs only. A second scene is required
for the two offshore AOIs.

## Scene 2 (TBD) — offshore AOIs

To be selected. Requirements:
- Sentinel-1 IW GRD VV+VH
- Over a genuine open-water shipping lane, ≥50 km from nearest coast
- No offshore wind farm contamination in the target AOI
- Wind speed ≤ 5 m/s (Beaufort ≤ 3) at acquisition time for clean sea state
- Traffic-verified against optical quicklook or public maritime data before
  download

Two offshore AOIs will be extracted from the chosen scene. Both labeled
per `docs/labeling_protocol.md`.

## Scope summary

| Scene | AOIs | Role |
|-------|------|------|
| 2025-04-30 Rotterdam | maasvlakte, ijmuiden | inshore pair |
| TBD offshore | offshore_a, offshore_b | offshore pair |

4 AOIs total across 2 acquisitions. Cross-domain mAP reported per AOI
and averaged within inshore/offshore groups.
