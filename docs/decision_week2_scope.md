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

## Scene 2 — Central North Sea attempt (ABANDONED)

Product: `S1A_IW_GRDH_1SDV_20260505T172626_20260505T172651_064385_081B91_65D0.SAFE`
UUID: `76df1f03-82dc-44ce-946c-e55590e9d2f6`
Downloaded: 2026-05-26. Still on disk at
`/Volumes/Tejas SSD/datasets/sentinel1/…_65D0.SAFE` (do not use).

### Selection rationale (flawed)

ERA5 wind at scene center (56°N, 4°E) at acquisition time: 1.8 m/s Bft 2.
This passed a single-point wind check. Scene downloaded.

### Failure

Post-download density grid search returned **0 candidate ships** across all
sampled cells. Background σ⁰ = −11 to −13 dB (expected −20 to −25 dB for
calm ocean) — a clear rough-sea signature. Multi-point ERA5 check then
confirmed:

| Corner | Location | Wind speed | Beaufort |
|--------|----------|-----------|---------|
| SW | 56.1°N, 2.2°E | 10.6 m/s | Bft 5 |
| NW | 57.5°N, 2.5°E | 10.4 m/s | Bft 5 |
| SE | 56.1°N, 5.8°E | 4.8 m/s | Bft 3 |
| NE | 57.5°N, 5.5°E | 5.6 m/s | Bft 4 |

The center-point Bft 2 reading was a small calm pocket inside an otherwise
Bft 4–5 wind field across the western two-thirds of the scene.

### Root cause

Single-point ERA5 wind check is insufficient for the heterogeneous North Sea
wind environment. A localized calm at scene center does not guarantee calm
conditions across the full footprint. The result was a ~1.8 GB wasted download
and no usable data.

### Revised wind-check rule

All future scene candidates must pass a **4-corner ERA5 check** — SW, NW, SE,
NE corners of the expected scene footprint — with **all four corners ≤ Bft 3**
before any download is initiated. A single failing corner rejects the scene.

### Extended search outcome

A year-long search (Jun 2025–May 2026) across the Central North Sea
(55–58°N, 1–6°E) found **no scenes passing the 4-corner Bft ≤ 3 criterion**.
The open North Sea at these latitudes is rarely calm across a full IW swath
(~250 km wide).

## Scene 2 — Outer Thames S1A 2026-05-17 (DOWNLOADED, AOIs UNDER REVIEW)

Product: `S1A_IW_GRDH_1SDV_20260517T172536_20260517T172601_064560_0821BD_460B.SAFE`
UUID: `fd83153d-4953-4bad-837c-9e10708c7eab`
Downloaded: 2026-05-26. On disk at
`/Volumes/Tejas SSD/datasets/sentinel1/…_460B.SAFE`

Footprint (from catalog + geolocation grid): 52.893–54.799°N, 2.937–7.335°E
Wind (4-corner ERA5): SW=4.0 Bft3, NW=0.9 Bft1, SE=4.1 Bft3, NE=2.9 Bft2 — all pass

### AOI A — `ots_aoi_a` — 53.30–53.80°N, 3.80–4.50°E

Status: **PROVISIONAL — plausible, pending density search**

Quicklook preview rendered (2026-05-26). Visual assessment: sparse open
ocean, dark background, a small number of isolated bright points. Consistent
with a genuine low-traffic offshore scene at Bft 3. Density grid search
required to count candidate ships and rule out infrastructure.

### AOI B — `ots_aoi_b` — 53.50–54.05°N, 5.00–5.80°E

Status: **FLAGGED — NOT CLEARED FOR CHIPPING**

Quicklook preview rendered (2026-05-26). Visual assessment: a cluster of
bright targets at center-right of the AOI showing suspicious regularity.
The AOI's eastern edge (5.80°E) is only 0.1° west of the Gemini offshore
wind farm cluster (~54°N, 5.9°E).

Required before any decision on AOI B:
1. Density grid search to measure local-max count and spacing regularity
2. Zoomed native-resolution crop of the bright cluster for manual inspection
   (turbines produce regular-grid sidelobe spikes; ships produce elongated
   compact hulls with aspect ratio ~2:1 to 10:1)
3. If contamination confirmed: propose new eastern AOI bounds shifted west,
   or promote a second western-corridor AOI instead

No preprocessing of AOI B until the turbine check is complete and the AOI
is explicitly signed off.

### Next session (2026-05-27 morning)

1. Run density grid search across both `ots_aoi_a` and `ots_aoi_b`
2. Render zoomed native-resolution crop of AOI B bright cluster
3. Based on findings: clear AOI B, rebound it, or replace it
4. Only after both AOIs are signed off: run `preprocess_s1_ots.py`

## Scope summary

| Scene | AOIs | Role | Status |
|-------|------|------|--------|
| 2025-04-30 Rotterdam | maasvlakte, ijmuiden | inshore pair | maasvlakte chipped; ijmuiden pending |
| 2026-05-17 OTS (S1A) | ots_aoi_a, ots_aoi_b | offshore pair | downloaded; AOI A provisional, AOI B flagged |
