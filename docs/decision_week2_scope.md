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

## Scene 2 — Status as of 2026-05-26

**OPEN — no offshore scene selected yet.** Three candidate scenes from
2026-05-17 (Outer Thames / Southern North Sea, 52–53.5°N, 1–3°E) pass the
4-corner wind check at max Bft 3. Pending scope decision: whether to pursue a
second scene at all (see trade-off analysis in git history / conversation).

Candidate scenes if the offshore path is chosen:

| Scene | UUID | 4-corner max |
|-------|------|-------------|
| S1A 2026-05-17 17:25 UTC | fd83153d-4953-4bad-837c-9e10708c7eab | Bft 3 |
| S1D 2026-05-17 17:32 UTC | a10c05bf-08d8-4386-aa90-5dcf7b1fa8d2 | Bft 3 |
| S1D 2026-05-17 17:33 UTC | 23b528f8-0042-49c0-9b45-987af6ffb604 | Bft 3 |

Requirements for any future offshore scene:
- Sentinel-1 IW GRD VV+VH
- Over a genuine open-water shipping lane, ≥50 km from nearest coast
- No offshore wind farm contamination in the target AOI
- ALL four footprint corners ≤ Bft 3 (ERA5) at acquisition time
- Scene footprint verified from geolocation grid BEFORE AOI bounds are chosen
- Density grid search run within actual footprint to confirm ship traffic
- Quicklook preview passes visual inspection before any chipping

## Scope summary

| Scene | AOIs | Role | Status |
|-------|------|------|--------|
| 2025-04-30 Rotterdam | maasvlakte, ijmuiden | inshore pair | chips ready (maasvlakte), ijmuiden pending |
| TBD offshore | offshore_a, offshore_b | offshore pair | OPEN |

Scope decision pending: 4-AOI cross-domain evaluation (2 scenes) vs.
2-AOI inshore-only evaluation (1 scene, Rotterdam only).
