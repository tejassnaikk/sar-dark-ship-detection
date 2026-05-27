# Ground Truth Labeling Protocol — Week 2

## Purpose

All ground truth labels for the Week 2 cross-domain evaluation are produced
by manual visual inspection of SAR chips. This protocol defines what to label,
what to exclude, and how to handle ambiguous returns. It must be applied
identically across all AOIs so that the inshore/offshore mAP comparison is not
confounded by inconsistent labeling decisions.

## What to label: vessels only

Label a return as a vessel if ALL of the following hold:

1. **Compact bright core** — main energy concentrated in a tight blob, typically
   2–15 pixels at Sentinel-1 GRD ~10 m/px resolution.
2. **Plausible aspect ratio** — elongated consistent with a ship hull
   (L/W roughly 2:1 to 10:1), or approximately square for small vessels seen
   bow-on.
3. **Not fixed infrastructure** (see exclusions).
4. **In water** — not continuously connected to a bright land or port mass.

## What to exclude: fixed infrastructure

Do NOT label any of the following as vessels:

- **Offshore wind turbines** — bright point targets, often in regular grids.
  Individual turbines may produce azimuth-ambiguity sidelobes identical in
  appearance to ships. Exclude all turbines regardless of brightness.
- **Oil and gas platforms** — stationary, typically +30–50 dB above ocean
  background, may have attached structures. Exclude.
- **Navigation buoys and beacons** — stationary, sub-pixel to 2-pixel returns.
  Exclude.
- **Port cranes, gantries, container stacks** — linear or blocky high-backscatter
  structures inside terminal footprints. Exclude.
- **Bridges and breakwaters** — continuous bright lines. Exclude.

## SAR-specific labeling rules

### Azimuth ambiguity sidelobes

Bright metallic targets (large ships, platforms) produce "spike" or "cross"
arms extending in the azimuth direction. These are a real SAR physics artifact,
not separate objects. **Label only the compact bright core.** Draw the OBB
tightly around the hull return; do not include sidelobe arms.

### Ship wakes

Wakes appear as faint linear features trailing from a vessel. **Do not include
the wake in the bounding box.** Label the hull return only.

### Ambiguous returns

If a return cannot be confidently classified as vessel vs. fixed infrastructure
(e.g., an isolated bright blob at grid spacing consistent with a wind farm,
or a return at the edge of a platform cluster), **do not label it.** Log it
as "uncertain — excluded" with pixel coordinates in the annotation session
notes. Uncertain exclusions are reported in the metrics commentary alongside
the final mAP numbers.

## Bounding box format

Oriented bounding boxes (OBB) in YOLOv11-OBB polygon format:
- 4 corners as 8 flat floats in chip-pixel coordinates (not full-scene coords)
- Long axis aligned with the vessel's principal axis (long-edge convention:
  h ≥ w; see `src/data/rsdd_to_yolo.py` for the enforcement function)
- Single class: `ship` (class index 0)

## Quality rules

- Label one AOI per session; do not continue when fatigued.
- A sparse clean GT is strictly better than a dense noisy one. If in doubt,
  exclude.
- Every label decision is final for that AOI; do not revisit after the session
  closes. Retroactive corrections invalidate the eval.
- AIS data may be used as a cross-check to identify vessel locations, but
  AIS is NOT a label source. Dark ships have no AIS record by definition;
  using AIS as ground truth excludes the targets the project exists to find.
