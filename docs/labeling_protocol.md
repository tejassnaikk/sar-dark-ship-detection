# Ground Truth Labeling Protocol — Week 2

**Applies to all four AOIs: maasvlakte · ijmuiden · ots_aoi_a · ots_aoi_b**  
Last updated: 2026-05-27

---

## Purpose

All ground truth labels for the Week 2 cross-domain evaluation are produced
by manual visual inspection of SAR chips. This protocol defines what to label,
what to exclude, and how to handle ambiguous returns. It must be applied
**identically across all four AOIs** so that the inshore/offshore mAP comparison
is not confounded by inconsistent labeling decisions.

---

## Rule 1 — Label vessels only; exclude all fixed infrastructure

Label a return as a vessel if ALL of the following hold:

1. **Compact bright core** — main energy concentrated in a tight blob, typically
   2–15 pixels at Sentinel-1 GRD ~10 m/px resolution.
2. **Plausible aspect ratio** — elongated consistent with a ship hull
   (L/W roughly 2:1 to 10:1), or approximately square for small vessels seen
   bow-on.
3. **Not fixed infrastructure** (see exclusions below).
4. **In water** — not continuously connected to a bright land or port mass.

**Do NOT label any of the following:**

- **Quays, jetties, piers** — continuous bright edges bounding port water.
- **Cranes and loading arms** — linear high-backscatter structures inside
  terminal footprints.
- **Container stacks and cargo piles** — blocky bright areas on land.
- **Breakwaters and seawalls** — long continuous bright lines.
- **Buoys and mooring dolphins** — stationary sub-pixel to 2-pixel returns.
- **Offshore wind turbines** — bright point targets in regular grids. Individual
  turbines may produce azimuth-ambiguity sidelobes identical in appearance to
  ships. Exclude all turbines regardless of brightness.
- **Oil/gas platforms** — stationary, +30–50 dB above ocean background.

If a chip contains only port infrastructure and no distinguishable vessel hulls,
press `e` (confirmed empty).

---

## Rule 2 — Berthed ships

Berthed ships **count as ships if the hull is distinguishable from the dock**.

If the ship-dock boundary cannot be confidently identified (e.g., the SAR return
from hull and quay merges into a single bright blob), press `.` to defer rather
than guess.

Never label a quay wall as a ship.

---

## Rule 3 — Label the hull; not sidelobe spikes, not wakes

Draw the OBB around the **compact bright hull return only**:

- **Azimuth sidelobe spikes** — bright linear artefacts extending in the range
  or azimuth direction from a strong point scatterer. Do not include them in the
  box. Label only the compact bright core.
- **Corner-reflector blooms** — saturated specular returns on superstructure may
  appear slightly larger than the physical hull. Fit the OBB to the main bright
  blob, not the full saturated extent.
- **Ship wakes** — faint linear features trailing behind a vessel. Do not include
  the wake in the bounding box.

The OBB should represent the best estimate of the ship's physical footprint.

---

## Rule 4 — Edge-cut ships (partially outside chip boundary)

- **> 50 % of hull visible inside chip** → label it. The OBB covers only the
  visible portion.
- **≤ 50 % of hull visible** → press `.` to defer. The adjacent chip (stride =
  256 px, 50 % overlap) will likely show the full hull.

---

## Rule 5 — Uncertainty → defer, never guess

When genuinely unsure whether a target is a ship, fixed infrastructure, or
clutter:

> **Press `.` (defer). Never press `e` or draw a box while uncertain.**

The deferred CSV is reviewed after the first pass. Defer is always the correct
choice when the target is ambiguous. A deferred chip is excluded from evaluation
(no label file written), which is safe. A guessed label corrupts the ground truth.

**Specific cases that must be deferred (not labeled, not marked empty):**
- Target smaller than ~3×3 pixels with no clear elongated hull shape
- Bright blob that could be a ship *or* a buoy/beacon
- Merged ship-dock return where separation is impossible
- Any target where you would want to see the adjacent chip before deciding

---

## Bounding box format

Oriented bounding boxes (OBB) in YOLOv11-OBB polygon format:
- 4 corners as 8 flat floats in chip-pixel coordinates (not full-scene coords)
- Long axis aligned with the vessel's principal axis (long-edge convention:
  h ≥ w; enforced automatically by `fit_obb_from_corners`)
- Single class: `ship` (class index 0)

### OBB drawing procedure

1. Click 4 corners of the ship hull in any order.
2. After the 4th click, a green OBB is fitted and displayed.
3. If the box looks wrong, press `c` to clear all clicks and redraw.
4. If the chip contains multiple ships, click 4 corners for each one in sequence.
5. Press `w` to commit all drawn boxes and advance to the next chip.

Click order does not matter — PCA is used to fit the minimum-area OBB.

---

## Key-binding reference

| Key | Action | File written |
|-----|--------|--------------|
| `e` | Confirmed empty — zero ships visible | Empty `.txt` (0 bytes) |
| `w` | Save drawn OBBs — one or more ships labeled | `.txt` with OBB lines |
| `.` | Defer — ambiguous or uncertain | None (→ `{aoi}_deferred.csv`) |
| `q` | Quit and save session | None for unreviewed chips |
| `backspace` | Undo last corner click | — |
| `c` | Clear all drawn boxes for this chip | — |

**Empty `.txt` = confirmed negative → counted in evaluation.**  
**Missing `.txt` = unreviewed → excluded from evaluation.**

---

## Quality constraints

- Label one AOI per session; stop when fatigued.
- A sparse, clean GT is strictly better than a dense, noisy one. If in doubt,
  defer.
- AIS data may be used as a cross-check to identify vessel locations, but AIS
  is NOT a label source. Dark ships have no AIS record by definition; using AIS
  as ground truth excludes the targets this project exists to find.
