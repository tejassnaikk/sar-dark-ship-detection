# Week 2 Retrospective

*Working note — not polished. For future-me picking this back up.*

---

## What I set out to do

Take the Week 1 baseline (YOLOv11n-OBB, mAP@0.5 = 0.938 on RSDD-SAR in-domain) and run
it against real Sentinel-1 GRD imagery to measure the cross-domain gap. The Week 1 handoff
predicted the gap would be real: offshore ships in open water the easier half, inshore port
environments the harder. Goal was 1,000+ hand-labeled chips across at least two scenes with
a clean evaluation at standard IoU.

---

## What actually happened

First offshore candidate (snsl) abandoned — wind-farm contamination. Second (CNS,
2026-05-05) downloaded and preprocessed before discovering Beaufort 5 across the western
two-thirds. Missed because the wind check was at scene center, not four corners. Third try
(Outer Thames, 2026-05-17) worked. Original maasvlakte AOI was only 20 chips — expanded
both inshore AOIs to 90 and 81 before labeling.

Labeling hit two bugs: matplotlib figure-window closing after one chip (fixed by fresh
figure per chip); and early in maasvlakte I labeled two bright building returns as ships
before internalizing that on-land cross signatures are infrastructure. Both caught in the
structured re-review pass.

---

## Key decisions and why

**Two scenes, four AOIs:** time constraint. Beaufort 3 over Beaufort 1–2 was deliberate —
the earlier cleaner scene would have looked better on paper but been unrepresentative of
conditions where dark-ship detection actually matters.

**Auto-labeling empty chips rejected:** the screener is a brightness detector; its blind
spots are exactly where low-σ⁰ dark ships hide. Writing empty labels for screener negatives
would have baked the model's failure mode into the ground truth.

**Reporting mAP@0.25 alongside @0.50:** not to soften the result — to separate whether
the model finds the right location from whether it draws the right box size. Those are
orthogonal failure modes; collapsing them into one threshold obscures both.

---

## Surprises

RSDD-SAR uses hull-scale labels (median 434 px², h/w ~3.6), not tight point-scatterer
fits as assumed. That killed the labeling-convention hypothesis for offshore box undersizing
and forced the honest conclusion: genuine model under-regression.

The screener (median+40 uint8) completely failed offshore. Per-AOI 1st/99th percentile
stretch pushes ocean speckle across the full 0–255 range, so an absolute threshold is
meaningless there. Required a different signal (tail_snr) for offshore triage.

Two failure modes turned out to need two different fixes — not one fine-tuning run.
Knowing that early would have saved analysis time.

---

## What I'd do differently

- **Append-only audit log.** Overwriting session JSON on every save was wrong. A simple
  append-only decision log (chip, action, timestamp) would have made the audit meaningful
  and cost nothing.
- **4-corner wind check before downloading.** One extra API call. Should be step 0, not
  a post-hoc diagnostic.
- **Quicklook before chipping.** Rendering a downsampled preview before full preprocessing
  would have caught the CNS scene immediately and saved several hours.

---

## What I learned

**Technical:** SAR speckle local maxima above median+40 are everywhere in ocean chips —
not ship candidates. Sigma-nought calibration is mandatory before log conversion. Ship SAR
signatures are cross/star shaped from superstructure corner reflectors; sidelobe arms extend
in range/azimuth and must not enter the OBB. Per-AOI percentile clipping changes the
effective dynamic range between domains in ways that break thresholds tuned for one domain.

**Process:** Defer over guess — a missing label excludes the chip cleanly; a wrong label
corrupts it permanently. Audit the audit — 2.5% spot-check hit rate means the audited
subset is a floor, not a ceiling. Honest small-N beats fake large-N: 130 offshore GT boxes
is a weak eval, saying so is more useful than not.
