"""
Render quicklook previews of both Outer Thames offshore AOIs.

Run AFTER download_s1_ots.py completes. No chipping is done here — this
produces only visual sanity-check images before any inference or labeling.

Scene:    S1A_IW_GRDH_1SDV_20260517T172536_20260517T172601_064560_0821BD_460B
Footprint (catalog): 52.893–54.799°N, 2.937–7.335°E
Wind (4-corner):     SW=4.0 Bft3, NW=0.9 Bft1, SE=4.1 Bft3, NE=2.9 Bft2

Proposed AOIs (preliminary — density search will confirm):
  AOI A (western corridor):  53.30–53.80°N, 3.80–4.50°E
  AOI B (central-east):      53.50–54.05°N, 5.00–5.80°E

Outputs:
  /Volumes/Tejas SSD/datasets/s1_chips/ots_aoi_a_preview.png
  /Volumes/Tejas SSD/datasets/s1_chips/ots_aoi_b_preview.png

Usage:
    cd "/Volumes/Tejas SSD/sar-dark-ship-detection"
    source .venv/bin/activate
    python scripts/preview_ots.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import rasterio
from rasterio.windows import Window

# ── Scene root (auto-discovered — works for S1A/S1C/S1D) ──────────────────
SENTINEL_DIR = Path("/Volumes/Tejas SSD/datasets/sentinel1")
PRODUCT_NAME = "S1A_IW_GRDH_1SDV_20260517T172536_20260517T172601_064560_0821BD_460B.SAFE"
SAFE_DIR = SENTINEL_DIR / PRODUCT_NAME

if not SAFE_DIR.exists():
    print(f"ERROR: scene not found at {SAFE_DIR}")
    print("Run scripts/download_s1_ots.py first.")
    sys.exit(1)

# Auto-discover VV files — skips ._* metadata files
def find_vv(directory, suffix=".tiff"):
    return [f for f in directory.iterdir()
            if not f.name.startswith("._") and "vv" in f.name.lower() and f.suffix == suffix]

vv_tiffs = find_vv(SAFE_DIR / "measurement", ".tiff")
if len(vv_tiffs) != 1:
    sys.exit(f"ERROR: expected 1 VV tiff, found {vv_tiffs}")

annot_xmls = [f for f in (SAFE_DIR / "annotation").iterdir()
              if not f.name.startswith("._") and "vv" in f.name.lower()
              and f.suffix == ".xml" and f.parent.name == "annotation"]
if len(annot_xmls) != 1:
    sys.exit(f"ERROR: expected 1 VV annotation XML, found {annot_xmls}")

cal_xmls = [f for f in (SAFE_DIR / "annotation" / "calibration").iterdir()
            if not f.name.startswith("._") and "calibration" in f.name.lower()
            and "vv" in f.name.lower() and f.suffix == ".xml"]
if len(cal_xmls) != 1:
    sys.exit(f"ERROR: expected 1 VV calibration XML, found {cal_xmls}")

TIFF_PATH = vv_tiffs[0]
ANNOT_XML = annot_xmls[0]
CAL_XML   = cal_xmls[0]

print(f"TIFF:    {TIFF_PATH.name}")
print(f"Annot:   {ANNOT_XML.name}")
print(f"Cal:     {CAL_XML.name}")

OUT_DIR = Path("/Volumes/Tejas SSD/datasets/s1_chips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── AOI definitions ────────────────────────────────────────────────────────
# Preliminary bounds — confirmed to be inside catalog footprint with margin.
# The post-download density grid search is the real gate; if either AOI shows
# a regular wind-farm grid pattern, update bounds here before chipping.
AOIS = {
    "ots_aoi_a": (53.30, 53.80, 3.80, 4.50),  # western shipping corridor (Terschelling TSS)
    "ots_aoi_b": (53.50, 54.05, 5.00, 5.80),  # central-east corridor (west of Gemini WF)
}

# ── Geolocation grid ───────────────────────────────────────────────────────
print("\nParsing geolocation grid …")
root = ET.parse(ANNOT_XML).getroot()
geo_rows, geo_cols, geo_lats, geo_lons = [], [], [], []
for pt in root.findall(".//geolocationGridPoint"):
    geo_rows.append(float(pt.find("line").text))
    geo_cols.append(float(pt.find("pixel").text))
    geo_lats.append(float(pt.find("latitude").text))
    geo_lons.append(float(pt.find("longitude").text))

geo_rows = np.array(geo_rows); geo_cols = np.array(geo_cols)
geo_lats = np.array(geo_lats); geo_lons = np.array(geo_lons)
print(f"  {len(geo_rows)} points | "
      f"lat {geo_lats.min():.3f}–{geo_lats.max():.3f}°N | "
      f"lon {geo_lons.min():.3f}–{geo_lons.max():.3f}°E")

# Confirm AOIs are inside scene hull before doing any I/O
print("\nVerifying AOI bounds against scene hull …")
latlon_pts    = np.column_stack([geo_lats, geo_lons])
interp_to_row = LinearNDInterpolator(latlon_pts, geo_rows)
interp_to_col = LinearNDInterpolator(latlon_pts, geo_cols)

def check_corner(lat, lon, label):
    r = float(interp_to_row([[lat, lon]])[0])
    c = float(interp_to_col([[lat, lon]])[0])
    if np.isnan(r) or np.isnan(c):
        print(f"  ERROR: corner {label} ({lat}, {lon}) is OUTSIDE the scene hull.")
        return False
    print(f"  OK  {label} ({lat}, {lon}) → row {r:.0f}, col {c:.0f}")
    return True

all_ok = True
for aoi_name, (lat_min, lat_max, lon_min, lon_max) in AOIS.items():
    print(f"  {aoi_name}:")
    for la, lo, tag in [(lat_min, lon_min, "SW"), (lat_min, lon_max, "SE"),
                        (lat_max, lon_min, "NW"), (lat_max, lon_max, "NE")]:
        if not check_corner(la, lo, tag):
            all_ok = False

if not all_ok:
    print("\nERROR: one or more AOI corners are outside the scene hull.")
    print("Update AOI bounds in this script and re-run.")
    sys.exit(1)

print("  All corners inside scene hull.\n")

def latlon_to_rowcol(lat, lon):
    r = float(interp_to_row([[lat, lon]])[0])
    c = float(interp_to_col([[lat, lon]])[0])
    return r, c

def aoi_window(lat_min, lat_max, lon_min, lon_max):
    corners = [(lat_min,lon_min),(lat_min,lon_max),(lat_max,lon_min),(lat_max,lon_max)]
    rows = [latlon_to_rowcol(la, lo)[0] for la, lo in corners]
    cols = [latlon_to_rowcol(la, lo)[1] for la, lo in corners]
    return (max(0, int(np.floor(min(rows)))),
            max(0, int(np.floor(min(cols)))),
            int(np.ceil(max(rows))),
            int(np.ceil(max(cols))))

# ── Calibration LUT ────────────────────────────────────────────────────────
print("Parsing calibration LUT …")
cal_root = ET.parse(CAL_XML).getroot()
cal_lv, cal_pr, cal_sr = [], [], []
for cv in cal_root.findall(".//calibrationVector"):
    cal_lv.append(int(cv.find("line").text))
    cal_pr.append(np.fromstring(cv.find("pixel").text, sep=" ", dtype=np.float32))
    cal_sr.append(np.fromstring(cv.find("sigmaNought").text, sep=" ", dtype=np.float32))
cal_lines  = np.array(cal_lv, dtype=np.float32)
cal_pixels = cal_pr[0]
cal_sig    = np.vstack(cal_sr)

if not (np.all(np.diff(cal_lines) > 0) and np.all(np.diff(cal_pixels) > 0)):
    sys.exit("ERROR: calibration axes not strictly increasing.")

cal_interp = RegularGridInterpolator(
    (cal_lines, cal_pixels), cal_sig,
    method="linear", bounds_error=False, fill_value=None)

def build_lut(r0, c0, h, w):
    rows_ = np.arange(r0, r0+h, dtype=np.float32)
    cols_ = np.arange(c0, c0+w, dtype=np.float32)
    rr, cc = np.meshgrid(rows_, cols_, indexing="ij")
    lut = cal_interp(np.column_stack([rr.ravel(), cc.ravel()])).reshape(h, w)
    return lut.astype(np.float32)

# ── Render previews ────────────────────────────────────────────────────────
# Target ~1400 px on the longest side — same density as maasvlakte control
TARGET_PX = 1400

with rasterio.open(TIFF_PATH) as ds:
    full_h, full_w = ds.height, ds.width
    print(f"TIFF: {full_h} × {full_w} px\n")

    for aoi_name, (lat_min, lat_max, lon_min, lon_max) in AOIS.items():
        r0, c0, r1, c1 = aoi_window(lat_min, lat_max, lon_min, lon_max)
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(full_h, r1); c1 = min(full_w, c1)
        H, W = r1 - r0, c1 - c0
        print(f"── {aoi_name}  ({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)")
        print(f"   Pixel window: rows {r0}–{r1} ({H} px)  ×  cols {c0}–{c1} ({W} px)")

        print("   Reading raw DN …", end=" ", flush=True)
        raw = ds.read(1, window=Window(col_off=c0, row_off=r0, width=W, height=H)).astype(np.float32)
        print("done")

        print("   Calibrating …", end=" ", flush=True)
        lut = build_lut(r0, c0, H, W)
        lut = np.where(lut == 0.0, 1e-10, lut)
        sigma0 = (raw**2) / (lut**2)
        del raw, lut
        print("done")

        valid = sigma0 > 0
        db = np.full(sigma0.shape, np.nan, dtype=np.float32)
        db[valid] = 10.0 * np.log10(sigma0[valid])
        del sigma0

        valid_vals = db[valid]
        p1, p99 = np.percentile(valid_vals, 1), np.percentile(valid_vals, 99)
        print(f"   dB 1st–99th pct: {p1:.1f} – {p99:.1f} dB")

        clipped = np.clip(db, p1, p99)
        clipped[~valid] = 0.0
        u8 = ((clipped - p1) / (p99 - p1) * 255.0).astype(np.uint8)
        del db, clipped, valid, valid_vals

        scale = max(1, max(H, W) // TARGET_PX)
        preview = u8[::scale, ::scale]
        del u8

        out_path = OUT_DIR / f"{aoi_name}_preview.png"
        Image.fromarray(preview).save(out_path)
        print(f"   Preview → {out_path}")
        print(f"   ({preview.shape[1]}×{preview.shape[0]} px, 1:{scale} scale)\n")

print("Both previews saved.")
print("Open them and confirm:")
print("  - Isolated bright points (ships) on a dark background")
print("  - NO regular grid of closely-spaced bright targets (wind farm signature)")
print("  - Reasonable ocean texture (speckle, not uniform black)")
print("  - Background σ⁰ reads visually consistent with Bft 2–3 ocean")
print("\nIf both look clean: python scripts/preprocess_s1_ots.py")
