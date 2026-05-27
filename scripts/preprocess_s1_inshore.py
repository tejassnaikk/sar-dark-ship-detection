"""
Inshore GRD preprocessing pipeline — Rotterdam scene (2025-04-30).

Supersedes the maasvlakte-only portion of preprocess_s1.py. Both inshore
AOIs are processed here with verified expanded bounds; the snsl AOI is
permanently excluded (wind-farm contamination, see docs/decision_week2_scope.md).

Scene: S1A_IW_GRDH_1SDV_20250430T055840_20250430T055905_058982_07507C_E57F.SAFE

AOIs (all four corners verified inside scene geolocation hull):
  maasvlakte  51.88–52.08°N, 3.88–4.25°E  Port of Rotterdam outer terminal +
              North Sea approach. Expanded from original 0.10°×0.20° (20 chips)
              to 0.20°×0.37° (~90 chips) so inshore evaluation is not chip-starved.
  ijmuiden    52.40–52.60°N, 4.38–4.72°E  North Sea Canal entrance (Amsterdam).
              0.20°×0.34° → ~81 chips.

Steps (identical to preprocess_s1_ots.py):
  A. Parse geolocation grid → scipy interpolators
  B. Compute pixel windows for each AOI
  C. Parse sigma-nought calibration LUT (bilinear)
  D. Read DN → σ⁰ → dB → 1st/99th pct clip per AOI → uint8
  E. Tile into 512×512 PNG chips, stride=256, skip >80% black
  F. Render downsampled preview PNG of each AOI
  G. Print chip inventory

Usage:
    python scripts/preprocess_s1_inshore.py

WARNING: clears and regenerates s1_chips/maasvlakte/ (old 20 chips from the
narrow bounds are replaced; filenames encode scene row/col so they differ).
s1_chips/ijmuiden/ is created fresh.
"""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import rasterio
from rasterio.windows import Window

# ── Paths ──────────────────────────────────────────────────────────────────
SAFE_DIR = Path("/Volumes/Tejas SSD/datasets/sentinel1") / \
    "S1A_IW_GRDH_1SDV_20250430T055840_20250430T055905_058982_07507C_E57F.SAFE"

TIFF_PATH = SAFE_DIR / "measurement" / \
    "s1a-iw-grd-vv-20250430t055840-20250430t055905-058982-07507c-001.tiff"
ANNOT_XML = SAFE_DIR / "annotation" / \
    "s1a-iw-grd-vv-20250430t055840-20250430t055905-058982-07507c-001.xml"
CAL_XML   = SAFE_DIR / "annotation" / "calibration" / \
    "calibration-s1a-iw-grd-vv-20250430t055840-20250430t055905-058982-07507c-001.xml"

CHIPS_DIR = Path("/Volumes/Tejas SSD/datasets/s1_chips")

# ── AOI definitions (lat_min, lat_max, lon_min, lon_max) ───────────────────
AOIS = {
    "maasvlakte": (51.88, 52.08, 3.88, 4.25),
    "ijmuiden":   (52.40, 52.60, 4.38, 4.72),
}

CHIP_SIZE    = 512
STRIDE       = 256
BLACK_THRESH = 0.80
PREVIEW_PX   = 1400


# ───────────────────────────────────────────────────────────────────────────
# Clear stale chip directories before regenerating
# ───────────────────────────────────────────────────────────────────────────
for aoi_name in AOIS:
    d = CHIPS_DIR / aoi_name
    if d.exists():
        n_old = sum(1 for f in d.glob("*.png") if not f.name.startswith("._"))
        print(f"Clearing {d}  ({n_old} existing chips)")
        shutil.rmtree(d)


# ───────────────────────────────────────────────────────────────────────────
# Step A: Geolocation grid
# ───────────────────────────────────────────────────────────────────────────
print("\nStep A: parsing geolocation grid …")

root = ET.parse(ANNOT_XML).getroot()
geo_rows, geo_cols, geo_lats, geo_lons = [], [], [], []
for pt in root.findall(".//geolocationGridPoint"):
    geo_rows.append(float(pt.find("line").text))
    geo_cols.append(float(pt.find("pixel").text))
    geo_lats.append(float(pt.find("latitude").text))
    geo_lons.append(float(pt.find("longitude").text))

geo_rows = np.array(geo_rows); geo_cols = np.array(geo_cols)
geo_lats = np.array(geo_lats); geo_lons = np.array(geo_lons)

print(f"  {len(geo_rows)} grid points | "
      f"lat {geo_lats.min():.3f}–{geo_lats.max():.3f}°N | "
      f"lon {geo_lons.min():.3f}–{geo_lons.max():.3f}°E")

latlon_pts    = np.column_stack([geo_lats, geo_lons])
interp_to_row = LinearNDInterpolator(latlon_pts, geo_rows)
interp_to_col = LinearNDInterpolator(latlon_pts, geo_cols)


def latlon_to_rowcol(lat: float, lon: float) -> tuple[float, float]:
    row = float(interp_to_row([[lat, lon]])[0])
    col = float(interp_to_col([[lat, lon]])[0])
    if np.isnan(row) or np.isnan(col):
        sys.exit(f"ERROR: ({lat}, {lon}) is outside the scene's geolocation hull.")
    return row, col


# ───────────────────────────────────────────────────────────────────────────
# Step B: AOI → pixel windows
# ───────────────────────────────────────────────────────────────────────────
print("\nStep B: computing AOI pixel windows …")

aoi_windows: dict[str, tuple[int, int, int, int]] = {}
for name, (lat_min, lat_max, lon_min, lon_max) in AOIS.items():
    corners = [
        (lat_min, lon_min), (lat_min, lon_max),
        (lat_max, lon_min), (lat_max, lon_max),
    ]
    rows, cols = [], []
    for lat, lon in corners:
        r, c = latlon_to_rowcol(lat, lon)
        rows.append(r); cols.append(c)

    row_off = max(0, int(np.floor(min(rows))))
    col_off = max(0, int(np.floor(min(cols))))
    row_end = int(np.ceil(max(rows)))
    col_end = int(np.ceil(max(cols)))
    aoi_windows[name] = (row_off, col_off, row_end, col_end)

    print(f"  {name:12s}: rows {row_off:5d}–{row_end:5d} ({row_end-row_off} px) | "
          f"cols {col_off:5d}–{col_end:5d} ({col_end-col_off} px)")


# ───────────────────────────────────────────────────────────────────────────
# Step C: Calibration LUT
# ───────────────────────────────────────────────────────────────────────────
print("\nStep C: parsing sigma-nought calibration LUT …")

cal_root = ET.parse(CAL_XML).getroot()
cal_line_vals, cal_pix_rows, cal_sig_rows = [], [], []
for cv in cal_root.findall(".//calibrationVector"):
    cal_line_vals.append(int(cv.find("line").text))
    cal_pix_rows.append(np.fromstring(cv.find("pixel").text,       sep=" ", dtype=np.float32))
    cal_sig_rows.append(np.fromstring(cv.find("sigmaNought").text, sep=" ", dtype=np.float32))

cal_lines  = np.array(cal_line_vals, dtype=np.float32)
cal_pixels = cal_pix_rows[0]
cal_sig    = np.vstack(cal_sig_rows)

if not np.allclose(cal_pix_rows[0], cal_pix_rows[-1]):
    sys.exit("ERROR: calibration pixel positions differ between vectors.")
assert np.all(np.diff(cal_lines)  > 0), "Calibration lines not strictly increasing"
assert np.all(np.diff(cal_pixels) > 0), "Calibration pixels not strictly increasing"

cal_interp = RegularGridInterpolator(
    (cal_lines, cal_pixels), cal_sig,
    method="linear", bounds_error=False, fill_value=None)

print(f"  {len(cal_lines)} vectors × {len(cal_pixels)} pixel samples | "
      f"lines {int(cal_lines[0])}–{int(cal_lines[-1])} | "
      f"pixels {int(cal_pixels[0])}–{int(cal_pixels[-1])}")


def build_cal_lut(row_off: int, col_off: int, height: int, width: int) -> np.ndarray:
    rows = np.arange(row_off, row_off + height, dtype=np.float32)
    cols = np.arange(col_off, col_off + width,  dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    lut = cal_interp(np.column_stack([rr.ravel(), cc.ravel()])).reshape(height, width)
    return lut.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
# Process each AOI
# ───────────────────────────────────────────────────────────────────────────
CHIPS_DIR.mkdir(parents=True, exist_ok=True)
chip_counts: dict[str, int] = {}

with rasterio.open(TIFF_PATH) as ds:
    full_h, full_w = ds.height, ds.width
    print(f"\nTIFF: {full_h} rows × {full_w} cols | dtype {ds.dtypes[0]}")

    for aoi_name, (row_off, col_off, row_end, col_end) in aoi_windows.items():
        row_off = max(0, row_off)
        col_off = max(0, col_off)
        row_end = min(full_h, row_end)
        col_end = min(full_w, col_end)
        height  = row_end - row_off
        width   = col_end - col_off

        lat_min, lat_max, lon_min, lon_max = AOIS[aoi_name]
        print(f"\n── AOI: {aoi_name}  ({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)")
        print(f"   Pixel size: {height} × {width} px")

        # Step D
        print("   Reading raw DN …", end=" ", flush=True)
        win = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        raw = ds.read(1, window=win).astype(np.float32)
        print("done")

        print("   Applying sigma-nought calibration …", end=" ", flush=True)
        lut = build_cal_lut(row_off, col_off, height, width)
        lut = np.where(lut == 0.0, 1e-10, lut)
        sigma0 = (raw ** 2) / (lut ** 2)
        del raw, lut
        print("done")

        valid = sigma0 > 0
        sigma0_db = np.full(sigma0.shape, np.nan, dtype=np.float32)
        sigma0_db[valid] = 10.0 * np.log10(sigma0[valid])
        del sigma0

        valid_vals = sigma0_db[valid]
        p1, p99 = np.percentile(valid_vals, 1), np.percentile(valid_vals, 99)
        print(f"   dB 1st–99th pct: {p1:.1f} – {p99:.1f} dB")
        del valid_vals

        clipped = np.clip(sigma0_db, p1, p99)
        clipped[~valid] = 0.0
        u8 = ((clipped - p1) / (p99 - p1) * 255.0).astype(np.uint8)
        del sigma0_db, clipped, valid

        # Step F: preview
        scale = max(1, max(height, width) // PREVIEW_PX)
        preview = u8[::scale, ::scale]
        preview_path = CHIPS_DIR / f"{aoi_name}_chipped_preview.png"
        Image.fromarray(preview).save(preview_path)
        print(f"   Preview → {preview_path}  "
              f"({preview.shape[1]}×{preview.shape[0]} px, 1:{scale} scale)")

        # Step E: chip tiling
        aoi_chip_dir = CHIPS_DIR / aoi_name
        aoi_chip_dir.mkdir(exist_ok=True)

        n_saved = 0
        n_skipped = 0
        for r0 in range(0, height - CHIP_SIZE + 1, STRIDE):
            for c0 in range(0, width - CHIP_SIZE + 1, STRIDE):
                chip = u8[r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE]
                if (chip == 0).mean() > BLACK_THRESH:
                    n_skipped += 1
                    continue
                fname = (f"{aoi_name}"
                         f"_r{row_off + r0:05d}"
                         f"_c{col_off + c0:05d}.png")
                Image.fromarray(chip).save(aoi_chip_dir / fname)
                n_saved += 1

        chip_counts[aoi_name] = n_saved
        print(f"   Chips saved:   {n_saved}")
        print(f"   Chips skipped: {n_skipped}  (>{BLACK_THRESH:.0%} black)")


# Step G: full inventory
print("\n── Chip inventory — all AOIs ───────────────────────────────────────────")
all_aois = ["maasvlakte", "ijmuiden", "ots_aoi_a", "ots_aoi_b"]
total = 0
for name in all_aois:
    d = CHIPS_DIR / name
    n = sum(1 for f in d.glob("*.png")
            if not f.name.startswith("._")) if d.exists() else 0
    domain = "inshore" if name in ("maasvlakte", "ijmuiden") else "offshore"
    print(f"  {name:15s} ({domain:8s}): {n:4d} chips")
    total += n
print(f"\n  Total: {total} chips across 4 AOIs")
print(f"\nOutput root: {CHIPS_DIR}")
