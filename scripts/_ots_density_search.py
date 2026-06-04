"""
Density grid search + AOI-B bright-cluster crop for the OTS scene.

Produces:
  s1_chips/ots_aoi_a_density.png   — annotated density map, AOI A
  s1_chips/ots_aoi_b_density.png   — annotated density map, AOI B
  s1_chips/ots_aoi_b_cluster_native.png  — native-res crop of the bright cluster
  s1_chips/ots_aoi_a_sample_native.png   — native-res crop of AOI A mid-section (reference)

Methodology (same as Rotterdam scouting pass):
  Grid 512×512 px blocks at 1024-px stride across each AOI.
  Per block: median dB as local background; count scipy local-maxima
  above bg+15 dB (candidate ships) and bg+30 dB (likely fixed infrastructure).
  These counts are SCOUTING PROBES ONLY — not ship counts. Real counts
  come from hand-labeling only.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import maximum_filter, label
import rasterio
from rasterio.windows import Window

# ── Paths ──────────────────────────────────────────────────────────────────
SAFE_DIR = Path("/Volumes/Tejas SSD/datasets/sentinel1") / \
    "S1A_IW_GRDH_1SDV_20260517T172536_20260517T172601_064560_0821BD_460B.SAFE"

TIFF_PATH = SAFE_DIR / "measurement" / \
    "s1a-iw-grd-vv-20260517t172536-20260517t172601-064560-0821bd-001.tiff"
ANNOT_XML = SAFE_DIR / "annotation" / \
    "s1a-iw-grd-vv-20260517t172536-20260517t172601-064560-0821bd-001.xml"
CAL_XML   = SAFE_DIR / "annotation" / "calibration" / \
    "calibration-s1a-iw-grd-vv-20260517t172536-20260517t172601-064560-0821bd-001.xml"

OUT_DIR = Path("/Volumes/Tejas SSD/datasets/s1_chips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AOIS = {
    "ots_aoi_a": (53.30, 53.80, 3.80, 4.50),
    "ots_aoi_b": (53.50, 54.05, 5.00, 5.80),
}

# ── Geolocation grid ───────────────────────────────────────────────────────
print("Parsing geolocation grid …")
root = ET.parse(ANNOT_XML).getroot()
geo_rows, geo_cols, geo_lats, geo_lons = [], [], [], []
for pt in root.findall(".//geolocationGridPoint"):
    geo_rows.append(float(pt.find("line").text))
    geo_cols.append(float(pt.find("pixel").text))
    geo_lats.append(float(pt.find("latitude").text))
    geo_lons.append(float(pt.find("longitude").text))

geo_rows = np.array(geo_rows); geo_cols = np.array(geo_cols)
geo_lats = np.array(geo_lats); geo_lons = np.array(geo_lons)
print(f"  {len(geo_rows)} pts | lat {geo_lats.min():.3f}–{geo_lats.max():.3f}°N "
      f"| lon {geo_lons.min():.3f}–{geo_lons.max():.3f}°E")

latlon_pts    = np.column_stack([geo_lats, geo_lons])
interp_to_row = LinearNDInterpolator(latlon_pts, geo_rows)
interp_to_col = LinearNDInterpolator(latlon_pts, geo_cols)

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

# ── Density search helper ──────────────────────────────────────────────────
BLOCK   = 512
STRIDE  = 1024   # one probe per ~10 km
MF_SIZE = 11     # local-max footprint: peak must be max in 11×11 window (~110 m)

def count_peaks(block_db, bg_thresh_db, infra_thresh_db):
    """
    Return (n_candidate, n_infra):
      n_candidate: local maxima > median(block) + bg_thresh_db
      n_infra:     local maxima > median(block) + infra_thresh_db
    """
    valid = np.isfinite(block_db)
    if valid.sum() < 0.5 * block_db.size:
        return 0, 0
    bg = float(np.nanmedian(block_db))
    mf = maximum_filter(block_db, size=MF_SIZE, mode="reflect")
    is_max = (block_db == mf) & valid
    candidate = int(np.sum(is_max & (block_db > bg + bg_thresh_db)))
    infra      = int(np.sum(is_max & (block_db > bg + infra_thresh_db)))
    return candidate, infra

# ── Process each AOI ───────────────────────────────────────────────────────
aoi_results = {}

with rasterio.open(TIFF_PATH) as ds:
    full_h, full_w = ds.height, ds.width
    print(f"\nTIFF: {full_h} × {full_w} px")

    for aoi_name, (lat_min, lat_max, lon_min, lon_max) in AOIS.items():
        r0, c0, r1, c1 = aoi_window(lat_min, lat_max, lon_min, lon_max)
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(full_h, r1); c1 = min(full_w, c1)
        H, W = r1 - r0, c1 - c0

        print(f"\n══ {aoi_name} ({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E) "
              f"→ {H}×{W} px ══")

        # Read & calibrate full AOI
        print("  Reading DN …", end=" ", flush=True)
        raw = ds.read(1, window=Window(col_off=c0, row_off=r0, width=W, height=H)).astype(np.float32)
        print("done  | Calibrating …", end=" ", flush=True)
        lut = build_lut(r0, c0, H, W)
        lut = np.where(lut == 0.0, 1e-10, lut)
        sigma0 = (raw**2) / (lut**2)
        del raw, lut

        valid = sigma0 > 0
        db = np.full(sigma0.shape, np.nan, dtype=np.float32)
        db[valid] = 10.0 * np.log10(sigma0[valid])
        del sigma0
        print("done")

        # Global stats
        vals = db[valid]
        p1, p25, p50, p75, p99 = np.percentile(vals, [1, 25, 50, 75, 99])
        print(f"  Global dB: p1={p1:.1f}  p25={p25:.1f}  median={p50:.1f}"
              f"  p75={p75:.1f}  p99={p99:.1f}")

        # ── Density grid ────────────────────────────────────────────────────
        grid_rows = range(0, H - BLOCK + 1, STRIDE)
        grid_cols = range(0, W - BLOCK + 1, STRIDE)

        grid_cand  = np.zeros((len(grid_rows), len(grid_cols)), dtype=np.int16)
        grid_infra = np.zeros_like(grid_cand)
        grid_bg    = np.full((len(grid_rows), len(grid_cols)), np.nan)

        print(f"  Grid search: {len(grid_rows)}×{len(grid_cols)} blocks …")
        for gi, br in enumerate(grid_rows):
            for gj, bc in enumerate(grid_cols):
                blk = db[br:br+BLOCK, bc:bc+BLOCK]
                bg_med = float(np.nanmedian(blk))
                grid_bg[gi, gj] = bg_med
                n_cand, n_infra = count_peaks(blk, bg_thresh_db=15, infra_thresh_db=30)
                grid_cand[gi, gj]  = n_cand
                grid_infra[gi, gj] = n_infra

        total_cand  = int(grid_cand.max())   # report max-per-block as "hottest cell"
        total_infra = int(grid_infra.max())
        print(f"  Candidate peaks (>bg+15dB): max per block = {total_cand}   "
              f"[blocks with ≥1: {int((grid_cand > 0).sum())} / {grid_cand.size}]")
        print(f"  Infra peaks    (>bg+30dB): max per block = {total_infra}   "
              f"[blocks with ≥1: {int((grid_infra > 0).sum())} / {grid_cand.size}]")

        # Check for regular-grid pattern (wind farm signature)
        # Measure: if infra peaks form a grid, spacing is ~800–1000 m = ~80–100 px
        infra_blocks_gi, infra_blocks_gj = np.where(grid_infra >= 2)
        if len(infra_blocks_gi) >= 4:
            # Compute centroid spacing of infra-heavy blocks
            spacings = []
            for k in range(len(infra_blocks_gi)-1):
                dr = (infra_blocks_gi[k+1] - infra_blocks_gi[k]) * STRIDE
                dc = (infra_blocks_gj[k+1] - infra_blocks_gj[k]) * STRIDE
                spacings.append(np.sqrt(dr**2 + dc**2))
            if spacings:
                print(f"  ⚠ Infra-heavy blocks: {len(infra_blocks_gi)}, "
                      f"typical spacing {np.median(spacings):.0f} px "
                      f"(~{np.median(spacings)*10/1000:.1f} km)")
        else:
            print(f"  Infra-heavy blocks (≥2 targets): {len(infra_blocks_gi)} "
                  f"— no regular grid pattern detected")

        # ── Density map image ───────────────────────────────────────────────
        # Render AOI at 1:8 scale, annotate grid cells with counts
        scale = 8
        preview_h, preview_w = H // scale, W // scale
        # u8 for rendering
        clipped = np.clip(db, p1, p99)
        clipped = np.where(np.isfinite(clipped), clipped, p1)
        u8 = ((clipped - p1) / max(p99 - p1, 1e-6) * 255).astype(np.uint8)
        del clipped

        img = Image.fromarray(u8[::scale, ::scale]).convert("RGB")
        draw = ImageDraw.Draw(img)

        for gi, br in enumerate(grid_rows):
            for gj, bc in enumerate(grid_cols):
                px = bc // scale
                py = br // scale
                nc, ni = int(grid_cand[gi, gj]), int(grid_infra[gi, gj])
                color = (255, 80, 80) if ni >= 2 else \
                        (255, 200, 0) if nc >= 3 else \
                        (80, 255, 80)
                draw.rectangle([px, py, px + BLOCK//scale, py + BLOCK//scale],
                               outline=color, width=1)
                draw.text((px+3, py+2), f"{nc}/{ni}", fill=color)

        legend = ("[candidate/infra peaks per block]  "
                  "green=few  yellow=ships?  red=infra suspect")
        draw.text((4, preview_h-16), legend, fill=(200, 200, 200))

        density_path = OUT_DIR / f"{aoi_name}_density.png"
        img.save(density_path)
        print(f"  Density map → {density_path}")

        aoi_results[aoi_name] = {
            "db": db, "H": H, "W": W, "r0": r0, "c0": c0,
            "p1": p1, "p99": p99, "p50": p50,
            "grid_cand": grid_cand, "grid_infra": grid_infra,
            "grid_rows": list(grid_rows), "grid_cols": list(grid_cols),
        }
        del u8

# ── Native-res crops ────────────────────────────────────────────────────────
CROP_SIZE = 1024   # native-resolution crop side (px) — ~10 km square

for aoi_name, label_str, target_fn in [
    ("ots_aoi_b", "cluster — AOI B (center-right, near Gemini WF)",
     "ots_aoi_b_cluster_native.png"),
    ("ots_aoi_a", "reference — AOI A mid-section",
     "ots_aoi_a_sample_native.png"),
]:
    res = aoi_results[aoi_name]
    db  = res["db"]
    H, W = res["H"], res["W"]
    gc = res["grid_cand"]; gi_arr = res["grid_infra"]
    gr = res["grid_rows"]; gcols = res["grid_cols"]

    if aoi_name == "ots_aoi_b":
        # Find hottest infra block; if none, hottest candidate block
        if gi_arr.max() > 0:
            flat = gi_arr.argmax()
        else:
            flat = gc.argmax()
        best_gi, best_gj = np.unravel_index(flat, gi_arr.shape)
        center_r = gr[best_gi] + BLOCK // 2
        center_c = gcols[best_gj] + BLOCK // 2
        print(f"\nAOI B: hottest block at grid ({best_gi},{best_gj}), "
              f"scene-pixel row≈{res['r0']+center_r}, col≈{res['c0']+center_c}")
    else:
        # AOI A: mid-AOI sample
        center_r = H // 2
        center_c = W // 2

    r_start = max(0, center_r - CROP_SIZE // 2)
    c_start = max(0, center_c - CROP_SIZE // 2)
    r_end   = min(H, r_start + CROP_SIZE)
    c_end   = min(W, c_start + CROP_SIZE)
    crop_db = db[r_start:r_end, c_start:c_end]

    p1c = float(np.nanpercentile(crop_db, 1))
    p99c = float(np.nanpercentile(crop_db, 99))
    # Use wider dynamic range for native crop so ships stand out
    p5c = float(np.nanpercentile(crop_db, 5))
    p95c = float(np.nanpercentile(crop_db, 95))

    clipped = np.clip(crop_db, p5c, p95c)
    clipped = np.where(np.isfinite(clipped), clipped, p5c)
    u8 = ((clipped - p5c) / max(p95c - p5c, 1e-6) * 255).astype(np.uint8)

    img = Image.fromarray(u8).convert("RGB")
    draw = ImageDraw.Draw(img)
    ch, cw = u8.shape
    draw.text((4, 4), f"{aoi_name}  {label_str}", fill=(255, 255, 80))
    draw.text((4, 20), f"native res (1:1, ~10m/px)  {cw}×{ch} px crop", fill=(200,200,200))
    draw.text((4, 36), f"dB range p5–p95: {p5c:.1f} – {p95c:.1f} dB", fill=(200,200,200))

    out_path = OUT_DIR / target_fn
    img.save(out_path)
    print(f"Native-res crop ({cw}×{ch}) → {out_path}")

print("\nDone. Open:")
print("  ots_aoi_a_density.png      — AOI A grid search")
print("  ots_aoi_b_density.png      — AOI B grid search")
print("  ots_aoi_b_cluster_native.png  — native-res crop of hottest block")
print("  ots_aoi_a_sample_native.png   — native-res AOI A mid reference")
