"""
Download the Outer Thames / Southern North Sea offshore scene (2026-05-17).

Scene: S1A_IW_GRDH_1SDV_20260517T172536_20260517T172601_064560_0821BD_460B.SAFE
Date:  2026-05-17 17:25 UTC  (9 days old at time of selection)
Footprint: 52.893–54.799°N, 2.937–7.335°E
Wind:  SW=4.0 m/s Bft3, NW=0.9 Bft1, SE=4.1 Bft3, NE=2.9 Bft2  (all ≤ Bft 3)

Usage — run in your terminal with credentials already exported:
    cd "/Volumes/Tejas SSD/sar-dark-ship-detection"
    source .venv/bin/activate
    caffeinate -i python scripts/download_s1_ots.py

Reads CDSE_USER and CDSE_PASS from environment — never from a file.
Downloads to /Volumes/Tejas SSD/datasets/sentinel1/
"""
import os
import sys
from pathlib import Path

# ── Credentials from environment only ─────────────────────────────────────
CDSE_USER = os.environ.get("CDSE_USER")
CDSE_PASS = os.environ.get("CDSE_PASS")

if not CDSE_USER or not CDSE_PASS:
    print("ERROR: CDSE_USER and CDSE_PASS must be set as environment variables.")
    print("  export CDSE_USER='your@email.com'")
    print("  export CDSE_PASS='your-password'")
    sys.exit(1)

# ── Target product ─────────────────────────────────────────────────────────
PRODUCT_ID   = "fd83153d-4953-4bad-837c-9e10708c7eab"
PRODUCT_NAME = "S1A_IW_GRDH_1SDV_20260517T172536_20260517T172601_064560_0821BD_460B.SAFE"
OUTPUT_DIR   = Path("/Volumes/Tejas SSD/datasets/sentinel1")
EXPECTED_GB  = 1.8

# ── Check if already downloaded ────────────────────────────────────────────
output_path = OUTPUT_DIR / PRODUCT_NAME
if output_path.exists():
    size_gb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
    print(f"Already exists: {output_path}  ({size_gb:.2f} GB)")
    print("Delete the directory to re-download.")
    sys.exit(0)

# ── Authenticate ───────────────────────────────────────────────────────────
import requests

print("Authenticating with Copernicus Data Space …")
token_r = requests.post(
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    data={
        "grant_type": "password",
        "username": CDSE_USER,
        "password": CDSE_PASS,
        "client_id": "cdse-public",
    },
    timeout=20,
)
if not token_r.ok:
    try:
        err = token_r.json().get("error_description", token_r.text[:120])
    except Exception:
        err = token_r.text[:120]
    print(f"Auth failed (HTTP {token_r.status_code}): {err}")
    sys.exit(1)

token = token_r.json()["access_token"]
print(f"Auth OK — token received ({len(token)} chars)")

# ── Download ────────────────────────────────────────────────────────────────
import zipfile
import time

download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({PRODUCT_ID})/$value"
zip_path = OUTPUT_DIR / f"{PRODUCT_NAME}.zip"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading {PRODUCT_NAME}")
print(f"  → {zip_path}")
print(f"  Expected size: ~{EXPECTED_GB:.1f} GB  (this will take several minutes)")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {token}"})

t0 = time.time()
with session.get(download_url, stream=True, timeout=60) as resp:
    if not resp.ok:
        print(f"Download failed: HTTP {resp.status_code}  {resp.text[:200]}")
        sys.exit(1)

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    last_print = 0

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total else 0
            if downloaded - last_print > 100 << 20 or pct >= 100:
                elapsed = time.time() - t0
                speed = downloaded / elapsed / 1e6
                print(f"  {pct:5.1f}%  {downloaded/1e9:.2f}/{total/1e9:.2f} GB  "
                      f"{speed:.1f} MB/s  {elapsed:.0f}s elapsed")
                last_print = downloaded

elapsed = time.time() - t0
print(f"\nDownload complete: {zip_path.stat().st_size/1e9:.2f} GB in {elapsed:.0f}s")

# ── Unzip ───────────────────────────────────────────────────────────────────
print(f"\nUnzipping to {OUTPUT_DIR} …")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(OUTPUT_DIR)

print("Removing zip …")
zip_path.unlink()

# ── Clean macOS metadata files (exFAT SSD regenerates them) ───────────────
import subprocess
print("Cleaning ._* metadata files …")
subprocess.run(
    ["find", str(output_path), "-type", "f", "-name", "._*", "-delete"],
    capture_output=True,
)
print("  Done.")

# ── Verify VV .tiff is present ─────────────────────────────────────────────
meas_dir = output_path / "measurement"
vv_tiffs = [f for f in meas_dir.iterdir()
            if f.suffix == ".tiff" and "vv" in f.name and not f.name.startswith("._")]

if not vv_tiffs:
    print("\nERROR: No VV .tiff found in measurement/")
    sys.exit(1)

vv_tiff = vv_tiffs[0]
size_mb = vv_tiff.stat().st_size / 1e6
final_gb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
print(f"\nDone. {output_path}")
print(f"  Total size on disk: {final_gb:.2f} GB")
print(f"\nVV measurement TIFF:")
print(f"  {vv_tiff.relative_to(OUTPUT_DIR)}  ({size_mb:.0f} MB)")
print("\nNext step:")
print("  python scripts/preview_ots.py")
