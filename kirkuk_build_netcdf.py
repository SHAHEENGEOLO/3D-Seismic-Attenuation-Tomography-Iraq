# -*- coding: utf-8 -*-
"""
=====================================================================
  NetCDF Builder for ArcGIS Pro - Kirkuk Attenuation Tomography
=====================================================================
  Reads all CSV outputs from AST v3.1 and creates a single CF-1.8
  compliant multidimensional NetCDF file for use in ArcGIS Pro.

  Contents of the NetCDF:
    DIMENSIONS:
      longitude       (100)  38.0 - 49.0 deg E
      latitude        (108)  29.0 - 39.0 deg N
      depth           (16)   0 - 15 km
      band            (4)    frequency bands
      ensemble_member (10)   bootstrap members
      iteration       (60)   inversion iterations

    PRIMARY VARIABLES (4D: band x depth x lat x lon):
      Qi_inv          Intrinsic attenuation (ensemble mean)
      Qi_inv_std      Intrinsic attenuation uncertainty
      Qsct_inv        Scattering attenuation (ensemble mean)
      Qsct_inv_std    Scattering attenuation uncertainty

    DERIVED VARIABLES (4D):
      Qt_inv          Total attenuation (Qi^-1 + Qsct^-1)
      Qsct_Qi_ratio   Scattering/intrinsic ratio
      Qi_SNR          Qi reliability (|mean|/std)
      Qsct_SNR        Qsct reliability (|mean|/std)

    DIAGNOSTICS:
      loss_total/data/tv/aniso/pnp  (member x iteration)
      slice_Qi/Qsct_min/max/mean/std  (band x depth)

    SPATIAL REFERENCE:
      WGS84 (EPSG:4326) via CF grid_mapping + esri_pe_string

  Requirements:
      pip install scipy pandas numpy

  Usage:
      python kirkuk_build_netcdf.py

  ArcGIS Pro:
      Add Data > Multidimensional Raster Layer > select .nc file
      Use band/depth dimension sliders to browse 4D volume
=====================================================================
"""

import numpy as np
import pandas as pd
import os, sys
from scipy.io import netcdf_file
from datetime import datetime, timezone

# ═════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════
ROOT = r"C:\Users\shahe\Desktop\Kirkuk SAC files\ast_outputs_kirkuk"
OUT  = os.path.join(ROOT, "Kirkuk_Attenuation_Tomography.nc")

# Frequency band definitions (Hz)
BAND_FREQS = np.array([
    [0.5,  1.5],
    [1.5,  3.0],
    [3.0,  6.0],
    [6.0, 12.0],
], dtype=np.float64)

FILL = np.float32(-9999.0)

# ═════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════
def load_csv(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {name}")
        return None
    df = pd.read_csv(path)
    print(f"  {name}: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def csv_to_4d(df, lons, lats, depths, nb):
    """Reshape flat CSV into [band, depth, lat, lon] array."""
    nx, ny, nz = len(lons), len(lats), len(depths)
    vol = np.full((nb, nz, ny, nx), np.nan, dtype=np.float32)
    for _, r in df.iterrows():
        bi = int(r.band_index)
        ix = np.searchsorted(lons, r.lon)
        iy = np.searchsorted(lats, r.lat)
        iz = np.searchsorted(depths, r.depth_km)
        if ix < nx and iy < ny and iz < nz and bi < nb:
            vol[bi, iz, iy, ix] = r.value
    return vol


def add_coord(nc, name, dim, data, **attrs):
    """Add a coordinate variable with attributes."""
    v = nc.createVariable(name, "d", (dim,))
    v[:] = data
    for k, val in attrs.items():
        setattr(v, k, val)


def add_4d_var(nc, name, data, long_name, comment="", **extra):
    """Add a 4D data variable with full CF attributes."""
    v = nc.createVariable(name, "f", ("band", "depth", "latitude", "longitude"))
    v[:] = data
    v.long_name = long_name
    v.units = "1"
    v.missing_value = FILL
    v._FillValue = FILL
    v.grid_mapping = "crs"
    v.coordinates = "band_freq_center depth latitude longitude"
    if comment:
        v.comment = comment
    fdata = data[data != FILL]
    if len(fdata) > 0:
        v.valid_min = float(fdata.min())
        v.valid_max = float(fdata.max())
    for k, val in extra.items():
        setattr(v, k, val)


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  NetCDF Builder - Kirkuk Attenuation Tomography")
    print("=" * 65)

    if not os.path.isdir(ROOT):
        sys.exit(f"[ERROR] Directory not found: {ROOT}")

    # ── 1. Load CSVs ──
    print("\n[1/5] Loading CSV files...")
    qi_df     = load_csv("Qi_voxels.csv")
    qi_std_df = load_csv("Qi_std_voxels.csv")
    qs_df     = load_csv("Qsct_voxels.csv")
    qs_std_df = load_csv("Qsct_std_voxels.csv")
    sl_df     = load_csv("slice_stats.csv")

    if qi_df is None:
        sys.exit("[ERROR] Qi_voxels.csv is required")

    loss_dfs = {}
    for fn in sorted(os.listdir(ROOT)):
        if fn.startswith("inversion_loss_m") and fn.endswith(".csv"):
            m = int(fn.replace("inversion_loss_m", "").replace(".csv", ""))
            loss_dfs[m] = pd.read_csv(os.path.join(ROOT, fn))
    print(f"  Loaded {len(loss_dfs)} inversion loss files")

    # ── 2. Build coordinates ──
    print("\n[2/5] Building coordinate arrays...")
    lons   = np.sort(qi_df["lon"].unique()).astype(np.float64)
    lats   = np.sort(qi_df["lat"].unique()).astype(np.float64)
    depths = np.sort(qi_df["depth_km"].unique()).astype(np.float64)
    bands  = np.array(sorted(qi_df["band_index"].unique()), dtype=np.int32)
    nx, ny, nz, nb = len(lons), len(lats), len(depths), len(bands)
    band_centers = np.sqrt(BAND_FREQS[:nb, 0] * BAND_FREQS[:nb, 1])

    print(f"  lon:   {nx} pts [{lons[0]:.4f} - {lons[-1]:.4f}] deg E")
    print(f"  lat:   {ny} pts [{lats[0]:.4f} - {lats[-1]:.4f}] deg N")
    print(f"  depth: {nz} pts [{depths[0]:.1f} - {depths[-1]:.1f}] km")
    print(f"  bands: {nb}")

    # ── 3. Reshape to 4D ──
    print("\n[3/5] Reshaping CSVs into 4D volumes...")
    Qi = csv_to_4d(qi_df, lons, lats, depths, nb)
    print(f"  Qi_inv: {np.isfinite(Qi).sum():,} valid voxels")

    Qi_s = csv_to_4d(qi_std_df, lons, lats, depths, nb) if qi_std_df is not None else np.full_like(Qi, np.nan)
    Qs   = csv_to_4d(qs_df, lons, lats, depths, nb) if qs_df is not None else np.full_like(Qi, np.nan)
    Qs_s = csv_to_4d(qs_std_df, lons, lats, depths, nb) if qs_std_df is not None else np.full_like(Qi, np.nan)

    # Derived
    Qt    = np.where(np.isfinite(Qi) & np.isfinite(Qs), Qi + Qs, np.nan)
    ratio = np.where(np.isfinite(Qi) & np.isfinite(Qs) & (np.abs(Qi) > 1e-8), Qs / Qi, np.nan)
    snr_q = np.where(np.isfinite(Qi) & np.isfinite(Qi_s) & (Qi_s > 1e-10), np.abs(Qi) / Qi_s, np.nan)
    snr_s = np.where(np.isfinite(Qs) & np.isfinite(Qs_s) & (Qs_s > 1e-10), np.abs(Qs) / Qs_s, np.nan)

    # Fill NaN
    for vol in [Qi, Qi_s, Qs, Qs_s, Qt, ratio, snr_q, snr_s]:
        vol[np.isnan(vol)] = FILL

    # Loss arrays
    max_iter = max(len(df) for df in loss_dfs.values()) if loss_dfs else 1
    n_mem = max(len(loss_dfs), 1)
    loss_arr = {}
    for f in ["total", "data", "tv", "aniso", "pnp"]:
        a = np.full((n_mem, max_iter), FILL, dtype=np.float32)
        for mi, (mn, df) in enumerate(sorted(loss_dfs.items())):
            v = df[f].values.astype(np.float32)
            a[mi, :len(v)] = v
        loss_arr[f] = a

    # ── 4. Write NetCDF ──
    print("\n[4/5] Writing NetCDF file...")
    if os.path.exists(OUT):
        os.remove(OUT)

    nc = netcdf_file(OUT, 'w', version=2)

    # Global attributes
    nc.Conventions = "CF-1.8"
    nc.title = "Seismic Attenuation Tomography - Kirkuk Region, Iraq"
    nc.institution = "ATTEN-SPEC-TOMO v3.1"
    nc.source = "Physics-Informed Self-Supervised Inversion with Ensemble UQ"
    nc.history = f"Created {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    nc.references = "Dr. Shaheen - Kirkuk SAC Waveform Inversion"
    nc.comment = (
        "4D seismic attenuation tomography (Qi_inv and Qsct_inv) with uncertainty. "
        f"Grid: {nx}x{ny}x{nz} (lon x lat x depth), {nb} frequency bands. "
        "Region: Northern Iraq / Kirkuk / Zagros Foothills. "
        f"Ensemble: {n_mem} bootstrap members, MC-Dropout PnP prior."
    )
    nc.geospatial_lat_min = float(lats[0])
    nc.geospatial_lat_max = float(lats[-1])
    nc.geospatial_lon_min = float(lons[0])
    nc.geospatial_lon_max = float(lons[-1])
    nc.geospatial_vertical_min = float(depths[0])
    nc.geospatial_vertical_max = float(depths[-1])
    nc.geospatial_vertical_units = "km"
    nc.geospatial_vertical_positive = "down"
    nc.frequency_bands_hz = "0.5-1.5, 1.5-3.0, 3.0-6.0, 6.0-12.0"
    nc.esri_pe_string = (
        'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
        'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
        'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    )

    # Dimensions
    nc.createDimension("longitude", nx)
    nc.createDimension("latitude", ny)
    nc.createDimension("depth", nz)
    nc.createDimension("band", nb)
    nc.createDimension("ensemble_member", n_mem)
    nc.createDimension("iteration", max_iter)

    # Coordinate variables
    add_coord(nc, "longitude", "longitude", lons,
              standard_name="longitude", long_name="Longitude",
              units="degrees_east", axis="X")
    add_coord(nc, "latitude", "latitude", lats,
              standard_name="latitude", long_name="Latitude",
              units="degrees_north", axis="Y")
    add_coord(nc, "depth", "depth", depths,
              standard_name="depth", long_name="Depth below surface",
              units="km", axis="Z", positive="down")
    add_coord(nc, "band", "band", bands.astype(np.float64),
              long_name="Frequency band index", units="1",
              comment="0=0.5-1.5Hz, 1=1.5-3.0Hz, 2=3.0-6.0Hz, 3=6.0-12.0Hz")
    add_coord(nc, "band_freq_center", "band", band_centers,
              long_name="Band center frequency (geometric mean)", units="Hz")
    add_coord(nc, "band_freq_min", "band", BAND_FREQS[:nb, 0],
              long_name="Band lower frequency edge", units="Hz")
    add_coord(nc, "band_freq_max", "band", BAND_FREQS[:nb, 1],
              long_name="Band upper frequency edge", units="Hz")
    add_coord(nc, "ensemble_member", "ensemble_member",
              np.arange(1, n_mem + 1, dtype=np.float64),
              long_name="Ensemble member index", units="1")
    add_coord(nc, "iteration", "iteration",
              np.arange(max_iter, dtype=np.float64),
              long_name="Inversion iteration index", units="1")

    # CRS grid mapping (critical for ArcGIS Pro)
    crs = nc.createVariable("crs", "i", ())
    crs._FillValue = np.int32(0)
    crs.data = np.int32(0)
    crs.grid_mapping_name = "latitude_longitude"
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563
    crs.longitude_of_prime_meridian = 0.0
    crs.crs_wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433],'
        'AUTHORITY["EPSG","4326"]]'
    )
    crs.spatial_ref = crs.crs_wkt

    # Primary tomography variables
    add_4d_var(nc, "Qi_inv", Qi,
               "Intrinsic attenuation (Qi inverse)",
               "Ensemble mean Qi^-1 from bootstrap inversion",
               cell_methods="ensemble_member: mean")
    add_4d_var(nc, "Qi_inv_std", Qi_s,
               "Intrinsic attenuation uncertainty (std)",
               "Ensemble std of Qi^-1 (epistemic uncertainty)",
               cell_methods="ensemble_member: standard_deviation")
    add_4d_var(nc, "Qsct_inv", Qs,
               "Scattering attenuation (Qsct inverse)",
               "Ensemble mean Qsct^-1 from bootstrap inversion",
               cell_methods="ensemble_member: mean")
    add_4d_var(nc, "Qsct_inv_std", Qs_s,
               "Scattering attenuation uncertainty (std)",
               "Ensemble std of Qsct^-1 (epistemic uncertainty)",
               cell_methods="ensemble_member: standard_deviation")

    # Derived variables
    add_4d_var(nc, "Qt_inv", Qt,
               "Total attenuation (Qt^-1 = Qi^-1 + Qsct^-1)",
               "Sum of intrinsic and scattering attenuation")
    add_4d_var(nc, "Qsct_Qi_ratio", ratio.astype(np.float32),
               "Scattering-to-intrinsic ratio (Qsct^-1 / Qi^-1)",
               ">1 = scattering dominant, <1 = intrinsic dominant")
    add_4d_var(nc, "Qi_SNR", snr_q.astype(np.float32),
               "Qi reliability (|mean|/std)",
               "Higher values = more reliable inversion")
    add_4d_var(nc, "Qsct_SNR", snr_s.astype(np.float32),
               "Qsct reliability (|mean|/std)",
               "Higher values = more reliable inversion")

    # Inversion loss curves
    for fname in ["total", "data", "tv", "aniso", "pnp"]:
        v = nc.createVariable(f"loss_{fname}", "f",
                              ("ensemble_member", "iteration"))
        v[:] = loss_arr[fname]
        v.long_name = f"Inversion loss: {fname}"
        v.units = "1"
        v.missing_value = FILL
        v._FillValue = FILL

    # Slice statistics
    if sl_df is not None:
        for field in ["Qi", "Qsct"]:
            sub = sl_df[sl_df["field"] == field]
            for stat in ["min", "max", "mean", "std"]:
                arr = np.full((nb, nz), FILL, dtype=np.float32)
                for _, r in sub.iterrows():
                    bi = int(r.band_index); zi = int(r.z_index)
                    if bi < nb and zi < nz:
                        arr[bi, zi] = float(r[stat])
                v = nc.createVariable(f"slice_{field}_{stat}", "f",
                                      ("band", "depth"))
                v[:] = arr
                v.long_name = f"{field}^-1 per-slice {stat}"
                v.units = "1"
                v.missing_value = FILL
                v._FillValue = FILL

    nc.close()

    # ── 5. Verify ──
    print("\n[5/5] Verification...")
    fsize = os.path.getsize(OUT)
    nc2 = netcdf_file(OUT, 'r', mmap=False)

    print(f"\n  File: {OUT}")
    print(f"  Size: {fsize / 1e6:.1f} MB")
    print(f"  Dimensions: {len(nc2.dimensions)}")
    print(f"  Variables:  {len(nc2.variables)}")
    print(f"  Convention: CF-1.8")

    for vn in ["Qi_inv", "Qi_inv_std", "Qsct_inv", "Qsct_inv_std",
               "Qt_inv", "Qsct_Qi_ratio", "Qi_SNR", "Qsct_SNR"]:
        v = nc2.variables[vn]
        d = v.data.copy()
        valid = d[d != -9999.0]
        print(f"  {vn:22s}  shape={str(v.shape):22s}  "
              f"valid={len(valid):>7,}  [{valid.min():.4f}, {valid.max():.4f}]")

    nc2.close()

    print(f"\n{'='*65}")
    print(f"  NetCDF ready for ArcGIS Pro: {OUT}")
    print(f"{'='*65}")
    print(f"\n  HOW TO USE IN ARCGIS PRO:")
    print(f"    1. Map > Add Data > Multidimensional Raster Layer")
    print(f"    2. Browse to: {os.path.basename(OUT)}")
    print(f"    3. Select variable (Qi_inv, Qsct_inv, Qt_inv, etc.)")
    print(f"    4. Band + Depth dimensions auto-detected as sliders")
    print(f"    5. Multidimensional tab > animate through slices")
    print(f"    6. Geoprocessing > Make Multidimensional Raster Layer")
    print(f"    7. For voxel: 3D Analyst > Create NetCDF Raster Layer")
    print(f"    8. For profiles: 3D Analyst > Profile Graph")
    print()


if __name__ == "__main__":
    main()
