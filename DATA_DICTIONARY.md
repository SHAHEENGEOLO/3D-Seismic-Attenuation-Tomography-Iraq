# Data Dictionary

This file describes the main data products provided in the repository.

## NetCDF File

**File:** `data/netcdf/Kirkuk_Attenuation_Tomography.nc`

**Format:** CF-1.8 multidimensional NetCDF  
**Coordinate reference system:** WGS84 / EPSG:4326  
**Longitude range:** 38.0°E to 49.0°E  
**Latitude range:** 29.0°N to 39.0°N  
**Depth range:** 0 to 15 km  
**Grid size:** 100 × 108 × 16  
**Frequency bands:** 0.5–1.5 Hz, 1.5–3.0 Hz, 3.0–6.0 Hz, 6.0–12.0 Hz  
**Ensemble members:** 10  
**Inversion iterations stored:** 60

### Principal Variables

| Variable | Description | Dimensions |
|---|---|---|
| `Qi_inv` | Intrinsic attenuation, ensemble mean | band × depth × latitude × longitude |
| `Qi_inv_std` | Uncertainty of intrinsic attenuation | band × depth × latitude × longitude |
| `Qsct_inv` | Scattering attenuation, ensemble mean | band × depth × latitude × longitude |
| `Qsct_inv_std` | Uncertainty of scattering attenuation | band × depth × latitude × longitude |
| `Qt_inv` | Total attenuation, calculated as Qi⁻¹ + Qsct⁻¹ | band × depth × latitude × longitude |
| `Qsct_Qi_ratio` | Scattering-to-intrinsic attenuation ratio | band × depth × latitude × longitude |
| `Qi_SNR` | Reliability proxy for intrinsic attenuation | band × depth × latitude × longitude |
| `Qsct_SNR` | Reliability proxy for scattering attenuation | band × depth × latitude × longitude |
| `loss_total`, `loss_data`, `loss_tv`, `loss_aniso`, `loss_pnp` | Inversion diagnostics | ensemble_member × iteration |
| `slice_Qi_min`, `slice_Qi_max`, `slice_Qi_mean`, `slice_Qi_std` | Slice statistics for intrinsic attenuation | band × depth |
| `slice_Qsct_min`, `slice_Qsct_max`, `slice_Qsct_mean`, `slice_Qsct_std` | Slice statistics for scattering attenuation | band × depth |

## CSV Files

### Voxel files

| File | Description | Main columns |
|---|---|---|
| `Qi_voxels.csv` | Intrinsic attenuation values at each voxel | `band_index`, `lon`, `lat`, `depth_km`, `value` |
| `Qi_std_voxels.csv` | Uncertainty of intrinsic attenuation at each voxel | `band_index`, `lon`, `lat`, `depth_km`, `value` |
| `Qsct_voxels.csv` | Scattering attenuation values at each voxel | `band_index`, `lon`, `lat`, `depth_km`, `value` |
| `Qsct_std_voxels.csv` | Uncertainty of scattering attenuation at each voxel | `band_index`, `lon`, `lat`, `depth_km`, `value` |

### Statistical files

| File | Description |
|---|---|
| `Qi_subregion_stats.csv` | Subregion statistics for intrinsic attenuation |
| `Qsct_subregion_stats.csv` | Subregion statistics for scattering attenuation |
| `slice_stats.csv` | Minimum, maximum, mean, standard deviation, NaN fraction, and quality flag for each field, band, and depth slice |
| `inversion_loss_m1.csv` ... `inversion_loss_m10.csv` | Inversion loss curves for the 10 ensemble members |

## GIS Files

### Fault segments

The `Fault segments/` directory contains the extracted 27 fault-segment shapefile components:

- `Iraq_Fault_Segments_27.shp`
- `Iraq_Fault_Segments_27.shx`
- `Iraq_Fault_Segments_27.dbf`
- `Iraq_Fault_Segments_27.prj`

### Fault segment summary

**File:** `Fault_Segments_Summary.csv`

Important attributes include:

| Attribute | Meaning |
|---|---|
| `FaultName` | Name or ID of the interpreted fault segment |
| `FaultType` | Structural style, such as thrust, strike-slip, or oblique |
| `SegLen_km` | Segment length in kilometers |
| `MinDep_km`, `MaxDep_km` | Minimum and maximum interpreted depth |
| `Width_km` | Interpreted width |
| `Trend_deg` | Segment trend in degrees |
| `Dip_deg` | Dip angle in degrees |
| `DipDir_deg` | Dip direction in degrees |
| `Activity` | Interpreted activity class |
| `Qi_mean`, `Qi_std` | Intrinsic attenuation statistics |
| `Qsct_mean` | Scattering attenuation statistic |
| `Qt_mean` | Total attenuation statistic |
| `Ratio_QsQi` | Scattering-to-intrinsic attenuation ratio |
| `SNR` | Reliability proxy |
| `Evidence` | Composite geophysical evidence value |
| `Confid` | Confidence class |
| `CenterLon`, `CenterLat` | Segment center coordinates |
| `Proximity` | Distance/proximity measure |
| `N_pixels` | Number of pixels/lineament cells associated with the segment |
| `Rank` | Ranked segment order |
