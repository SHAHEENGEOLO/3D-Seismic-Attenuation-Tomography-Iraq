# 3D Seismic Attenuation Tomography of Iraq

This repository contains supplementary data products, GIS files, and Python codes associated with a 3D seismic attenuation tomography study of the Kirkuk region, northern Iraq. The repository is prepared to support transparency, reproducibility, independent verification, and future extension of the results presented in the related manuscript.

## Study Area and Model Domain

The tomography model covers northern Iraq and the surrounding Kirkuk–Zagros foothill region using a geographic grid from 38.0°E to 49.0°E and 29.0°N to 39.0°N. The depth interval extends from 0 to 15 km. The multidimensional attenuation model is provided for four frequency bands: 0.5–1.5 Hz, 1.5–3.0 Hz, 3.0–6.0 Hz, and 6.0–12.0 Hz.

## Repository Contents

```text
3D-Seismic-Attenuation-Tomography-Iraq/
│
├── README.md
├── requirements.txt
├── CITATION.cff
├── DATA_DICTIONARY.md
├── .gitignore
│
├── codes/
│   ├── kirkuk_build_netcdf.py
│   └── kirkuk_fault_extraction.py
│
├── data/
│   ├── netcdf/
│   │   └── Kirkuk_Attenuation_Tomography.nc
│   │
│   ├── csv/
│   │   ├── Qi_voxels.csv
│   │   ├── Qi_std_voxels.csv
│   │   ├── Qsct_voxels.csv
│   │   ├── Qsct_std_voxels.csv
│   │   ├── Qi_subregion_stats.csv
│   │   ├── Qsct_subregion_stats.csv
│   │   ├── slice_stats.csv
│   │   └── inversion_loss_m1.csv ... inversion_loss_m10.csv
│   │
│   └── gis/
│       ├── Fault segments/
│       │   ├── Iraq_Fault_Segments_27.shp
│       │   ├── Iraq_Fault_Segments_27.shx
│       │   ├── Iraq_Fault_Segments_27.dbf
│       │   └── Iraq_Fault_Segments_27.prj
│       │
│       ├── Iraq administrative/
│       │   ├── iraq_administrative.shp
│       │   ├── iraq_administrative.shx
│       │   ├── iraq_administrative.dbf
│       │   └── iraq_administrative.prj
│       │
│       ├── Iraq_Fault_Segments.shp
│       ├── Iraq_Fault_Segments.shx
│       ├── Iraq_Fault_Segments.dbf
│       ├── Iraq_Fault_Segments.prj
│       ├── Iraq_Fault_Segments.cpg
│       └── Fault_Segments_Summary.csv
│
└── docs/
    └── Final_Kirkuk_AST_v4_Report.pdf
```

## Main Data Products

### 1. Multidimensional NetCDF file

The file `Kirkuk_Attenuation_Tomography.nc` is a CF-1.8 multidimensional NetCDF product containing the final 4D attenuation volumes. It includes:

- intrinsic attenuation: `Qi_inv`
- intrinsic attenuation uncertainty: `Qi_inv_std`
- scattering attenuation: `Qsct_inv`
- scattering attenuation uncertainty: `Qsct_inv_std`
- total attenuation: `Qt_inv`
- scattering-to-intrinsic attenuation ratio: `Qsct_Qi_ratio`
- reliability proxies: `Qi_SNR` and `Qsct_SNR`
- inversion loss diagnostics
- depth- and frequency-dependent slice statistics
- WGS84 spatial reference information

### 2. CSV voxel and statistical tables

The CSV files provide tabular outputs from the attenuation tomography workflow. They include voxel-level intrinsic and scattering attenuation values, uncertainty estimates, slice statistics, subregion statistics, and inversion loss curves for the ensemble members.

### 3. GIS shapefiles

The GIS folder contains ESRI Shapefile datasets for the extracted fault segments and the Iraq administrative boundary used for mapping and spatial interpretation. The fault-segment shapefiles include geophysical attributes such as segment length, depth range, trend, dip, activity class, attenuation statistics, signal-to-noise values, confidence class, central coordinates, proximity, and rank.

### 4. Python codes

The Python scripts support the construction of the NetCDF product and the extraction of fault-segment GIS outputs from the attenuation tomography results.

## Software Requirements

The codes were developed in Python and require common scientific and geospatial libraries. The main packages are listed in `requirements.txt`.

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Basic Usage

To rebuild the NetCDF product from the CSV outputs, place the CSV files in the configured output directory and run:

```bash
python codes/kirkuk_build_netcdf.py
```

To extract GIS fault-segment products from the NetCDF attenuation model, run:

```bash
python codes/kirkuk_fault_extraction.py
```

The paths inside the scripts may need to be adjusted according to the local location of the data files.

## ArcGIS Pro Usage

The NetCDF file can be opened in ArcGIS Pro as a multidimensional raster layer:

```text
Add Data → Multidimensional Raster Layer → Select Kirkuk_Attenuation_Tomography.nc
```

The frequency-band and depth dimensions can then be explored using the multidimensional layer controls.

## Data Availability

The data products supporting the findings of the related study are provided in this repository and may also be submitted as supplementary material or deposited in an institutional repository upon publication. These materials include the NetCDF attenuation volume, GIS fault-segment shapefiles, CSV statistical tables, and Python scripts used for processing and visualization. The repository is intended to enable independent verification, reproducibility, and extension of the results.

## Citation

If you use this repository, please cite the associated manuscript after publication.

## Author

Dr. Shaheen Mohammed Saleh Ahmed  
Department of Applied Geology  
Kirkuk University, Iraq
