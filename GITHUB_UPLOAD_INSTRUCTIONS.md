# GitHub Upload Instructions

## 1. Create folders

In GitHub, use:

```text
Add file → Create new file
```

Create each folder by typing the folder name followed by a placeholder README file:

```text
codes/README.md
data/netcdf/README.md
data/csv/README.md
data/gis/README.md
docs/README.md
```

## 2. Upload files to the correct folders

### codes/

Upload:

```text
kirkuk_build_netcdf.py
kirkuk_fault_extraction.py
```

### data/netcdf/

Upload:

```text
Kirkuk_Attenuation_Tomography.nc
```

### data/csv/

Extract `csv files(1).zip` and upload all CSV files:

```text
Qi_voxels.csv
Qi_std_voxels.csv
Qsct_voxels.csv
Qsct_std_voxels.csv
Qi_subregion_stats.csv
Qsct_subregion_stats.csv
slice_stats.csv
inversion_loss_m1.csv ... inversion_loss_m10.csv
```

### data/gis/

Extract `GIS Files(1).zip` and upload the GIS folders and files:

```text
Fault segments/
Iraq administrative/
Fault_Segments_Summary.csv
Iraq_Fault_Segments.*
kirkuk_fault_extraction.py
```

If the script is already uploaded to `codes/`, you can omit the duplicate GIS script or keep it in both places.

### docs/

Upload:

```text
Final_Kirkuk_AST_v4_Report.pdf
```

## 3. Recommended commit messages

```text
Add NetCDF attenuation tomography volume
Add CSV attenuation voxel tables and ensemble diagnostics
Add GIS fault-segment shapefiles and summary table
Add Python processing and NetCDF builder scripts
Add final workflow documentation
Update README and data dictionary
```

## 4. Recommended repository topics

```text
seismic-tomography
attenuation-tomography
kirkuk
iraq
geophysics
seismology
netcdf
gis
python
fault-segments
```
