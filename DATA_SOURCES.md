# Data sources, provenance, and licences

This file documents **every dataset** used in the study, mirroring **Table 2** of the manuscript and
adding the exact file(s) shipped in this archive for each one. All modelling layers were harmonised to
a common **1.5 km analysis grid in NAD 1983 BC Albers (EPSG:3005)**; the *native* source resolution is
listed below alongside the harmonised grid.

Access date for all layers: as reported in Table 2 of the article. Indices (ONI, MEI v2, PDO) were
retrieved from the official providers; the copies shipped here were downloaded on **2026-06-17**.

---

## Conditioning factors (predictor rasters) — `data/rasters/` (1.5 km, EPSG:3005)

| Group | Layer(s) | Source / provider | Native res. | Access URL |
|---|---|---|---|---|
| Topographic | Elevation (DEM), Slope, Aspect, Plan curvature, Profile curvature, TWI | USGS EarthExplorer — SRTM (derived for slope/curvature/TWI) | 30 m | https://earthexplorer.usgs.gov/ |
| Vegetative | NDVI | Landsat-8 (USGS EarthExplorer) | 30 m | https://earthexplorer.usgs.gov/ |
| Climatic | Max Temperature, Wind Speed (WS), Precipitation, Soil Moisture, Actual Evapotranspiration (AET), Drought Severity Index / PDSI | TerraClimate (Abatzoglou et al., 2018) | 4.63 km | https://www.climatologylab.org/terraclimate.html |
| Climatic | Specific Humidity | ERA5-Land reanalysis (ECMWF/Copernicus) | 11.1 km | https://cds.climate.copernicus.eu/ |
| Anthropogenic | Distance to rivers | Government of Canada — Lakes and Rivers (polygons) | vector → 30 m | https://open.canada.ca/ |
| Anthropogenic | Distance to roads | Statistics Canada — Road Network File | vector → 30 m | https://www.statcan.gc.ca/ |
| Anthropogenic | Distance to households | Government of Canada — Pseudo-Household Demographic Distribution | vector → 30 m | https://open.canada.ca/en |
| Anthropogenic | Land-use / land-cover (LULC) | Esri Sentinel-2 10 m Land Cover | 10 m | https://livingatlas.arcgis.com/landcover/ |

Distances are Euclidean, computed on a 30 m grid then resampled to the analysis grid.

**Source vectors now included (geometry-only).** The original vector layers behind two anthropogenic
predictors are provided under `data/anthropogenic_sources/` as geometry-only shapefiles (locations only;
large source attribute tables removed):

| File | Layer | Features | CRS | Provider |
|---|---|---|---|---|
| `data/anthropogenic_sources/households_PHH/PHH-BC.*` | Pseudo-household points (distance-to-households) | 1,842,788 | EPSG:4326 | Statistics Canada — Pseudo-Household Demographic Distribution (PHH 2021) |
| `data/anthropogenic_sources/road_network/bc_roads.*` | BC road-network lines (distance-to-roads) | 278,798 | NAD83 Lambert Conformal Conic (Canada, custom) | Statistics Canada — Road Network File |


## Wildfire inventory — `data/processed/fires_geq_70ha.*` (vector, EPSG:3005)

| Layer | Source / provider | Type | Access URL |
|---|---|---|---|
| Historical wildfire events / National Burned Area Composite (≥70 ha, 2000–2024) | Canadian Wildland Fire Information System (CWFIS) / National Fire Database (NFDB) | Point / polygon shapefile | https://cwfis.cfs.nrcan.gc.ca/ |

## Climate-oscillation indices — `data/raw/`

| Index | File | Source / provider | Type | Access URL |
|---|---|---|---|---|
| ENSO — Oceanic Niño Index (ONI) | `oni.ascii.txt` | NOAA Climate Prediction Center | Monthly time series (3-month running Niño-3.4 SST anomaly) | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt |
| ENSO — Multivariate ENSO Index v2 (MEI v2) | `meiv2.txt` | NOAA Physical Sciences Laboratory | Bimonthly time series | https://psl.noaa.gov/enso/mei/ |
| Pacific Decadal Oscillation (PDO) | `pdo.csv` | NOAA Physical Sciences Laboratory | Monthly time series | https://psl.noaa.gov/pdo/ |

> **Note on the ENSO index.** Table 2 of the manuscript lists the **ONI** (NOAA CPC). The
> teleconnection analysis script (`code/teleconnections.py`) was run on the **MEI v2** series. Both the
> official ONI series and the MEI v2 series used are therefore included here for full transparency; the
> PDO series is identical in both. See `data/raw/README.md`.

## Cross-border comparison (Section 3.10 / Comment 4) — `data/cross_border_US/`

| Layer | File | Source / provider | Native res. / type | Access URL |
|---|---|---|---|---|
| U.S. annual burn probability (FSim) for WA, ID, MT | `BP_WA_ID_MT.tif` | USDA Forest Service — Wildfire Risk to Communities (FSim) | Raster, 270 m (reprojected to EPSG:3005 in analysis) | https://wildfirerisk.org/ |
| 49°N BC sampling strip | `bc_strip.*` | Derived in this study (EPSG:3005) | Vector | — |
| 49°N U.S. sampling strip | `us_strip.*` | Derived in this study (EPSG:3005) | Vector | — |
| BC–U.S. border line | `bc_us_border.*` | Derived from boundary data (EPSG:3005) | Vector | — |

## U.S. fire records — `data/fire_records_US/`

| Layer | File | Source / provider | Type | Access URL |
|---|---|---|---|---|
| FPA-FOD fire occurrences, subset to WA, ID, MT | `fpa_fod_WA_ID_MT.*` | USDA Forest Service — Fire Program Analysis Fire-Occurrence Database (FPA-FOD) | Point shapefile (NAD83 geographic, EPSG:4269) | https://www.fs.usda.gov/rds/archive/ |

## Boundaries — `data/boundaries/`

| Layer | File | Source / provider | Type | Access URL |
|---|---|---|---|---|
| British Columbia provincial boundary | `bc_boundary.*` | Statistics Canada (NAD83 / Statistics Canada Lambert) | Vector | https://www.statcan.gc.ca/ |
| U.S. state polygons (cartographic, 2023, 1:5 m) | `cb_2023_us_state_5m.*` | U.S. Census Bureau (NAD83 geographic, EPSG:4269) | Vector | https://www.census.gov/geographies/mapping-files.html |

---

## Licensing

- **Code** (`code/`): MIT Licence (see `LICENSE`).
- **Derived data produced in this study** (processed rasters, training tables, susceptibility maps,
  SHAP values, derived border strips): Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Third-party source data** (SRTM, Landsat-8, TerraClimate, ERA5-Land, CWFIS/NFDB, NOAA CPC/PSL,
  USDA FSim & FPA-FOD, Statistics Canada, U.S. Census Bureau, Esri Sentinel-2) remain under the terms
  of their respective providers, cited above and in Table 2 of the article.
