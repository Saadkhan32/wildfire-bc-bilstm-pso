# `data/` — inventory

Every input and processed dataset used in the study. All modelling layers are on the **1.5 km analysis
grid, NAD 1983 BC Albers (EPSG:3005)**, study period **2000–2024**, wildfire response **≥ 70 ha**.
Provenance, providers, and licences for each dataset are in `../DATA_SOURCES.md`; machine-readable
metadata are in `../metadata/` (ISO 19115-2 XML per layer, CSVW per table, SHA-256 checksums).

| Sub-folder / file | What it is | Format |
|---|---|---|
| `rasters/` | 18 conditioning-factor layers (topographic, vegetative, climatic, anthropogenic) | GeoTIFF (EPSG:3005) |
| `climate/` | Monthly ERA5-Land + TerraClimate climate tables (2000–2024) | CSV |
| `raw/` | Climate-oscillation indices: ONI, MEI v2, PDO | text / CSV |
| `processed/` | BC wildfire perimeters (≥70 ha) and the derived monthly teleconnection table | shapefile + CSV |
| `boundaries/` | BC provincial boundary; U.S. state polygons; (BC–U.S. border line in `cross_border_US/`) | shapefile |
| `cross_border_US/` | U.S. FSim annual burn probability (WA/ID/MT) + 49°N BC/U.S. sampling strips + border line | GeoTIFF + shapefile |
| `fire_records_US/` | U.S. FPA-FOD fire occurrences subset to WA, ID, MT | shapefile |
| `anthropogenic_sources/` | Source vectors (geometry-only) for the road & household distance predictors | shapefile |
| `susceptibility/` | Modelled susceptibility maps (continuous + 5-class) for LSTM, BiLSTM, LSTM-PSO, BiLSTM-PSO | GeoTIFF |
| `shap/` | SHAP attribution values for the BiLSTM-PSO model | pickle |
| `training_points_70ha_seed42.csv` | **Primary modelling table**: 1,992 fire + 1,992 non-fire points × 18 factors | CSV |
| `training_points_70ha_seed{101…909}.csv` | Same design, 9 further random seeds (sensitivity analysis) | CSV |
| `training_points_{100,200}ha_seed42.csv` | Alternative burned-area thresholds (sensitivity analysis) | CSV |
| `analysis_dataset.csv` | Annual table (area burned vs climate / teleconnection indices) | CSV |

Column definitions for the modelling tables: `../metadata/data_dictionary.csv` and
`../metadata/training_points.csvw.json`.
