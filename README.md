# Wildfire susceptibility and exposure modelling in British Columbia

Code and processed data to reproduce the wildfire susceptibility and exposure
analysis for British Columbia, Canada, using LSTM and BiLSTM neural networks
optimized with Particle Swarm Optimization (PSO), evaluated under spatial
cross-validation and interpreted with SHAP.

## Repository structure
- `code/`   analysis scripts (Python)
- `data/`   processed modelling datasets (point samples with conditioning factors)
- `metadata/` data dictionary and dataset metadata

## Data
`data/training_points_70ha_seed42.csv` is the primary modelling dataset: 1,992
wildfire points (burned area >= 70 ha, 2000-2024) and 1,992 spatially
independent non-wildfire points, each with the conditioning factors listed in
`metadata/data_dictionary.csv`. Files for the other random seeds (101-909) and
for the 100 ha and 200 ha thresholds support the sensitivity analysis. All
spatial layers use the NAD 1983 BC Albers equal-area projection (EPSG:3005).

Input raster and vector layers are publicly available from their providers:
Canadian Wildland Fire Information System and National Fire Database (wildfire
records), SRTM (topography), ESRI/Sentinel-2 (land cover), and TerraClimate and
ERA5-Land (climate).


## Geospatial layers
`data/rasters/` holds the 18 gridded conditioning-factor layers used by the model
(climate: Max_Temperature, Precipitation, WS, Specific_Humidity, AET, DSI,
Soil_Moisture; topography: Elevation, Slope, Aspect, TWI, Plan/Profile curvature;
vegetation: NDVI, LULC; anthropogenic: Distance_roads/rivers/households), each a
GeoTIFF on the 1.5 km analysis grid in EPSG:3005. `data/susceptibility/` holds the
modelled susceptibility maps (continuous and 5-class) for the four models.

The original source datasets (SRTM, ESRI/Sentinel-2 land cover, TerraClimate,
ERA5-Land, CWFIS/NFDB) are public and obtainable from their providers (Table 1 of
the article). The trained model weights (~100 MB) are archived on the Zenodo deposit
(DOI 10.5281/zenodo.20389083), as they exceed GitHub size limits.

## Quick reproducibility check
```bash
conda env create -f environment.yml && conda activate wildfire
python test_reproducibility.py
```
This checks the environment, confirms the shipped data are present, and runs the
deterministic spatial-autocorrelation analysis end-to-end (expected Moran's I ~ 0.70).
Exit code 0 = no blockers.

## Reproducing the analysis
```bash
conda env create -f environment.yml
conda activate wildfire
python code/train_models_spatialcv.py      # LSTM/BiLSTM with RFE + 10-fold spatial CV
python code/bilstm_pso.py                   # PSO-optimized BiLSTM
python code/lstm_pso.py                     # PSO-optimized LSTM
python code/predict_susceptibility.py       # province-wide susceptibility maps
python code/shap_analysis.py                # SHAP feature attribution
python code/spatial_autocorrelation.py      # Moran's I / Geary's C / Getis-Ord
python code/climate_analysis.py             # seasonal climate comparison
python code/teleconnections.py              # ENSO/PDO teleconnections
python code/cross_border_comparison.py      # BC vs US FSim comparison
python code/assumption_checks.py            # statistical assumption tests
```
All stochastic steps use fixed random seeds (see `code/seeds.py`).

> Steps that need the trained weights (`predict_susceptibility.py`, `shap_analysis.py`,
> `roc_curves.py`) require the model archive from Zenodo (DOI 10.5281/zenodo.20389083)
> unzipped into `models/`. `teleconnections.py` uses the indices in `data/raw/`
> (shipped). All other analyses run from the data in this repository.

## Trained models (for Figs 13/14, SHAP-from-model, prediction)

The small preprocessing pipeline (`static_preprocessor.joblib`) and selected-feature
lists (`selected_features_final.csv`) for the four thr70/seed42 models are included
in `models/<MODEL>/thr70/seed42/`. The large trained weights are on Zenodo
(DOI 10.5281/zenodo.20389083). Unzip `models_for_zenodo.zip` so the `final_model.keras`
files land next to them, i.e. into the repo root:

```bash
# after unzip you should have, e.g.:
#   models/BiLSTM_PSO/thr70/seed42/final_model.keras            (from Zenodo)
#   models/BiLSTM_PSO/thr70/seed42/static_preprocessor.joblib   (shipped here)
#   models/BiLSTM_PSO/thr70/seed42/selected_features_final.csv  (shipped here)
python code/roc_curves.py            # Fig. 13 train/test ROC (BiLSTM-PSO test AUC = 0.92)
```

## Reproducibility map (paper result -> how to reproduce)

| Item | How to reproduce | Inputs |
|------|------------------|--------|
| Moran's I / spatial dependence | `python code/spatial_autocorrelation.py` | shipped training points |
| Table 4 & 5, Fig. 12 (climate) | `python code/climate_analysis.py` | `data/climate/` |
| Fig. 16 (susceptibility class areas, 4 models) | `python code/susceptibility_class_area.py` | `data/susceptibility/BC_susceptibility_*.tif` (equal-interval breaks) |
| Fig. 17 (SHAP) | `python code/figure17_shap_beeswarm.py` | `data/shap/SHAP_BiLSTM_PSO_values.pkl` |
| Fig. 18 + Section 3.9 (ENSO/PDO) | `python code/figure18_teleconnection.py` and `python code/teleconnections.py` | `data/analysis_dataset.csv`, `data/raw/` |
| Table 6 (model AUCs) | stored in `models/<MODEL>/thr70/seed42/metrics_summary.json` (Zenodo); ROC regenerated by `code/roc_curves.py` | Zenodo archive |
| Fig. 13 (train/test ROC, test AUC 0.92) | `python code/roc_curves.py` | `models/<MODEL>/thr70/seed42/` (Zenodo .keras + shipped preprocessor/features) |
| Table 7 (seed sensitivity) | per-seed `metrics_summary.json` in the Zenodo archive + `spatial_autocorrelation.py` | Zenodo archive |
| Fig. 14 (susceptibility maps) | shipped rasters, open in QGIS/ArcGIS | `data/susceptibility/*.tif` |
| Fig. 19 / Table 8 (cross-border) | `python code/cross_border_comparison.py --bc-susc ... --us-bp ... --border ...` | needs US FSim burn-probability layer (provider/Zenodo) |
| Fig. 14 prediction (province-wide) | `python code/predict_susceptibility.py --model models/BiLSTM_PSO/thr70/seed42/final_model.keras --preprocessor models/BiLSTM_PSO/thr70/seed42/static_preprocessor.joblib --features models/BiLSTM_PSO/thr70/seed42/selected_features_final.csv --rasters data/rasters --out outputs/BC_susc.tif` | Zenodo .keras + shipped preprocessor/features + `data/rasters/` |
| Figs 6-8 (wildfire trend / district / monthly counts) | EDA from the shipped fire records | `data/processed/fires_geq_70ha.shp` |
| Figs 1-5, 9-11 (study-area / workflow / density maps) | descriptive GIS / diagram figures - not script-generated | - |

Model performance (Table 6 / Fig. 13): AUCs are in each `models/<MODEL>/thr70/seed42/metrics_summary.json` (Zenodo); the train/test ROC is regenerated by `python code/roc_curves.py` (BiLSTM-PSO test AUC = 0.92).

## Licence
Code: MIT. Processed data: CC-BY 4.0.
