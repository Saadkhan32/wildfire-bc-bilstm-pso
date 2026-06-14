# Reviewer reproduction guide

This walks through reproducing the results in Khan et al. (2026) from a clean
clone. Steps 1-4 reproduce the deterministic results from data shipped in the
repo (no GPU, no downloads). Steps 5-7 need the trained model weights from Zenodo
and/or a GPU.

## Prerequisites
- git, and Miniconda or Anaconda
- ~2 GB free disk; a GPU only for optional retraining (Step 7)

## Step 1 - Clone the reviewed release
```bash
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
git checkout v1.0-revision1
```

## Step 2 - Build the environment
```bash
conda env create -f environment.yml
conda activate wildfire
```

## Step 3 - One-command sanity check
```bash
python test_reproducibility.py
```
Expected: all checks PASS (exit code 0), ending with
`spatial_autocorrelation.py runs and prints Moran's I -> 0.699`.

## Step 4 - Reproduce the model-free results (fast, deterministic)

### 4a. Spatial dependence (Methods)
```bash
python code/spatial_autocorrelation.py
```
Expected (mean +/- SD across 10 seeds): Moran's I = 0.699 +/- 0.003,
Geary's C = 0.301 +/- 0.003, Getis-Ord z = 79.6 +/- 0.35; Moran (KNN8) = 0.698.

### 4b. Seasonal climate contrasts - Table 4, Table 5, climate panels of Fig. 12
```bash
python code/climate_analysis.py
```
Writes `tables/T_climate_stats_publication.csv` and panels in `figs/climate_v2/`.
Expected Table 5 (wildfire May-Aug vs non-wildfire; Welch t, BH-FDR, Cohen's d):

| Variable | mean (W) | mean (NW) | Cohen's d | sig |
|----------|----------|-----------|-----------|-----|
| AET (mm/mo) | 75.52 | 13.23 | 4.36 | *** |
| Max temperature (C) | 17.63 | 2.36 | 2.65 | *** |
| Specific humidity (kg/kg) | 0.0061 | 0.0028 | 2.61 | *** |
| Mean temperature (C) | 11.36 | -2.41 | 2.54 | *** |
| Min temperature (C) | 5.10 | -7.17 | 2.39 | *** |
| Wind speed (m/s) | 2.65 | 2.77 | -0.51 | *** |
| Precipitation (mm) | 72.32 | 81.41 | -0.41 | *** |
| Soil moisture (mm) | 43.17 | 48.18 | -0.43 | ** |
| PDSI | -0.42 | -0.43 | 0.01 | n.s. |

(The 2024 PDSI outlier is replaced by 2014-2023 climatology, as in the pipeline.)

### 4c. ENSO/PDO teleconnections - Section 3.9, Fig. 18
```bash
python code/teleconnections.py
```
Writes `data/processed/teleconnections_monthly.csv` and `figs/`, and prints
fire-season ENSO (MEI/ONI) and PDO correlations with annual area burned (n = 25).

## Step 5 - Get the trained models (for Steps 5-6)
The preprocessing pipeline and selected-feature lists for the four thr70/seed42
models are already in `models/<MODEL>/thr70/seed42/`. Download
`models_for_zenodo.zip` (DOI 10.5281/zenodo.20389083) and unzip it into the repo
root so each `final_model.keras` lands beside them. Then:
```
python code/roc_curves.py        # Fig. 13 train/test ROC (BiLSTM-PSO test AUC = 0.92)
```

### 5a. Province-wide susceptibility maps
```bash
python code/predict_susceptibility.py --models models/ --rasters data/rasters
```
Reproduces `data/susceptibility/BC_susceptibility_*.tif` (also shipped, so you can
diff your output against the archived rasters).

### 5b. SHAP interpretation - Fig. 17
```bash
python code/shap_analysis.py --model models/BiLSTM_PSO --data data/training_points_70ha_seed42.csv
```
Expected top factors: drought index (DSI) and forested land use, then temperature
and topography; specific humidity ranks mid-list.

### 5c. ROC / PR curves
```bash
python code/roc_curves.py --models models/ --data data/training_points_70ha_seed42.csv
```

### 5d. Statistical assumption checks (VIF etc.)
```bash
python code/assumption_checks.py
```
Reads the per-seed training outputs (`holdout_predictions.csv`,
`selected_features_final.csv`, `feature_meta.json`) from Step 7 or the Zenodo
archive; default `--model-dir` is `outputs/lstm_bilstm_spatialcv/seed_42/BILSTM`.

## Step 6 - Cross-border benchmark - Fig. 19 (Spearman rho = +0.42)
Needs the US FSim burn-probability layer and the 49 deg N border strips (in the
Zenodo archive). Then:
```bash
python code/cross_border_comparison.py \
  --bc-susc data/susceptibility/BC_susceptibility_BiLSTM_PSO.tif \
  --us-bp <us_burn_probability.tif> --border <border_49N.shp>
```

## Step 7 - (Optional) Full retraining - heavy, GPU recommended
Random seeds are fixed in `code/seeds.py`.
```bash
python code/train_models_spatialcv.py     # LSTM/BiLSTM + RFE + 10-fold spatial CV
python code/bilstm_pso.py                  # PSO-optimized BiLSTM
python code/lstm_pso.py                    # PSO-optimized LSTM
```
Each script defaults to the shipped `data/training_points_70ha_seed42.csv` and
writes to `outputs/`. AUC/PR values reproduce the paper within small run-to-run
deep-learning variation across hardware.

## Notes
- All scripts are non-interactive (`--help` shows options).
- All spatial layers are EPSG:3005 (NAD83 BC Albers).
- Steps 1-4 require no downloads beyond the repo; Steps 5-6 require the Zenodo models.

## Figure and result generators
- `code/figure17_shap_beeswarm.py` -> Fig. 17 (SHAP), from `data/shap/`
- `code/figure18_teleconnection.py` -> Fig. 18 (ENSO/PDO 3-panel), from `data/analysis_dataset.csv`
- `code/susceptibility_class_area.py` -> Fig. 16 class areas, from `data/susceptibility/`
See the Reproducibility map in `README.md` for the full paper-result -> script mapping.
