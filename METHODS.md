# Methods — and how each step maps to the code and data

This document summarises the methodology (Section 2 of the manuscript) in plain language and links
**every step to the exact script, input data, and output** it produces, so the analysis can be
followed end to end. Full reproduction commands are in `REVIEWER_GUIDE.md`.

All spatial layers use **NAD 1983 BC Albers (EPSG:3005)** on a **1.5 km analysis grid**; the study
period is **2000–2024** and the wildfire response is **burned area ≥ 70 ha**.

## Pipeline overview

```
conditioning factors (data/rasters)            wildfire records (data/processed)
            │                                            │
            └──────────────┬─────────────────────────────┘
                           ▼
        fire / non-fire sampling + pseudo-absences   →  data/training_points_*.csv
                           │
        ┌──────────────────┼───────────────────────────────────────┐
        ▼                  ▼                       ▼                 ▼
 spatial dependence   LSTM/BiLSTM (+RFE,      PSO optimisation   SHAP interpretation
 (Moran/Geary/Getis)  10-fold spatial CV)     (BiLSTM/LSTM)      (factor attribution)
        │                  │                       │                 │
        ▼                  ▼                       ▼                 ▼
   Methods stats     province-wide prediction → susceptibility maps  Fig. 17
                           │                  (data/susceptibility)
                           ▼
        climate contrasts · ENSO/PDO teleconnections · BC↔U.S. FSim cross-border check
```

## Step-by-step crosswalk

| # | Method step (manuscript) | Script (`src/`) | Input data | Output |
|---|---|---|---|---|
| 1 | Assemble conditioning factors on the 1.5 km BC Albers grid | (GIS pre-processing; layers provided) | `data/rasters/*.tif` | 18 predictor rasters |
| 2 | Sample fire / non-fire points; generate spatially independent pseudo-absences | `generate_pseudo_absence.py` | `data/processed/fires_geq_70ha.*`, `data/rasters/` | `data/training_points_*.csv` |
| 3 | Quantify spatial dependence of fire occurrence | `spatial_autocorrelation.py` | `data/training_points_70ha_seed42.csv` | Moran's I = 0.699, Geary's C = 0.301, Getis-Ord z = 79.6 |
| 4 | Train LSTM and BiLSTM with recursive feature elimination under 10-fold spatial cross-validation | `train_models_spatialcv.py` | `data/training_points_*.csv` | per-seed models, selected features (`models/`) |
| 5 | Optimise BiLSTM / LSTM hyper-parameters with Particle Swarm Optimization | `bilstm_pso.py`, `lstm_pso.py` | `data/training_points_70ha_seed42.csv` | PSO-tuned models |
| 6 | Predict province-wide susceptibility for the four models | `predict_susceptibility.py` | `models/`, `data/rasters/` | `data/susceptibility/BC_susceptibility_*.tif` |
| 7 | Evaluate discrimination (ROC / PR) | `roc_curves.py` | `models/`, training tables | BiLSTM-PSO test AUC = 0.92 |
| 8 | Classify susceptibility into 5 classes; compute class areas | `susceptibility_class_area.py` | `data/susceptibility/` | class-area summary (Fig. 16) |
| 9 | Interpret the model with SHAP | `shap_analysis.py`, `figure17_shap_beeswarm.py` | model + `data/shap/` | Fig. 17 (factor attribution) |
| 10 | Seasonal climate contrasts (wildfire vs non-wildfire, May–Aug) | `climate_analysis.py` | `data/climate/*.csv` | Tables 4–5, climate panels (Fig. 12) |
| 11 | ENSO / PDO teleconnections with annual area burned | `teleconnections.py`, `figure18_teleconnection.py` | `data/raw/meiv2.txt`, `data/raw/pdo.csv`, `data/analysis_dataset.csv` | `data/processed/teleconnections_monthly.csv`, Fig. 18 (Section 3.9) |
| 12 | Cross-border consistency vs U.S. FSim along the 49th parallel | `cross_border_comparison.py` | `data/cross_border_US/` (FSim + strips), `data/susceptibility/` | Fig. 19, Table 8 (Section 3.10) |
| 13 | Statistical assumption checks (multicollinearity / VIF, etc.) | `assumption_checks.py` | per-seed training outputs | diagnostics |

Helpers: `seeds.py` (fixed random seeds for reproducibility) and `figure_style.py` (shared plotting style).

## Reproducibility controls
- **Environment:** `environment.yml` pins Python 3.10 and all geospatial / ML dependencies.
- **Determinism:** random seeds are fixed in `src/seeds.py`; deterministic steps reproduce exactly,
  and deep-learning steps reproduce within small run-to-run variation across hardware.
- **Provenance:** `metadata/checksums_sha256.txt` lists a SHA-256 hash for every file.
