# Reviewer reproduction guide

This walks through reproducing the results in Khan et al. (2026) from a clean clone. Steps 1–4
reproduce deterministic results from data shipped in this archive (no GPU, no downloads). Steps 5–7
use the trained model weights and the largest input layers from the Zenodo archive (DOI 10.5281/zenodo.20389083) and/or a GPU.
The data for Steps 1–4 is included in `data/`; see `data/README.md` and `DATA_SOURCES.md`.

## Prerequisites
- git, and Miniconda or Anaconda
- ~2 GB free disk; a GPU only for optional retraining (Step 7)

## Step 1 — Clone and enter
```bash
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
```

## Step 2 — Build the environment
```bash
conda env create -f environment.yml
conda activate wildfire
```

## Step 3 — One-command sanity check
```bash
python test_reproducibility.py
```
Expected: all checks PASS (exit code 0), ending with the deterministic spatial-autocorrelation
result (Moran's I ≈ 0.70).

## Step 4 — Deterministic, model-free results (fast)
```bash
python code/spatial_autocorrelation.py
python code/climate_analysis.py
python code/teleconnections.py
```
Expected: `spatial_autocorrelation.py` prints Moran's I = 0.699, Geary's C = 0.301 and Getis-Ord z = 79.6;
`climate_analysis.py` writes Tables 4–5 and the Fig. 12 climate panels; `teleconnections.py` writes
`data/processed/teleconnections_monthly.csv` and the Fig. 18 inputs. `teleconnections.py` reads
`data/raw/meiv2.txt` (MEI v2) and `data/raw/pdo.csv` (PDO); the NOAA CPC Oceanic Niño Index from Table 2
is also provided as `data/raw/oni.ascii.txt` (see `data/raw/README.md`).

## Step 5 — Model inference (requires the trained weights)

These steps load the trained networks. First download `models.zip` from the Zenodo archive
(DOI 10.5281/zenodo.20389083) and unzip it so each `final_model.keras` sits in
`models/<MODEL>/thr70/seed42/`, beside the preprocessor and feature list already in this repository.

Regenerate the per-model probability tables (writes CSV to `outputs/susceptibility/`, not rasters):
```bash
python code/predict_susceptibility.py
```
The `data/susceptibility/` rasters shipped in this repo are the canonical inputs for ROC, the class shares, and the cross-border step, so the figures reproduce exactly; the matching `BC_susceptibility_*.csv` tables are in the Zenodo `data.zip`. To turn any table into a raster, use ArcGIS Pro: XY Table To Point, then Point to Raster at 1500 m (EPSG:3005).
To predict a single model instead, pass `--model/--preprocessor/--features/--out` (run `python code/predict_susceptibility.py -h`).

ROC / precision-recall evaluation and SHAP interpretation (also use the weights):
```bash
python code/roc_curves.py
python code/shap_analysis.py
```

The five susceptibility-class shares read the shipped maps and run without the weights:
```bash
python code/susceptibility_class_area.py
```
Expected: the four per-model probability tables in `outputs/susceptibility/` (mapping to the Fig. 14 susceptibility surfaces); ROC / precision-recall curves (BiLSTM-PSO test AUC = 0.92); SHAP attribution (Fig. 17); and the five susceptibility-class shares (Fig. 16).

## Step 6 — Cross-border consistency vs U.S. FSim (Section 3.10, Fig. 19, Table 8)
The 49°N sampling strips are in `data/cross_border_US/`; download the U.S. FSim raster `BP_WA_ID_MT.tif`
from the Zenodo archive (DOI 10.5281/zenodo.20389083) into that folder, then the default paths work:
```bash
python code/cross_border_comparison.py --bc-susc data/susceptibility/BC_susceptibility_BiLSTM_PSO.tif
```
This compares the BC susceptibility surface with U.S. FSim annual burn probability in tiles straddling
the 49th parallel; across 20 km tiles (n = 34) the two agree in rank (Spearman ρ ≈ +0.42).

## Step 7 — (Optional) Full retraining — heavy, GPU recommended
Random seeds are fixed in `code/seeds.py`.
```bash
python code/train_models_spatialcv.py
python code/bilstm_pso.py
python code/lstm_pso.py
python code/assumption_checks.py
```

## Notes
- All scripts are non-interactive (`--help` shows options where relevant).
- All BC analysis layers are EPSG:3005 (NAD83 BC Albers) on the 1.5 km grid; U.S. FSim is reprojected
  to EPSG:3005 on the fly.
- Steps 1–4 require no downloads; Steps 5–6 use the trained weights and the U.S. FSim raster from the Zenodo archive.
- Provenance for every file: `metadata/checksums_sha256.txt`.
