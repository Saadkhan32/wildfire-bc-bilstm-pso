# Reviewer guide

How to verify and reproduce the study from the archived package (concept DOI
10.5281/zenodo.20389083 → newest version; the manuscript cites
10.5281/zenodo.21899021) or from GitHub
(https://github.com/Saadkhan32/wildfire-bc-bilstm-pso, tag `v1.1-revision2`,
byte-identical to the archived `code.zip`). This procedure has been executed
end to end on an independent Windows machine.

## The 10-minute check

1. **Integrity** — checksums (Section 1). Expected: every line `OK`.
2. **Environment + headline numbers** — `python test_reproducibility.py`
   re-derives the manuscript's Theil-Sen climate trends deterministically.
   Expected: `SUMMARY:  35 pass | 0 warn | 0 fail`.
3. **Results** — `python reproduce.py` rebuilds the four-model ROC curves and
   Figs. 5a, 12, 17, 18 from the shipped data and trained weights.
   Expected: six steps, `All steps passed.` (~2–3 minutes after setup).

## 1. Get and verify the package

Either clone GitHub and add `data.zip` + `models.zip` (Path A in the README —
recommended), or assemble everything from Zenodo alone:

```bash
sha256sum -c MANIFEST_sha256.txt
unzip code.zip
unzip data.zip -d data
unzip models.zip -d models
unzip metadata.zip -d metadata
cp data/climate/BC_*.csv data/
sha256sum -c metadata/checksums_sha256.txt
```

The first checksum line verifies the download against the manifest; the last
verifies the unpacked content (409 files across `data/`, `models/`, `src/`,
`R/`, `notebooks/`). Every line must report OK. Each archive holds its folder
contents directly (hence `-d`); the `cp` line places five monthly-climate
CSVs where a few scripts expect them.

## 2. Set up the environment

```bash
conda env create -f environment.yml
conda activate wildfire
Rscript -e "renv::restore()"
python test_reproducibility.py
```

Line 1 builds the pinned Python 3.10 environment (TensorFlow/Keras 2.15;
exact pins: `environment.lock.yml`; a few minutes on first run); line 3 is
needed only for the R-based figures; line 4 must end
`SUMMARY:  35 pass | 0 warn | 0 fail`, including the deterministic
reproduction of the May–August Theil-Sen trends (avg. temperature
+0.0702 °C/yr; specific humidity +2.88e-05 kg/kg/yr). Seeds are fixed (42).

## 3. One-command reproduction

`python reproduce.py` rebuilds everything derivable from the shipped data and
trained models — the four-model ROC figure (each AUC cross-checked against
the stored CV metrics), the annual trend figure, the Fig. 12 climate
composites, the Fig. 17 SHAP beeswarm and the Fig. 18 teleconnection figure —
and prints a per-step PASS/FAIL summary ending `All steps passed.` Outputs:
`figs/`.

## 4. Every manuscript result, mapped

**Tier 1** results are rebuilt on your machine by `reproduce.py`. **Tier 2**
results are shipped in the package (`tables/`, `data/susceptibility/`,
`models/`) together with the script that produced them; those scripts expect
the staged training workspace (`revision_c8c11/`) created by
`src/*_setup_folders.py`, so rerunning them is optional for verification.
Script index: `src/README.md`.

| Manuscript result | How to verify / reproduce |
|---|---|
| Climate statistics (Table 5) | shipped `tables/T_climate_stats_publication.csv`; headline Theil–Sen slopes re-derived by the smoke test |
| Seasonal climate composites (Fig. 12) | `reproduce.py` step 4 — from the five monthly CSVs in `data/`, incl. the Welch/effect-size verification table on the console; single-page manuscript layout: `python make_fig12_onepage.py` |
| Annual trend analysis (Fig. 5a) | `reproduce.py` step 3 (`src/fig_wildfire_trend.py` + `figure_style.py`); CI register: `tables/TrendRegister_TheilSen_CI.csv` |
| ROC curves, train/test | `reproduce.py` step 2 or `python make_roc_figure.py` — recomputed from trained weights; AUCs cross-checked against `models/*/thr70/seed42/metrics_summary.json` |
| CV metrics and bootstrap CIs | shipped `models/*/cv_metrics_10fold.csv`, `models/*/cv_oof_predictions.csv`, `tables/T_metrics_bootstrap.csv`; recompute: `src/phase_b_bootstrap_ci.py` |
| Spatial autocorrelation (Moran's I, Geary's C) | shipped `tables/T_autocorrelation_global.csv`, `T_gearys_c.csv`, `T_morans_per_threshold.csv`; recompute: `src/c8_assumption_checks.py` |
| Susceptibility maps, four models | grids shipped in `data/susceptibility/`; recompute: `src/predict_bc_4models.py` (staged workspace, long runtime); manuscript maps are GIS renderings of these grids (EPSG:3005) |
| Class-area summary | `src/predict_bc_susceptibility.py` over the shipped grids |
| SHAP explanation (Fig. 17) | `reproduce.py` step 5 from `data/shap/SHAP_BiLSTM_PSO_values.pkl`; from scratch: `src/build_shap_beeswarm.py` |
| ENSO/PDO teleconnections (Fig. 18, Sect. 3.9) | `reproduce.py` step 6; correlations shipped: `tables/T_teleconnection_burnedarea_corr.csv`; stats: `src/specific_humidity_update/Section3-9_teleconnection_stats_specific_humidity.py` |
| Cross-border comparison (Fig. S6b) | inputs shipped (`data/cross_border_US/`, `data/susceptibility/`); script: `src/cross_border_c4.py` (staged workspace) |
| Model training from scratch | staged workflow `src/revision_step*.py` / `src/c8c11_step*.py`; optional — trained models are shipped |
| Study-area / data-prep maps | GIS cartography (ArcGIS Pro, EPSG:3005) from layers listed in `DATA_SOURCES.md` |

*ROC sanity line:* each model's in-sample AUC is compared against its stored
out-of-fold CV AUC. In-sample is expectedly optimistic — most for the
best-fitted BiLSTM-PSO, whose informational `[MISMATCH]` flag marks a gap
above 0.05. The manuscript reports the cross-validated/test performance.

## 5. Data documentation

- `data/README.md` (inside data.zip) — dataset index with ISO 19115-2 record, provider and licence per group
- `metadata/iso19115/` — ISO 19115-2 XML per dataset group; `metadata/csvw/` — CSVW dictionary per tabular dataset
- `DATA_SOURCES.md` — third-party inputs: provider, access link, licence
- All spatial layers: NAD83 / BC Albers (EPSG:3005), 1.5 km analysis grid

Contact: Aitazaz A. Farooque — afarooque@upei.ca
