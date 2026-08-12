# Reviewer guide

This guide walks through verifying and reproducing the study from the archived
package (Zenodo concept DOI 10.5281/zenodo.20389083, which always resolves to
the newest version; the version cited in the manuscript is
10.5281/zenodo.21899021) or from the GitHub repository
(https://github.com/Saadkhan32/wildfire-bc-bilstm-pso, tag `v1.1-revision2` —
byte-identical to the archived `code.zip`). The full procedure below has been
executed end to end on an independent Windows machine.

## The 10-minute check

If you verify only three things, verify these (details in the numbered
sections below):

1. **Integrity** — every downloaded and unpacked file matches its published
   SHA-256 checksum (Section 1). Expected: every line `OK`.
2. **Environment + headline numbers** — `python test_reproducibility.py`
   rebuilds the manuscript's Theil-Sen climate trends deterministically.
   Expected: `SUMMARY:  35 pass | 0 warn | 0 fail`.
3. **Results** — `python reproduce.py` rebuilds, from the shipped data and
   trained model weights, the four-model ROC curves and the manuscript's
   Figs. 5a, 12, 17 and 18. Expected: six steps, `All steps passed.`
   (~2–3 minutes after setup).

## 1. Get and verify the package

Either clone GitHub and add the two data archives (Path A in the repository
README — recommended), or assemble everything from Zenodo alone:

```bash
sha256sum -c MANIFEST_sha256.txt
unzip code.zip
unzip data.zip -d data
unzip models.zip -d models
unzip metadata.zip -d metadata
cp data/climate/BC_*.csv data/
sha256sum -c metadata/checksums_sha256.txt
```

The first checksum line verifies the downloaded files against the manifest;
the last verifies the unpacked content (409 files across `data/`, `models/`,
`src/`, `R/`, `notebooks/`). Run both from the package root — every line must
report OK. The archives contain their folder contents directly, so
`data.zip`, `models.zip` and `metadata.zip` are extracted into folders of the
same name (`-d`); the `cp` line places the five monthly-climate CSVs where a
few scripts expect them.

## 2. Set up the environment

```bash
conda env create -f environment.yml
conda activate wildfire
Rscript -e "renv::restore()"
python test_reproducibility.py
```

Line 1 builds the pinned Python 3.10 environment (TensorFlow/Keras 2.15;
exact versions: `environment.lock.yml`) and takes a few minutes on first run;
line 3 restores the R components (terra, sf, blockCV) and is needed only for
the R-based figures; line 4 is the environment + headline-metric smoke test —
with the archives unpacked it must end:

```
SUMMARY:  35 pass | 0 warn | 0 fail
```

including the deterministic reproduction of the manuscript's May–August
Theil-Sen trends (avg. temperature +0.0702 °C/yr; specific humidity
+2.88e-05 kg/kg/yr). Random seeds are fixed (42); deterministic steps
reproduce exactly.

## 3. One-command reproduction

```bash
python reproduce.py
```

runs the environment checks and rebuilds everything derivable from the
shipped data and trained models — the four-model ROC figure (with each AUC
cross-checked against the stored cross-validation metrics), the annual trend
figure, the Fig. 12 climate composites, the Fig. 17 SHAP beeswarm and the
Fig. 18 teleconnection figure — and prints a per-step PASS/FAIL summary
ending `All steps passed.` Outputs are written to `figs/`. The table below
covers every quantitative manuscript result.

## 4. Map manuscript results to scripts

Every quantitative result in the manuscript is covered by one of two tiers.
**Tier 1** — rebuilt on your machine by `python reproduce.py` from the shipped
data and trained models. **Tier 2** — the computed result itself is shipped in
the package (`tables/`, `data/susceptibility/`, `models/`) together with the
script that produced it; these scripts document the full procedure and expect
the staged training workspace (`revision_c8c11/`) that
`src/*_setup_folders.py` creates, so rerunning them is optional, not
required, for verification. Scripts live in `src/` (index: `src/README.md`).

| Manuscript result | How to verify / reproduce |
|---|---|
| Climate statistics (Table 5) | shipped `tables/T_climate_stats_publication.csv`; the headline Theil–Sen slopes are re-derived deterministically by the smoke test (`test_reproducibility.py`) |
| Seasonal climate composites (Fig. 12) | `reproduce.py` step 4 — rebuilt from the five monthly CSVs in `data/` into `figs/climate_v2/`, incl. the Welch/effect-size verification table printed to the console |
| Annual trend analysis (Fig. 5a) | `reproduce.py` step 3 (`src/fig_wildfire_trend.py` + `figure_style.py`); CI register shipped: `tables/TrendRegister_TheilSen_CI.csv` |
| ROC curves, train/test (model performance) | `reproduce.py` step 2 or `python make_roc_figure.py` — recomputed from the trained weights; AUCs cross-checked against `models/*/thr70/seed42/metrics_summary.json` |
| Cross-validation metrics and bootstrap CIs | shipped `models/*/cv_metrics_10fold.csv`, `models/*/cv_oof_predictions.csv`, `tables/T_metrics_bootstrap.csv`; recompute: `src/phase_b_bootstrap_ci.py` |
| Spatial autocorrelation (Moran's I, Geary's C) | shipped `tables/T_autocorrelation_global.csv`, `tables/T_gearys_c.csv`, `tables/T_morans_per_threshold.csv`; recompute: `src/c8_assumption_checks.py` |
| Susceptibility maps, four models | per-model grids shipped in `data/susceptibility/`; recompute: `src/predict_bc_4models.py` from `models/` + `data/rasters/` (staged workspace, long runtime); the manuscript map figures are GIS renderings of these grids (EPSG:3005) |
| Class-area summary | `src/predict_bc_susceptibility.py` over the shipped susceptibility grids |
| SHAP explanation (Fig. 17) | `reproduce.py` step 5 from shipped `data/shap/SHAP_BiLSTM_PSO_values.pkl`; from scratch: `src/build_shap_beeswarm.py` (loads the trained models) |
| ENSO/PDO teleconnections (Fig. 18, Section 3.9) | `reproduce.py` step 6; correlation table shipped: `tables/T_teleconnection_burnedarea_corr.csv`; stats script: `src/specific_humidity_update/Section3-9_teleconnection_stats_specific_humidity.py` |
| Cross-border comparison (Fig. S6b) | inputs shipped (`data/cross_border_US/`, `data/susceptibility/`); script: `src/cross_border_c4.py` (staged workspace) |
| Model training from scratch | staged workflow `src/revision_step*.py` / `src/c8c11_step*.py`; optional — the trained models are shipped |
| Study-area and data-preparation maps | GIS cartography (ArcGIS Pro, EPSG:3005) from the shipped and source layers listed in `DATA_SOURCES.md` |

*Note on the ROC step's sanity line:* for each model it compares the
in-sample AUC on the full dataset against the stored cross-validation
out-of-fold AUC. In-sample values are expectedly optimistic; the gap is
largest for the best-fitted model (BiLSTM-PSO) and the informational
`[MISMATCH]` flag there simply marks a gap above 0.05. The manuscript reports
the cross-validated/test performance.

## 5. Data documentation

- `data/README.md` (shipped inside data.zip) — dataset index with ISO 19115-2 record,
  provider and licence per dataset group
- `metadata/iso19115/` — one ISO 19115-2 XML record per dataset group
- `metadata/csvw/` — CSVW (JSON-LD) dictionary per tabular dataset
- `DATA_SOURCES.md` — third-party inputs: provider, access link, licence
- All spatial layers: NAD83 / BC Albers (EPSG:3005), 1.5 km analysis grid

Contact: Aitazaz A. Farooque — afarooque@upei.ca
