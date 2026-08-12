# Reviewer guide

This guide walks through verifying and reproducing the study from the archived
package (Zenodo, concept DOI 10.5281/zenodo.20389083) or the GitHub repository
(https://github.com/Saadkhan32/wildfire-bc-bilstm-pso, release v1.1-revision2).

## 1. Get and verify the package

Download every file from the Zenodo record into one empty folder, then:

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
the last verifies the unpacked content. Run both from the package root — every
line must report OK. The archives contain their folder contents directly, so
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

Line 1 builds the pinned Python environment (exact versions:
`environment.lock.yml`); line 3 restores the R components (terra, sf,
blockCV) and is needed only for the R-based figures; line 4 is the
environment + headline-metric smoke test.

Random seeds are fixed (42); deterministic steps reproduce exactly.

## 3. Map manuscript results to scripts

Scripts live in `src/` (stage-by-stage index: `src/README.md`). Each script
documents its inputs and arguments in its header.

| Manuscript result | Script |
|---|---|
| Climate statistics (Table 5, Fig. 12) | `src/run_climate_full_stats.py` |
| Annual trend analysis (Fig. 5a) | `src/fig_wildfire_trend.py` |
| Spatial autocorrelation checks | `src/c8_assumption_checks.py` |
| Susceptibility prediction (four models) | `src/predict_bc_4models.py` |
| Class-area summary | `src/predict_bc_susceptibility.py` |
| ROC curves (train/test) | `make_roc_figure.py` (package root) |
| SHAP explanation (Fig. 17) | `src/build_shap_beeswarm.py`; `src/specific_humidity_update/Figure17_SHAP_beeswarm_specific_humidity.py` |
| ENSO/PDO teleconnections (Fig. 18) | `src/phase_f_enso_pdo.py`; `src/specific_humidity_update/Figure18_ENSO_PDO_teleconnection_specific_humidity.py` |
| Cross-border comparison (Fig. S6b) | `src/cross_border_c4.py` |

## 4. Data documentation

- `data/README.md` (shipped inside data.zip) — dataset index with ISO 19115-2 record,
  provider and licence per dataset group
- `metadata/iso19115/` — one ISO 19115-2 XML record per dataset group
- `metadata/csvw/` — CSVW (JSON-LD) dictionary per tabular dataset
- `DATA_SOURCES.md` — third-party inputs: provider, access link, licence
- All spatial layers: NAD83 / BC Albers (EPSG:3005), 1.5 km analysis grid

Contact: Aitazaz A. Farooque — afarooque@upei.ca
