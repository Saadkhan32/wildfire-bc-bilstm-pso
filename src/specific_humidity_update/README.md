# Specific-humidity correction — code and updates

The model was trained on **ERA5-Land specific humidity (kg/kg)**, but several
figures, tables, and the text originally labelled this predictor "relative
humidity / RH". This folder contains every script needed to reproduce the
correction, plus a record of the manuscript/table edits.

All paths are repo-relative. Run scripts from the repository root.
Install dependencies first: `pip install -r src/specific_humidity_update/requirements.txt`

Each corrected figure has its own self-contained script (named by figure number):

| Script | Produces |
|--------|----------|
| `Figure12_full_climate_composites_specific_humidity.py` | Fig. 12 full climate figure, all 9 panels a-i (panel d = specific humidity) |
| `Figure17_SHAP_beeswarm_specific_humidity.py` | Fig. 17 SHAP beeswarm |
| `Figure18_ENSO_PDO_teleconnection_specific_humidity.py` | Fig. 18 ENSO/PDO 3-panel (standalone) |
| `Section3-9_teleconnection_stats_specific_humidity.py` | §3.9 ONI/PDO correlation table |
| `Figure_inputs_GEE_ERA5Land_specific_humidity.js` | derives the specific-humidity input (GEE) |

## Inputs

| File | Description |
|------|-------------|
| `data/BC_2000_2024_monthly_climate_wide.csv` | Wide monthly climate (temp, wind, RH, precip) |
| `data/BC_ERA5Land_monthly_specific_humidity_2000_2024.csv` | ERA5-Land specific humidity, q (kg/kg) |
| `data/raw/pdo.csv` | ERSST PDO index (Year + Jan..Dec) |
| `reviewer_response_C2_C15/data/raw/BC_TerraClimate_monthly_PDSI_2000_2024.csv` | TerraClimate PDSI |
| `revision_c8c11/06_Final_Tables/SHAP_BiLSTM_PSO_values.pkl` | Stored SHAP values + feature names |

## 1. Derive specific humidity from ERA5-Land (Google Earth Engine)

`Figure_inputs_GEE_ERA5Land_specific_humidity.js` — paste into
https://code.earthengine.google.com and Run, then export the CSV.

Specific humidity is derived from ERA5-Land 2 m dewpoint and surface pressure:

```
e = 6.112 * exp(17.67 * Td / (Td + 243.5))      # vapour pressure, hPa (Td in degC)
q = 0.622 * e / (P - 0.378 * e)                 # specific humidity, kg/kg (P in hPa)
```

Region: British Columbia via `FAO/GAUL/2015/level1`, ADM1_NAME =
`'British Columbia / Colombie-Britannique'` (the bilingual GAUL name — the
plain `'British Columbia'` filter returns an empty geometry).

## 2. Section 3.9 teleconnection analysis

`python src/specific_humidity_update/Section3-9_teleconnection_stats_specific_humidity.py`

Fire-season (May-Aug) Spearman correlations with ONI and PDO, n = 25 years:

| Variable | ONI rho (p) | PDO rho (p) |
|----------|-------------|-------------|
| Specific humidity (new) | +0.21 (0.32) n.s. | +0.05 (0.80) n.s. |
| Relative humidity (old) | -0.61 (0.00)      | -0.16 (0.45)      |
| Precipitation | -0.47 (0.02) | -0.26 (0.22) |
| Drought (PDSI) | -0.51 (0.01) | -0.28 (0.17) |

Specific humidity is **not** significantly coupled to ENSO/PDO (warm air holds
more moisture — opposite sign to RH), so it is removed from the "lower in warm
phases" group; precipitation and drought carry the warm-phase drying signal.

## 3. Fig. 12 — full climate figure (all nine panels a-i)

`python src/specific_humidity_update/Figure12_full_climate_composites_specific_humidity.py`
-> `figs/climate_v2/Fig12_temperature_abc.{png,jpg,pdf}`  (a,b,c)
-> `figs/climate_v2/Fig12_atmospheric_def.{png,jpg,pdf}`  (d,e,f)
-> `figs/climate_v2/Fig12_hydrology_ghi.{png,jpg,pdf}`    (g,h,i)

Single-file generator for the COMPLETE figure (the three composite images that
make up Fig. 12). The humidity panel (d) shows specific humidity (kg/kg). The
script also prints the full statistical-verification table. 2024 PDSI is an
extreme outlier and is replaced by 2014-2023 climatology (`infill_recent_year`),
as in the original pipeline. All BH-FDR q values are computed across the nine
variables together.

| Panel | Variable | mean_W | mean_NW | Cohen d | sig |
|-------|----------|--------|---------|---------|-----|
| (a) | Max temperature (°C) | 17.63 | 2.36 | 2.65 | *** |
| (b) | Avg temperature (°C) | 11.36 | -2.41 | 2.54 | *** |
| (c) | Min temperature (°C) | 5.10 | -7.17 | 2.39 | *** |
| (d) | Specific humidity (kg/kg) | 0.0061 | 0.0028 | 2.61 | *** |
| (e) | Avg wind speed (m/s) | 2.65 | 2.77 | -0.51 | *** |
| (f) | PDSI | -0.42 | -0.43 | 0.01 | n.s. |
| (g) | AET (mm/month) | 75.52 | 13.23 | 4.36 | *** |
| (h) | Precipitation (mm) | 72.32 | 81.41 | -0.41 | *** |
| (i) | Soil moisture (mm) | 43.17 | 48.18 | -0.43 | ** |

All three composites are rendered with one consistent font (Times New Roman, or
its metric twin Liberation Serif as fallback) and embedded in the manuscript.

## 4. Fig. 18 — ENSO/PDO teleconnection figure

Generator: `reviewer_response_C2_C15/code/make_figures.py` (`fig_combined`),
reading `reviewer_response_C2_C15/data/analysis_dataset.csv`.

Changes made:
- Added an `SH` column to `analysis_dataset.csv` = fire-season (May-Aug) annual
  mean specific humidity (same construction as the existing `RH` column).
- In `make_figures.py`, the humidity feature was switched from `RH`/"Humidity"
  to `SH`/"Sp. humidity" in `DCOL`/`DLAB`, `ROWS_A`, `ROWS_B`, `_detrend`, and
  `_paired`.
- Rebuild (original generator): `cd reviewer_response_C2_C15/code && python make_figures.py combined`
  -> `reviewer_response_C2_C15/figures/Fig_ENSO_combined.{pdf,tif,png}`
- Or run the **self-contained standalone** (no repo-internal imports):
  `python src/specific_humidity_update/Figure18_ENSO_PDO_teleconnection_specific_humidity.py`
  -> `figs/Fig18_ENSO_teleconnection_specifichumidity.{pdf,tif,png}`

Resulting specific-humidity correlations (replacing the old RH values):

| Link | Specific humidity | (old relative humidity) |
|------|-------------------|--------------------------|
| ENSO (ONI summer) -> humidity | +0.18 (n.s.) | -0.64 *** |
| humidity -> area burned | +0.36 (n.s.) | -0.81 *** |
| panel (c) raw / detrended &#124;rho&#124; | 0.36 / 0.19 | 0.61 / 0.55 |

Specific humidity is no longer a significant teleconnection mediator, matching
the revised Section 3.9.

## 5. Fig. 17 — SHAP beeswarm relabel

`python src/specific_humidity_update/Figure17_SHAP_beeswarm_specific_humidity.py`
-> `Fig17_SHAP_specifichumidity.{png,pdf}`

Loads the stored SHAP values and only relabels feature `RH` -> `Specific
Humidity` (no model re-run). Ordering and styling are identical to the original.

> Note: figures here render with Liberation Serif (a Times New Roman
> metric-equivalent) when TNR is not installed. For a pixel-perfect font match,
> run on a machine with Times New Roman installed.

## 6. Manuscript / table edits (OOXML, tracked changes)

These were applied as surgical replacements on `word/document.xml` inside the
`.docx` (author "Muhammad Saadullah Khan", tracked via `<w:ins>`/`<w:del>`).
Delivered in `Manuscript_R1_tracked_v61.docx`.

- **Predictor relabel** throughout: Table 2 predictor name, the `RH -> SH`
  abbreviation entry, Methods, SHAP sections (3.8 / 4.6), and Conclusion now
  read "specific humidity (kg/kg)". Active-text count: "relative humidity" = 0,
  "specific humidity" = 15.
- **Fig. 12** all three composite images replaced (image19 temperature a-c,
  image20 atmospheric d-f, image21 hydrology g-i) so the whole figure is one
  consistent font with specific humidity in panel (d).
- **Table 4** (descriptive) humidity row -> Sp