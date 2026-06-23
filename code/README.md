# `code/` — analysis scripts

Python scripts that reproduce the study. They read from `../data/`, write figures to
`../figs/` and tables to `../tables/`. Run order and expected numbers are in
`../REVIEWER_GUIDE.md`; the method-to-script-to-data crosswalk is in `../METHODS.md`.

## Scripts and what they produce

| Script | Produces |
|---|---|
| `generate_pseudo_absence.py` | spatially independent non-fire (pseudo-absence) points -> training tables |
| `spatial_autocorrelation.py` | Moran's I, Geary's C, Getis-Ord spatial-dependence statistics |
| `train_models_spatialcv.py` | LSTM/BiLSTM with recursive feature elimination, 10-fold spatial CV |
| `bilstm_pso.py`, `lstm_pso.py` | PSO-optimised BiLSTM / LSTM |
| `predict_susceptibility.py` | province-wide susceptibility maps (Fig. 14) |
| `roc_curves.py` | ROC / precision-recall evaluation (Fig. 13) |
| `susceptibility_class_area.py` | per-model susceptibility class shares (Fig. 16) |
| `shap_analysis.py`, `figure17_shap_beeswarm.py` | SHAP factor attribution (Fig. 17) |
| `climate_analysis.py` | seasonal wildfire vs non-wildfire climate contrasts (Fig. 12; Tables 4-5) |
| `fig_wildfire_trend.py` | annual large-fire count + area-burned trend with Mann-Kendall / Theil-Sen (Fig. 6) |
| `teleconnections.py`, `figure18_teleconnection.py` | ENSO/PDO teleconnections (Fig. 18) |
| `cross_border_comparison.py` | BC susceptibility vs U.S. FSim along the 49th parallel (Fig. 19; Table 8) |
| `assumption_checks.py` | multicollinearity / VIF, calibration, residual Moran's I |
| `seeds.py` | fixed random seeds |
| `figure_style.py` | shared plotting style |

## Figures produced in ArcGIS Pro (no Python script)

These manuscript figures are cartographic outputs assembled in **ArcGIS Pro** from the
spatial layers in `../data/`, and are not reproduced by a Python script:

- **Fig. 1** — study area
- **Fig. 3** — climatic, topographic and anthropogenic input-factor maps
- **Fig. 5** — cross-border (49th parallel) sampling-strip design
- **Fig. 7** — wildfire incidents across the top-ten districts
- **Fig. 8** — average monthly wildfire incidences
- **Fig. 9, 10, 11** — annual / monthly / seasonal spatiotemporal wildfire-density maps
- **Fig. 15** — susceptibility-and-exposure maps; the per-district exposure shares
  (BiLSTM-PSO Very-High class, and households, road length and trees/rangeland in the
  Very-High class) were computed in ArcGIS Pro from the shipped susceptibility raster,
  the road and household layers in `../data/anthropogenic_sources/`, and the land-cover layer.
