# src/ — script index

Scripts keep their original names (they are cited in the reviewer
correspondence); this index groups them by pipeline stage. Python 3.11,
TensorFlow 2.15; fixed seed 42 throughout. R components are in `../R/`.

| Stage | Scripts |
|---|---|
| 1 · Data preparation | `generate_pseudo_absence.py`, `c8_step3_generate_pseudo_absence.py`, `c8c11_step2_generate_datasets.py`, `gee/GEE_ERA5Land_specific_humidity.js` |
| 2 · Exploratory & climate statistics | `run_climate_full_stats.py`, `climate_analysis.py`, `day2_automation.py` |
| 3 · Trend analysis (Theil–Sen + CI) | `fig_wildfire_trend.py`, `assumption_checks.py` |
| 4 · Model training & PSO | `lstm_pso.py`, `bilstm_pso.py`, `random_seed_sensitivity_train.py`, `c8_step4_train_sensitivity.py`, `c8c11_non_pso_cli.py` |
| 5 · Prediction & mapping | `predict_susceptibility.py`, `predict_bc_4models.py`, `phase_d_to_g.py` |
| 6 · Evaluation | `roc_curves.py`, `seeds.py` |
| 7 · Explanation (SHAP) | `shap_analysis.py`, `figure17_shap_beeswarm.py` |
| 8 · Teleconnections & cross-border | `figure18_teleconnection.py`, `cross_border_comparison.py`, `cross_border_c4.py`, `Comment3_4_xborder.py`, `comment4_xborder_tier1_tier3.py` |
| Shared style / tools | `figure_style.py`, `tools/check_dois.py`, `tools/check_dois_R2.py` |

Reproduction order and the mapping to manuscript figures/tables:
see `../REVIEWER_GUIDE.md` and `../METHODS.md`.
