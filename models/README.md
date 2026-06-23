# `models/`

Per-model preprocessing pipeline (`static_preprocessor.joblib`), selected-feature lists
(`selected_features_final.csv`) and cross-validation outputs (`best_params.json`,
`cv_metrics_10fold.csv`, `cv_oof_predictions.csv`, `metrics_summary.json`) for the four models
(LSTM, BiLSTM, LSTM-PSO, BiLSTM-PSO), including the threshold/seed sensitivity runs.

The trained network weights (`final_model.keras`) are not stored in this GitHub mirror because of
size. Download them from the Zenodo archive (DOI 10.5281/zenodo.20389083) and place each one in the
matching `models/<MODEL>/thr70/seed42/` folder, then follow Steps 5-6 of `../REVIEWER_GUIDE.md`.
