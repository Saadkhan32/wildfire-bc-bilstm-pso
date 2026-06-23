# `data/susceptibility/` — modelled susceptibility maps and probability tables

Province-wide wildfire susceptibility for the four models (EPSG:3005, 1.5 km grid).

- Continuous raster (0–1 relative index): `BC_susceptibility_{LSTM,BiLSTM,LSTM_PSO,BiLSTM_PSO}.tif`
- 5-class raster: `BC_susceptibility_class_{LSTM,BiLSTM,LSTM_PSO,BiLSTM_PSO}.tif`
- Probability table per model (in the Zenodo `data.zip`): `BC_susceptibility_{LSTM,BiLSTM,LSTM_PSO,BiLSTM_PSO}.csv` — one row per valid 1.5 km cell, columns `x_epsg3005`, `y_epsg3005` (cell-centre coordinates) and `probability`. Same values as the rasters, in tabular form; to rebuild a raster, import a table in ArcGIS Pro (XY Table To Point, then Point to Raster at 1500 m, EPSG:3005).

`predict_susceptibility.py` regenerates the probability tables to `outputs/susceptibility/` (not rasters). BiLSTM-PSO is the primary model in the article. The 0–1 values are a *relative* susceptibility index (not an absolute probability).
