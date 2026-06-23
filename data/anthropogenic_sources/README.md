> **GitHub mirror note:** the large file(s) described below are archived on Zenodo (DOI 10.5281/zenodo.20389083), not stored in this lightweight repository. See the top-level README.

# `data/anthropogenic_sources/` — source vectors for the anthropogenic predictors

The original vector layers from which the anthropogenic distance rasters were derived
(`data/rasters/Distance_households.tif`, `data/rasters/Distance_roads.tif`). **Geometry-only**
versions are provided here: the point/line locations actually used in the analysis, with a single
`ID` field. The large source attribute tables (demographic and road-network fields not used in this
study) were removed to keep the archive compact; the fully attributed originals are available from the
providers listed below and in `../../DATA_SOURCES.md`.

| Sub-folder / file | What it is | Features | CRS | Provider |
|---|---|---|---|---|
| `households_PHH/PHH-BC.*` | Pseudo-household point locations for British Columbia (PHH 2021), used to compute distance-to-households | 1,842,788 points | EPSG:4326 (WGS84) | Statistics Canada — Pseudo-Household Demographic Distribution (PHH), 2021 |
| `road_network/bc_roads.*` | British Columbia road-network lines, used to compute distance-to-roads | 278,798 lines | NAD83 Lambert Conformal Conic (Canada, custom var 5) | Statistics Canada — Road Network File (BC) |

Geometry is byte-identical to the source `.shp`/`.shx`; only the `.dbf` was replaced with a minimal
one-field table. Distances in the model were computed as Euclidean distance to the nearest feature on a
30 m grid, then resampled to the 1.5 km analysis grid (EPSG:3005).
