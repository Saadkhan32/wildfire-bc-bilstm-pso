# data/ — dataset index

Only lightweight, author-generated **processed** data are tracked in git.
Large rasters and inputs live in the Zenodo archive
(https://doi.org/10.5281/zenodo.20389083 → `data.zip`), which unpacks to this
same layout. Every dataset group has an ISO 19115-2 record in
`../metadata/iso19115/` and tabular files have CSVW dictionaries in
`../metadata/csvw/`. All spatial layers: **NAD83 / BC Albers, EPSG:3005**.

| Path (after unpacking data.zip) | Contents | ISO 19115-2 record | Source / licence |
|---|---|---|---|
| `processed/fires_geq_{50,70,100}ha.*` | Wildfire perimeter centroids ≥50/70/100 ha, 2000–2024 | `bc-fire-perimeters` | CWFIS (NRCan), Open Government Licence – Canada |
| `processed/centroids_*.{shp,dbf,shx,prj}` | Modelling point sets per threshold | `modelling_points_dataset` | author-generated, CC-BY 4.0 |
| `processed/BC_ge70ha_annual_2000_2024.csv` | Annual ≥70 ha fire count and area burned (trend analysis) | `bc-fire-perimeters` | derived from CWFIS, CC-BY 4.0 |
| `rasters/` *(Zenodo)* | 19 conditioning-factor rasters, 1.5 km grid | `predictor-rasters-bc` | see `../DATA_SOURCES.md` per layer |
| `susceptibility/` *(Zenodo)* | Per-model susceptibility surfaces + 5-class maps | `susceptibility-maps-bc` | author-generated, CC-BY 4.0 |
| `climate/` *(Zenodo)* | Monthly provincial climate series 2000–2024 | `predictor-rasters-bc` | TerraClimate (CC0), ERA5-Land (Copernicus) |
| `cross_border_US/` *(Zenodo)* | 49th-parallel strips, US FSim + FPA-FOD extracts | `us-fsim-burn-probability`, `us-fpa-fod-fires` | USDA-FS / USGS, public domain |
| `raw/` *(Zenodo)* | Inputs as downloaded + provider README | per-source records | see `../DATA_SOURCES.md` |

Integrity: every file is hashed in `../metadata/checksums_sha256.txt`
(`sha256sum -c` from the repo root after unpacking).

ESRI sidecar files (`.shp.xml`, `.sbn`, `.ovr`, `.aux.xml`) are not tracked:
they are software-generated caches; the authoritative metadata is the ISO
19115-2 record set.
