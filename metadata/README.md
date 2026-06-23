# `metadata/` — machine-readable metadata

Standards-based metadata for every dataset in this archive, so each layer can be discovered,
understood, and provenance-checked independently.

| Item | What it is |
|---|---|
| `iso19115/*.iso19115.xml` | **ISO 19115-2** geographic metadata, one record per geospatial dataset (predictor rasters, susceptibility maps, BC fire perimeters, U.S. FSim burn probability, U.S. FPA-FOD fires, BC boundary, U.S. states) plus the point modelling dataset. Each record gives title, abstract, CRS, geographic/temporal extent, provider, licence, lineage, and distribution links. |
| `csvw/*.csvw.json` | **CSVW (JSON-LD)** data dictionaries for the tabular datasets (modelling tables, monthly climate tables, the annual analysis table, and the ONI index), giving per-column names, datatypes, and descriptions. |
| `data_dictionary.csv` | Human-readable column dictionary for the modelling tables. |
| `dataset_inventory.csv` | One-row-per-dataset index: file/folder, title, type, CRS, provider, licence. |
| `checksums_sha256.txt` | SHA-256 hash of **every file** in the archive (relative paths), for integrity and provenance verification. |

All spatial layers are documented in their native CRS; BC analysis layers use NAD 1983 BC Albers
(EPSG:3005). To verify integrity after download:

```bash
cd wildfire-bc-bilstm-pso
sha256sum -c metadata/checksums_sha256.txt
```
