# wildfire-bc-bilstm-pso

Code, data, and metadata supporting **Khan et al. (2026)**, *Ecological Informatics*
(manuscript **ECOINF-D-26-01275**):

> **Geospatial Deep Learning and SHAP-Based Explanation for Wildfire Susceptibility
> and Exposure Assessment in Western Canada**

## Where to find what

| You want… | Go to |
|---|---|
| Source code (version-controlled) | this GitHub repository |
| The exact frozen version behind the paper, incl. **data + trained models** | Zenodo: **https://doi.org/10.5281/zenodo.20389083** |
| Metadata (ISO 19115-2, CSVW, checksums) | `metadata/` — here and on Zenodo |
| Third-party input data | not redistributed — `DATA_SOURCES.md` lists provider, link, and licence for each |

**GitHub hosts the code; Zenodo is the permanent citable archive of code + data + metadata.**
The DOI above is the *concept DOI* and always resolves to the newest archived version;
each version also has its own DOI, shown on the Zenodo page.

## Repository layout

- `code/`, `src/` — Python modules (TensorFlow 2.15, PySAL, SHAP)
- `notebooks/` — Jupyter, numbered in execution order
- `R/` — R scripts (terra, sf, blockCV, ggplot2)
- `data/{raw,interim,processed}` — inputs and derived datasets (large files on Zenodo)
- `models/` — trained Keras weights (archived on Zenodo)
- `figs/`, `tables/` — manuscript figures and exported tables
- `metadata/iso19115/` — **ISO 19115-2 XML records** for every dataset group used or created
- `metadata/csvw/` — **CSVW (JSON-LD) data dictionaries** for every tabular dataset
- `metadata/checksums_sha256.txt` — SHA-256 integrity hashes
- `docs/` — manuscript and supplementary drafts

## Coordinate reference system

All spatial layers use **NAD 1983 BC Environment Albers, EPSG:3005** — the British
Columbia provincial standard. Equal-area conformity is required for the area-based
statistics in the manuscript. Analysis grid: **1.5 km**.

## Metadata standards

ISO 19115-2:2009 records (one per dataset group: fire perimeters, predictor rasters,
susceptibility maps, modelling points, roads, households, boundaries, US FPA-FOD,
US FSim); W3C CSVW JSON-LD dictionaries; Citation File Format 1.2 (`CITATION.cff`);
DataCite via Zenodo; SHA-256 checksums. These records follow the ISO 19115-2 schema
(ISO-aligned); no formal certification is claimed. Organised per FAIR principles.

Verify integrity:
```bash
cd metadata && sha256sum -c checksums_sha256.txt
```

## Reproducing the analysis

```bash
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
git checkout v1.1-revision2
conda env create -f environment.yml     # pinned versions: environment.lock.yml
conda activate wildfire
R -e "renv::restore()"
```
Download `data.zip` and `models.zip` from the Zenodo record and unpack into `data/`
and `models/`. Seeds are fixed (42); `test_reproducibility.py` re-checks headline metrics.

## Licensing

| Component | Licence |
|---|---|
| Code (this repository) | MIT (`LICENSE`) |
| Author-generated derived data on Zenodo | CC-BY 4.0 |
| Third-party inputs | original providers' licences — see `DATA_SOURCES.md` |

## Citation

Cite the article and the archive (machine-readable: `CITATION.cff`):

> Khan, M.S., Farooque, A.A., Al-Mughrabi, T., Malekian, R., Wang, X., Esau, T.J.,
> Uz Zaman, Q. (2026). Geospatial Deep Learning and SHAP-Based Explanation for
> Wildfire Susceptibility and Exposure Assessment in Western Canada.
> *Ecological Informatics*.
>
> Khan, M.S. et al. (2026). wildfire-bc-bilstm-pso (v1.1). Zenodo.
> https://doi.org/10.5281/zenodo.20389083

## Contact
Aitazaz A. Farooque — afarooque@upei.ca — University of Prince Edward Island
