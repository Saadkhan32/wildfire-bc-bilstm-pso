# wildfire-bc-bilstm-pso

Code, data and metadata supporting Khan et al. (2026), "An Interpretable Deep
Learning Framework for Wildfire Susceptibility and Exposure Assessment in
Western Canada", Ecological Informatics (manuscript ECOINF-D-26-01275).

## Citation
Cite both the article and the archived release on Zenodo (DOI: see
`metadata/10.5281/zenodo.20389084.txt`). A machine-readable citation is in `CITATION.cff`.

## Licence
Code: MIT (see `LICENSE`). Derived raster products on Zenodo: CC-BY 4.0.
Raw input datasets retain their original licences (see `data/raw/README.md`).

## Repository layout
- `data/{raw,interim,processed}` — inputs and derived datasets
- `src/` — Python modules
- `notebooks/` — Jupyter (numbered in execution order)
- `R/` — R scripts (terra, sf, blockCV, ggplot2)
- `models/` — saved Keras weights (gitignored; on Zenodo)
- `figs/` — manuscript figures (TIFF + PDF; .drawio for Fig 2)
- `metadata/iso19115/` — ISO 19115-2 XML for each derived raster
- `metadata/datadict/` — CSVW JSON-LD data dictionaries
- `docs/` — manuscript and supplementary drafts

## Reproducing the analysis
```bash
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
git checkout v1.0-revision1
conda env create -f environment.yml
conda activate wildfire
R -e "renv::restore()"
```

## Coordinate reference system
All spatial layers are in NAD 1983 BC Environment Albers Equal-Area Conic,
EPSG:3005. This is the BC provincial standard; equal-area conformity is
required for the area-based statistics in the manuscript.

## Authors
Muhammad S. Khan, Aitazaz A. Farooque (afarooque@upei.ca), Tatiana Al-Mughrabi,
Raheleh Malekian, Xander Wang, Travis J. Esau, Qamar Uz Zaman.

