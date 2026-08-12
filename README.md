# wildfire-bc-bilstm-pso

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20389083.svg)](https://doi.org/10.5281/zenodo.20389083)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, data, and metadata supporting **Khan et al. (2026)**, *Ecological Informatics*
(manuscript **ECOINF-D-26-01275**):

> **Geospatial Deep Learning and SHAP-Based Explainable AI for Wildfire Susceptibility and Exposure Mapping in Western Canada**

## What this study does

Wildfire susceptibility and exposure are mapped for British Columbia, Canada
(2000–2024) on a 1.5 km analysis grid (BC Albers, EPSG:3005). LSTM and BiLSTM
deep-learning models are trained on topographic, environmental, vegetation and
climate predictors, with hyperparameters tuned by particle swarm optimization
(PSO); the PSO-optimized BiLSTM performed best (ROC-AUC = 0.92). SHAP is used
as post-hoc explainable AI to attribute predictions to individual variables,
and the susceptibility maps are combined with road and household data for an
exposure assessment, including a cross-border consistency check against US
estimates. This repository contains the complete pipeline: data preparation,
model training, evaluation, SHAP analysis, trend statistics and every
manuscript figure.

## Where to find what

| You want… | Go to |
|---|---|
| Source code (version-controlled, latest) | this GitHub repository |
| The exact frozen version behind the paper, incl. **data + trained models** | Zenodo: **https://doi.org/10.5281/zenodo.20389083** |
| Dataset index (what every data file is) | `data/README.md`, shipped inside `data.zip`; summary: `metadata/dataset_inventory.csv` |
| Metadata (ISO 19115-2, CSVW, checksums) | `metadata/` — here and on Zenodo |
| Third-party input data | not redistributed — `DATA_SOURCES.md` lists provider, link, and licence for each |

**How GitHub and Zenodo relate.** GitHub hosts the living code; Zenodo is the
permanent, citable archive of code + data + metadata. The DOI above is the
*concept DOI*: it always resolves to the newest archived version. Each archived
version additionally has its own immutable DOI — the version used in this study
is **https://doi.org/10.5281/zenodo.21899021**, which matches git tag
`v1.1-revision2` in this repository (the archived `code.zip` is byte-identical
to that tag).

## Two ways to use this package

The Zenodo record contains four archives. **They do not collide with this
repository**: no file path tracked in git also exists inside `data.zip` or
`models.zip`, so the archives unpack over a clone cleanly, with no
"replace file?" prompts.

A fresh clone already contains a small `data/` folder. That is intentional:
it holds the lightweight, git-tracked script inputs (monthly climate CSVs,
LISA layers, fire subsets) that are *not* inside `data.zip`. `models/` does
not exist in a clone at all — `models.zip` creates it.

| Archive | Contains | Needed by |
|---|---|---|
| `data.zip` (~358 MB) | `data/` — rasters, climate series, SHAP values, training points | everyone reproducing results |
| `models.zip` (~71 MB) | `models/` — trained Keras weights + CV metrics | everyone reproducing results |
| `code.zip` (small) | `src/`, `R/`, `notebooks/` — snapshot of the repository code | **Zenodo-only users** (no git) |
| `metadata.zip` (small) | `metadata/` — snapshot of the repository metadata | **Zenodo-only users** (no git) |

**Path A — GitHub clone (recommended).** Clone this repository (steps below),
then download **only `data.zip` and `models.zip`** from Zenodo and unpack both
into the repository root (the folder containing this README). Do **not**
unpack `code.zip` or `metadata.zip` over a clone — they are complete copies of
the code and metadata for Path B users and would only duplicate what git
already gave you.

Each archive contains a single top-level folder (`data/` or `models/`), so
unpack them *directly into the repository root*. On Windows, the safest way
(no wrapper folder, no prompts) is the built-in `tar`, run from the
repository folder:

```
tar -xf "%USERPROFILE%\Downloads\data.zip"
tar -xf "%USERPROFILE%\Downloads\models.zip"
```

If you use Explorer's "Extract All…" instead, set the destination to the
repository folder itself — not the suggested new subfolder — otherwise you
end up with a nested `data\data\` folder.

**Path B — Zenodo only (no git needed).** Download all four archives and the
loose documentation files from the Zenodo record and unpack the archives side
by side into one folder. `code.zip` provides `src/`, `R/` and `notebooks/`;
`metadata.zip` provides `metadata/`. Everything the archives deliver is
verifiable with `metadata/checksums_sha256.txt`.

*Path B note:* a few scripts read five small monthly-climate CSVs from
`data/` directly. Four of them are byte-identical to the copies shipped in
`data/climate/` (copy them up one level); the fifth,
`BC_2000_2024_monthly_climate_wide.csv`, additionally carries the
relative-humidity column used by the Figure 12 script — take that one from
the repository (browse or download the tag archive on GitHub, no git
required).

## Repository layout

```
src/            Python pipeline (see src/README.md for the stage-by-stage index)
R/              R scripts (terra, sf, blockCV, ggplot2)
notebooks/      Jupyter notebooks, numbered in execution order
data/
  processed/    lightweight derived datasets tracked in git (LISA layers,
                fire subsets, annual series); the >=70 ha fire perimeters
                arrive with data.zip
  climate/, rasters/, susceptibility/, shap/, raw/, ...
                filled by data.zip from Zenodo (not tracked in git);
                full dataset index: data/README.md inside data.zip
models/         filled by models.zip from Zenodo (not tracked in git)
figs/           manuscript figures (figs/manuscript_R2/ = revision-2 finals,
                each as editable PPTX + vector PDF + 400 dpi PNG)
tables/         exported result tables incl. the Theil-Sen CI trend register
metadata/
  iso19115/     ISO 19115-2 XML -- one record per dataset group
  csvw/         CSVW (JSON-LD) dictionaries -- one per tabular dataset
  checksums_sha256.txt, dataset_inventory.csv, data_dictionary.csv
docs/           see docs/README.md (drafts are not distributed here)
CHANGELOG.md  CITATION.cff  DATA_SOURCES.md  METHODS.md  REVIEWER_GUIDE.md
environment.yml / environment.lock.yml / renv.lock   (pinned environments)
```

## Reproducing the analysis

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Git | any recent | https://git-scm.com |
| Miniconda (or Anaconda) | any recent | https://docs.conda.io — provides Python 3.10 via `environment.yml` |
| R | ≥ 4.2 | **optional** — only for the R-based figures |

On Windows, open the **Anaconda Prompt** or **Anaconda PowerShell Prompt**
(Start menu → Anaconda). Plain PowerShell and CMD do not know `conda`.

### Step by step

Each command is explained below the block; paste the commands exactly as
written, one at a time.

```
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
git checkout v1.1-revision2
conda env create -f environment.yml
conda activate wildfire
python test_reproducibility.py
```

1. `git clone` + `cd` — download the repository (~270 MB) and enter it.
2. `git checkout v1.1-revision2` — switch to the exact version archived on
   Zenodo and used in the paper. Git prints a "detached HEAD" notice; that is
   normal for tag checkouts and nothing needs fixing.
3. `conda env create` — build the pinned Python 3.10 environment
   (TensorFlow/Keras 2.15; exact versions in `environment.lock.yml`). The
   first run takes several minutes.
4. `conda activate wildfire` — switch into that environment.
5. `python test_reproducibility.py` — run the smoke test (next section).

Optional, only if you want to rebuild the R-based figures (requires R ≥ 4.2):

```
Rscript -e "renv::restore()"
```

Use `Rscript`, not `R` — in PowerShell, `R` is an alias for `Invoke-History`.

The smoke test ends with a summary line:

```
SUMMARY:  <n> pass | <n> warn | 0 fail
```

- **Fresh clone (code only):** all code checks pass; the large data files are
  reported as warnings. That is the expected result and confirms the
  environment is correct.
- **Full reproduction:** download `data.zip` (~358 MB) and `models.zip`
  (~71 MB) from the Zenodo record — these two only, see *Two ways to use this
  package* above — and unpack both into the repository root. They add files
  without touching any file git put there, so no overwrite prompts appear.
  Then re-run step 4. The test then also reproduces the manuscript's Theil-Sen
  climate trend values deterministically (seeds fixed at 42) and must end with
  `0 fail`.

To run the full pipeline afterwards, follow the stage-by-stage index in
`src/README.md`; the reviewer-oriented walkthrough is `REVIEWER_GUIDE.md`.

### Verify file integrity

`sha256sum` is a Linux/macOS/Git-Bash command (not PowerShell). From the
repository root:

```bash
sha256sum -c metadata/checksums_sha256.txt
```

Every line must print `OK`. The checksum file covers `data/`, `models/`,
`src/`, `R/` and `notebooks/`, so run it after unpacking the Zenodo archives.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda : The term 'conda' is not recognized` | plain PowerShell/CMD | use the **Anaconda PowerShell Prompt**, or install Miniconda first |
| `R -e ...` does nothing or errors | in PowerShell, `R` aliases `Invoke-History` | use `Rscript -e "renv::restore()"` |
| `python` opens the Microsoft Store | Windows app-alias stub | activate the conda env first (`conda activate wildfire`) |
| `'src' is not recognized...` when running a script | Windows does not execute `.py` files by path alone | prefix with `python`, e.g. `python src\build_roc_train_test.py` |
| nested `data\data\` or `models\models\` folder after extracting | Explorer's "Extract All…" adds a wrapper folder named after the zip | delete the nested folder and re-extract with the `tar` commands above, or set the extraction destination to the repository root |
| `sha256sum` not found on Windows | not a PowerShell command | run it in **Git Bash** (installed with Git) |
| smoke test warns about missing data files | fresh clone without archives | expected; download `data.zip` / `models.zip` from Zenodo for the full check |
| `'#' is not recognized...` / `error: pathspec '#'` | comment lines pasted into CMD, where `#` is not a comment | paste only the commands, without any `#` lines |
| `detached HEAD` notice after `git checkout` | normal for tag checkouts | nothing to fix |

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

## Licensing

| Component | Licence |
|---|---|
| Code (this repository) | MIT (`LICENSE`) |
| Author-generated derived data on Zenodo | CC-BY 4.0 |
| Third-party inputs | original providers' licences — see `DATA_SOURCES.md` |

## Citation

Cite the article and the archive (machine-readable: `CITATION.cff`):

> Khan, M.S., Farooque, A.A., Al-Mughrabi, T., Malekian, R., Wang, X., Esau, T.J.,
> Uz Zaman, Q. (2026). Geospatial Deep Learning and SHAP-Based Explainable AI for
> Wildfire Susceptibility and Exposure Mapping in Western Canada.
> *Ecological Informatics*.
>
> Khan, M.S. et al. (2026). wildfire-bc-bilstm-pso (v1.1). Zenodo.
> https://doi.org/10.5281/zenodo.20389083

## Contact
Aitazaz A. Farooque — afarooque@upei.ca — University of Prince Edward Island
