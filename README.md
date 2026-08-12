# wildfire-bc-bilstm-pso

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20389083.svg)](https://doi.org/10.5281/zenodo.20389083)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](environment.yml)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](environment.lock.yml)

Code, data, and metadata supporting **Khan et al. (2026)**, *Ecological Informatics*
(manuscript **ECOINF-D-26-01275**):

> **Geospatial Deep Learning and SHAP-Based Explainable AI for Wildfire Susceptibility and Exposure Mapping in Western Canada**

## Quick start

```
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
git checkout v1.1-revision2
conda env create -f environment.yml
conda activate wildfire
python test_reproducibility.py
```

Then download `data.zip` and `models.zip` from the
[Zenodo record](https://doi.org/10.5281/zenodo.20389083), unpack them into the
repository root, and run `python reproduce.py` — one command that rebuilds
every result derivable from the shipped data and trained models. Details,
expected outputs and troubleshooting below. This full path has been executed
end to end on an independent Windows machine.

**Contents:**
[What this study does](#what-this-study-does) ·
[Where to find what](#where-to-find-what) ·
[Two ways to use this package](#two-ways-to-use-this-package) ·
[Repository layout](#repository-layout) ·
[Reproducing the analysis](#reproducing-the-analysis) ·
[Troubleshooting](#troubleshooting) ·
[CRS](#coordinate-reference-system) ·
[Metadata standards](#metadata-standards) ·
[Licensing](#licensing) ·
[Citation](#citation)

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
| The permanent citable archive incl. **data + trained models** | Zenodo: **https://doi.org/10.5281/zenodo.20389083** |
| Result-by-result reproducibility map | `REVIEWER_GUIDE.md` |
| Dataset index (what every data file is) | `data/README.md`, shipped inside `data.zip`; summary: `metadata/dataset_inventory.csv` |
| Metadata (ISO 19115-2, CSVW, checksums) | `metadata/` — here and on Zenodo |
| Third-party input data | not redistributed — `DATA_SOURCES.md` lists provider, link, and licence for each |

**How GitHub and Zenodo relate.** GitHub hosts the living code; Zenodo is the
permanent, citable archive of code + data + metadata. The DOI above is the
*concept DOI*: it always resolves to the newest archived version (currently
v1.2.0, https://doi.org/10.5281/zenodo.21910108). Each archived version also
has its own immutable DOI — the version cited in the manuscript is
**https://doi.org/10.5281/zenodo.21899021**; later versions changed packaging
and documentation only. The analysis code is identical throughout: the
archived `code.zip` is byte-identical to git tag `v1.1-revision2` in this
repository.

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
| `data.zip` (~239 MB) | `data/` — rasters, climate series, SHAP values, training points | everyone reproducing results |
| `models.zip` (~63 MB) | `models/` — trained Keras weights + CV metrics | everyone reproducing results |
| `code.zip` (small) | `src/`, `R/`, `notebooks/` — snapshot of the repository code | **Zenodo-only users** (no git) |
| `metadata.zip` (small) | `metadata/` — snapshot of the repository metadata | **Zenodo-only users** (no git) |

**Path A — GitHub clone (recommended).** Clone this repository (steps below),
then download **only `data.zip` and `models.zip`** from Zenodo and unpack both
into the repository root (the folder containing this README). Do **not**
unpack `code.zip` or `metadata.zip` over a clone — they are complete copies of
the code and metadata for Path B users and would only duplicate what git
already gave you.

Each archive contains the folder *contents* directly — there is no wrapper
folder inside the zip. That means Windows' "Extract All…" with its **default
destination** produces exactly one correctly named folder:

1. Right-click `data.zip` → Extract All… → keep the suggested destination.
   You get a single folder named `data`.
2. Move that `data` folder into the repository root (the folder containing
   this README). Windows merges it with the small git-tracked `data` folder;
   no file exists in both, so there are no "replace?" prompts.
3. Repeat for `models.zip` → a `models` folder → move it into the repository
   root.

Command-line alternative, from the repository root:

```
tar -xf "%USERPROFILE%\Downloads\data.zip" -C data
mkdir models
tar -xf "%USERPROFILE%\Downloads\models.zip" -C models
```

**Path B — Zenodo only (no git needed).** Download all four archives and the
loose documentation files from the Zenodo record and unpack the archives side
by side into one folder. `code.zip` provides `src/`, `R/` and `notebooks/`;
`metadata.zip` provides `metadata/`. Everything the archives deliver is
verifiable with `metadata/checksums_sha256.txt`. Full walkthrough:
`HOW_TO_ASSEMBLE.txt` on the record.

*Path B note:* a few scripts read five small monthly-climate CSVs from
`data/` directly; copy them from `data/climate/` up one level into `data/`
after unpacking.

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
reproduce.py            one-command reproduction (checks + all rebuildable figures)
make_roc_figure.py      ROC curves only, from the trained models
test_reproducibility.py environment + headline-metric smoke test
figure_style.py         shared Matplotlib style helper
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
5. `python test_reproducibility.py` — run the smoke test (below).

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
- **Full reproduction:** download `data.zip` (~239 MB) and `models.zip`
  (~63 MB) from the Zenodo record — these two only, see *Two ways to use this
  package* above — and unpack both into the repository root. They add files
  without touching any file git put there, so no overwrite prompts appear.
  Then re-run the smoke test: it now also reproduces the manuscript's
  Theil-Sen climate trend values deterministically (seeds fixed at 42) and
  must end `35 pass | 0 warn | 0 fail`.

### One-command reproduction from the trained models

Once `data.zip` and `models.zip` are unpacked, a single command runs the
checks and rebuilds every result derivable from the shipped data and
already-trained models (no retraining; a few minutes total):

```
python reproduce.py
```

Expected final summary:

```
REPRODUCTION SUMMARY
  [PASS] Environment / package checks
  [PASS] ROC curves from trained models
  [PASS] Annual wildfire trend figure
  [PASS] Seasonal climate composites (Fig. 12)
  [PASS] SHAP beeswarm (Fig. 17)
  [PASS] ENSO/PDO teleconnections (Fig. 18)
All steps passed.
```

Outputs land in `figs/`. The complete result-by-result reproducibility map —
covering every quantitative manuscript result, including those shipped as
computed tables and grids — is in `REVIEWER_GUIDE.md`.

Each step is also available on its own — for only the ROC curves from the
trained models:

```
python make_roc_figure.py
```

To run the full pipeline beyond that, follow the stage-by-stage index in
`src/README.md`.

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

Cite the article and the archive. `CITATION.cff` is machine-readable — GitHub's
"Cite this repository" button uses it.

> Khan, M.S., Farooque, A.A., Al-Mughrabi, T., Malekian, R., Wang, X., Esau, T.J.,
> Uz Zaman, Q. (2026). Geospatial Deep Learning and SHAP-Based Explainable AI for
> Wildfire Susceptibility and Exposure Mapping in Western Canada.
> *Ecological Informatics*.
>
> Khan, M.S. et al. (2026). wildfire-bc-bilstm-pso. Zenodo.
> https://doi.org/10.5281/zenodo.20389083

## Contact
Aitazaz A. Farooque — afarooque@upei.ca — University of Prince Edward Island
