# wildfire-bc-bilstm-pso

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20389083.svg)](https://doi.org/10.5281/zenodo.20389083)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](environment.yml)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](environment.lock.yml)

Code, data, and metadata for **Khan et al. (2026)**, *Ecological Informatics*
(ECOINF-D-26-01275):

> **Geospatial Deep Learning and SHAP-Based Explainable AI for Wildfire Susceptibility and Exposure Mapping in Western Canada**

Wildfire susceptibility and exposure are mapped for British Columbia
(2000–2024) on a 1.5 km grid (EPSG:3005). LSTM/BiLSTM networks tuned by
particle swarm optimization are trained on topographic, environmental,
vegetation and climate predictors — the PSO-optimized BiLSTM performs best
(ROC-AUC = 0.92) — with SHAP as post-hoc explainable AI, an exposure
assessment against roads and households, and a US cross-border consistency
check. This repository holds the complete pipeline.

## Quick start

```
git clone https://github.com/Saadkhan32/wildfire-bc-bilstm-pso.git
cd wildfire-bc-bilstm-pso
git checkout v1.1-revision2
conda env create -f environment.yml
conda activate wildfire
python test_reproducibility.py
```

Then download `data.zip` (~239 MB) and `models.zip` (~63 MB) from the
[Zenodo record](https://doi.org/10.5281/zenodo.20389083), unpack them into
the repository root, and run `python reproduce.py`. Everything below explains
these steps; the whole path has been validated end to end on an independent
Windows machine.

[Where to find what](#where-to-find-what) ·
[Using the archives](#using-the-zenodo-archives) ·
[Layout](#repository-layout) ·
[Reproduce](#reproducing-the-analysis) ·
[Troubleshooting](#troubleshooting) ·
[Standards](#spatial-reference-and-metadata-standards) ·
[Licensing](#licensing) ·
[Citation](#citation)

## Where to find what

| You want… | Go to |
|---|---|
| Source code (latest) | this repository |
| Permanent citable archive incl. **data + trained models** | Zenodo: **https://doi.org/10.5281/zenodo.20389083** |
| Result-by-result reproducibility map | `REVIEWER_GUIDE.md` |
| Dataset index | `data/README.md` (inside `data.zip`); summary: `metadata/dataset_inventory.csv` |
| Metadata (ISO 19115-2, CSVW, checksums) | `metadata/` |
| Third-party inputs (not redistributed) | provider, link and licence per dataset: `DATA_SOURCES.md` |

GitHub hosts the living code; Zenodo is the permanent archive. The badge DOI
is the *concept DOI* and always resolves to the newest version (currently
v1.2.0, 10.5281/zenodo.21910108); the version cited in the manuscript is
**10.5281/zenodo.21899021** — later versions changed packaging and
documentation only. The archived `code.zip` is byte-identical to git tag
`v1.1-revision2`.

## Using the Zenodo archives

| Archive | Contains | Needed by |
|---|---|---|
| `data.zip` (~239 MB) | `data/` — rasters, climate series, SHAP values, training points | everyone |
| `models.zip` (~63 MB) | `models/` — trained Keras weights + CV metrics | everyone |
| `code.zip`, `metadata.zip` (small) | snapshots of `src/`+`R/`+`notebooks/` and `metadata/` | Zenodo-only users (no git) |

The archives **never collide with the repository**: no git-tracked file path
exists inside them, so they unpack over a clone with zero "replace?" prompts.
A fresh clone already has a small `data/` folder (git-tracked script inputs
not in `data.zip`); `models/` appears only when `models.zip` creates it.

**With a clone (recommended):** download only `data.zip` + `models.zip`.
Each zip holds its folder *contents* directly, so Explorer's "Extract All…"
with the default destination yields clean `data` and `models` folders — move
both into the repository root. Command-line equivalent:

```
tar -xf "%USERPROFILE%\Downloads\data.zip" -C data
mkdir models
tar -xf "%USERPROFILE%\Downloads\models.zip" -C models
```

**Zenodo only (no git):** download all four archives, unpack side by side
(`unzip data.zip -d data`, etc.), copy the five monthly-climate CSVs from
`data/climate/` up into `data/`, and verify with
`metadata/checksums_sha256.txt`. Full walkthrough: `HOW_TO_ASSEMBLE.txt`.

## Repository layout

```
src/            Python pipeline (stage index: src/README.md)
R/              R scripts (terra, sf, blockCV, ggplot2)
notebooks/      Jupyter notebooks, numbered in execution order
data/           processed/ tracked in git; climate/, rasters/,
                susceptibility/, shap/, raw/ filled by data.zip
models/         filled by models.zip (trained weights + CV metrics)
figs/           manuscript figures (manuscript_R2/ = revision-2 finals:
                editable PPTX + vector PDF + 400 dpi PNG)
tables/         exported result tables incl. the Theil-Sen CI register
metadata/       ISO 19115-2 records, CSVW dictionaries, SHA-256 checksums
docs/           see docs/README.md
reproduce.py            one-command reproduction (checks + all figures)
make_roc_figure.py      ROC curves only, from the trained models
test_reproducibility.py environment + headline-metric smoke test
figure_style.py         shared Matplotlib style helper
CHANGELOG.md  CITATION.cff  DATA_SOURCES.md  METHODS.md  REVIEWER_GUIDE.md
environment.yml / environment.lock.yml / renv.lock    (pinned environments)
```

## Reproducing the analysis

**Prerequisites:** Git and Miniconda (https://docs.conda.io); optionally
R ≥ 4.2 for the R-based figures. On Windows use the **Anaconda Prompt** or
**Anaconda PowerShell Prompt** — plain PowerShell/CMD do not know `conda`.

The Quick start commands, explained: cloning (~270 MB) and checking out the
archived tag (`detached HEAD` notice is normal); building the pinned
Python 3.10 environment (TensorFlow/Keras 2.15, exact pins in
`environment.lock.yml`; first run takes several minutes); activating it; and
running the smoke test. For the R figures additionally run
`Rscript -e "renv::restore()"` (`Rscript`, not `R` — PowerShell aliases `R`).

**Smoke test** (`python test_reproducibility.py`): on a code-only clone the
data files are reported as warnings — expected. With the archives unpacked it
must end

```
SUMMARY:  35 pass | 0 warn | 0 fail
```

including the deterministic reproduction of the manuscript's Theil-Sen
climate trends (seeds fixed at 42).

**One-command reproduction** (`python reproduce.py`, a few minutes): rebuilds
every result derivable from the shipped data and trained models — no
retraining — and must end

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

Outputs land in `figs/`. For the ROC curves alone: `python
make_roc_figure.py`. The result-by-result map of every quantitative
manuscript result is in `REVIEWER_GUIDE.md`; the full pipeline index is
`src/README.md`.

**Integrity check** (Linux/macOS/Git Bash, from the repository root, after
unpacking the archives) — every line must print `OK`:

```bash
sha256sum -c metadata/checksums_sha256.txt
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda` is not recognized | plain PowerShell/CMD | use the **Anaconda PowerShell Prompt** |
| `R -e ...` does nothing | PowerShell aliases `R` to `Invoke-History` | use `Rscript -e "renv::restore()"` |
| `python` opens the Microsoft Store | Windows app-alias stub | `conda activate wildfire` first |
| `'src' is not recognized...` | Windows does not run `.py` by path alone | prefix with `python` |
| nested `data\data\` after extracting | Explorer added a wrapper folder | re-extract with the `tar` commands above, or set the destination to the repository root |
| `sha256sum` not found | not a PowerShell command | run in **Git Bash** |
| smoke test warns about data files | clone without archives | expected; add `data.zip` / `models.zip` |
| `'#' is not recognized` / `pathspec '#'` | comment lines pasted into CMD | paste commands only |
| `detached HEAD` notice | normal for tag checkouts | nothing to fix |

## Spatial reference and metadata standards

All spatial layers use **NAD 1983 BC Environment Albers (EPSG:3005)**, the BC
provincial standard; equal-area conformity underpins the manuscript's
area-based statistics. Analysis grid: 1.5 km.

Metadata: ISO 19115-2:2009 records (one per dataset group), W3C CSVW JSON-LD
dictionaries, Citation File Format 1.2, DataCite via Zenodo, SHA-256
checksums. Records follow the ISO 19115-2 schema (ISO-aligned; no formal
certification claimed), organised per FAIR principles.

## Licensing

| Component | Licence |
|---|---|
| Code (this repository) | MIT (`LICENSE`) |
| Author-generated derived data on Zenodo | CC-BY 4.0 |
| Third-party inputs | original providers' licences — `DATA_SOURCES.md` |

## Citation

`CITATION.cff` is machine-readable (GitHub's "Cite this repository" uses it):

> Khan, M.S., Farooque, A.A., Al-Mughrabi, T., Malekian, R., Wang, X., Esau, T.J.,
> Uz Zaman, Q. (2026). Geospatial Deep Learning and SHAP-Based Explainable AI for
> Wildfire Susceptibility and Exposure Mapping in Western Canada.
> *Ecological Informatics*.
>
> Khan, M.S. et al. (2026). wildfire-bc-bilstm-pso. Zenodo.
> https://doi.org/10.5281/zenodo.20389083

## Contact

Aitazaz A. Farooque — afarooque@upei.ca — University of Prince Edward Island
