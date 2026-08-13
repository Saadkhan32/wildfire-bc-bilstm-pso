# Changelog

## v1.2.1 - 2026-08-12
- New ArcGIS Pro cartography pipeline for the conditioning-factor figure:
  arcgis_make_maps.py exports the eighteen 1.5 km predictor rasters from
  ArcGIS Pro with native symbology at 600 dpi, and arcgis_compose_figure.py
  assembles the single-page journal figure (labelled graticule frames,
  per-panel colour bars sampled from the Pro rendering, LULC class legend,
  north arrow, exact 500 km scale bar) -> figs/Fig_predictors_final.png/.tif.
- New make_fig_predictors.py (package root): journal-standard single-page
  grid of the eighteen 1.5 km conditioning-factor rasters (panel letters,
  units, rounded 2-98% colour scales, LULC class legend, north arrow, exact
  500 km scale bar), rendered from the shipped data/rasters/; outputs
  figs/Fig_predictors_grid.{png,pdf,tif} at 600 dpi.
- New make_fig12_onepage.py (package root): single-page portrait composite of
  the nine seasonal climate variables (manuscript Fig. 7) - Theil-Sen trends
  with 95% CI bands plus wildfire/non-wildfire distributions - computed
  entirely from the five git-tracked monthly CSVs; outputs
  figs/Fig12_climate_onepage.{png,pdf,tif} at 600 dpi.
- Documentation rewrite (GitHub): README condensed around a quick start,
  contents bar, current version DOIs and the expected outputs of the smoke
  test and of reproduce.py; the reviewer guide gains a 10-minute
  verification path, expected summary lines, and a note explaining the ROC
  step's informational sanity flag. No code or data changes; the published
  Zenodo v1.2.0 record is unaffected.

## v1.2.0 - 2026-08-12
- Zenodo archives restructured (new Zenodo version): data.zip, models.zip and
  metadata.zip now contain the folder contents directly, without a wrapper
  folder, so Windows "Extract All" with its default destination yields exactly
  one correctly named folder (no more data/data or models/models nesting).
  code.zip is unchanged and remains byte-identical to git tag v1.1-revision2
  (tag name retained as cited in the manuscript).
- data.zip: data/climate/BC_2000_2024_monthly_climate_wide.csv updated to the
  current version carrying the avg_relative_humidity column required by the
  Figure 12 script; the archive is now self-sufficient for Zenodo-only users.
- metadata/checksums_sha256.txt updated accordingly.
- README: extraction instructions rewritten for the flat archives.
- New make_roc_figure.py (package root): rebuilds the four-model train/test
  ROC figure from the archived training table and trained weights using
  package paths; --check mode verifies inputs. Replaces the working-copy-only
  src/build_roc_train_test.py in the reviewer guide.
- New reproduce.py (package root): one command that runs the smoke test and
  rebuilds every result derivable from the shipped data and trained models -
  ROC curves, annual trend figure, Fig. 12 climate composites, Fig. 17 SHAP
  beeswarm and Fig. 18 teleconnections - with a PASS/FAIL summary.
- REVIEWER_GUIDE: complete result-by-result reproducibility matrix covering
  every quantitative manuscript result (rebuilt by reproduce.py, or shipped
  in tables/, data/susceptibility/ and models/ with its producing script).
- figure_style.py added at the package root: shared Matplotlib style helper
  required by src/fig_wildfire_trend.py that was previously missing from the
  distribution.

## v1.1.4 - 2026-08-12
- No-overlap packaging: the six files that existed both in git and in the
  Zenodo data.zip (data/README.md and the >=70 ha fire-perimeter shapefile)
  are now delivered by data.zip only, so the archives unpack over a clone
  without overwrite prompts; .gitignore extended to keep git status clean
  after unpacking.
- README: "Two ways to use this package" - archive-by-archive table and
  separate GitHub (data.zip + models.zip only) and Zenodo-only paths;
  Windows extraction guidance (tar; avoiding Explorer's nested-folder
  wrapper) and a note on the five git-tracked monthly-climate CSVs.
- Removed empty-folder placeholders (models/.gitkeep,
  data/processed/.gitkeep): a clone now contains no models folder at all
  until models.zip is unpacked.
- Tag v1.1-revision2 moved to this state so the tagged tree shows the
  authors' final figures; the archived code.zip remains byte-identical to
  the tag (src/, R/, notebooks/ unchanged).

## v1.1.3 - 2026-08-12
- Figures: replaced the reconstructed Figure 2 and Figure 4 sources with the
  authors' final versions (Fig2_in_manuscript; LSTM_Final, BiLSTM_Final and
  PSO_LightPalette_with_Legend panels), each as editable PPTX with vector PDF
  and 400 dpi PNG exports (`figs/manuscript_R2/`).
- README: rewritten for clarity - study overview, step-by-step reproduction
  guide with expected outputs, troubleshooting table, DOI badges, and the
  version DOI of the archived release.

## v1.1.2 - 2026-08-12
- Smoke test (test_reproducibility.py) rewritten for the actual package layout
  and given a deterministic numeric check (May-August Theil-Sen climate trends
  reproduce the manuscript values from the shipped data). Verified: exit 0 on
  the assembled package and on a code-only clone.
- fig_wildfire_trend.py restored to src/ (referenced by the reviewer guide).

## v1.1.1 - 2026-08-12
- Replication-drill fixes: content checksums regenerated for the actual package
  tree (root-relative paths, scoped to data/models/src/R/notebooks); verification
  commands corrected to run from the package root; HOW_TO_ASSEMBLE and the
  reviewer guide updated to the real code layout (src/, R/, notebooks/) with a
  result-to-script mapping table.

## v1.1-revision2 — 2026-08-11
- Metadata: replaced placeholder templates with populated records — 10 ISO 19115-2 XML
  (one per dataset group), 11 CSVW JSON-LD dictionaries, SHA-256 checksums, dataset
  inventory and data dictionary (`metadata/`).
- README rewritten: GitHub-vs-Zenodo roles, single concept DOI
  (10.5281/zenodo.20389083), CRS/EPSG statement, metadata standards, verified
  reproduce steps.
- Citation metadata: CITATION.cff (CFF 1.2, with ORCIDs and preferred-citation) and
  .zenodo.json aligned with the article and archive identifiers.
- Figures: reconstructed Figures 2 and 4 (editable PPTX + vector PDF + 400 dpi PNG)
  and revised trend figures (`figs/manuscript_R2/`).
- Added annual ≥70 ha fire series to `data/processed/` with CSVW dictionary.
- Repository cleanup: reviewer working folders untracked; GEE script consolidated to
  `src/gee/`; DOI-checker tools in `src/tools/`.

## v1.0-revision1 — 2026-06-23
- Initial archived release supporting revision 1 (Zenodo record 20820293 family).
