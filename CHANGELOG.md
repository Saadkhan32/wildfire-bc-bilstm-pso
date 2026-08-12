# Changelog

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
