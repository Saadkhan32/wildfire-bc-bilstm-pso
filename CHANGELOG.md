# Changelog

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
