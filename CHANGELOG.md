# Changelog

## v1.1-revision2 — 2026-08-11
- Metadata: replaced placeholder templates with populated records — 10 ISO 19115-2 XML
  (one per dataset group), 11 CSVW JSON-LD dictionaries, SHA-256 checksums, dataset
  inventory and data dictionary (`metadata/`).
- README rewritten: GitHub-vs-Zenodo roles, single concept DOI
  (10.5281/zenodo.20389083), CRS/EPSG statement, metadata standards, verified
  reproduce steps.
- Citation metadata: CITATION.cff (CFF 1.2, with ORCIDs and preferred-citation) and
  .zenodo.json aligned with the revised article title.
- Figures: reconstructed Figures 2 and 4 (editable PPTX + vector PDF + 400 dpi PNG)
  and revised trend figures with Theil–Sen 95% CIs (`figs/manuscript_R2/`).
- Trend analysis register with confidence intervals (`tables/`).
- Added annual ≥70 ha fire series to `data/processed/` with CSVW dictionary.
- Repository cleanup: reviewer working folders untracked; GEE script consolidated to
  `src/gee/`; DOI-checker tools in `src/tools/`.

## v1.0-revision1 — 2026-06-23
- Initial archived release supporting revision 1 (Zenodo record 20820293 family).
