# Metadata

| File / folder | Standard | Contents |
|---|---|---|
| `iso19115/<group>.iso19115.xml` | ISO 19115-2:2009 | One record per dataset group: CRS + EPSG code, extent, resolution, temporal coverage, lineage, constraints, contact, distribution |
| `csvw/<dataset>.csvw.json` | W3C CSVW (JSON-LD) | Column-level dictionary per tabular dataset |
| `checksums_sha256.txt` | SHA-256 | Integrity hashes for the unpacked scientific content (`data/`, `models/`, `src/`, `R/`, `notebooks/`) |
| `dataset_inventory.csv`, `data_dictionary.csv` | — | Human-readable inventories |

Derived layers: NAD83 / BC Albers (EPSG:3005), 1.5 km grid. Source-dataset
records keep their native CRS with reprojection documented in lineage.

Verify the unpacked package (run from the package root):

```bash
sha256sum -c metadata/checksums_sha256.txt
```

The deposit-level files themselves are verified by `MANIFEST_sha256.txt` at the
package root. These records follow the ISO 19115-2 schema (ISO-aligned); no
formal certification is claimed.
