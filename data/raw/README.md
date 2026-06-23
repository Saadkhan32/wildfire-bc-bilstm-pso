# `data/raw/` — climate-oscillation indices

Source index series used in the teleconnection analysis (Section 3.9, Fig. 18).

| File | Index | Provider | Notes |
|---|---|---|---|
| `oni.ascii.txt` | Oceanic Niño Index (ONI) | NOAA Climate Prediction Center | Official 3-month-running Niño-3.4 SST anomaly; columns `SEAS YR TOTAL ANOM`. Listed in Table 2. Retrieved 2026-06-17 from https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt |
| `meiv2.txt` | Multivariate ENSO Index v2 (MEI v2) | NOAA Physical Sciences Laboratory | Bimonthly values; the ENSO series actually read by `teleconnections.py`. |
| `pdo.csv` | Pacific Decadal Oscillation (PDO) | NOAA Physical Sciences Laboratory | Monthly values. |

**ENSO index note.** Table 2 of the manuscript names the **ONI** as the ENSO source, while
`code/teleconnections.py` computes correlations from **MEI v2**. To be fully transparent, both the
official ONI series (`oni.ascii.txt`) and the MEI v2 series (`meiv2.txt`) are provided; the PDO series
is the same in either case. The two indices are strongly correlated measures of the same ENSO
phenomenon, so the reported teleconnection conclusions are unaffected.
