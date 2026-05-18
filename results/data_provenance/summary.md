# Data Provenance Summary -- SP01
Generated: 2026-05-17T20:10:58.698350+00:00

## Study sample
- **Countries:** 19 European equity markets
- **Study period:** 2017-01-02 to 2025-10-15
- **Trading days:** 2278
- **Dropped return dates** (no conflict index coverage): 15
- **Total panel rows:** 43,282 (19 countries x 2278 days)

## Countries and clusters
  - Cluster 1: 4 countries
  - Cluster 2: 9 countries
  - Cluster 3: 6 countries

Full list: Austria, Belgium, Bulgaria, Croatia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Netherlands, Norway, Poland, Portugal, Romania, Spain, Sweden, United Kingdom

## Return statistics
| Statistic | Value |
|-----------|-------|
| Mean return | 0.000274 |
| Std return  | 0.011113 |
| Min return  | -0.185411 |
| Max return  | 0.108501 |
| NaN returns | 0 |

## Shock series (declustered EVT+FDR)
- **Source vintage:** 20251222_paper_candidate (locked -- same as BIR paper)
- **Raw events (before alignment):** 278
- **Events on original trading day:** 97
- **Events forward-filled from weekend/holiday:** 181
- **Events after alignment (collisions removed):** 278
- **Days with >=1 country shocked:** 163
- **S = -log(q) range:** 0.0728 to 2.8598

## Aggregate shock measures (daily)
| Measure | Days > 0 | Max | Mean (excl. 0) |
|---------|----------|-----|----------------|
| max_shock | 163 | 2.8598 | 0.2808 |
| breadth_shock | 163 | 16 | 1.71 |
