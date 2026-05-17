# Paper_GFJ — When Geopolitical Shocks Become Systemic

**Working title:** When Geopolitical Shocks Become Systemic: Volatility Connectedness and Market-Implied Stress in European Equity Markets

**Target journal:** Global Finance Journal (GFJ)

**Status:** Active development — Phase 0 (Setup)

---

## Overview

This paper studies whether geopolitical shocks become systemic financial stress events in European equity markets. It combines a validated high-frequency geopolitical shock series (inherited from the companion BIR paper) with new market-based measures of systemic fragility, volatility connectedness, and HMM-implied stress regimes estimated over 19 European equity markets from 2017 to 2025.

The central research question is:

> Do geopolitical shocks become systemic financial stress events by increasing volatility connectedness across European equity markets, and can market-implied stress regimes distinguish financially material geopolitical shocks from those absorbed by markets?

### Three-layer design

1. **Layer 1 — Trigger:** Validated EVT+FDR geopolitical shock series S_{i,t} (from companion BIR paper, `../results/shocks/`). Used as an external trigger, not as a methodological contribution.

2. **Layer 2 — Systemic fragility object (main contribution):** Daily European Market Fragility Index (EMFI) = PCA composite of volatility stress, tail co-exceedance breadth, rolling cross-market correlation, and total volatility connectedness index (TCI) from rolling Diebold–Yilmaz VAR.

3. **Layer 3 — Market-implied stress classification:** Gaussian HMM on fragility indicators, estimated without reference to shock dates, producing daily P(SystemicStress_t). Used to classify shocks as absorbed vs. systemic.

---

## Repository Structure

```
Paper_GFJ/
├── ACTION_PLAN.md          # Master phased action plan (update after each phase)
├── README.md               # This file
├── .gitignore
├── requirements.txt
│
├── data/                   # Input data (copied from parent repo)
│   ├── README.md           # Data provenance and source paths
│   ├── shocks_long.csv     # EVT shock events (copied from ../results/shocks/)
│   ├── panel_daily.parquet # Unified analysis panel (built in SP01)
│   └── clusters.csv        # Country cluster assignments
│
├── subprojects/            # Numbered analysis pipeline
│   ├── 01_data_preparation/
│   ├── 02_fragility_indicators/
│   ├── 03_connectedness/
│   ├── 04_emfi/
│   ├── 05_stress_regimes/
│   ├── 06_panel_lp/
│   ├── 07_state_dependent_lp/
│   ├── 08_event_classification/
│   └── 09_robustness/
│
├── scripts/                # Paper asset build scripts (figures + tables → LaTeX)
│
├── results/                # Computed outputs (CSVs, PNGs — git-ignored large files)
│   ├── fragility/
│   ├── connectedness/
│   ├── emfi/
│   ├── stress_regimes/
│   ├── panel_lp/
│   ├── state_lp/
│   ├── event_classification/
│   ├── robustness/
│   └── figures/            # Paper-ready figures (committed)
│
└── Paper_LaTeX/            # Manuscript (LaTeX)
    ├── main.tex
    ├── references.bib
    ├── figures/            # Figures included in paper
    └── tables/             # Table .tex files
```

---

## Running the pipeline

```bash
# Phase 1: prepare data
python subprojects/01_data_preparation/prepare_data.py

# Phase 2: fragility indicators
python subprojects/02_fragility_indicators/build_fragility_indicators.py

# Phase 3: connectedness
python subprojects/03_connectedness/build_connectedness.py

# Phase 4: EMFI composite
python subprojects/04_emfi/build_emfi.py

# Phase 5: HMM stress regimes
python subprojects/05_stress_regimes/build_hmm_regimes.py

# Phase 6: panel local projections
python subprojects/06_panel_lp/run_panel_lp.py

# Phase 7: state-dependent LP
python subprojects/07_state_dependent_lp/run_state_lp.py

# Phase 8: event classification
python subprojects/08_event_classification/classify_events.py

# Phase 9: robustness
python subprojects/09_robustness/run_robustness.py
```

Each script is self-contained: it reads from `data/` and `results/` and writes to `results/<subproject>/`.

---

## Relationship to companion papers

| Paper | Folder | Status | Role in GFJ |
|-------|--------|--------|-------------|
| BIR paper (Borsa Istanbul Review) | `../Paper_BIR/` | Under review | Provides the validated shock measure used here as input |
| Rejected FI paper | `../Paper/` | Rejected — retired | Earlier attempt; GFJ is a fundamentally new paper |

---

## Key variables

| Variable | Phase | Description |
|----------|-------|-------------|
| S_{i,t} | Input | Geopolitical shock intensity for country i on day t (from BIR paper) |
| MaxShock_t | SP01 | max_i S_{i,t} — aggregate daily shock intensity |
| VolStress_t | SP02 | mean_i \|r_{i,t}\| — average absolute return across 19 markets |
| TailVolBreadth_t | SP02 | Count of markets with \|r_{i,t}\| > Q_{i,95} |
| AvgCorr_t | SP02 | Average pairwise return correlation (60-day rolling) |
| TCI_t | SP03 | Total Connectedness Index from rolling DY VAR |
| EMFI_t | SP04 | European Market Fragility Index (PCA composite) |
| P_stress_t | SP05 | HMM probability of systemic stress state |

---

## Contact

Radu Lupu — radulupu.ase@gmail.com
