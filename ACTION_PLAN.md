# Action Plan — Paper_GFJ

**Working title:** When Geopolitical Shocks Become Systemic: Volatility Connectedness and Market-Implied Stress in European Equity Markets

**Target journal:** Global Finance Journal (GFJ)

**Created:** 2026-05-17  
**Last updated:** 2026-05-17  
**Overall status:** Phase 1 — Data Preparation ✅ Complete

---

## Scientific Framing

This paper asks a single sharp question:

> **When do geopolitical shocks become systemic financial stress events?**

The paper is built around three conceptual layers:

- **Layer 1 — Trigger (inherited):** The validated EVT+FDR geopolitical shock series from the companion BIR paper is reused *as a trigger*, not as a contribution. The framing shift is explicit: "We use a validated high-frequency geopolitical conflict shock series to study systemic financial-market responses."
- **Layer 2 — New object (core contribution):** A suite of daily market-based systemic fragility indicators constructed from the 19-country European equity return panel: volatility stress, tail co-exceedance breadth, rolling average cross-market correlation, a total volatility connectedness index (TCI via rolling Diebold–Yilmaz), and a composite European Market Fragility Index (EMFI) built by PCA.
- **Layer 3 — Stress classification (enhancement):** A Hidden Markov Model (HMM) estimated on the fragility indicators — without reference to geopolitical shock dates — to assign each trading day a market-implied probability of systemic stress.

**Separation from BIR paper:**
- BIR = shock construction and validation methodology.
- GFJ = financial stability and systemic transmission, using that shock measure as input.

---

## Phases Overview

| Phase | Title | Status |
|-------|-------|--------|
| 0 | Repository setup and scaffolding | ✅ Complete |
| 1 | Data preparation | ✅ Complete |
| 2 | Daily fragility indicators | ⬜ Pending |
| 3 | Volatility connectedness (TCI) | ⬜ Pending |
| 4 | Composite EMFI | ⬜ Pending |
| 5 | HMM market-implied stress regimes | ⬜ Pending |
| 6 | Panel local projections | ⬜ Pending |
| 7 | State-dependent local projections | ⬜ Pending |
| 8 | Event classification | ⬜ Pending |
| 9 | Robustness | ⬜ Pending |
| 10 | Paper writing (LaTeX) | ⬜ Pending (parallel with 6–9) |

---

## Phase 0 — Repository Setup and Scaffolding

**Status:** ✅ Complete  
**Date started:** 2026-05-17  
**Date completed:** 2026-05-17

### Objectives
- Create folder structure following the conventions established in the parent repository.
- Initialize a standalone git repository inside `Paper_GFJ/`.
- Set up the LaTeX skeleton for the manuscript.
- Write this action plan.

### Deliverables
- [x] `Paper_GFJ/` directory with `data/`, `subprojects/`, `results/`, `scripts/`, `Paper_LaTeX/`
- [x] `git init` inside `Paper_GFJ/`
- [x] `ACTION_PLAN.md` (this file)
- [x] `README.md`
- [x] `.gitignore`
- [x] `requirements.txt`
- [x] Subproject skeleton READMEs (01–09)
- [x] `Paper_LaTeX/main.tex` skeleton with adapted preamble
- [x] `Paper_LaTeX/references.bib` (seeded from FI paper bibliography)
- [ ] Initial git commit

### Conventions established
- Subprojects numbered 01–09, each with a `README.md` and Python scripts.
- Results organized under `results/<subproject_name>/`.
- All scripts write output to `results/`; no hardcoded absolute paths.
- `scripts/` contains paper-asset build scripts (figures, tables for LaTeX).
- `Paper_LaTeX/` contains the manuscript; figures and tables are populated by build scripts.

---

## Phase 1 — Data Preparation

**Status:** ✅ Complete  
**Date completed:** 2026-05-17  
**Subproject:** `subprojects/01_data_preparation/`  
**Script:** `subprojects/01_data_preparation/prepare_data.py`  
**Tests:** `subprojects/01_data_preparation/test_prepare_data.py` — 43/43 passed

### Objectives
- Copy the locked paper-candidate shock series from the parent repository into `data/shocks/`.
- Copy the clean 19-country equity return panel into `data/`.
- Copy the country cluster assignments.
- Build a single unified analysis-ready panel: `data/panel_daily.parquet` with columns [date, country, return, abs_return, sq_return, shock_intensity].
- Verify date alignment, coverage gaps, and report a data provenance summary.

### Inputs (from parent repo — relative paths)
- `../data/gdelt_indices/conflict_intensity/20251222_paper_candidate/` — locked shock vintage
- `../results/shocks/` — EVT shock events and shock series (long format)
- `../data/stock_prices.xlsx` — 19-country daily equity prices
- `../data/clusters_input.csv` — country cluster assignments

### Key design decisions
- Use the **same locked vintage** as the BIR paper (`20251222_paper_candidate`) to ensure consistency.
- The shock measure used is `S_{i,t} = -log(q_{i,t})` (continuous, declustered), identical to BIR.
- Study period: **2017-01-02 to 2025-10-15** (first return date to last available date).
- **Weekend shock alignment:** 181 of 278 raw shock events fall on weekends (the conflict index runs on all calendar days). These are forward-filled to the next trading day. 15 return dates with no conflict index coverage are dropped. Final study period: 2,278 trading days.

### Deliverables
- [x] `data/panel_daily.parquet` — unified daily panel: 43,282 rows (2,278 days × 19 countries)
- [x] `data/shocks_events.csv` — 278 shock events with S, original_date, trading-day-aligned date
- [x] `data/aggregate_shocks.csv` — daily max_shock, breadth_shock, sum_shock, avg_shock_pos
- [x] `data/clusters.csv` — country → cluster mapping
- [x] `results/data_provenance/summary.md` — coverage report
- [x] `results/data_provenance/manifest.json` — machine-readable provenance

### Key facts from implementation
- **2,278 trading days** in study period (2017-01-02 to 2025-10-15)
- **278 shock events** total: 97 on trading days, 181 forward-filled from weekends
- **163 shock days** (days with max_shock > 0) after forward-fill
- **S range:** 0.0728 to 2.8598 (mean 0.3183)
- **15 return dates** dropped (not present in conflict index)

---

## Phase 2 — Daily Fragility Indicators

**Status:** ⬜ Pending  
**Subproject:** `subprojects/02_fragility_indicators/`  
**Script:** `subprojects/02_fragility_indicators/build_fragility_indicators.py`

### Objectives
Construct a daily cross-country panel of market-based fragility measures from the 19-country equity return series.

### Indicators to construct (all daily, cross-sectional aggregates)

| Variable | Definition |
|----------|-----------|
| `VolStress_t` | Average absolute return: mean_i \|r_{i,t}\| |
| `SqStress_t` | Average squared return: mean_i r²_{i,t} |
| `TailVolBreadth_t` | Count of markets with \|r_{i,t}\| > Q_{i,95} (rolling 252-day threshold) |
| `TailLossBreadth_t` | Count of markets with r_{i,t} < Q_{i,5} (rolling 252-day threshold) |
| `TailGainBreadth_t` | Count of markets with r_{i,t} > Q_{i,95} (rolling 252-day) |
| `AvgCorr_t` | Average pairwise return correlation over rolling 60-day window |
| `AvgCorr30_t` | Same over rolling 30-day window (robustness) |

### Implementation notes
- Country-specific Q_{i,95} and Q_{i,5} thresholds: estimated on a rolling 252-day backward window (no look-ahead).
- Rolling correlation: use pairwise Pearson over a rolling 60-day window; average the N*(N-1)/2 pairs.
- Output all indicators in a single `results/fragility/fragility_daily.csv` plus a verification plot.

### Deliverables
- `results/fragility/fragility_daily.csv` — all indicators, daily
- `results/fragility/Fig_FragilityTimeline.png` — time-series plot of each indicator with major event overlays
- `results/fragility/summary_stats.csv` — descriptive statistics

---

## Phase 3 — Volatility Connectedness (TCI)

**Status:** ⬜ Pending  
**Subproject:** `subprojects/03_connectedness/`  
**Script:** `subprojects/03_connectedness/build_connectedness.py`

### Objectives
Construct a daily Total Connectedness Index (TCI) from the 19 volatility proxies (\|r_{i,t}\|) using a rolling VAR forecast-error variance decomposition (Diebold–Yilmaz methodology).

### Method: Rolling Diebold–Yilmaz VAR connectedness

For each rolling window of W days (primary: W=100, secondary: W=200):
1. Estimate a VAR(p) on the 19-dimensional vector of absolute returns v_{i,t} = |r_{i,t}|. Use p=1 or select via BIC (cap at p=3).
2. Compute the generalized forecast-error variance decomposition (GFEVD) at H-step horizon (H=10).
3. Extract the TCI = (sum of all off-diagonal FEVD shares) / N × 100.
4. Also extract: directional FROM/TO spillovers per country; net spillover position.

### Robustness
- Alternative window sizes: W=60, W=150.
- Alternative VAR order selection: p fixed at 2 vs BIC.
- Correlation-network connectedness (average of rolling pairwise correlations, network density, minimum spanning tree length) as a simpler parallel measure.

### Key implementation note
Use the `statsmodels` VAR implementation. For GFEVD: implement the Pesaran-Shin (1998) generalized decomposition (order-invariant), not the Cholesky decomposition.

### Deliverables
- `results/connectedness/tci_daily.csv` — daily TCI for each window specification
- `results/connectedness/directional_connectedness.csv` — daily FROM/TO per country
- `results/connectedness/Fig_TCI_Timeline.png` — TCI over time with event overlays
- `results/connectedness/Fig_Network_MajorEvents.png` — network topology before/after key shocks
- `results/connectedness/summary_stats.csv`

---

## Phase 4 — Composite EMFI (European Market Fragility Index)

**Status:** ⬜ Pending  
**Subproject:** `subprojects/04_emfi/`  
**Script:** `subprojects/04_emfi/build_emfi.py`

### Objectives
Combine the fragility indicators from Phase 2 and the TCI from Phase 3 into a single composite daily index using PCA. This index — the European Market Fragility Index (EMFI) — is the paper's **primary outcome variable**.

### Method
1. Standardize each component to mean 0, std 1 (using the full-sample mean and std, not rolling — the EMFI is not used for detection, only as an outcome variable).
2. Apply PCA to [VolStress, TailVolBreadth, AvgCorr, TCI].
3. Extract the first principal component; orient it so that higher values = more fragility (flip sign if needed).
4. Report variance explained by PC1 (target: ≥50% for the index to be credible).
5. Also report loadings to motivate the weighting.

### Deliverables
- `results/emfi/emfi_daily.csv` — daily EMFI series
- `results/emfi/pca_loadings.csv` — component loadings
- `results/emfi/Fig_EMFI_Timeline.png` — EMFI with shaded stress episodes and vertical event lines
- `results/emfi/Fig_EMFI_Components.png` — all four components alongside EMFI

---

## Phase 5 — HMM Market-Implied Stress Regimes

**Status:** ⬜ Pending  
**Subproject:** `subprojects/05_stress_regimes/`  
**Script:** `subprojects/05_stress_regimes/build_hmm_regimes.py`

### Objectives
Estimate a Hidden Markov Model on the systemic fragility indicators to classify each trading day into latent market stress regimes, without reference to geopolitical shock dates. Extract the daily probability of being in the "systemic stress" state.

### Method
- Input vector: X_t = [VolStress_t, TailVolBreadth_t, AvgCorr_t, TCI_t] — standardized.
- Model: Gaussian emission HMM, fitted with `hmmlearn.GaussianHMM`.
- Primary specification: 3-state model (calm / elevated / systemic stress). States ordered post-hoc by mean EMFI to assign labels.
- Robustness: 2-state model.
- Initialization: multiple random restarts (n=50) to avoid local optima; keep the run with highest log-likelihood.
- **Critical constraint:** estimated on the full return sample with NO reference to shock dates. Circularity check: verify that shock dates are not an input to the HMM.

### Outputs
- `P_stress_t`: daily probability of being in the systemic-stress state (State 3 in 3-state model).
- `regime_t`: most likely regime assignment (Viterbi path).

### Validation
- Overlay regime periods on a timeline with known major events (Ukraine invasion, Hamas-Israel, COVID crash, etc.) to verify face validity.
- Compute fraction of shock dates falling in each regime — report in paper as a characterization, not identification.

### Deliverables
- `results/stress_regimes/hmm_daily.csv` — P_stress_t, regime_t for each day
- `results/stress_regimes/hmm_params.json` — fitted parameters
- `results/stress_regimes/Fig_HMM_Regimes.png` — regime probability timeline
- `results/stress_regimes/Fig_HMM_Validation.png` — alignment with known events

---

## Phase 6 — Panel Local Projections

**Status:** ⬜ Pending  
**Subproject:** `subprojects/06_panel_lp/`  
**Script:** `subprojects/06_panel_lp/run_panel_lp.py`

### Objectives
Estimate how geopolitical shocks dynamically affect systemic fragility using Jordà (2005) local projections. This is the paper's **primary empirical design**.

### Specification
For each horizon k ∈ {-5, ..., +15}:

    Y_{t+k} = α + β_k S_t + γ Y_{t-1} + δ r^{global}_{t} + μ_m + ε_{t+k}

where:
- Y is the outcome variable (one of EMFI, TCI, TailVolBreadth, P_stress, AvgCorr)
- S_t = geopolitical shock intensity on day t (aggregate measure: maximum country-level S across the 19 countries on day t)
- Y_{t-1} = lagged dependent variable (1 lag)
- r^{global}_{t} = equal-weighted return across all 19 markets on day t (global control)
- μ_m = month-of-year fixed effects
- Inference: Newey-West standard errors with bandwidth = 2×H (to account for overlapping projections)

**Note:** Unlike the FI paper, this is a **time-series LP** (outcomes are aggregate daily series, not a country-panel). Country-level analysis is in Phase 7 for the state-dependent specifications.

### Outcome variables (five separate regressions)
1. EMFI_t (primary)
2. TCI_t (primary)
3. TailVolBreadth_t
4. P_stress_t (from HMM)
5. AvgCorr_t

### Additional specifications
- Aggregate shock measure: MaxShock_t (max S across countries on day t) — primary.
- Alternative: AvgShock_t (mean S); BreadthShock_t (count of countries with shock on day t).
- Longer horizon: k up to +20 for persistence analysis.

### Key figures
- **Figure 2 (paper):** Impulse-response plots for each outcome variable, β_k ± 90% and 95% CI bands, pre-period shown to verify no pre-trends.

### Deliverables
- `results/panel_lp/lp_results_{outcome}.csv` — coefficients and SEs for each outcome
- `results/panel_lp/Fig_LP_{outcome}.png` — response function plots
- `results/panel_lp/Fig_LP_Combined.png` — multi-panel summary figure (paper-ready)

---

## Phase 7 — State-Dependent Local Projections

**Status:** ⬜ Pending  
**Subproject:** `subprojects/07_state_dependent_lp/`  
**Script:** `subprojects/07_state_dependent_lp/run_state_lp.py`

### Objectives
Test whether geopolitical shocks are more destabilizing when they occur in already-fragile markets. This is the paper's **most novel empirical contribution**.

### Specification

    Y_{t+k} = α + β_k S_t + θ_k (S_t × HighFragility_{t-1}) + φ HighFragility_{t-1} + controls + ε_{t+k}

where:
- `HighFragility_{t-1}` = 1 if EMFI_{t-1} > 75th percentile of EMFI (full-sample threshold)
- `HighConnectedness_{t-1}` = 1 if TCI_{t-1} > 75th percentile of TCI (robustness)
- β_k = effect of shock in normal markets
- θ_k = additional (incremental) effect when pre-shock market is fragile
- Total effect in fragile markets: β_k + θ_k

### Key figures
- **Figure 3 (paper):** Two IRF lines — normal state (β_k) vs. fragile pre-state (β_k + θ_k) — with confidence bands for each.

### Deliverables
- `results/state_lp/state_lp_results.csv`
- `results/state_lp/Fig_StateLPFragility.png` — normal vs. fragile IRF (primary figure)
- `results/state_lp/Fig_StateLPConnected.png` — normal vs. high-connectedness IRF (robustness)

---

## Phase 8 — Event Classification

**Status:** ⬜ Pending  
**Subproject:** `subprojects/08_event_classification/`  
**Script:** `subprojects/08_event_classification/classify_events.py`

### Objectives
Produce a 4-category taxonomy of geopolitical shock episodes, combining the news-based shock measure with the market-implied stress classification. This creates a concrete, interpretable typology.

### Four categories

| Category | Definition |
|----------|-----------|
| **Absorbed** | Shock detected (S_t > threshold), no EMFI spike, P_stress_t stays low |
| **Localized** | Shock detected, moderate EMFI rise but P_stress_t < 0.5 within 5 days |
| **Systemic** | Shock detected, EMFI rises above 75th percentile AND P_stress_t > 0.5 within 5 days |
| **Market-only stress** | P_stress_t > 0.5, no shock detected within ±3 days |

### Implementation
- Iterate over all declustered shock event dates.
- For each event: compute EMFI change (mean of t+1 to t+5 minus mean of t-5 to t-1), and max P_stress in window [t, t+5].
- Assign category using the definitions above.
- For each category: compute average volatility response, average TCI response, average duration of elevated stress.

### Key outputs
- **Table (paper):** Event listing with category, date, country, shock intensity, EMFI response, TCI response, P_stress peak, classification. This is **Figure 5 / Table** in the idea document.
- **Figure 4 (paper):** Connectedness network before/after the top 5 "Systemic" events.

### Deliverables
- `results/event_classification/event_taxonomy.csv`
- `results/event_classification/Fig_NetworkMajorEvents.png`
- `results/event_classification/summary_by_category.csv`

---

## Phase 9 — Robustness

**Status:** ⬜ Pending  
**Subproject:** `subprojects/09_robustness/`  
**Script:** `subprojects/09_robustness/run_robustness.py`

### Checks to implement

| Check | Description |
|-------|------------|
| Alternative connectedness | Correlation-network TCI (network density, avg correlation) instead of VAR-based |
| Alternative shock aggregation | AvgShock vs MaxShock vs BreadthShock |
| TVP-VAR connectedness | Time-varying parameter VAR (Antonakakis et al.) as alternative to rolling DY |
| Alternative EMFI weights | Equal-weight composite (vs PCA) |
| Alternative fragility threshold | 66th and 90th percentile fragility states (vs 75th) |
| COVID exclusion | Drop 2020-03 to 2020-12 |
| Ukraine exclusion | Drop 2022-02 to 2022-06 |
| Future-shock placebo | Replace S_t with S_{t+30} — must show no significant effects |
| Bootstrapped inference | Block bootstrap CIs for LP coefficients |

### Deliverables
- `results/robustness/robustness_summary.csv` — all checks with β_k at key horizons
- `results/robustness/Fig_Robustness_LP.png` — overlay of baseline + robustness IRFs

---

## Phase 10 — Paper Writing

**Status:** ⬜ Pending (begins in parallel with Phase 6)  
**Location:** `Paper_LaTeX/`

### Manuscript structure

1. Introduction
2. Data and geopolitical shock triggers
3. Market-based financial stability measures (Phases 2–4)
4. Market-implied systemic stress regimes (Phase 5)
5. Dynamic effects of geopolitical shocks on systemic fragility (Phase 6)
6. State dependence and event classification (Phases 7–8)
7. Robustness (Phase 9)
8. Conclusion

### Key figures (for paper)
- Figure 1: Timeline — EMFI + HMM shaded regimes + vertical shock lines
- Figure 2: LP impulse responses (EMFI, TCI, TailBreadth, P_stress)
- Figure 3: State-dependent LP (normal vs fragile pre-state)
- Figure 4: Connectedness network — before/after major systemic events
- Figure 5: Event taxonomy table

### Writing milestones
- [ ] Data section draft (after Phase 1)
- [ ] Sections 3–4 draft (after Phases 2–5)
- [ ] Sections 5–6 draft (after Phases 6–7)
- [ ] Full draft for co-author review
- [ ] GFJ submission package

---

## Key Design Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Shock measure | Reuse EVT+FDR S_{i,t} from BIR paper | Validated, avoids double contribution; BIR = methodology, GFJ = application |
| LP design | Time-series (aggregate outcomes) not panel | EMFI/TCI/P_stress are aggregate objects; country-panel LP less natural here |
| Shock aggregation | MaxShock_t as primary | Captures the most severe country shock on each day; interpretable |
| EMFI construction | PCA (not equal-weight) | Lets data determine loadings; variance-explained criterion validates the composite |
| HMM | Gaussian emission, 3-state | Interpretable states; 2-state as robustness. Estimated on market features only |
| Connectedness | Rolling DY (W=100) as primary | Established methodology; TVP-VAR as robustness |
| Fragility state threshold | 75th percentile of EMFI | Standard in state-dependent LP literature; 66th and 90th as robustness |

---

## Dependencies Between Phases

```
Phase 1 (data)
    ├── Phase 2 (fragility indicators)
    │       └── Phase 4 (EMFI) ─────────────────── Phase 6 (LP) ─── Phase 7 (state LP)
    └── Phase 3 (connectedness) ─── Phase 4 ─────── Phase 6 ─────── Phase 7
                                                    Phase 5 (HMM) ── Phase 6 ── Phase 7
                                                                        └── Phase 8 (classification)
Phases 2–5 → Phase 9 (robustness)
Phases 6–9 → Phase 10 (writing)
```

---

## Change Log

| Date | Phase | Change |
|------|-------|--------|
| 2026-05-17 | 0 | Initial action plan created. Repository scaffolded. Git repo initialized (`init_git.sh`). LaTeX skeleton created. |
| 2026-05-17 | 1 | Data preparation complete. Key discovery: 181/278 shock events fall on weekends (conflict index runs on calendar days); implemented forward-fill to next trading day. Panel: 43,282 rows, 163 shock days, S in [0.07, 2.86]. 43/43 tests pass. |
