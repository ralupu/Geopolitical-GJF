# Action Plan — Paper_GFJ

**Working title:** When Geopolitical Shocks Become Systemic: Volatility Connectedness and Market-Implied Stress in European Equity Markets

**Target journal:** Global Finance Journal (GFJ)

**Created:** 2026-05-17  
**Last updated:** 2026-05-18 (Phase 7 complete; paper Section 7 drafted)  
**Overall status:** Phase 7 — State-Dependent LP ✅ Complete; paper Sections 1–6 drafted

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
| 2 | Daily fragility indicators | ✅ Complete |
| 3 | Volatility connectedness (TCI) | ✅ Complete |
| 4 | Composite EMFI | ✅ Complete |
| 5 | HMM market-implied stress regimes | ✅ Complete |
| 6 | Panel local projections | ✅ Complete |
| 7 | State-dependent local projections | ✅ Complete |
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

**Status:** ✅ Complete  
**Date completed:** 2026-05-17  
**Subproject:** `subprojects/02_fragility_indicators/`  
**Script:** `subprojects/02_fragility_indicators/build_fragility_indicators.py`  
**Tests:** `subprojects/02_fragility_indicators/test_fragility_indicators.py` — 39/39 passed

### Objectives
Construct a daily cross-country panel of market-based fragility measures from the 19-country equity return series.

### Indicators constructed (all daily, cross-sectional aggregates)

| Variable | Definition |
|----------|-----------|
| `VolStress_t` | Average absolute return: mean_i \|r_{i,t}\| |
| `SqStress_t` | Average squared return: mean_i r²_{i,t} |
| `TailVolBreadth_t` | Count of markets with \|r_{i,t}\| > Q_{i,95} (rolling 252-day threshold) |
| `TailLossBreadth_t` | Count of markets with r_{i,t} < Q_{i,5} (rolling 252-day threshold) |
| `TailGainBreadth_t` | Count of markets with r_{i,t} > Q_{i,95} (rolling 252-day) |
| `AvgCorr60_t` | Average pairwise return correlation over rolling 60-day window (171 pairs) |
| `AvgCorr30_t` | Same over rolling 30-day window (robustness) |

### Implementation notes
- Country-specific Q_{i,95} and Q_{i,5} thresholds: rolling 252-day backward window, **shifted by 1 day** (strictly out-of-sample, no look-ahead).
- Rolling correlation: pairwise Pearson over rolling 60-day window; average the N*(N-1)/2 = 171 pairs.
- Warm-up NaNs: 60 rows for tail breadth indicators, 29 rows for AvgCorr60, 19 rows for AvgCorr30.

### Deliverables
- [x] `results/fragility/fragility_daily.csv` — all 7 indicators, 2,278 rows
- [x] `results/fragility/Fig_FragilityTimeline.png` — 4-panel timeline with COVID/Ukraine shading
- [x] `results/fragility/summary_stats.csv` — descriptive statistics
- [x] `results/fragility/manifest.json` — parameters and provenance

### Key facts from implementation
- **VolStress max: 0.119** (2020-03-16, COVID crash) — cross-sectional mean of |r| across 19 markets
- **TailVolBreadth on 2020-03-16: 18/19** — nearly all markets simultaneously in extreme tail
- **AvgCorr60 range: 0.207 to 0.871** (mean 0.476) — high baseline cross-market correlation
- **AvgCorr30 range: 0.145 to 0.885** (mean 0.461) — more volatile, captures short bursts of co-movement
- Script runtime: ~3 seconds

---

## Phase 3 — Volatility Connectedness (TCI)

**Status:** ✅ Complete  
**Date completed:** 2026-05-17  
**Subproject:** `subprojects/03_connectedness/`  
**Script:** `subprojects/03_connectedness/build_connectedness.py`  
**Tests:** `subprojects/03_connectedness/test_connectedness.py` — 32/32 passed

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
- [x] `results/connectedness/tci_daily.csv` — daily TCI for W=60/100/150/200, 2219 rows
- [x] `results/connectedness/directional_connectedness.csv` — daily FROM/TO/NET per country (W=100), 41,401 rows
- [x] `results/connectedness/Fig_TCI_Timeline.png` — 2-panel: primary TCI + all window specs
- [x] `results/connectedness/Fig_Network_MajorEvents.png` — TO/FROM bar charts for 3 regimes
- [x] `results/connectedness/summary_stats.csv`
- [x] `results/connectedness/manifest.json`

### Key facts from implementation
- **TCI (W=100) mean: 70.5%** — European equity markets are very highly connected
- **TCI range: 46.7% to 94.6%** — substantial time variation
- **COVID peak (Mar-Apr 2020): TCI > 80%** — near-total volatility co-movement
- **FROM spillover (Germany, Mar 2020): ~91%** — market effectively became a single entity
- Script runtime: ~25 seconds for all 4 window specs
- BIC-selected VAR order: predominantly p=1 across windows

### Analytical observations (for paper)
- **High baseline TCI level (70.5%)** reflects the deep structural integration of European equity markets rather than acute stress. This means TCI captures two different phenomena: (1) structural co-movement (always high), and (2) crisis amplification (spike above the baseline). This distinction matters for interpretation in the paper: TCI elevation above its own time-series mean is the relevant signal, not the absolute level.
- **TCI vs. AvgCorr60 overlap concern**: the correlation between TCI and AvgCorr60 is 0.797. Both series measure cross-market integration but via different methods (VAR-FEVD vs. simple correlation). The high overlap raises the question of whether TCI provides incremental information over AvgCorr60 in the LP regressions. This must be checked in Phase 9 robustness — if LP results are unchanged when TCI is replaced by AvgCorr60, the theoretical superiority of TCI does not translate into empirical content here.
- **Low cross-cluster correlation**: VolStress/TailBreadth correlate at 0.784 with each other; AvgCorr60/TCI correlate at 0.797 with each other; but the cross-cluster correlations are only 0.16–0.38. This reveals two structurally distinct dimensions of market fragility that the EMFI will need to bridge. See Phase 4 observations for implications.

---

## Phase 4 — Composite EMFI (European Market Fragility Index)

**Status:** ✅ Complete  
**Date completed:** 2026-05-18  
**Subproject:** `subprojects/04_emfi/`  
**Script:** `subprojects/04_emfi/build_emfi.py`  
**Tests:** `subprojects/04_emfi/test_emfi.py` — 28/28 passed

### Objectives
Combine the fragility indicators from Phase 2 and the TCI from Phase 3 into a single composite daily index using PCA. This index — the European Market Fragility Index (EMFI) — is the paper's **primary outcome variable**.

### Method
1. Standardise each component to mean 0, std 1 using the *full-sample* mean and std (not rolling — the EMFI is an outcome variable, not a detector).
2. Apply PCA to [VolStress, TailVolBreadth, AvgCorr60, TCI_w100].
3. Extract PC1; orient so that higher values = more fragility (flip sign if VolStress anti-correlated).
4. Report variance explained (target ≥50%).

### Deliverables
- [x] `results/emfi/emfi_daily.csv` — daily EMFI, 2,179 rows (2017-05-19 to 2025-10-15)
- [x] `results/emfi/pca_loadings.csv` — PC1 loadings per component
- [x] `results/emfi/Fig_EMFI_Timeline.png` — EMFI with stress-episode shading and event lines
- [x] `results/emfi/Fig_EMFI_Components.png` — 5-panel: EMFI + all 4 standardised components
- [x] `results/emfi/summary_stats.csv`
- [x] `results/emfi/manifest.json`

### Key facts from implementation
- **PC1 variance explained: 58.3%** — above the 50% credibility threshold
- **PC1 loadings (all positive):** VolStress=0.532, AvgCorr60=0.516, TCI=0.495, TailVolBreadth=0.454
- **EMFI max: 15.2** on 2020-03-12 (COVID crash peak)
- **EMFI range:** [-2.74, 15.22]; mean=0.00, std=1.53
- **75th pctile threshold: 0.55** (used for HighFragility state in Phase 7)
- Sample starts 2017-05-19 (TCI W=100 warm-up + TailBreadth warm-up combined)
- Note: Sign flip was applied — raw PC1 had all negative loadings; flipped for interpretability

### Analytical observations (for paper)

- **Two-cluster structure limits PC1 dominance:** PC1 explains 58.3%, but PC2 explains ~31.7% — a substantial residual. The reason is the two-cluster covariance structure identified in Phase 3: (VolStress, TailBreadth) correlate at 0.784 with each other but only 0.16–0.38 with (AvgCorr60, TCI), which themselves correlate at 0.797. PC1 loads roughly equally on all four components (0.45–0.53), effectively averaging the two clusters; PC2 captures the contrast between them. The 58.3% variance share is credible but the composite is not overwhelmingly one-dimensional. **Implication for paper:** this must be discussed explicitly in Section 3 — the EMFI is a balanced composite, not a pure first factor. The decision to use PCA rather than equal-weighting is validated by the variance explained, but the second factor's size should be acknowledged.

- **EMFI is an acute-stress indicator, not a pre-crisis fragility buildup detector:** the EMFI peaks sharply during realized crises (COVID=15.2, Ukraine=~3–4) and returns quickly to baseline thereafter. It does not build up gradually before known events, which means it will not function as a leading indicator of fragility accumulation — only as a contemporaneous or lagging stress measure. **Implication:** in the state-dependent LP (Phase 7), the HighFragility conditioning variable (EMFI_{t-1} > 75th pctile) will overwhelmingly classify days *during* crises as "fragile", not days *before* them. The state-dependent effect will therefore capture "shocks hitting already-stressed markets" not "shocks hitting pre-fragile markets". This distinction must be stated clearly in the paper and is already implicit in the EMFI construction choice.

- **COVID dominance in extreme tail:** the top decile of EMFI observations is heavily concentrated in Feb–May 2020. The 75th percentile threshold (EMFI=0.55) is already above baseline, but the extreme tail is almost entirely COVID. **Implication:** in Phase 7, the HighFragility dummy will fire during COVID for approximately 60 consecutive trading days. Any state-dependent LP result must be robust to COVID exclusion (already listed in Phase 9). This is a priority robustness check — if the state-dependent amplification result disappears when 2020-Q1/Q2 is excluded, the finding is identified from a single episode.

- **HighFragility dummy composition must be explicitly reported:** when implementing Phase 7, compute and report: (a) total number of HighFragility days (expected ~570, i.e., 25% of 2,279), (b) breakdown by year/episode, (c) number of shock events landing in HighFragility vs. normal state. If fewer than ~20 shock events fall within HighFragility periods, statistical power for θ_k identification will be low. This number must be verified before interpreting significance.

- **TCI incremental value over AvgCorr60 needs Phase 9 verification:** given the 0.797 TCI–AvgCorr60 correlation, it is possible that substituting AvgCorr60 for TCI in the EMFI (yielding a simpler index using only three indicators, all from SP02) would produce nearly identical LP results. This substitution test is explicitly added to Phase 9 robustness (see below). If results are robust, it strengthens the paper's argument that the underlying market state — not the specific measurement technology — is what matters.

- **Loadings are balanced but not equal:** VolStress (0.532) > AvgCorr60 (0.516) > TCI (0.495) > TailBreadth (0.454). The VolStress cluster loads slightly more than the connectedness cluster. The difference is small but means EMFI tilts marginally toward acute realized volatility vs. co-movement. Equal-weighted robustness in Phase 9 is important to verify results are not driven by this differential loading.

---

## Phase 5 — HMM Market-Implied Stress Regimes

**Status:** ✅ Complete  
**Date completed:** 2026-05-18  
**Subproject:** `subprojects/05_stress_regimes/`  
**Script:** `subprojects/05_stress_regimes/build_hmm_regimes.py`  
**Tests:** `subprojects/05_stress_regimes/test_hmm_regimes.py` — 45/45 passed

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
- [x] `results/stress_regimes/hmm_daily.csv` — P_stress_t, regime_t, P_calm, P_elevated for each day (2,179 rows)
- [x] `results/stress_regimes/hmm_params.json` — fitted HMM parameters (transmat, means, covars, startprob)
- [x] `results/stress_regimes/Fig_HMM_Regimes.png` — 3-panel: EMFI coloured by regime, posterior probs, Viterbi path
- [x] `results/stress_regimes/Fig_HMM_Validation.png` — P_stress vs. EMFI vs. MaxShock overlay
- [x] `results/stress_regimes/manifest.json` — provenance, state counts, diagnostics

### Key facts from implementation
- **Sample:** 2,179 dates (2017-05-19 to 2025-10-15), same as EMFI
- **State counts:** Calm=1,389 (63.7%), Elevated=567 (26.0%), Systemic stress=223 (10.2%)
- **COVID share of systemic-stress days: 22.9%** (51/223) — COVID is the dominant crisis episode but not the only one; the HMM is a genuine multi-episode detector
- **P_stress on COVID peak (2020-03-12): 1.00** — fully assigned to systemic state
- **P_stress on Ukraine invasion (2022-02-24): 1.00** — also fully assigned
- **Hamas attack (2023-10-07 is a Sunday):** nearest trading day 2023-10-09 has P_stress=0.296 — elevated but not systemic
- **Transition matrix (calm→calm=0.719, elevated→elevated=0.377, stress→stress=0.371):** calm state is strongly persistent; stressed states are transient (~2–3 day average duration)
- **Model convergence:** best log-likelihood -1367.11 over 50 random restarts; model converged
- Script runtime: ~6.6 seconds

### Analytical observations (for paper)

- **COVID is the dominant but not sole episode (22.9% of stress days):** the HMM identified 223 systemic-stress days, of which 51 fall in the Feb–Jun 2020 COVID window. This is important: the HMM is detecting multiple distinct episodes (COVID, Ukraine, and others), not simply learning a COVID dummy. This strengthens the validity of P_stress_t as a general fragility measure for the LP regressions.
- **Stressed states are transient (persistence ~0.37):** the stress state has a transition probability back to itself of only 0.371, implying an average duration of roughly 1/(1-0.371) ≈ 1.6 days in continuous sequence. The calm state is much stickier (0.719, ≈3.6 days). This means the HMM captures sharp, episodic spikes rather than prolonged stress periods — consistent with the EMFI's acute-stress nature observed in Phase 4. The regime path therefore captures the same information as the EMFI spike, not a slow-moving fragility buildup.
- **P_stress = 1.00 on COVID and Ukraine peak days:** the HMM assigns full posterior probability to the systemic state on the peak crisis days. This validates the model's face validity but also means P_stress_t is a sharp indicator with near-binary behavior during extreme events. In the LP, this will look similar to using an extreme-EMFI dummy. A smooth-transition LP using the continuous P_stress_t (rather than a threshold-based HighFragility dummy) may therefore add more information.
- **Hamas attack (2023-10-07, Sunday) → P_stress=0.296 on 2023-10-09:** the first trading day after the Hamas attack shows elevated (but sub-0.5) P_stress. The market partially absorbed the shock without transitioning into the systemic regime. This is a useful characterization for the event taxonomy in Phase 8 — a "Localized" rather than "Systemic" event by the HMM classification.
- **P_stress and EMFI are complements, not substitutes:** Corr(P_stress, EMFI) = 0.647 — moderate, not collinear. EMFI captures the severity/magnitude of fragility (continuous, large COVID spike); P_stress captures the certainty of being in the systemic regime (bounded [0,1]). They can be used as separate LP outcomes without multicollinearity concerns.
- **P_stress is near-bimodal, not smoothly continuous:** median P_stress = 0.0001, but 95th pctile = 1.00. It is essentially 0 for ~90% of days and then spikes to 1.0 during crises. This means linear LP on P_stress will behave similarly to a binary indicator LP. Note in the paper.
- **Critical: only 16 shock events land in the HMM systemic-stress regime** (10.4% of 154 shock days in the HMM sample). This is below the ~20-observation threshold for reliable interaction-term identification. **Implication: the binary state-dependent LP conditioned on HMM regime = systemic is underpowered and cannot be the primary specification.** The smooth-transition LP using P_stress_{t-1} as a continuous conditioning variable must be promoted to the primary specification in Phase 7. The HighFragility dummy (EMFI-based, 38 shock events) remains as the binary secondary specification.
- **63% of shocks hit calm markets (regime=0), 27% hit elevated markets (regime=1), 10% hit systemic:** this empirically validates the state-dependence hypothesis — most geopolitical shocks strike when markets are not yet in crisis. The amplification question (θ_k in Phase 7) is about the 37% of shocks that hit non-calm markets.
- **"Other 2017–2019" is the largest stress-day category (54 days, 24.2%)**, ahead of COVID (51 days, 22.9%). The HMM is detecting genuine European market stress beyond COVID and Ukraine — likely the Turkey currency crisis (2018), Italian sovereign bond spread widening (2018–2019), and pre-Brexit uncertainty. This multi-episode coverage strengthens the generalizability argument.
- **143 distinct stress episodes, median duration = 1 calendar day:** the HMM functions as a spike detector rather than a regime identifier. The smooth-transition specification using P_stress_{t-1} is more appropriate than a regime dummy precisely because the stressed regime is so transient.

---

## Phase 6 — Panel Local Projections

**Status:** ✅ Complete  
**Date completed:** 2026-05-18  
**Subproject:** `subprojects/06_panel_lp/`  
**Script:** `subprojects/06_panel_lp/run_panel_lp.py`  
**Tests:** `subprojects/06_panel_lp/test_panel_lp.py` — 40/40 passed

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
- [x] `results/panel_lp/lp_results_{outcome}.csv` — coefficients and SEs for each of 5 outcomes × 21 horizons
- [x] `results/panel_lp/Fig_LP_{outcome}.png` — individual IRF figures (5 files)
- [x] `results/panel_lp/Fig_LP_Combined.png` — 5-panel paper-ready combined figure
- [x] `results/panel_lp/lp_summary.csv` — peak β, peak horizon, pre-trend count per outcome
- [x] `results/panel_lp/manifest.json` — provenance

### Key facts from implementation
- **Sample:** 2,179 dates, 154 shock days (MaxShock > 0)
- **All five outcomes show positive β_k0** (correct direction): EMFI=0.458, TCI=0.396, TailBreadth=0.950, P_stress=0.106, AvgCorr60=0.003
- **No result reaches 5% significance at k=0;** AvgCorr60 marginally significant (p=0.067)
- **Peak TCI response: β_k6=1.064 (p=0.049)** — only 5%-significant result in the baseline LP
- **TailVolBreadth decays cleanly:** peak at k=1 (1.17), near-zero by k=10 (0.05)
- **EMFI and TCI show persistent positive responses** through k=15 (~0.94 pp for TCI), though not significant
- **Pre-trends: clean for P_stress (0/5) and AvgCorr60 (0/5);** TCI has 2/5 significant (rolling-window smoothness artifact — explained in paper)
- **k=-1 degeneracy fixed:** use Y_{t-2} as lag control when k=-1
- Script runtime: ~1.8 seconds

### Additional checks to implement (motivated by Phase 4 observations)

- **Pre-trend verification (negative horizons):** plot β_k for k ∈ {-5,...,-1} and verify they are not significantly different from zero. Given that EMFI is an acute-stress indicator (not a leading predictor), we would expect no pre-trends. If negative-horizon coefficients are significantly positive, the shock measure may be anticipating realized market stress, which would require careful framing.
- **COVID exclusion as priority robustness:** run the LP for all outcomes dropping 2020-03-01 to 2020-12-31. This is the single most important robustness check given the COVID dominance in the upper tail of EMFI. Report this as a primary robustness panel, not a secondary one.
- **EMFI pre-level check on shock dates:** before running the LP, inspect the distribution of EMFI_{t-1} on shock dates versus non-shock dates. If EMFI is already elevated when shocks arrive (i.e., shocks cluster in already-stressed periods), the LP coefficient β_k is identified from periods when markets are both stressed and shocked simultaneously — a confound. This check informs whether a simultaneous-equation concern needs to be addressed.
- **AvgCorr60 as LP outcome (added to list):** separate from the main five outcomes, run the LP with AvgCorr60 as the dependent variable. Given the two-cluster structure (AvgCorr60 and TCI are near-substitutes), if TCI responds significantly but AvgCorr60 does not (or vice versa), it tells us something about the mechanism — VAR-based connectedness vs. simple correlation co-movement. This is a mechanism-identification exercise, not just robustness.
- **Shock intensity bins:** report median, 75th, and 90th percentiles of MaxShock_t on shock days (S > threshold). In the LP, distinguish between moderate and large shocks by interacting S_t with an indicator for S_t > 90th pctile of shock days. This is secondary but useful for the event classification narrative.
- **P_stress as LP outcome — near-binary caveat:** Corr(P_stress, EMFI) = 0.647 (Phase 5 computation). P_stress median is 0.0001 and its 95th pctile is 1.00 — it is near-binary, not a smooth probability. Linear LP on P_stress is still valid (LPM interpretation) but the IRF will essentially show: "does a geopolitical shock raise the probability of being in the systemic-stress regime?" Note the bounded nature in the paper and report the fraction of days where P_stress > 0.5 (226 days, 10.4% of the sample) as context.

---

## Phase 7 — State-Dependent Local Projections

**Status:** ✅ Complete  
**Date completed:** 2026-05-18  
**Subproject:** `subprojects/07_state_dependent_lp/`  
**Script:** `subprojects/07_state_dependent_lp/run_state_lp.py`  
**Tests:** `subprojects/07_state_dependent_lp/test_state_lp.py` — **42/42 pass**

### Objectives
Test whether geopolitical shocks are more destabilizing when they occur in already-fragile markets. This is the paper's **most novel empirical contribution**.

### Specification — PRIMARY: Smooth-Transition LP

Given that only 16 shock events land in the HMM systemic-stress regime (below the power threshold), the primary specification uses P_stress_{t-1} as a **continuous** conditioning variable:

    Y_{t+k} = α + β_k S_t + θ_k (S_t × P_stress_{t-1}) + φ P_stress_{t-1} + controls + ε_{t+k}

where:
- `P_stress_{t-1}` = lagged posterior probability of systemic-stress state from HMM (continuous, [0,1])
- β_k = effect of shock when P_stress_{t-1} = 0 (calm baseline)
- β_k + θ_k × p = total effect when P_stress_{t-1} = p
- Evaluate at p ∈ {0.0, 0.5, 0.9} for three IRF lines in the figure

### Specification — SECONDARY: Binary HighFragility LP

    Y_{t+k} = α + β_k S_t + θ_k (S_t × HighFragility_{t-1}) + φ HighFragility_{t-1} + controls + ε_{t+k}

where:
- `HighFragility_{t-1}` = 1 if EMFI_{t-1} > 75th percentile (38 shock events, workable but marginal)
- `HighConnectedness_{t-1}` = 1 if TCI_{t-1} > 75th percentile (robustness)
- Note: 38 shock events in HighFragility periods is the full LP identification set for θ_k. Confidence intervals will be wide — interpret with care and report the sample composition explicitly.

### Key figures
- **Figure 3 (paper):** Three IRF lines from smooth-transition LP — P_stress=0 (calm), P_stress=0.5 (elevated), P_stress=0.9 (near-systemic) — with confidence bands for each.

### Deliverables
- `results/state_lp/state_lp_results.csv`
- `results/state_lp/Fig_StateLPFragility.png` — normal vs. fragile IRF (primary figure)
- `results/state_lp/Fig_StateLPConnected.png` — normal vs. high-connectedness IRF (robustness)

### Key results (from run_state_lp.py, 2026-05-18)

#### Smooth-transition LP — impact at k=0 (p=0.0 vs p=0.9)

| Outcome | β (p=0) | θ | IRF at p=0 | IRF at p=0.9 | Amplification |
|---------|---------|---|-----------|-------------|---------------|
| EMFI | 0.306 | 1.854 | 0.306 | 1.975 | 6.4× |
| TCI | 0.241 | 2.500 | 0.241 | 2.491 | 10.3× |
| TailVolBreadth | 0.776 | 2.642 | 0.776 | 3.154 | 4.1× |
| P_stress | 0.071 | 0.537 | 0.071 | 0.554 | 7.8× |
| AvgCorr60 | 0.004 | −0.011 | 0.004 | −0.006 | — (no amplification) |

**AvgCorr60 exception:** θ_k ≈ 0 and negative at k=0 for AvgCorr60 — geopolitical shocks do not meaningfully amplify cross-market correlation above its pre-existing elevated level in stressed markets. Disclose this in Section 6.

#### Binary LP — irf_fragile vs irf_normal at k=0

| Outcome | Normal | Fragile | Δ (fragile−normal) |
|---------|--------|---------|---------------------|
| EMFI | −0.204 | 1.012 | +1.217 |
| TCI | −0.048 | 0.791 | +0.839 |
| TailVolBreadth | −0.325 | 2.097 | +2.422 |
| P_stress | −0.060 | 0.247 | +0.306 |

Pattern: irf_normal < 0 for EMFI, TCI, TailVolBreadth at k=0 in the binary LP. This is not a contradiction — in non-fragile markets, geopolitical shocks on average do not produce detectable immediate stress; the response is entirely concentrated in pre-fragile markets.

#### COVID exclusion (nocovid variant)
- Sample reduction: 2178 → 1960 obs (−218, ≈ one year of trading)
- β at k=0 remains positive for EMFI (0.233), TCI (0.227), TailVolBreadth (0.623), AvgCorr60 (0.001)
- **P_stress nocovid β at k=0 = −0.029 (slightly negative):** the P_stress baseline shock impact is largely COVID-driven; the amplification (θ) carries the signal in the non-COVID subsample. This is an honest finding to report — the baseline P_stress result is concentrated in the COVID episode.

#### Composition report (HighFragility by year)

| Year | Shock days | HF days | % HF |
|------|-----------|---------|-------|
| 2017 | 13 | 0 | 0% |
| 2018 | 14 | 4 | 28.6% |
| 2019 | 10 | 0 | 0% |
| 2020 | 16 | 10 | 62.5% |
| 2021 | 11 | 0 | 0% |
| 2022 | 21 | 11 | 52.4% |
| 2023 | 21 | 1 | 4.8% |
| 2024 | 25 | 3 | 12.0% |
| 2025 | 23 | 7 | 30.4% |
| **TOTAL** | **154** | **36** | **23.4%** |

Key: 2022 (Ukraine year) has the second-highest HF concentration (52.4%), confirming the state-dependent identification is not solely COVID-driven.

### Analytical observations for paper writing

1. **State amplification is the paper's headline finding.** At p=0.9 (near-systemic pre-shock stress), EMFI response is 6.4× larger than at p=0 (calm baseline). TCI amplification is 10.3×. These are large and economically meaningful.

2. **AvgCorr60 is the notable exception.** θ_k ≈ 0 at k=0 for AvgCorr60 — correlations are already elevated in stressed markets and shocks do not amplify them further. This is an informative null: the shock-amplification mechanism operates through volatility and connectedness channels, not through incremental correlation.

3. **Binary LP confirms: shock impact is regime-specific.** In non-fragile markets, the average shock response is near-zero or slightly negative for several outcomes. The entire positive effect is concentrated in the HighFragility subsample. This is a cleaner story than a uniform (if larger) positive effect.

4. **COVID exclusion: main result survives, P_stress baseline does not.** EMFI, TCI, TailVolBreadth β at k=0 all remain positive in the nocovid subsample. P_stress β at k=0 flips to −0.029 — the positive baseline for P_stress is COVID-specific. Disclose this: the state-dependent amplification (θ) remains, but the unconditional P_stress impact is a COVID artifact.

5. **2022 (Ukraine) contributes 11/36 HF shock events (30.6%).** This gives meaningful multi-episode identification for the binary LP alongside COVID (10 events). The 2022 Ukraine contribution reduces the COVID-only critique for θ_k in the binary specification.

6. **Pre-trends are clean at p=0 (calm).** At p=0, pre-trend betas are mixed in sign and small (range: −0.14 to +0.29), satisfying the identification condition that shocks are conditionally exogenous. The p=0.9 pre-trend has wider CIs due to sparse high-P_stress observation count.

### Deliverables (actual outputs)
- `results/state_lp/state_lp_smooth_{outcome}.csv` — smooth-transition LP (5 outcomes)
- `results/state_lp/state_lp_smooth_{outcome}_nocovid.csv` — COVID-excluded variant (5 outcomes)
- `results/state_lp/state_lp_binary_{outcome}.csv` — binary HF LP (5 outcomes)
- `results/state_lp/Fig_StateLPSmooth_{outcome}.png` — smooth-transition IRF figure (5 figures)
- `results/state_lp/Fig_StateLPSmooth_{outcome}_nocovid.png` — COVID-excluded figures (5 figures)
- `results/state_lp/Fig_StateLPBinary_{outcome}.png` — binary LP figure (5 figures)
- `results/state_lp/Fig_StateLPSmooth_Combined.png` — **Figure 3 (paper)**, combined 5-panel
- `results/state_lp/composition_report.csv` — HF shock composition by year
- `results/state_lp/manifest.json`

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
- Iterate over all declustered shock event dates (154 events in the HMM sample).
- For each event: compute EMFI change (mean of t+1 to t+5 minus mean of t-5 to t-1), and max P_stress in window [t, t+5].
- Assign category using the definitions above.
- For each category: compute average volatility response, average TCI response, average duration of elevated stress.

### Empirical anchors from Phase 5 (to verify in Phase 8)
- **COVID peak (2020-03-12):** P_stress=1.00, EMFI=15.2 → expected classification: Systemic
- **Ukraine invasion (2022-02-24):** P_stress=1.00 → expected: Systemic
- **Hamas attack (2023-10-09, nearest trading day):** P_stress=0.296 → expected: Localized
- **Base rate:** 63% of shock events hit calm markets (P_stress≈0) → expected majority of events to be "Absorbed" or "Localized"
- **Overall distribution expected:** few Systemic events (likely 5–20), majority Absorbed/Localized

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

| Check | Description | Priority |
|-------|------------|----------|
| COVID exclusion | Drop 2020-03 to 2020-12 from LP and state-dependent LP | **Critical** |
| HighFragility COVID share | Verify state-dependent result survives COVID exclusion | **Critical** |
| TCI → AvgCorr60 substitution in EMFI | Replace TCI with AvgCorr60 in PCA inputs (3-component EMFI); re-run LP and state LP | **High** |
| "AcuteEMFI" from VolStress + TailBreadth only | PCA on only the two acute-volatility indicators; compare LP results | **High** |
| Alternative EMFI weights | Equal-weight composite (vs PCA) | **High** |
| Alternative fragility threshold | 66th and 90th percentile fragility states (vs 75th) | **High** |
| P_stress continuous interaction | Smooth-transition LP using P_stress_{t-1} as continuous conditioning variable | **High** |
| Alternative connectedness | Correlation-network TCI (avg pairwise correlation) instead of VAR-based | Medium |
| Alternative shock aggregation | AvgShock vs MaxShock vs BreadthShock | Medium |
| TVP-VAR connectedness | Time-varying parameter VAR (Antonakakis et al.) as alternative to rolling DY | Medium |
| Ukraine exclusion | Drop 2022-02 to 2022-06 | Medium |
| Future-shock placebo | Replace S_t with S_{t+30} — must show no significant effects | Medium |
| Alternative TCI window | W=60 and W=150 in EMFI instead of W=100 | Low |
| Bootstrapped inference | Block bootstrap CIs for LP coefficients | Low |

### Rationale for new priority robustness checks (motivated by Phase 4 analytical observations)

**TCI → AvgCorr60 substitution:** given the 0.797 correlation between TCI and AvgCorr60, TCI may not add empirical content beyond what AvgCorr60 already captures. If LP results are unchanged when using the simpler 3-component EMFI (VolStress, TailBreadth, AvgCorr60), this argues that the two-cluster structure is the robust feature, not the VAR-FEVD measurement technology. Conversely, if TCI-based EMFI gives different LP results, TCI is earning its place in the composite.

**"AcuteEMFI" from VolStress + TailBreadth only:** this tests whether the co-movement cluster (AvgCorr60 + TCI) is necessary for the LP results. The VolStress+TailBreadth cluster captures acute realized stress; the AcuteEMFI would be a pure volatility-breadth index. If LP results with AcuteEMFI match those with the full EMFI, the connectedness dimension does not add explanatory power for the shock→fragility transmission, which is an interesting null result in itself.

**COVID exclusion as critical:** this is not a standard robustness check here — it is a potential falsification. Given that COVID accounts for the dominant mass of the EMFI upper tail and most HighFragility shock days, a statistically significant state-dependent result that disappears upon COVID exclusion would indicate the finding is not generalizable beyond that single episode.

### Deliverables
- `results/robustness/robustness_summary.csv` — all checks with β_k at key horizons
- `results/robustness/Fig_Robustness_LP.png` — overlay of baseline + robustness IRFs
- `results/robustness/Fig_Robustness_StateLPComparison.png` — state-dependent LP baseline vs TCI-substitution vs AcuteEMFI

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
- [x] Data section draft (after Phase 1) — drafted 2026-05-18
- [x] Sections 3–4 draft (after Phases 2–5) — drafted 2026-05-18
- [x] Section 5 draft (HMM, after Phase 5) — drafted 2026-05-18
- [x] Section 6 baseline LP prose (after Phase 6) — drafted 2026-05-18
- [x] Section 6 state-dependent LP prose (after Phase 7) — drafted 2026-05-18
- [ ] Full draft for co-author review
- [ ] GFJ submission package

### Writing notes from implementation (Phases 2–4 analytical observations)

**Section 3 — Market-Based Financial Stability Measures:**
- When describing the EMFI, explicitly discuss the **two-cluster structure**: VolStress+TailBreadth (cluster 1, within-corr=0.784) vs. AvgCorr60+TCI (cluster 2, within-corr=0.797), with cross-cluster correlations of only 0.16–0.38. Explain that PCA bridges these two dimensions: PC1 loads equally on all four components (0.45–0.53), while PC2 captures the contrast. Acknowledge that 58.3% explained variance reflects a genuine two-dimensional fragility space, not a near-perfect common factor.
- **TCI vs. AvgCorr60 discussion**: note that TCI (VAR-FEVD) and AvgCorr60 (simple rolling correlation) correlate at 0.797. The theoretical superiority of TCI (order-invariance, captures indirect spillover paths) is argued, but the empirical overlap is explicitly acknowledged and the TCI-substitution robustness check is referenced.
- **EMFI nature**: be explicit that EMFI is a contemporaneous/lagging acute-stress indicator. It does not build up gradually before crises — it spikes during them. Contrast with VIX-type implied volatility measures if space allows. This framing matters for interpreting the state-dependent LP: HighFragility captures *ongoing* stress episodes, not *pre-fragility*.
- **HighFragility composition disclosure**: in a table or footnote, report the breakdown of HighFragility days by year/episode. COVID 2020 will dominate the upper tail; this must be stated, not buried.

**Section 4 — HMM Market-Implied Stress Regimes:**
- Explicitly describe the HMM as a spike detector rather than a regime classifier: 143 distinct stress episodes, median duration = 1 calendar day. Contrast with traditional HMM applications where regimes persist for months — this is a fundamentally different use case.
- Report the episode breakdown of systemic-stress days: COVID=22.9%, Ukraine=13.0%, "Other 2017–2019"=24.2%, "Other 2024–2025"=17.0%. The dominant category is actually non-labeled European stress events, which argues for generalizability beyond any single crisis.
- Report shock-regime d