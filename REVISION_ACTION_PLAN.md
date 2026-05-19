# Revision Action Plan — Paper_GFJ
# "When Geopolitical Shocks Become Systemic"

**Based on:** Review1.pdf (co-author/advisor pre-submission review, 14 pages)  
**Created:** 2026-05-18  
**Last updated:** 2026-05-18 (Phase R2 complete)  
**Target journal:** Global Finance Journal (GFJ)  
**Overall reviewer verdict:** "Promising paper, potentially suitable for GFJ after substantial revision, but not submission-ready in the current form."

---

## Executive Summary

The reviewer confirms the paper is conceptually on the right path and identifies the **state-dependent amplification** finding (6.4× EMFI, 10.3× TCI) as the paper's strongest and most publishable contribution. Ten submission-blocking issues are identified, along with nine internal inconsistencies and several methodological upgrades. This revision plan addresses all of them in a structured sequence of phases ordered by dependency and priority.

**Strategic reframe (reviewer recommended):**
> *Geopolitical shocks are not systematically destabilizing on average; they become systemic only when they interact with pre-existing market fragility.*

This is the paper's true claim. The weak unconditional LP result is not a failure — it is the empirical setup for the state-dependence finding. The paper must be rewritten to embrace this logic explicitly.

**Key note on citation style:**  
The "companion paper" / "companion BIR paper" / "companion study" language must be eliminated throughout. The BIR paper is cited as a reference like any other: **Lupu et al. (2026)**. The GFJ paper stands on its own; it reuses the shock series as an input, citing the source. No special relationship is claimed in the text.

---

## Current Paper Status

- Full draft: 23 pages, compiles cleanly
- All phases R0–R9 complete with passing tests
- Git commit: `832cabb` (Phase 10 complete)
- Reviewer: co-author/advisor, thorough pre-submission review
- Data are internally consistent: the 19-country equity panel and shock series both use {Austria, Belgium, Bulgaria, Croatia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Netherlands, Norway, Poland, Portugal, Romania, Spain, Sweden, United Kingdom} — but **Table 1 in the LaTeX manuscript incorrectly lists Czech Republic, Denmark, and Switzerland** instead of Bulgaria, Croatia, and United Kingdom. This is a text error, not a data error.

---

## Revision Phases Overview

| Phase | Title | Priority | Type | Est. Effort |
|-------|-------|----------|------|-------------|
| R0 | Text corrections and internal consistency fixes | ✅ **Complete** | Text only | — |
| R1 | References, citations, and appendices | ✅ **Complete** | Text + bib | — |
| R2 | COVID reclassification as non-geopolitical event | ✅ **Complete** | Code + text | — |
| R3 | TCI strengthening (window and robustness) | ✅ **Complete** | Code + text | — |
| R4 | HMM enhancements (terminology + robustness) | ✅ **Complete** | Code + text | — |
| R5 | Formal inference for state-dependent amplification θ | 🟠 High | Code + text | 3–4 hours |
| R6 | Predictive event classification (logit/probit) | 🟠 High | Code + text | 2–3 hours |
| R7 | Figure improvements | 🟡 Medium | Code | 1–2 hours |
| R8 | Paper rewrite: framing, abstract, policy claims | 🟡 Medium | Text | 3–4 hours |
| R9 | Submission package | 🟡 Medium | Text | 1–2 hours |

**Sequencing dependencies:**
```
R0 (text) → R1 (references)
R2 (COVID) must precede R8 (paper rewrite)
R3 (TCI) must precede R4 (HMM uses TCI)
R5 (inference) is independent
R6 (predictive) is independent
R7 (figures) is independent
R0 + R1 + R2 + R3 + R4 + R5 + R6 + R7 → R8 → R9
```

---

## Phase R0 — Text Corrections and Internal Consistency

**Status:** ✅ Complete (2026-05-18)  
**Git commit:** `d962572` — R0: text consistency fixes  
**Priority:** 🔴 Critical (submission-blocking)  
**Type:** Text-only edits in `Paper_LaTeX/main.tex`  
**Estimated effort:** 1–2 hours

### R0.1 — Fix Table 1: Country list

**Problem:** Table 1 (lines ~251–280 of main.tex) lists Czech Republic (PX), Denmark (OMXC25), and Switzerland (SMI). The actual data contains Bulgaria, Croatia, and United Kingdom.

**Fix:** Replace in Table 1:
- `Czech Republic & PX & 2` → `Bulgaria & SOFIX & 2`
- `Denmark & OMXC25 & 1` → `Croatia & CROBEX & 2`
- `Switzerland & SMI & 1` → `United Kingdom & FTSE 100 & 1`

Verify cluster assignments match `data/clusters.csv`. Add a sentence to the data section clarifying that the 19-country sample was constructed to maximize GDELT coverage and equity data availability, and differs from some other European financial market studies.

### R0.2 — Fix Hamas contradiction

**Problem:** Abstract (line 73) says "the 2023 Hamas–Israel shock is absorbed without regime transition." Introduction (line 157) says it "produced elevated (P_stress = 0.30) but not systemic market conditions." But Section 6.2/7.2 classifies Hamas as **Systemic** (P_stress_peak = 1.0 on day+1 within the 5-day post-window).

**Fix:** The empirical result (Systemic, per the classification algorithm) is correct. Fix the abstract and introduction to be consistent:
- Abstract: replace "the 2023 Hamas–Israel shock is absorbed without regime transition" with "the 2023 Hamas–Israel attack produces a delayed one-day stress-state transition before abating"
- Introduction: revise the characterization of Hamas accordingly
- Decide on final narrative: Hamas = Systemic (delayed, one-day spike), documented as an interesting case of delayed stress crystallization

### R0.3 — Fix conclusion significance overclaim

**Problem:** The conclusion (and possibly Section 8) states that β_k0(EMFI) = 0.458 is "statistically significant." The results section correctly reports p = 0.174 — not significant at any conventional level.

**Fix:** Everywhere the conclusion calls β_k0 significant, replace with accurate language. Suggested replacement:
> "The baseline local projection yields a positive impact on EMFI of 0.458 standard deviations at impact (k=0), though this estimate is statistically imprecise (p = 0.174). This imprecision is itself informative: it reflects the fact that most geopolitical shocks are absorbed without measurable market fragility response."

This framing is both honest and strategically correct — the weak unconditional result motivates the state-dependence analysis.

### R0.4 — Clarify shock count mapping (278 / 163 / 154)

**Problem:** The paper mentions three different shock counts without explaining how they relate:
- 278 raw shock events (all calendar days in the conflict intensity series where EVT/FDR threshold is exceeded)
- 163 shock-positive trading days (days with max_shock > 0 after forward-filling weekends to next trading day)
- 154 shock events used in the LP/taxonomy

**Fix:** Add a clear footnote or data section sentence explaining the full mapping:
> "The raw shock series contains 278 shock events on calendar days. Of these, 181 fall on weekends or holidays and are forward-filled to the next trading day. After forward-filling, 163 distinct trading days carry non-zero max-shock intensity. Nine of these trading days fall outside the EMFI/HMM estimation window (which begins in May 2017 due to TCI/EMFI warm-up), leaving 154 shock-day observations used in the local projections and event taxonomy."

### R0.5 — Eliminate "companion paper" language; cite as Lupu et al. (2026)

**Problem:** The paper references the BIR paper as "companion paper" / "companion BIR paper" (3 occurrences in main.tex). This language implies a co-submission or multi-paper package, which can create issues with journals that accept single-paper submissions.

**Fix:** Replace all three occurrences:
1. Line 113: "geopolitical conflict shock series from a companion paper" → "geopolitical conflict shock series developed by \citet{lupu2026bir}"
2. Line 244: "inherited from the companion BIR paper" → "inherited from \citet{lupu2026bir}"
3. Line 294: "contribution of that companion paper" → "contribution of \citet{lupu2026bir}"

Also add a footnote at the first mention: "Lupu et al. (2026) is an independent working paper on geopolitical shock construction and validation. The present paper uses their validated shock series as an external trigger input, making no claim on shock-construction methodology."

### R0.6 — Trim CAMEO/Goldstein vs GDELT Document 2.0 description

**Problem:** The paper describes the conflict intensity index using "CAMEO action codes weighted by the Goldstein scale" but the BIR paper uses GDELT Document 2.0 conflict coverage. This is methodologically inconsistent and exposes the paper to reviewer criticism.

**Fix:** Replace the current shock-construction description paragraph with a short, self-contained summary that makes no CAMEO/Goldstein claim:
> "We use the validated conflict-shock series from \citet{lupu2026bir}. The series is derived from country-level conflict-related news coverage in GDELT Document 2.0, transformed into tail-based shock intensities via extreme value theory and multiple-testing adjustments. For each country $i$ and trading day $t$, the shock intensity $S_{i,t} = -\log(q_{i,t})$ where $q_{i,t}$ is a BH-FDR adjusted exceedance probability. Days with raw tail probability $p_{i,t} < 0.005$ are classified as shock events. The present paper does not contribute to shock construction; it takes the validated series as given."

Remove any other references to CAMEO codes, Goldstein weights, or detailed EVT methodology — those belong in the BIR paper.

**Deliverables — Phase R0:**
- [x] Table 1 corrected (Bulgaria/SOFIX Cl.2, Croatia/CROBEX Cl.2, UK/FTSE100 Cl.1 replacing Czech Republic, Denmark, Switzerland)
- [x] Abstract Hamas sentence revised ("one-day delayed systemic stress episode")
- [x] Introduction Finding 4 Hamas revised (consistent with Systemic classification, explains day+1 crystallisation)
- [x] Conclusion significance overclaim corrected ("statistically imprecise, p=0.174"; added interpretive sentence)
- [x] Shock count footnote added (explicit 278→163→154 mapping with warm-up explanation)
- [x] All "companion paper" / "companion BIR paper" language removed (0 occurrences)
- [x] Introduction citation fixed (malformed `\citep[Lupu et al.\ 2026,][]` → `\citet{lupu2026bir}`)
- [x] Footnote added at first shock-series mention (paper takes series as given; construction is Lupu et al. 2026)
- [x] Shock-construction paragraph trimmed; CAMEO/Goldstein removed; GDELT Doc 2.0 / GPD / FDR description kept
- [x] `\bibliography{references}` + `\bibliographystyle{plainnat}` added (were missing)
- [x] LaTeX compiles cleanly: 23 pages, 0 LaTeX errors (undefined-citation warnings remain → Phase R1)
- [x] Git commit: `d962572` — "R0: text consistency fixes — country table, Hamas, significance, shock counts, companion→lupu2026bir"

---

## Phase R1 — References, Citations, and Appendices

**Status:** ✅ Complete (2026-05-18)  
**Git commit:** (next commit) — R1: all citations resolved, five appendices drafted  
**Priority:** 🔴 Critical (automatic desk-rejection risk)  
**Type:** Text + bibliography  
**Estimated effort:** 2–3 hours

### R1.1 — Fix missing/unresolved citations

**Problem:** The reviewer reports many `[? ? ?]` and `? ]` placeholder citations. Current grep of main.tex shows these may not appear as literal `?` characters but as unresolved `\cite{}` keys. Run a full BibTeX compilation and collect all undefined-reference warnings.

**Action:**
1. Compile `main.tex` with `pdflatex` + `bibtex` and capture all "undefined citation" warnings
2. For each undefined citation key, either: (a) find the correct BibTeX entry and add it to `references.bib`, or (b) find an equivalent citation already in the bib file
3. Verify the following key citations are fully specified in `references.bib`:
   - Caldara and Iacoviello (2022) — Measuring Geopolitical Risk
   - Diebold and Yilmaz (2012) — connectedness
   - Pesaran and Shin (1998) — GFEVD
   - Jordà (2005) — local projections
   - Newey and West (1987) — HAC standard errors
   - Hamilton (1989) — Markov switching/HMM
   - Adrian and Brunnermeier (2016 or 2011) — CoVaR
   - Auerbach and Gorodnichenko — state-dependent multipliers
   - Stock and Watson (2018) — SVAR-IV
   - Berkman, Jacobsen, Lee (2011) — war and financial markets
   - Glick and Taylor (2010) — war and markets
   - Liu and Zhang (2021) — state-dependent LP
   - Recent geopolitical risk and connectedness papers (2020–2025)

### R1.2 — Add Lupu et al. (2026) to references.bib

**Action:** Add a proper `@unpublished` or `@techreport` entry:
```bibtex
@unpublished{lupu2026bir,
  author    = {Lupu, Radu and [co-authors]},
  title     = {[Full title of BIR paper]},
  note      = {Working paper, submitted to Borsa Istanbul Review},
  year      = {2026}
}
```
Fill in co-authors and full title from the BIR paper's title page.

### R1.3 — Complete all five appendices

**Problem:** Five appendix sections in `main.tex` contain only `% TODO` placeholders. Empty appendices are not submission-ready.

**Appendices to write:**
1. **Appendix A — Data Details:** Country list with full index names and date coverage; data sources; treatment of weekend/holiday shock alignment (181 of 278 events); 15 dropped return dates; provenance statement
2. **Appendix B — EMFI Construction Details:** Full PCA specification; loading table (all 4 components, all PCs); variance explained by PC1–PC4; sign-flip rule; comparison of PCA vs equal-weight EMFI; full-sample vs rolling standardization discussion
3. **Appendix C — HMM Parameter Estimates:** Transition matrix (full 3×3); emission means and covariances per state; log-likelihood; number of restarts; 2-state HMM comparison parameters; start probabilities
4. **Appendix D — Additional LP Results:** LP IRF figures for all 5 outcomes separately (or individual horizon tables); pre-trend coefficients (k = −5 to −1) for all outcomes; BIC-selection for VAR order
5. **Appendix E — Robustness Tables:** Full robustness table with all 12 checks (R00–R12) for all outcomes and all key horizons (k = 0, 1, 5, 10, 20); state LP theta for all EMFI-construction variants

**Deliverables — Phase R1:**
- [x] Full BibTeX compilation run; all undefined-citation warnings listed (19 keys)
- [x] All 19 undefined citations resolved in references.bib (16 new entries + 3 alias duplicates)
- [x] `lupu2026bir` entry added to references.bib as working paper
- [x] Appendix A — Data Details written (~400 words + 19-country table)
- [x] Appendix B — EMFI Details written (~300 words + PCA loading table)
- [x] Appendix C — HMM Parameters written (~250 words + transition matrix + mean vectors table)
- [x] Appendix D — Additional LP Results written (~300 words + component LP table)
- [x] Appendix E — Robustness Tables written (R00–R12, k = 0, 1, 5, 10)
- [x] LaTeX compiles cleanly: 28 pages, 0 errors, 0 undefined-citation warnings
- [x] Git commit: "R1: 19 citations resolved, five appendices drafted"

**Notes:**
- Two bib entries (jordan2020state, liuzhang2021) are approximate placeholders — exact paper details need verification before final submission
- Float layout warnings for oversized figures remain (addressed in Phase R7)

---

## Phase R2 — COVID Reclassification as Non-Geopolitical Event

**Status:** ✅ Complete (2026-05-18)  
**Git commit:** `c9555f7` — R2: COVID reclassified as non-geopolitical; new baseline LP  
**Priority:** 🔴 Critical  
**Type:** Code + text  
**Estimated effort:** 3–4 hours  
**Script:** `subprojects/10_covid_reclassify/run_covid_reclassify.py`

### The Problem

The paper currently treats COVID-19 shock days as geopolitical conflict shocks. The reviewer flags this as a serious credibility risk:
> "COVID-19 is an enormous systemic financial stress event, but it is not naturally a geopolitical conflict shock. If your geopolitical shock series flags COVID-related days, that is itself a problem: it may mean the shock series is detecting general crisis news rather than geopolitical conflict."

COVID days are currently present in:
1. The shock treatment series (S_t > 0 on COVID-related news days)
2. The LP regressions as treatment observations
3. The event taxonomy (classified as Systemic)
4. The state-dependent LP identification set (HF events)

### Decision

**Primary specification:** Reclassify COVID. COVID shock days (2020-02-01 to 2020-06-30) remain in the **sample** for EMFI/HMM estimation (market-stress outcomes) but are **excluded from the treatment series** (S_t is set to 0 or dropped for the LP regression). This is not merely a robustness exclusion — it becomes the baseline.

**COVID's role becomes:** A validation benchmark for EMFI and HMM (they correctly detect COVID as a systemic stress episode), but not a geopolitical treatment event in the LP.

**Ukraine** (2022-02-24 onward) remains the primary geopolitical systemic event.

### Implementation

1. In `run_panel_lp.py` and `run_state_lp.py`, add a `covid_shock_excl` flag that zeroes out `max_shock` on days where the shock is plausibly COVID-driven. Define COVID shock days as any shock day between 2020-01-01 and 2020-12-31. The outcomes (EMFI etc.) remain in the sample; only the treatment is modified.
2. Re-run Panel LP (Phase 6) with COVID shock days excluded from treatment → new primary baseline β
3. Re-run State LP (Phase 7) with COVID shock days excluded → new primary θ
4. Keep the full-sample version as a robustness check (not vice versa)
5. Update the event taxonomy: COVID shock days are flagged as "market-stress events, COVID reclassified" — not treated as geopolitical treatment events. This reduces the 154 LP shock days.
6. Report the before/after: how many shock days are COVID-period? What happens to β and θ?
7. Update the event taxonomy text: COVID-19 is presented as a benchmark stress event that EMFI/HMM correctly detect as systemic, while geopolitical shocks (Ukraine, Hamas, etc.) are the paper's treatment events.

### Expected results

From Phase 7 COVID-exclusion runs already completed: β_k0(EMFI) drops from 0.458 to 0.370 (survives, 81% of baseline). The state-dependent θ is also expected to survive because 2022 Ukraine (11 HF events) provides meaningful identification independent of COVID.

### Writing implications

The Introduction must be revised: remove COVID as an opening example of a geopolitical event. Instead: "From the Russian invasion of Ukraine in February 2022 to the Hamas–Israel conflict of October 2023, geopolitical events periodically disrupt global equity markets." COVID is mentioned separately in the EMFI validation context: "We validate EMFI and HMM against well-known stress episodes, including the COVID-19 market crash of March 2020..."

**Deliverables — Phase R2:**
- [x] `subprojects/10_covid_reclassify/run_covid_reclassify.py` written and run
- [x] New primary LP results (COVID shock days excluded from treatment) saved to `results/panel_lp_nocovid/`
- [x] New primary state LP results saved to `results/state_lp_nocovid/`
- [x] Updated event taxonomy (COVID shock days flagged separately, `covid_reclassify/event_taxonomy_nocovid.csv`)
- [x] `test_covid_reclassify.py` test suite, 24/24 passing
- [x] Table comparing full-sample vs COVID-excl results (`covid_reclassify/comparison_table.csv`)
- [x] LaTeX updated: new baseline β_k0=0.245, COVID→Ukraine framing, robustness section swapped; compiles 29 pages, 0 errors
- [x] Git commit: "R2: COVID reclassified as non-geopolitical; new baseline LP"

**Key results:**
- 16 COVID-period shock days zeroed in treatment (2020-01-01 – 2020-12-31); sample unchanged (2,179 obs)
- Panel LP β_k0(EMFI) = 0.245 (was 0.458); outcomes remain in sample for EMFI/HMM validation
- State LP: β=0.148, θ=1.455 for EMFI; amplification ratio 9.9× (fragile vs calm), up from 6.4× baseline
- Full-sample version demoted to robustness check (Appendix E)

---

## Phase R3 — TCI Strengthening

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** `177bffcb` — R3: TCI strengthened — W=250 alternative, simpler network metrics  
**Priority:** 🟠 High  
**Type:** Code + text  
**Estimated effort:** 3–4 hours  
**Script:** New `subprojects/11_tci_robustness/run_tci_robustness.py`

### The Problem

The reviewer flags that a 19-variable rolling VAR with W=100 days is likely overparameterized:
> "A 19-variable VAR with even one lag has many parameters relative to only 100 observations per window. With p=1, each equation has 19 lag coefficients plus intercept; across 19 equations, that is a very large parameter set for a short window."

This is a technically valid concern that a GFJ referee could use to reject the paper.

### Solutions

**Option A (preferred): Add W=250 as alternative primary window**
- Rerun TCI with W=250 (250 trading days ≈ 1 calendar year) as the primary specification alongside W=100
- Report both: present W=250 as the better-powered specification, W=100 as a short-window alternative
- If EMFI constructed using W=250 TCI gives similar state LP results, show this as robustness
- Note: W=250 window reduces the early sample (first 250 days are NaN), but this is acceptable

**Option B (complementary): Add correlation-based network metrics as parallel measures**
- Already partially implemented (AvgCorr60 is in EMFI)
- Add: first eigenvalue of the rolling correlation matrix (connectivity measure); network density at 0.5 correlation threshold; minimum spanning tree total length
- If TCI and these simpler metrics give the same LP results, the overparameterization concern is answered: "the TCI finding is robust to simpler connectedness measures"

**Option C (if A and B are insufficient): LASSO-VAR**
- Estimate a penalized VAR using the `sklearn` elastic net; extract FEVD from penalized coefficients
- More robust to short windows but methodologically complex — reserve for R&R if needed

### Implementation plan

1. Add `W=250` to the existing TCI computation (already coded, just need to verify output)
2. Add rolling first-eigenvalue (`lambda1_t`) and network density (`density_t`, threshold=0.5) computation
3. Rebuild EMFI using W=250 TCI instead of W=100 TCI (new file `emfi_w250.csv`)
4. Re-run state LP with W=250 EMFI → compare θ_k
5. Report sensitivity table: TCI window W=60/100/150/250 → LP β and state LP θ
6. Update Section 3.2 (TCI description) to note that W=100 was chosen following standard practice (Diebold–Yilmaz 2012), and longer windows (W=250) give near-identical results (robustness reported in Appendix E)

**Deliverables — Phase R3:**
- [x] `tci_w250.csv` computed and saved (`results/tci_robustness/tci_w250.csv`)
- [x] EMFI recomputed for all windows W=60/100/150/200/250 (`results/tci_robustness/emfi_variants.csv`)
- [x] State LP rerun for all window variants — θ positive across all (0.68–0.96)
- [x] Rolling eigenvalue (λ₁/N) and network density computed (`results/tci_robustness/network_metrics.csv`)
- [x] Sensitivity table W=60/100/150/200/250: β_k0=0.15–0.21, θ_k0=0.68–0.96, ratio 4.8–9.9× (all positive)
- [x] `test_tci_robustness.py` test suite, 29/29 passing
- [x] Section 3.2 updated: overparameterization response + W=250 result + network metrics
- [x] Section 7 robustness paragraph updated: full W sensitivity range reported
- [x] Appendix E: new `tab:tci_window_sensitivity` added with 5-window results
- [x] LaTeX compiles cleanly: 29 pages, 0 errors, 0 undefined references
- [x] Git commit: "R3: TCI strengthened — W=250 alternative, simpler network metrics"

**Key results:**
- W=250 TCI (VAR(1)): mean=71.1% vs W=100 mean=70.5% — nearly identical dynamics
- β_k0 stable across windows (CV=14%): ranges 0.15 (W=60) to 0.21 (W=150)
- θ_k0 positive and substantial across all windows: 0.68 (W=250) to 0.96 (W=60)
- Amplification ratios: 4.8× to 9.9× — state dependence robust to window choice
- Network metrics (λ₁, density) spike at COVID/Ukraine confirming VAR not over-fit

---

## Phase R4 — HMM Enhancements and Terminology

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** (next commit) — R4: HMM terminology fixed, additional robustness specs, filtered probabilities  
**Priority:** 🟠 High  
**Type:** Code + text  
**Estimated effort:** 3–4 hours  
**Script:** New `subprojects/12_hmm_robustness/run_hmm_robustness.py`

### R4.1 — Terminology change throughout

**Problem:** The paper uses "systemic stress regime" and "stress regime" language, but the HMM identifies episodes averaging 1.6 trading days — too short to be a "regime."

**Fix:** Replace consistently throughout main.tex:
- "systemic stress regime" → "market-implied systemic stress state"
- "stress regime" → "stress episode" or "stress state"
- "regime probability" → "stress-state probability"
- "regime classification" → "stress-state classification"
- Keep "state" for HMM-specific language; use "episode" for temporal sequences

### R4.2 — Additional HMM robustness specifications

Implement and report the following HMM variants (adding to the existing 2-state HMM robustness):

1. **2-state HMM** (already estimated in Phase 5; verify results are accessible and reported in Appendix C)
2. **Student-t HMM:** Replace Gaussian emission with Student-t (use `hmmlearn` or manual EM with t-distribution). Robust to COVID-level outliers.
3. **COVID-excluded HMM:** Estimate HMM dropping 2020-01-01 to 2020-12-31 from training data; apply to full sample. This tests whether the stress-state classifications are COVID-driven or genuinely multi-episode.
4. **Pre-2020 trained, post-2020 applied:** Estimate HMM on 2017-2020-01-01 only; apply Viterbi path to 2020–2025. Fully out-of-sample.
5. **Diagonal vs full covariance HMM:** Compare diagonal covariance (default in `hmmlearn.GaussianHMM`) vs full covariance (`covariance_type='full'`).

For each variant, compute: (a) state count classification agreement with baseline, (b) Cohen's kappa with baseline Viterbi path, (c) correlation of P_stress with baseline P_stress.

### R4.3 — Filtered vs smoothed probabilities

**Problem:** The current P_stress_t is based on smoothed posterior probabilities (using the full forward-backward algorithm, which uses future observations). For any real-time or policy claim, filtered probabilities (using information up to time t only) are required.

**Fix:**
1. Compute filtered probabilities from the HMM's forward pass only (`hmmlearn` provides `predict_proba()` which gives smoothed; implement the forward algorithm manually for filtered)
2. Add `P_stress_filtered_t` to `hmm_daily.csv`
3. Report correlation between filtered and smoothed P_stress
4. In the paper: note that current EMFI/P_stress use ex-post standardization and smoothed probabilities, making them outcome measures rather than real-time detectors. Filtered probabilities are provided in Appendix C for the HMM and discussed in the policy section.
5. This also addresses the reviewer's concern about ex-post EMFI claims (see Phase R8)

**Deliverables — Phase R4:**
- [x] All "systemic stress regime" → "stress state/episode" replacements in main.tex (21 targeted replacements)
- [x] COVID-excluded HMM estimated; kappa=0.939 vs baseline
- [x] Pre-2020 trained HMM computed (out-of-sample); kappa=0.879
- [x] Diagonal vs full covariance HMM compared; kappa=0.971, ρ=0.958
- [x] 2-state HMM; kappa=0.629 (binary agreement), ρ=0.733
- [x] `P_stress_filtered` computed and added to `hmm_daily.csv`; ρ(filtered,smoothed)=0.994
- [x] `test_hmm_robustness.py` test suite, 22/22 passing
- [x] Appendix C updated: C.4 HMM robustness table, C.5 filtered vs smoothed discussion
- [x] Section 4 footnote: filtered proba correlation + reference to App C.4–C.5
- [x] LaTeX compiles cleanly: 31 pages, 0 errors
- [x] Git commit: "R4: HMM terminology fixed, additional robustness specs, filtered probabilities"

**Key results:**
- Filtered vs smoothed P_stress: ρ=0.994 (look-ahead bias economically negligible)
- Diagonal cov HMM: κ=0.971 — off-diagonal covariance provides minimal identification gain
- COVID-excluded training: κ=0.939 — stress classifications not COVID-driven artefacts
- Pre-2020 OOS: κ=0.879 — stable market structure confirmed out-of-sample
- "Regime" terminology replaced throughout with "state" / "episode"

---

## Phase R5 — Formal Inference for State-Dependent Amplification θ

**Status:** ⬜ Pending  
**Priority:** 🟠 High  
**Type:** Code + text  
**Estimated effort:** 3–4 hours  
**Script:** Extend `subprojects/07_state_dependent_lp/run_state_lp.py`

### The Problem

The reviewer notes:
> "Table 3 reports amplification magnitudes but not p-values or confidence intervals for theta. Without this, a reviewer may say the amplification story is visually suggestive but not formally established."

Currently the paper reports: θ_k0(EMFI) = 1.854, amplification at p=0.9 = 6.4×. But no standard error, no p-value, no confidence interval for θ.

### Implementation

**Step 1: OLS/Newey-West SE for θ directly**

The smooth-transition LP is a linear regression:
```
Y_{t+k} = α + β_k S_t + θ_k (S_t × P_stress_{t-1}) + φ_k P_stress_{t-1} + controls
```
θ_k has an OLS standard error already available from the regression output. Check whether the current code stores the SE for the interaction term. If not, extract it from `statsmodels` `OLS.fit().bse` and `OLS.fit().pvalues`.

**Step 2: Confidence intervals for total effect at p = 0.5 and p = 0.9**

The total effect at P_stress = p is: `IRF(p) = β_k + θ_k × p`
Standard error: `SE[IRF(p)] = SE[β + θ·p] = sqrt(Var(β) + p²·Var(θ) + 2p·Cov(β,θ))`

Implement `compute_irf_se(beta, theta, se_beta, se_theta, cov_bt, p_vals)` and report 95% CI for total effect at p = {0.5, 0.9}.

**Step 3: Circular block bootstrap (for non-standard inference)**

Add block bootstrap CIs as a robustness check on the Newey-West SEs:
- Block length: auto-selected as T^(1/3) or fixed at 20 days
- 1000 bootstrap replications
- For each replication: resample blocks of (Y_t, S_t, P_stress_t, controls) → re-estimate smooth-transition LP → store β̂, θ̂
- Report percentile bootstrap 95% CI for θ_k0 and for IRF at p=0.9

**Step 4: Number of identifying observations**

For Table 3, add a column: "N observations with P_stress > 0.5" (the high-stress region that primarily identifies θ). Currently known: 16 shock events with HMM systemic state (= 1), but the smooth-transition LP uses continuous P_stress, so the effective identifying mass is higher.

**Step 5: Updated Table 3 and Figure 3**

Add to Table 3:
- θ_hat (already present)
- SE(θ) — from OLS/Newey-West
- p-value for θ
- 95% CI for θ
- Total effect at p=0.9 with SE
- 95% CI for total effect at p=0.9

Update Figure 3: ensure confidence bands for each of the three IRF lines (p=0, 0.5, 0.9) are visible and properly labeled.

**Deliverables — Phase R5:**
- [ ] SE(θ_k) and p-value extracted for all outcomes and all horizons
- [ ] `compute_irf_se()` function implemented
- [ ] Block bootstrap CIs computed (1000 replications)
- [ ] Comparison table: Newey-West vs block bootstrap CIs for θ
- [ ] Table 3 updated with SE(θ), p-values, and total-effect CIs
- [ ] Figure 3 updated with proper CI bands for all three p-level IRFs
- [ ] Section 6 (state LP) updated with significance language for θ
- [ ] Git commit: "R5: formal inference for theta — SE, p-values, bootstrap CIs"

---

## Phase R6 — Predictive Event Classification

**Status:** ⬜ Pending  
**Priority:** 🟠 High  
**Type:** Code + text (new subsection)  
**Estimated effort:** 2–3 hours  
**Script:** `subprojects/11_predictive_classification/run_predictive_class.py`

### The Problem / Opportunity

The reviewer notes:
> "If you want [the taxonomy] to become stronger, estimate a predictive model: Pr(Systemic event) = f(pre-shock EMFI, pre-shock TCI, shock intensity, shock breadth). Then test whether pre-existing fragility predicts which geopolitical shocks become systemic. That would make the taxonomy far more valuable."

Currently the event taxonomy is descriptive (classified using post-event EMFI and P_stress). Adding a predictive model that uses only **pre-shock** information transforms the taxonomy from a classification exercise into an answer to the paper's title question: "Which shocks become systemic?"

### Implementation

**Data setup:**
- Sample: the 154 shock-day observations (or the COVID-excluded sample from R2)
- Outcome: `Systemic_i = 1` (42 events), `Systemic_i = 0` (112 events, Absorbed + Localized)
- Predictors (all pre-shock, i.e., lagged one day before the shock):
  - `EMFI_{t-1}` — pre-shock fragility level
  - `TCI_{t-1}` — pre-shock connectedness
  - `P_stress_{t-1}` — pre-shock stress-state probability
  - `S_t` — shock intensity (contemporaneous; this is known at the time of classification)
  - `breadth_t` — number of countries affected by the shock
  - Year dummies or time trend (to control for structural changes)

**Models:**
1. **Logit (primary):** `Pr(Systemic_i = 1) = Λ(α + β₁ EMFI_{t-1} + β₂ TCI_{t-1} + β₃ P_stress_{t-1} + β₄ S_t + β₅ breadth_t)`
2. **Probit (robustness):** Same specification
3. **Classification tree (illustrative):** CART tree for visual presentation in paper
4. **Out-of-sample performance:** Leave-one-out cross-validation AUC/ROC; report confusion matrix; Brier score

**Key results to report:**
- Coefficient table with odds ratios and p-values
- AUC from LOO-CV
- Predicted probability of Systemic classification for COVID (should be high), Ukraine (high), Hamas (intermediate), typical absorbed shock (low)
- Discussion: "Which features best predict systemic shocks?" Expected: P_stress_{t-1} and EMFI_{t-1} are significant; shock breadth matters; shock intensity has mixed effect

**Writing (new Section 6.3 or addition to Section 6.2):**
Title: "6.3 Predicting Geopolitical Systemic Events"

> "Having established that geopolitical shocks become systemic primarily when markets are already fragile, we now ask whether pre-shock market conditions can predict which shocks will become systemic. We estimate a binary response model..."

This section directly answers the paper's title question in a formal, predictive sense.

**Deliverables — Phase R6:**
- [ ] `subprojects/11_predictive_classification/run_predictive_class.py` written and run
- [ ] `results/predictive_class/logit_results.csv` — coefficient table
- [ ] `results/predictive_class/loocv_auc.csv` — LOO-CV AUC
- [ ] `results/predictive_class/Fig_ROC_Predictive.png` — ROC curve
- [ ] `results/predictive_class/Fig_Tree_Predictive.png` — classification tree (optional)
- [ ] `test_predictive_class.py` test suite, all passing
- [ ] New Section 6.3 in main.tex (~400 words + coefficient table + ROC figure)
- [ ] Git commit: "R6: predictive event classification — logit, LOO-CV, new Section 6.3"

---

## Phase R7 — Figure Improvements

**Status:** ⬜ Pending  
**Priority:** 🟡 Medium  
**Type:** Code (figure generation)  
**Estimated effort:** 1–2 hours

### R7.1 — Fix Figure 6 (Network correlation — too dense)

**Problem:** The reviewer says the network correlation figure is "conceptually useful but visually too dense. It needs clearer panels, maybe only pre/post density and top edges, not complete hairball networks."

**Fix:**
- Replace the current full hairball network with 2 cleaner panels:
  - **Panel A:** Pre-shock (t-5 to t-1) correlation network — show only edges with correlation > 0.7; label key nodes (Germany, France, Italy); node size proportional to degree centrality
  - **Panel B:** Post-shock (t to t+5) correlation network — same threshold; highlight new edges (increased connectivity in red)
- Alternative: replace network plot with a **heatmap** showing correlation matrix pre vs post for the top 5 systemic events — much cleaner and publishable
- Or: show only the top-N edges (N=20) pre and post, colored by correlation strength

**Script:** Modify `subprojects/08_event_classification/classify_events.py` → `generate_network_figure(df, events, max_edges=20)`

### R7.2 — Improve Figure 3 (State LP)

Following Phase R5 updates, ensure Figure 3 clearly shows:
- Three distinct IRF lines for p=0, 0.5, 0.9 in distinct colors
- Shaded confidence bands for each line
- A clear legend: "P_stress=0 (calm)", "P_stress=0.5 (elevated)", "P_stress=0.9 (near-systemic)"
- Horizontal reference line at 0

### R7.3 — Add ROC figure (from Phase R6)

Place the ROC curve figure from the predictive classification in the paper.

**Deliverables — Phase R7:**
- [ ] Figure 6 (network) redesigned: heatmap pre/post or filtered-edge network
- [ ] Figure 3 (state LP) updated with clear CI bands and labels
- [ ] All new figures verified ≥ 300 DPI, ≥ 20KB
- [ ] Git commit: "R7: figure improvements — network, state LP CI bands"

---

## Phase R8 — Paper Rewrite: Framing, Abstract, Policy Claims

**Status:** ⬜ Pending  
**Priority:** 🟡 Medium  
**Type:** Text revisions throughout main.tex  
**Estimated effort:** 3–4 hours  
**Prerequisite:** Phases R0–R7 complete

### R8.1 — Rewrite abstract (reviewer-suggested version)

Replace current abstract with a version that:
1. Leads with the research question (when do shocks become systemic?)
2. States the unconditional result honestly (positive but imprecise)
3. Leads with state-dependence as the main finding (this is the paper's core contribution)
4. Does not contradict the Hamas result
5. Mentions the predictive taxonomy
6. Tones down real-time monitoring language

Reference the reviewer's suggested abstract draft (Section 7 of Review1.pdf) as a starting point.

### R8.2 — Restructure contribution statement (Introduction)

Replace the current three-layer description with a cleaner three-contribution framing (as recommended in Review1.pdf, Section 6):

> **Contribution 1:** A market-based systemic fragility framework — EMFI and HMM stress-state probabilities, constructed independently of geopolitical shock dates.  
> **Contribution 2:** State-dependent geopolitical risk transmission — shocks have limited average effects, but large effects when markets are already fragile (6.4× EMFI, 10.3× TCI amplification at P_stress = 0.9).  
> **Contribution 3:** A predictive taxonomy of financially material geopolitical shocks — logit model predicts which shocks become systemic from pre-shock EMFI, TCI, and P_stress.

### R8.3 — Reframe unconditional LP result

Current framing (problematic): unconditional LP is presented as a positive result that is then supplemented by state-dependence.

New framing: unconditional LP is presented as an expected null (most shocks are absorbed), which then motivates and sets up the state-dependence analysis. The imprecision of β_k0 is the empirical fact that makes the state-dependent result interesting.

> "The unconditional average response of EMFI to a geopolitical shock is positive (β_k0 = 0.458) but statistically imprecise (p = 0.174). This imprecision is not a failure of the empirical design; it is the central fact. With 63% of shocks striking calm markets that exhibit minimal response, the average effect is pulled toward zero by the large absorbed-shock majority. The meaningful response is concentrated in the minority of shocks that arrive during already-fragile market conditions."

### R8.4 — Tone down policy/monitoring claims

Current conclusion language ("monitoring EMFI and P_stress may be more informative for crisis prediction...") is too strong for ex-post measures.

Replace with:
> "Our findings suggest that geopolitical risk monitoring should be combined with market-based measures of systemic fragility to improve identification of shock-stress interactions. The EMFI and HMM stress-state probability developed here are ex-post outcome measures; their real-time counterparts, constructed using rolling standardization and filtered probabilities, remain a direction for future research."

### R8.5 — Opening of Introduction

Replace the COVID-centered opening:
> "From the COVID-19 shock of February–March 2020 to the Russian invasion of Ukraine..."

With a geopolitically-focused opening:
> "From the Russian invasion of Ukraine in February 2022 to the Hamas–Israel conflict of October 2023, geopolitical shocks create episodes of uncertainty for global financial markets. Yet the vast majority of geopolitical events leave no discernible trace in aggregate market fragility..."

Keep COVID for EMFI/HMM validation only.

### R8.6 — Update title (optional)

The reviewer suggests: *"When Geopolitical Shocks Become Systemic: Market Fragility and Volatility Connectedness in European Equity Markets"* (removing "Market-Implied Stress" from the subtitle).

Consider whether to adopt the reviewer's suggestion or retain the current title. Either option is defensible; the current title is also fine.

**Deliverables — Phase R8:**
- [ ] Abstract rewritten (incorporating reviewer suggestion)
- [ ] Introduction contribution statement restructured (3 contributions)
- [ ] Unconditional LP framing revised (imprecision as central fact, not limitation)
- [ ] Introduction opening revised (remove COVID as geopolitical event)
- [ ] Conclusion policy claims toned down
- [ ] All "systemic stress regime" terminology verified updated (cross-check Phase R4)
- [ ] Full paper re-read for consistency
- [ ] LaTeX compiles cleanly; target page count ≤ 30 pages
- [ ] Git commit: "R8: major framing revision — abstract, contributions, LP framing, policy claims"

---

## Phase R9 — Submission Package

**Status:** ⬜ Pending  
**Priority:** 🟡 Medium  
**Type:** Text  
**Estimated effort:** 1–2 hours  
**Prerequisite:** Phases R0–R8 complete

### R9.1 — Cover letter

Write a GFJ cover letter addressing:
- Paper title, authors, contact
- One-paragraph summary of the paper
- Why GFJ is the right venue (European equity focus, systemic risk, international finance)
- Statement that the paper is not under review elsewhere
- Statement that the geopolitical shock series is reused from a separately submitted working paper (Lupu et al. 2026), with the shock-construction contribution attributed there
- Mention of data availability (if relevant)

### R9.2 — Final abstract polish

Ensure the abstract is exactly 150 words (GFJ guidelines) and mentions: research question, methodology, main finding (state dependence), and policy implication.

### R9.3 — Author information page

Prepare a separate title page with author information (for submission as a separate file), and a blind review version of main.tex with author information omitted.

### R9.4 — Final compilation check

- `pdflatex` + `bibtex` + `pdflatex` + `pdflatex` (full round-trip)
- Zero undefined citations, zero missing references
- Zero overfull hbox warnings above 5pt
- All figures appear, all tables compile
- Page count within GFJ limits
- Line numbers added (if required by GFJ)

### R9.5 — Data and code availability statement

GFJ increasingly requires data and code statements. Add to the paper:
> "Data and replication code are available at [repository or upon request]. Equity return data are sourced from [provider]. Conflict intensity data are from GDELT Document 2.0 (publicly available). All code is written in Python 3.11."

**Deliverables — Phase R9:**
- [ ] Cover letter written (1 page)
- [ ] Abstract polished to ≤ 150 words
- [ ] Title page (with author info) prepared separately
- [ ] Blind manuscript (`main_blind.tex`) — author info omitted
- [ ] Final compilation: 0 warnings, all figures/tables present
- [ ] Data and code availability statement added
- [ ] Final git tag: `v1.0-submission-ready`

---

## Checklist: All Reviewer Requirements

### From Review1.pdf Section 4 (Major concerns):

| Issue | Phase | Status |
|-------|-------|--------|
| 4.1 Country sample inconsistency (Czech Republic/Denmark/Switzerland vs actual data) | R0.1 | ✅ |
| 4.2 Shock-construction description conflict (CAMEO/Goldstein vs GDELT Doc 2.0) | R0.6 | ✅ |
| 4.3 COVID as geopolitical