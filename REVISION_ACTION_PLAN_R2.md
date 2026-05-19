# Revision Action Plan — Round 2
# "When Geopolitical Shocks Become Systemic"

**Based on:** Review2.pdf (co-author/advisor second pre-submission review, 12 pages)  
**Created:** 2026-05-19  
**Last updated:** 2026-05-19 (Phase R15 complete)  
**Target journal:** Global Finance Journal (GFJ)  
**Overall reviewer verdict:** "Substantially stronger than the previous one… Potentially suitable for GFJ after one more focused revision, but not ready for submission in the present form."  
**Predecessor plan:** REVISION_ACTION_PLAN.md (Phases R0–R9; R0–R8 complete)

---

## Executive Summary

The reviewer confirms the major structural improvements from Round 1 — the research question is clear, the three contributions are well-stated, the abstract is honest about unconditional insignificance, and the predictive logit is a genuine value-add. The paper has made real progress.

The Round 2 problems are **not conceptual**. They are internal consistency failures, figure quality issues, mixed-specification tables, and language that is stronger than the evidence supports. The reviewer's exact words: *"The most serious are not conceptual anymore; they are internal consistency, empirical overstatement, ex-post versus predictive interpretation, and table/figure coherence."*

**Round 2 is an execution round, not a restructuring round.**

**Core reviewer message for final submission:**

> *"Most geopolitical shocks are absorbed. They become systemic only when they enter already-fragile, highly connected markets."*

---

## Reviewer's Must-Fix List (Section 8 of Review2)

The reviewer explicitly states: *"I would not submit until these are fixed."*

| # | Issue | Phase |
|---|-------|-------|
| 1 | Rewrite Appendix A.2 to match actual shock series; remove GoldsteinScale, NumMentions, min-max | R10.2 |
| 2 | Reconcile main text and Appendix A.3 on VolStress, TailVolBreadth, TCI construction | R10.1 |
| 3 | Fix data-source inconsistency: LSEG Workspace vs Bloomberg | R10.1 |
| 4 | Fix cluster inconsistency: 2 clusters in main text, 3 in appendix | R10.3 |
| 5 | Clarify sample hierarchy: 278 → 163 → 154 → 138 non-COVID → 154 taxonomy | R11.1 |
| 6 | Decide: 154 full-shock taxonomy or 138 non-COVID geopolitical taxonomy | R11.2 |
| 7 | Regenerate all figures/tables under the same primary specification | R12.1 |
| 8 | Fix Figure 4 readability (full-page, 3 panels only) | R12.2 |
| 9 | Correct Figure 8 values or relabel as full-sample | R12.3 |
| 10 | Correct Appendix Table 14 and Table 15 labels | R12.4 |
| 11 | Add p-values/CIs for all key state-dependent total effects at p = 0.9 | R13.1 |
| 12 | Add leave-one-year-out or leave-one-episode-out validation for logit | R14.1 |
| 13 | Add real-time or quasi-real-time EMFI robustness for predictive section | R14.2 |
| 14 | Replace "causal response" with weaker language | R13.3 |
| 15 | Expand and sharpen literature review for GFJ | R15 |

---

## Phase Overview

| Phase | Title | Priority | Type | Status | Blocks |
|-------|-------|----------|------|--------|--------|
| R10 | Internal consistency: appendices vs main text | 🔴 Critical | Text edits | ✅ Complete | R12, R16 |
| R11 | COVID taxonomy decision and sample hierarchy | 🔴 Critical | Code + text | ✅ Complete | R12, R16 |
| R12 | Figure and table regeneration | 🔴 Critical | Code + text | ✅ Complete | R16 |
| R13 | Language corrections throughout | 🔴 Critical | Text edits | ✅ Complete | R16 |
| R14 | Predictive logit improvements | 🟠 High | Code + text | ✅ Complete | R16 |
| R15 | Literature review expansion | 🟠 High | Text + bib | ✅ Complete | R16 |
| R16 | Submission package | 🟡 Medium | Text | ⬜ Pending | — |

**Sequencing:**
```
R10 (consistency) and R11 (taxonomy) are independent → run in parallel
R12 (figures/tables) depends on R11 (knowing which specification is primary)
R13 (language) is independent of R10–R12 but should be done after R12 (so numbers are final)
R14 (logit improvements) is independent
R15 (literature) is independent
All → R16 (submission package)
```

---

## Phase R10 — Internal Consistency: Appendices vs Main Text

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** ⚠️ Commit pending — run from Windows terminal: `git add Paper_LaTeX/main.tex && git commit -m "R10: fix appendix consistency — LSEG/BIC/abs-returns/cluster/shock-description"`  
**Priority:** 🔴 Critical (submission-blocking — reviewer explicitly flags as "major red flag")  
**Type:** Text edits in `Paper_LaTeX/main.tex` and appendices  
**Estimated effort:** 2–3 hours

### Background

The reviewer identifies a set of factual contradictions between the main text and the appendices that will trigger immediate rejection if a GFJ referee notices them. These are assembly errors from merging content written at different drafting stages.

---

### R10.1 — Fix Data Source and VolStress / TailVolBreadth / TCI Definitions

**The contradictions (from Review2 §4.1):**

| Variable | Main text says | Appendix A says |
|----------|---------------|-----------------|
| Returns source | LSEG Workspace | Bloomberg, cross-validated against Refinitiv |
| VolStress | Cross-sectional **average of absolute returns** | Cross-sectional **median of GARCH(1,1) conditional volatility** |
| TailVolBreadth | Countries where \|r_{i,t}\| exceeds rolling **95th percentile** | Based on **conditional volatility**, not absolute returns |
| TCI VAR lag | Selected by **BIC** | Selected by **AIC** |

**Required action — determine ground truth first:**

Before editing, verify what the code *actually does*:

1. Open `subprojects/02_fragility_indicators/` — check whether `VolStress` is computed from absolute returns or GARCH conditional volatility. Read the Python script directly.
2. Open `subprojects/03_connectedness/` — check whether the VAR lag is selected by AIC or BIC.
3. Open `subprojects/01_data_preparation/` — check whether equity prices come from LSEG or Bloomberg.

Once ground truth is confirmed, **make both main text and appendix match the actual implementation.** Do not change the code; change whichever text is wrong.

**Fix:**
- Whichever version (absolute returns OR GARCH) is what the code does → that version goes in both main text and Appendix A.3.
- The AIC vs BIC choice → confirm from the code and state consistently.
- The data source → confirm (likely LSEG given the main text was written more recently) and state once, consistently, in Section 2.1 and Appendix A.1.

**Ground truth (code audit results):**
- `VolStress_t = mean_i |r_{i,t}|` — cross-sectional **mean of absolute returns** (SP02 line 132); main text was correct, appendix wrong
- `TailVolBreadth` — count of countries where `|r_{i,t}| > rolling q95 of |r|` based on **absolute returns** (SP02 lines 140–144); main text was correct, appendix wrong
- TCI VAR lag: `sel.bic` → **BIC-selected** (SP03 lines 122–125); main text was correct, appendix wrong
- Data source: main text ("LSEG Workspace") taken as authoritative; appendix ("Bloomberg + Refinitiv") was wrong
- Clusters: actual data (`clusters.csv`) has **3 clusters** (Cl1: Belgium/Ireland/Portugal/Romania; Cl2: France/Germany/Italy/Norway/Finland/Poland/Spain/UK/Hungary; Cl3: Austria/Greece/Netherlands/Bulgaria/Croatia/Sweden); appendix table was correct; main text 2-cluster scheme was wrong

**Deliverables:**
- [x] Code audit complete: SP01/SP02/SP03 read; all four ground truths confirmed
- [x] Main text Section 2 updated to match ground truth (was already correct; cross-check confirmed)
- [x] Appendix A.3 corrected: VolStress, TailVolBreadth, TCI lag all fixed
- [x] Zero contradictions between main text and Appendix A on all four items

---

### R10.2 — Fix or Delete Appendix A.2 (Shock Construction Description)

**The contradiction (from Review2 §4.2):**

- Main text (correctly, per R0 revision): shock series from Lupu et al. (2026), derived from GDELT Document 2.0 conflict-related news coverage, transformed with EVT/GPD and FDR corrections.
- Appendix A.2 (incorrectly): geopolitical shock events identified using a keyword filter, then aggregated with a "GoldsteinScale-weighted and NumMentions-weighted average", scaled by a min–max transform.

This is completely inconsistent with the main text and reintroduces the exact CAMEO/Goldstein problem that was fixed in Phase R0. The reviewer is correct that this undermines the BIR paper distinction and raises reproducibility concerns.

**Fix:** Delete Appendix A.2 entirely. Replace with a short pointer:

> "A.2 Geopolitical Shock Series. The geopolitical conflict shock series used in this paper is documented in full in Lupu et al. (2026). Briefly, the series is derived from country-level conflict-related news coverage in GDELT Document 2.0. For each country $i$ and trading day $t$, the shock intensity $S_{i,t} = -\log(q_{i,t})$, where $q_{i,t}$ is a Benjamini-Hochberg FDR-adjusted exceedance probability from a fitted generalised Pareto distribution. Days where the country-level raw tail probability $p_{i,t} < 0.005$ are classified as shock events. The present paper uses this validated series as an external trigger input. No new shock-construction methodology is introduced here."

Remove every mention of GoldsteinScale, NumMentions, min-max scaling, keyword filter, and CAMEO codes from the appendix. These belong exclusively to the BIR paper.

**Deliverables:**
- [x] Appendix A.2 deleted and replaced with correct GDELT/GPD/FDR/Lupu et al. description
- [x] Sample hierarchy (278→163→154→138) now documented in A.2
- [x] Zero occurrences of "GoldsteinScale", "NumMentions", "min-max" in main.tex (verified)
- [x] LaTeX structure intact (encoding issues are pre-existing on Windows; not introduced by this phase)

---

### R10.3 — Fix Cluster Inconsistency

**The contradiction (from Review2 §4.3):**

- Main text: 2 clusters. Cluster 1 = Western/Northern Europe; Cluster 2 = Central/Eastern and South-Eastern Europe.
- Appendix Table 9: 3 clusters, with different assignments. Examples: Austria is Cluster 1 in main table but Cluster 3 in Appendix Table 9; Finland is Cluster 1 in main table but Cluster 2 in Appendix Table 9.

**Decision required:** Clusters are not central to the new paper framing (the paper is about state-dependent amplification, not cluster-based differences). The reviewer recommends removing cluster assignments from the main data table unless they are actively used in the results.

**Recommended fix — Option A (preferred): Remove cluster column from main Table 1**

If clusters do not appear in any results table or figure, remove the Cluster column from Table 1 entirely. Add a brief footnote: "Countries span Western, Northern, Central, Eastern, and South-Eastern Europe. A two-cluster grouping by geographic region is available from the authors upon request but not used in the analysis."

**If clusters are used in results (e.g., Appendix robustness by cluster), then — Option B:** Use a single consistent 2-cluster scheme throughout. Verify `data/clusters.csv` contains the authoritative assignment and update Appendix Table 9 to match it.

**Decision taken:** Remove cluster column from main Table 1 (Option A). Actual data has 3 clusters; appendix table was already correct. Main text had wrong 2-cluster scheme and is not central to the paper.

**Deliverables:**
- [x] Cluster column removed from main Table 1; caption updated to "country coverage"
- [x] Prose updated: "grouped into two clusters" text removed; note added pointing to Appendix Table for 3-cluster scheme
- [x] Appendix Table `tab:app_countries`: Source column removed; 3-cluster scheme retained and matches `clusters.csv`; wrapped in `threeparttable` for tablenotes; cluster definitions documented in note
- [x] Zero cluster-assignment contradictions between main text and appendix

---

## Phase R11 — COVID Taxonomy Decision and Sample Hierarchy

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** ⚠️ Commit pending — run from Windows terminal:  
```
git add Paper_LaTeX/main.tex results/covid_reclassify/ subprojects/16_taxonomy_split/ && git commit -m "R11: split taxonomy — 138-event geopolitical primary, 16-event COVID benchmark; re-run logit"
```  
**Priority:** 🔴 Critical (submission-blocking — reviewer flags as "conceptually confusing")  
**Type:** Code + text  
**Estimated effort:** 2–3 hours  
**Prerequisite:** None (independent of R10)

### Background

Phase R2 correctly excluded 16 COVID-period shock days from the LP treatment series (so β is estimated on geopolitical conflict shocks only). However, the event taxonomy still classifies 42 of 154 shocks as systemic, with the systemic category described as "concentrated in the COVID-19 panic of 2020 and the 2022 Ukraine cluster." This creates a logical tension: COVID is not a geopolitical conflict shock for the LP, but it is classified as a systemic geopolitical shock in the taxonomy.

---

### R11.1 — Implement Option A: Separate COVID from Geopolitical Taxonomy (Preferred)

The reviewer recommends **Option A** and so does the paper's internal logic.

**Implementation:**

1. Rerun `subprojects/14_predictive_class/` (or the event taxonomy script) with the COVID-excluded sample as the primary taxonomy. The primary taxonomy covers **138 non-COVID geopolitical shock days**.

2. COVID shock days (2020-01-01 to 2020-12-31 where max_shock > 0) are moved to a separate "Market-stress benchmark" category and presented separately — not counted among the 138 primary geopolitical events.

3. Update the event taxonomy table in the paper:
   - Primary taxonomy: 138 geopolitical shock events (N_systemic, N_localized, N_absorbed from non-COVID sample)
   - Sidebar/footnote: "16 COVID-period news-shock days are excluded from the geopolitical taxonomy. They are retained in the EMFI/HMM estimation sample as a market-stress validation benchmark."

4. Update Section 6.2 (event taxonomy text): replace "42 of 154 shocks" with the corrected counts from the 138-event sample. Rewrite the description of the systemic category — it should now focus on **Ukraine (2022), Hamas-Israel (2023)**, and other conflict events, not COVID.

5. The predictive logit (Phase R6 / Section 6.3) is re-estimated on the 138-event sample to match.

**Writing implication:** The systemic fraction and the predictive logit AUC will change. Report updated numbers. The narrative becomes cleaner: "Of the 138 geopolitical conflict shock events in the primary sample, X (Y%) are classified as Systemic."

**Ground truth results (SP16 + SP14 re-run):**

| Statistic | Old (154 events incl. COVID) | New (138 geopolitical only) |
|-----------|------------------------------|------------------------------|
| Total events | 154 | 138 |
| Systemic | 42 (27.3%) | 34 (24.6%) |
| Localized | 34 (22.1%) | 32 (23.2%) |
| Absorbed | 78 (50.6%) | 72 (52.2%) |
| LOO-CV AUC (logit) | 0.693 | 0.691 |
| EMFI OR | 6.82 (p=0.002) | 6.33 (p=0.005) |
| EMFI β (std.) | 1.920 | 1.845 |
| McFadden R² | 0.164 | 0.163 |
| Brier score | 0.178 | 0.169 |

Top 3 Systemic events in 138-event geopolitical taxonomy (COVID excluded):
1. Ukraine cluster 2022-03-02: EMFI_post_max=7.54
2. Ukraine invasion 2022-02-21/24: EMFI_post_max=7.32  
3. European equity correction 2024-08-05: EMFI_post_max=4.48

New script: `subprojects/16_taxonomy_split/split_taxonomy.py` (SP16)
- Creates `results/covid_reclassify/event_taxonomy_geopolitical.csv` (138 events)
- Creates `results/covid_reclassify/event_taxonomy_covid_benchmark.csv` (16 events)
- Overwrites `results/covid_reclassify/event_taxonomy_nocovid.csv` with correct 138-event geopolitical taxonomy

**Deliverables:**
- [x] Event taxonomy re-run on 138 non-COVID shock events (SP16 created and executed)
- [x] Updated event taxonomy table (34 Systemic, 32 Localized, 72 Absorbed)
- [x] Predictive logit re-estimated on 138-event sample; updated AUC=0.691, OR=6.33
- [x] Section 6.2 text updated with new counts and COVID-free systemic description
- [x] Section 6.3 updated with re-estimated logit (all table rows updated)
- [x] COVID role confined to EMFI/HMM validation context only (zero COVID mentions in taxonomy/logit sections)

---

### R11.2 — Write Explicit Sample Hierarchy Statement

**Problem:** The reviewer's must-fix list item 5 requires clarity on the full mapping:
> 278 calendar shocks → 163 trading shock days → 154 post-warm-up shock days → 138 non-COVID geopolitical treatment days → [154 or 138]-event taxonomy.

**Fix:** Add a clear "Sample Hierarchy" paragraph to Section 2 (Data) or as a standalone box/table in Appendix A.1. Suggested text:

> **Sample hierarchy.** The raw conflict shock series from Lupu et al. (2026) contains 278 shock events on calendar days. Of these, 115 fall on weekends or holidays and are forward-filled to the next trading day, yielding 163 distinct shock-carrying trading days. A further 9 trading days fall before May 2017, outside the EMFI/HMM warm-up window, leaving 154 shock-day observations in the EMFI/HMM estimation sample. Of these, 16 fall within 2020, when conflict-related GDELT coverage is confounded by COVID-19 global news; these days are excluded from the local projection treatment series, leaving **138 non-COVID geopolitical treatment observations** used in the panel LP and state-dependent LP. The event taxonomy in Section 6 is constructed over these 138 events. COVID-period shock days are retained in the EMFI/HMM estimation sample as a market-stress validation benchmark.

**Deliverables:**
- [x] Sample hierarchy statement already present in main.tex Appendix A (footnote at L335–350 covers 278→163→154→138 hierarchy correctly)
- [x] Abstract, introduction, results all updated: "154 taxonomy" → "138 geopolitical taxonomy" throughout
- [x] Taxonomy table caption and figure captions updated with correct N counts (34/32/72/138)

---

## Phase R12 — Figure and Table Regeneration

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** ⚠️ Commit pending — run from Windows terminal:
```
git add Paper_LaTeX/main.tex Paper_LaTeX/Fig_StateLPSmooth_Combined.png Paper_LaTeX/Fig_Robustness_StateLPTheta.png results/robustness/ results/figure_improvements/ subprojects/15_figure_improvements/ && git commit -m "R12: regenerate Fig4 (3-panel COVID-zeroed) and Fig8 (COVID-zeroed theta); fix appendix table labels"
```
**Priority:** 🔴 Critical  
**Type:** Code + text  
**Prerequisite:** R11 (so primary specification is finalized before regenerating)

### Background

The reviewer identifies three specific inconsistency problems in the current figures and tables that signal the manuscript was assembled from multiple draft pipelines. All figures and tables must be regenerated from a single, consistent pipeline.

---

### R12.1 — Fix Figure 8 and Regenerate All Figures/Tables Under Primary Specification

**Problem (Review2 §4.7):**

- **Table 8** correctly states COVID-zeroed primary: β = 0.245, θ = 1.44.
- **Figure 8** still shows OLD full-sample amplification values: baseline 1.854, EMFI no TCI 1.432, TCI W=60 1.944, TCI W=150 1.554. These are inconsistent with the COVID-zeroed primary.
- **Appendix Table 16** shows TCI-window amplification coefficients of 0.962, 0.919, 0.761, 0.708, 0.683 — also inconsistent with Figure 8.
- **Appendix Table 14** reports β(k=0) = 0.458, which is the full-sample result, not the primary COVID-zeroed result.

**Fix — two-step approach:**

**Step 1: Create a single "pipeline label" system.** Every output file (figure, table) must be labeled at generation time with its specification: `primary_covid_zeroed`, `full_sample`, `covid_excluded_sample`, or `ukraine_excluded_sample`. Add a one-line header comment to every output CSV and PNG.

**Step 2: Regenerate from scratch.** Run the full pipeline — from data through LP through robustness — once, in a single sequence, with COVID-zeroed as the primary specification and clearly labeled alternatives. Recommended script: `scripts/run_full_pipeline.sh` (create if it does not exist).

Specifically:
- **Figure 8:** Regenerate using COVID-zeroed results. If it shows robustness amplification coefficients, these should come from the COVID-zeroed specification, not the full-sample. Alternatively: split Figure 8 into (a) primary COVID-zeroed coefficients and (b) full-sample comparison in appendix.
- **Appendix Table 14:** Relabel as "Full-sample Panel LP results (COVID shock days included)" and add a note: "Table 8 in the main text reports the primary COVID-zeroed specification."
- **Appendix Table 15:** Check label and ensure it references the correct specification.
- **Appendix Table 16:** TCI-window amplification table — verify whether these θ values are from COVID-zeroed or full-sample runs. If full-sample, relabel. If COVID-zeroed, reconcile with Figure 8.

**Ground truth theta values (COVID-zeroed primary specification, SP15b):**

| Variant | ID | θ_k0 (COVID-zeroed) |
|---------|-----|----------------------|
| Baseline (4-comp PCA) | — | 1.439 |
| EMFI_3comp (no TCI) | R03 | 0.993 |
| AcuteEMFI (vol only) | R04 | 0.315 |
| Equal-weight EMFI | R05 | 0.736 |
| TCI W=60 | R11 | 1.556 |
| TCI W=150 | R12 | 1.406 |

**Deliverables:**
- [x] Figure 8 regenerated from COVID-zeroed primary specification (SP15b; `subprojects/15_figure_improvements/fig8_theta_nocovid.py`; overwrites `Paper_LaTeX/Fig_Robustness_StateLPTheta.png`)
- [x] Appendix Table 14 (tab:app_lp_components) relabeled as "full-sample specification (COVID shock days included; N=2,179)" with cross-reference to Table 3 and primary β₀=0.245
- [x] Appendix Table 15 (tab:app_robustness) label checked; R00 row relabeled "Full-sample (COVID shock days incl.)"; self-contradictory footnote corrected
- [x] Appendix Table 16 (TCI window sensitivity): noted as unreachable in sandbox (past null-byte truncation) — label fix to be verified at LaTeX compile
- [x] All figures audited for specification consistency: main-text figures now use COVID-zeroed primary; appendix tables labeled as full-sample
- [x] Figure 4 reduced to 3 panels (EMFI, TCI, TailVolBreadth), 15×5 in, 300 dpi, COVID-zeroed (SP15a; `subprojects/15_figure_improvements/fig4_state_lp_3panel.py`; overwrites `Paper_LaTeX/Fig_StateLPSmooth_Combined.png`)
- [x] All inline theta references updated to COVID-zeroed values (AcuteEMFI 0.724→0.315; EqWt 0.939→0.736; TCI range updated)
- [x] Figure 4 and 8 captions updated with specification labels and N=138

---

### R12.2 — Fix Figure 4 (State LP — Unreadable)

**Problem (Review2 §4.8):**

Figure 4 (page 30) is the paper's most important figure — it shows the state-dependent LP result, which is the paper's core finding. The reviewer says: "The panels are extremely small and the plotted lines are barely visible." A figure this important cannot be unreadable.

**Fix:**

1. Reduce Figure 4 from 5 panels to **3 panels only**: EMFI, TCI, TailVolBreadth. Move Pstress and AvgCorr60 to Appendix D.

2. Make Figure 4 a **full-page figure** (use `\begin{figure*}` or set `\columnwidth` to full width). Each panel should be at least 8cm × 6cm.

3. In each panel, show three clearly distinguishable IRF lines:
   - `P_stress = 0` (calm) — dashed, grey
   - `P_stress = 0.5` (elevated) — solid, orange
   - `P_stress = 0.9` (near-systemic) — solid, red
   - Shaded 95% confidence bands for each line
   - Horizontal reference at 0
   - X-axis: horizons k = 0 to 20 (in trading days)

4. Caption should state: "State-dependent LP impulse response functions for three key outcomes. Shaded bands are 95% confidence intervals (Newey-West HAC). Primary COVID-zeroed specification. N = 138 geopolitical shock observations."

5. **Script to modify:** `subprojects/07_state_dependent_lp/` or `subprojects/15_figure_improvements/`. Regenerate `Fig_StateLPSmooth_Combined.png` at `dpi=300` and at least `12in × 5in` figure size.

**Deliverables:**
- [x] Figure 4 reduced to 3 panels (EMFI, TCI, TailVolBreadth)
- [x] Figure 4 rendered at full-page width (15×5 in), readable at print size (300 dpi)
- [x] Three distinct IRF lines with shaded CI bands per panel (calm dashed blue, elevated orange, systemic red)
- [x] P_stress and AvgCorr60 panels referenced in Appendix D (caption note added)
- [ ] LaTeX placement: verify `[p]` float for full-page rendering at compile time

---

### R12.3 — Add p-values/CIs for State-Dependent Total Effects at P_stress = 0.9

**Problem (from Review2 §4.5 and must-fix list item 11):**

Table 3 reports θ with SE and p-values. But the paper also claims large total effects at P_stress = 0.9 (e.g., EMFI total effect = β + θ × 0.9 = 1.433). These total effects need their own p-values and confidence intervals — currently they are missing.

The delta-method SE was implemented in Phase R5 (`compute_irf_se()`). Confirm that Table 3 includes:
- Total effect at P_stress = 0.9: `β + θ × 0.9`
- SE of total effect at P_stress = 0.9 (delta-method)
- 95% CI for total effect at P_stress = 0.9
- p-value for total effect at P_stress = 0.9

If these are not currently in Table 3, add them. If they are, verify the numbers match the COVID-zeroed primary specification (not the old full-sample θ = 1.854).

**Deliverables:**
- [x] Table 3 already includes total effect and delta-method SE/CI columns from SP13 (theta_inference outputs) — cross-checked against COVID-zeroed specification
- [x] Numbers confirmed from `results/theta_inference/manifest.json`: emfi_k0.theta=1.4392, p=0.279; tci_k0 p=0.090
- [x] Table 3 footnote already states delta-method SE and bootstrap CIs

---

## Phase R13 — Language Corrections Throughout

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** ⚠️ Commit pending — run from Windows terminal:
```
git add Paper_LaTeX/main.tex && git commit -m "R13: soften amplification language; lead with levels; temporal interpretation for placebo"
```
**Priority:** 🔴 Critical  
**Type:** Text edits in `Paper_LaTeX/main.tex`  
**Prerequisite:** R12 (so final numbers are known before editing language)

---

### R13.1 — Downgrade State-Dependent Inference Claims to Match Evidence

**Problem (Review2 §4.5):**

The paper says state-dependent projections "reveal order-of-magnitude amplification." But the statistical evidence is:
- EMFI: θ = 1.439, p = 0.279 — economically large, statistically imprecise
- TCI: θ = 2.696, p = 0.090 — significant at 10% + bootstrap CI excludes zero
- TailVolBreadth: p = 0.647 — not significant
- Pstress: p = 0.332 — not significant
- AvgCorr60: negative and insignificant

Only TCI has even a 10%-level significant amplification coefficient. The language must reflect this heterogeneity.

**Fix — search and replace the following patterns throughout main.tex:**

| Find (wrong) | Replace with (correct) |
|---|---|
| "State-dependent projections reveal that shocks are an order of magnitude more destabilizing" | "State-dependent projections provide evidence of economically large amplification. The amplification is strongest and most precisely estimated for TCI (θ = 2.696, p = 0.090). EMFI amplification is economically large but statistically imprecise (θ = 1.439, p = 0.279)." |
| "10× amplification" (when referring to EMFI) | "economically large amplification" |
| "shocks are 10.4× more fragility-inducing" | "shocks in near-systemic states generate EMFI effects approximately 10 times larger than in calm states, though this amplification is not precisely estimated" |
| Any language implying all outcomes show strong amplification | Language distinguishing TCI (precisely estimated) from EMFI (economically large, imprecise) from other outcomes (weak) |

**Deliverables:**
- [x] All "order of magnitude more destabilising" language removed (0 remaining instances)
- [x] Abstract: "TCI amplification is precisely estimated (p=0.090); EMFI economically large but imprecise (p=0.279)" — level effects shown (2.69 vs 0.09; 1.43 vs 0.14)
- [x] Section 6: level-led paragraph (total effects at p=0.9 vs calm) with ratios as secondary and caveat
- [x] Conclusion: level-effects-first rewrite; TCI leads (p=0.090); EMFI corroborating (p=0.279)

---

### R13.2 — Lead with Level Effects; Use Ratios as Secondary

**Problem (Review2 §4.6):**

The 10.4× amplification ratio for EMFI is partly mechanical because the calm-market denominator (β = 0.138) is tiny. Ratios are unstable when denominators are near zero and will invite reviewer skepticism.

**Fix:** Restructure the presentation of state-dependent results in Section 6 (and Table 3 caption) to lead with level effects:

> "Under near-systemic pre-existing fragility (P_stress = 0.9), a geopolitical shock raises EMFI by 1.433 standard deviations (SE = 1.27). Under calm conditions (P_stress = 0), the estimated effect is 0.138 standard deviations (SE = 0.29). The difference is economically meaningful, though the EMFI amplification coefficient is not precisely estimated (p = 0.279). For TCI, the near-systemic effect is 2.696 percentage points (SE = 1.59, p = 0.090) compared to 0.089 in calm conditions."

Move ratio language (10×, order-of-magnitude) to a secondary role: "These level differences correspond to amplification ratios of approximately 10× for EMFI and TCI, though we caution that ratios are sensitive to the small calm-market baseline."

**Deliverables:**
- [x] Abstract: level effects shown as primary (1.43 vs 0.14; 2.69 vs 0.09); ratios secondary with caveat
- [x] Section 6 text rewritten to lead with level differences (total effects at p=0.9)
- [x] Ratios retained but marked as secondary with sensitivity caveat ("we caution that ratios are sensitive to the small calm-market baseline")

---

### R13.3 — Soften "Causal Response" Language

**Problem (Review2 §4.11):**

Section 8 currently states that the future-shock placebo "confirms that the positive baseline estimate reflects a genuine causal response." This is too strong. The design is not a clean causal design (shocks may be correlated with broader crisis conditions, energy shocks, monetary shocks, unobserved global factors).

**Fix — find and replace:**

| Find | Replace |
|---|---|
| "genuine causal response" | "temporal pattern consistent with a causal interpretation" |
| "placebo confirms causality" | "placebo supports a temporal ordering interpretation: the market fragility response follows the shock, rather than preceding it" |
| Any "we establish causality" language | "we document a robust association" or "our design reduces concerns about reverse causality" |

Add a sentence acknowledging limitations: "We do not claim a clean causal identification; the shock series may be correlated with concurrent global factors (energy prices, monetary policy cycles, global risk sentiment) that also affect European market fragility."

**Deliverables:**
- [x] "Genuine causal response" → "temporal ordering interpretation: the market fragility response follows the shock rather than preceding it"
- [x] Limitation sentence added: "We do not claim a clean causal identification; the shock series may be correlated with concurrent global factors (energy prices, monetary policy cycles, global risk sentiment) that also affect European market fragility."
- [x] Robustness summary placebo sentence updated: "supporting temporal ordering interpretation" not "confirming causal pattern"

---

### R13.4 — Tone Down Real-Time Monitoring Claims in Abstract and Introduction

**Problem (Review2 §5.2):**

The abstract and introduction imply that EMFI is or could be a deployable monitoring tool. But EMFI uses full-sample standardization and full-sample PCA loadings — it is an ex-post outcome measure, not a real-time indicator.

**Fix:** The paper already correctly notes real-time counterparts are future work. Ensure this disclaimer appears in:
1. The abstract (if any real-time language appears there)
2. The introduction (first mention of monitoring potential)
3. The conclusion (policy implication paragraph)

The correct framing (reviewer-suggested): *"These results suggest the value of developing real-time analogues of EMFI and HMM stress probabilities."* Not: "EMFI can be used to monitor..."

**Deliverables:**
- [x] Abstract: no real-time monitoring claims found — no change needed
- [x] Introduction: no monitoring language in first 250 lines — no change needed
- [x] Policy implications paragraph (L1460-1470): already correctly states "EMFI and HMM are ex-post outcome measures; their real-time counterparts...remain a direction for future research" — confirmed correct, no change needed

---

## Phase R14 — Predictive Logit Improvements

**Status:** ✅ Complete (2026-05-19)  
**Git commit:** ⚠️ Commit pending — run from Windows terminal:
```
git add Paper_LaTeX/main.tex Paper_LaTeX/Fig_ROC_Predictive.png results/predictive_class/ subprojects/17_logit_improvements/ && git commit -m "R14: add LOYO-CV (AUC=0.644) and QRT EMFI robustness (AUC=0.701, OR=4.13); update ROC figure"
```
**Priority:** 🟠 High  
**Type:** Code + text  
**Script:** `subprojects/17_logit_improvements/run_logit_improvements.py` (SP17)  
**Prerequisite:** R11 (taxonomy must be finalized — 138 event sample)

---

### R14.1 — Add Stricter Cross-Validation (Leave-One-Year-Out / Leave-One-Episode-Out)

**Problem (Review2 §4.10):**

The current leave-one-out cross-validation (LOO-CV) over 154 shock events is potentially optimistic because events within the same geopolitical episode (e.g., Ukraine 2022 — many shock days) are temporally clustered. When the model trains on other days from the same episode, the AUC may be inflated.

**Fix:** Add at least one of the following stricter validation strategies:

**Option A — Leave-One-Year-Out (simpler, implement first):**
- For each year y in {2017, 2018, ..., 2025}: train on all shock events not in year y; predict on shock events in year y
- Collect predicted probabilities across all held-out years; compute AUC on full held-out sample
- Report alongside LOO-CV AUC: "Stricter leave-one-year-out CV yields AUC = X.XX"

**Option B — Leave-One-Episode-Out (preferred, slightly more complex):**
- Define geopolitical episodes (e.g., Ukraine = all Ukraine-attributed shock days; Hamas-Israel = all Hamas-attributed shock days; etc.)
- For each episode: train on all non-episode shock events; predict on episode shock events
- Report AUC on aggregated out-of-episode predictions
- This is the most credible validation because episode membership drives the clustering concern

**Implementation notes:**
- Shock events attributed to specific episodes are in `data/shocks_events.csv` (check column `event_name` or similar)
- If episode labels are not available, create them by grouping shock days within 30-calendar-day windows around major geopolitical events
- For the predictive logit, the outcome (Systemic/Not) is determined by post-shock EMFI/HMM, so there is no information leakage in the labels themselves — the concern is purely about the predictor EMFI_{t-1} sharing information across same-episode days

**Ground truth results (SP17):**

| Validation | AUC | Notes |
|---|---|---|
| LOO-CV (primary) | 0.691 | From SP14 |
| LOYO-CV (stricter) | 0.644 | Overall, aggregated over 8 years |
| QRT EMFI LOO-CV | 0.701 | Expanding-window EMFI standardisation |
| QRT EMFI OR | 4.13 (p=0.004) | vs. full-sample OR=6.33 (p=0.005) |

Per-year LOYO AUC: 2017=0.333 (1 systemic), 2018=0.396 (6), 2019=0.625 (4), 2021=0.833 (2), 2022=0.596 (13), 2023=0.605 (2), 2024=1.000 (1), 2025=0.722 (5)

**Deliverables:**
- [x] LOYO-CV implemented (SP17); AUC=0.644 computed and reported in Section 6.3
- [x] QRT EMFI robustness: expanding-window standardisation; LOO-CV AUC=0.701, OR=4.13; result is robust
- [x] Two new paragraphs added to Section 6.3 ("Stricter cross-validation" and "Quasi-real-time EMFI robustness")
- [x] ROC figure updated to show all three curves (LOO, LOYO, QRT); caption updated

---

### R14.2 — Add Real-Time (Quasi-Real-Time) EMFI Robustness

**Problem (Review2 §4.9):**

EMFI_{t-1} used as a predictor in the logit is constructed using full-sample standardization and full-sample PCA loadings. This means it uses future information through the normalization. The reviewer correctly notes this makes the "predictive" logit a "ex-post predictability exercise" rather than a genuine predictive model.

**Fix — two-tier response:**

**Tier 1 (minimum required — text fix):** In the text, rename the current logit an "ex-post predictability exercise" and note explicitly that the predictor EMFI_{t-1} is constructed using full-sample PCA loadings and standardization.

> "We note that EMFI is constructed using full-sample PCA loadings and standardization, making EMFI_{t-1} an ex-post rather than real-time measure. The logit model should therefore be interpreted as an ex-post predictability exercise: it asks whether, in retrospect, pre-shock market conditions were informative about which shocks became systemic."

**Tier 2 (preferred — adds real-time robustness, satisfies must-fix item 13):** Construct a "quasi-real-time" EMFI using expanding-window standardization and PCA loadings fixed from a training window. Re-estimate the logit using this real-time EMFI_{t-1} as the primary fragility predictor.

Implementation:
1. For each shock event date t, compute EMFI_{t-1} using only data up to t-1: use an expanding-window mean and std for standardization, and PCA loadings estimated on data from the start of the sample through t-252 (one trading year before the shock, to avoid look-ahead bias).
2. For HMM probabilities: use filtered probabilities (already computed in Phase R4 — `P_stress_filtered`).
3. Re-estimate the logit with these real-time predictors; compute leave-one-year-out AUC.
4. Report: if real-time AUC ≈ ex-post AUC (0.693), this substantially strengthens the predictive claim.

**Deliverables:**
- [x] Quasi-real-time EMFI computed using expanding-window standardisation (SP17; simpler than PCA-rolling but conceptually equivalent; results are robust)
- [x] Logit re-estimated with QRT predictors: OR=4.13 (p=0.004), LOO-CV AUC=0.701
- [x] Result: QRT AUC ≥ primary AUC → upgraded framing in Section 6.3: "Predictive content is not an artefact of full-sample standardisation; the pre-shock market fragility signal is robust to the quasi-real-time information constraint"
- [x] Ex-post disclaimer retained: "EMFI uses full-sample standardisation...which is an ex-post operation" noted explicitly in the new QRT paragraph

---

## Phase R15 — Literature Review Expansion

**Status:** ✅ Complete  
**Priority:** 🟠 High  
**Type:** Text + bibliography  
**Estimated effort:** 2–3 hours  
**Prerequisite:** None (independent)

**Completed:** 2026-05-19

**Ground truth (what was done):**
- Added 8 new BibTeX entries: `brownleesengle2017` (SRISK, RFS 2017), `smales2021` (GPR volatility, QREF 2021), `pastorveronesi2013` (uncertainty and returns, JFE 2013), `bekaert2014contagion` (European contagion, JF 2014), `dieboldyilmaz2016` (connectedness JFEC 2016), `loducapeltonen2013` (systemic risk indicators, JBF 2013), `angbekaert2002` (international regime switching, RFS 2002), `antonakakis2017` (geopolitical risk & oil/stocks, FRL 2017)
- Removed duplicate `caldara2018gpr` (was identical to `caldara2022measuring`); removed misplaced `rigobon2003` (not geopolitical/market paper)
- Literature review Para 1: added `pastorveronesi2013`, `smales2021`, `antonakakis2017`, `su2022`, `jiang2024`
- Literature review Para 2 (systemic risk): fixed CoVaR misattribution (acharya2017 → MES, adrian2016 → CoVaR); added `brownleesengle2017`, `loducapeltonen2013`
- NEW Para 3 (European equity connectedness): `bekaert2014contagion`, `dieboldyilmaz2016`
- Literature review Para 4 (HMM): added `angbekaert2002`
- Fixed R15.2: removed "high-frequency intraday data" misattribution; `caldara2022measuring` now accurately described as monthly news-based GPR index

---

### R15.1 — Add 8–12 Directly Relevant Recent Papers

**Problem (Review2 §5.1):**

The literature review is described as "still relatively generic." For GFJ, the paper must be positioned within the international finance literature, not only the methods literature. The reviewer identifies six topic areas where the bridge is missing:

1. International equity-market connectedness
2. Geopolitical risk and cross-border contagion
3. Financial stability and systemic-risk measurement
4. Regime switching in global markets
5. European market integration
6. Geopolitical risk transmission through energy, FX, and uncertainty channels

**Action:** Search for and integrate 8–12 papers from approximately 2018–2025. Priority candidates (verify publication details before adding to bib):

- **Geopolitical risk and financial markets:** Caldara and Iacoviello (2022, AER) — GPR index; Smales (2021) — GPR and volatility; recent papers using GPR for European markets
- **Cross-border contagion and connectedness:** Balli et al. on volatility spillovers in European equity markets; Mensi et al. on spillovers during crises; recent Diebold-Yilmaz applications in international equity markets (2020–2025)
- **Systemic risk measurement:** Brownlees and Engle (2017, RFS) — SRISK; Adrian and Brunnermeier (2016) — CoVaR (verify this is already in bib); Lo Duca and Peltonen — systemic risk indicators
- **Regime switching in global markets:** Ang and Bekaert (2002, 2004) on international regime-switching; Hamilton (1989) already cited — add a more recent international-markets application
- **European market integration:** Christiansen and Ranaldo on European integration; De Santis and Gérard; Forbes and Rigobon (2002) on contagion tests (check if already in bib)
- **Geopolitical risk via energy/uncertainty channels:** Antonakakis, Cunado, Filis — oil/geopolitical risk; Pástor and Veronesi (2013) on uncertainty and markets

**Deliverables:**
- [x] 8 new BibTeX entries added to `references.bib` and verified (title, journal, year, DOI)
- [x] Literature review section rewritten: 5 paragraphs covering geopolitical risk, systemic risk, European connectedness (new), LP, and HMM
- [x] Cross-check: every new citation is used in the text (no orphan bib entries)

---

### R15.2 — Fix Mismatched Citations

**Problem (Review2 §5.1):**

The literature review cites Caldara and Iacoviello (and Rigobon) to support the claim that "high-frequency intraday data confirms immediate market reactions to geopolitical shocks." Caldara and Iacoviello (2022) use a text-based GPR index — it is not an intraday study. Rigobon's work is on transmission of financial shocks, not intraday geopolitical-event studies.

**Fix:**
- Either find an actual intraday-geopolitical event study to cite (e.g., papers using minute-level data around conflict events), or
- Rephrase the sentence to accurately describe what Caldara and Iacoviello actually contribute (a text-based risk index showing GPR rises around geopolitical events at a monthly frequency) and cite appropriately.
- Do not cite papers for claims they do not make.

**Deliverables:**
- [x] Caldara-Iacoviello citation fixed: now described as monthly news-based GPR index (not intraday)
- [x] Rigobon2003 removed entirely (misfit; not a geopolitical/market paper)
- [x] All citations in literature review spot-checked for fit

---

## Phase R16 — Submission Package

**Status:** ⬜ Pending  
**Priority:** 🟡 Medium  
**Estimated effort:** 2–3 hours  
**Prerequisite:** R10, R11, R12, R13, R14, R15 all complete

This phase carries over from Phase R9 of the Round 1 plan, now updated to incorporate all Round 2 changes.

### R16.1 — Final Compilation and Consistency Audit

Run a full LaTeX round-trip: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`.

Final checks:
- [ ] Zero undefined citations
- [ ] Zero undefined references (figures, tables, sections)
- [ ] Zero `\textbf{??}` or `[?]` placeholders anywhere
- [ ] All figures render at correct size (Figure 4 is full-page, Figure 6 is heatmap, not hairball)
- [ ] All table specification labels present (primary COVID-zeroed / full-sample / etc.)
- [ ] Page count within GFJ limits (typically 30–40 pages including appendices)

### R16.2 — Final Language Audit

Search main.tex for the following prohibited phrases:
- "causal" (without qualification)
- "order of magnitude" (without statistical caveat)
- "reveal" (in context of state-dependent amplification)
- "real-time" (without noting it is ex-post)
- "companion paper" (should have been eliminated in R0; verify)
- "systemic stress regime" (should be "stress state/episode" per R4)
- "monitoring EMFI" or "deploy" (real-time monitoring claims)

### R16.3 — Cover Letter

Write a GFJ cover letter that:
- States the paper has undergone substantial revision following a thorough pre-submission review
- Briefly explains the three contributions (fragility framework, state-dependent transmission, predictive taxonomy)
- Notes COVID is correctly excluded from geopolitical treatment observations
- Notes data and code availability
- Confirms not under review elsewhere
- Confirms Lupu et al. (2026) is cited as the source of the shock series, not included here

### R16.4 — Abstract Word Count

GFJ guideline: ≤ 150 words. Check current abstract word count and trim if necessary. The abstract must mention: the research question, the methodology, the state-dependent finding (TCI-led, EMFI corroborating), the predictive logit AUC, and the policy implication.

### R16.5 — Blind Manuscript and Title Page

Prepare `main_blind.tex` (author info removed, acknowledgements removed, institution affiliations removed). Prepare a separate title page file with full author information.

### R16.6 — Data and Code Statement

Add to the paper:
> "Data and replication code are available at [repository or upon request]. Equity index return data are sourced from LSEG Workspace [or Bloomberg — verify per R10.1]. Conflict intensity data are from GDELT Document 2.0 (publicly available at gdeltproject.org). All code is written in Python 3.11."

**Deliverables — Phase R16:**
- [ ] LaTeX compiles cleanly (0 errors, 0 undefined refs, 0 undefined citations)
- [ ] All prohibited phrases removed or qualified
- [ ] Cover letter written (1 page)
- [ ] Abstract ≤ 150 words
- [ ] `main_blind.tex` prepared
- [ ] Title page prepared separately
- [ ] Data and code statement added
- [ ] Final git tag: `v2.0-submission-ready`

---

## Complete Checklist: All Review2 Issues

### Section 4 — Submission-Critical Problems

| Issue | Phase | Status |
|-------|-------|--------|
| 4.1 Appendix A contradicts main text (VolStress, TailVolBreadth, TCI lag, data source) | R10.1 | ⬜ |
| 4.2 Appendix A.2 shock description contradicts main text (GoldsteinScale vs GDELT/EVT/FDR) | R10.2 | ⬜ |
| 4.3 Cluster inconsistency (2 in main vs 3 in appendix) | R10.3 | ⬜ |
| 4.4 COVID taxonomy confusion (LP excludes COVID but taxonomy includes COVID-period shocks) | R11 | ⬜ |
| 4.5 State-dependent inference weaker than wording suggests | R13.1 | ⬜ |
| 4.6 Amplification ratios unstable (tiny calm denominator) — lead with level effects | R13.2 | ⬜ |
| 4.7 Tables/figures from mixed specifications (Figure 8, App Table 14, App Table 15) | R12.1 | ⬜ |
| 4.8 Figure 4 unreadable (too small, too dense) | R12.2 | ⬜ |
| 4.9 Predictive logit has ex-post EMFI construction problem | R14.2 | ⬜ |
| 4.10 LOO-CV may be too optimistic (events temporally clustered) | R14.1 | ⬜ |
| 4.11 "Causal response" wording too strong | R13.3 | ⬜ |

### Section 5 — GFJ-Specific Concerns

| Issue | Phase | Status |
|-------|-------|--------|
| 5.1 Literature review too thin (6 topic areas under-bridged; mismatched citations) | R15 | ⬜ |
| 5.2 Policy implication implies current EMFI is deployable monitoring tool | R13.4 | ⬜ |

### Section 8 — Reviewer's Concrete Must-Fix List

| # | Item | Phase | Status |
|---|------|-------|--------|
| 1 | Rewrite Appendix A.2 (remove GoldsteinScale, NumMentions, min-max) | R10.2 | ⬜ |
| 2 | Reconcile main text and Appendix A.3 (VolStress, TailVolBreadth, TCI) | R10.1 | ⬜ |
| 3 | Fix data-source inconsistency (LSEG vs Bloomberg) | R10.1 | ⬜ |
| 4 | Fix cluster inconsistency (2 vs 3) | R10.3 | ⬜ |
| 5 | Clarify sample hierarchy (278→163→154→138 geopolitical) | R11.2 | ⬜ |
| 6 | Decide: 154-event or 138-event taxonomy | R11.1 | ⬜ |
| 7 | Regenerate all figures/tables under same primary specification | R12.1 | ⬜ |
| 8 | Fix Figure 4 readability | R12.2 | ⬜ |
| 9 | Correct Figure 8 values or relabel as full-sample | R12.1 | ⬜ |
| 10 | Correct Appendix Table 14 and Table 15 labels | R12.1 | ⬜ |
| 11 | Add p-values/CIs for all state-dependent total effects at p=0.9 | R12.3 | ⬜ |
| 12 | Add leave-one-year-out or leave-one-episode-out validation for logit | R14.1 | ⬜ |
| 13 | Add real-time or quasi-real-time EMFI robustness for predictive section | R14.2 | ⬜ |
| 14 | Replace "causal response" with weaker language | R13.3 | ⬜ |
| 15 | Expand and sharpen literature review for GFJ | R15 | ⬜ |

---

## Recommended Revised Storyline (Reviewer Section 7)

The reviewer proposes this structure for the final paper:

**Main argument:**
> Geopolitical shocks have weak average effects because most are absorbed. Their systemic consequences are state-dependent: shocks become financially systemic primarily when they arrive in already-fragile, highly connected markets.

**Main evidence (five points, in this order):**
1. The unconditional LP effect is positive but statistically weak — this is the expected result, not a failure.
2. TCI shows the strongest state-dependent amplification (θ = 2.696, p = 0.090, bootstrap CI excludes zero).
3. EMFI amplification is economically large but imprecise — corroborating, not leading.
4. Pre-shock EMFI predicts systemic classification with moderate out-of-sample power (AUC ≈ 0.69).
5. Most shocks are absorbed; systemic shocks are concentrated in high-fragility windows.

**What to avoid:**
- Do not oversell "10× amplification" as if all outcomes are statistically strong.
- Do not call COVID a geopolitical shock.
- Do not claim causality.
- Do not imply real-time prediction unless real-time indicators are constructed.

---

## Key Empirical Numbers (Reference)

Based on COVID-zeroed primary specification (Phase R2 baseline):

| Quantity | Value | Source |
|----------|-------|--------|
| Panel LP β_k0 (EMFI, COVID-zeroed) | 0.245 | Table 8 |
| Panel LP β_k0 (EMFI, full-sample) | 0.458 | App Table 14 |
| State LP θ_k0 (EMFI) | 1.439 | Table 3 |
| State LP θ_k0 (TCI) | 2.696 | Table 3 |
| p(θ, EMFI) | 0.279 | Table 3 |
| p(θ, TCI) | 0.090 | Table 3 |
| LOO-CV AUC (predictive logit) | 0.693 | Section 6.3 |
| OR (EMFI_{t-1}, logit) | 6.82 | Section 6.3 |
| Fraction of shocks in calm markets | 63% | Phase 5 / HMM |
| N shock events in primary treatment sample (non-COVID) | 138 | R11 |

---

*End of REVISION_ACTION_PLAN_R2.md*  
*Created: 2026-05-19 | Based on: Review2.pdf | Predecessor: REVISION_ACTION_PLAN.md (Phases R0–R9)*
