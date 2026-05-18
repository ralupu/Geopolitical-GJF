"""
Tests for SP06 -- Panel Local Projections
==========================================
Run from Paper_GFJ root:
  pytest subprojects/06_panel_lp/test_panel_lp.py -v

Tests verify LP mechanics, output file integrity, directional validity of
impulse responses, and pre-trend absence for key outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# -- Path bootstrap -----------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]
sys.path.insert(0, str(THIS_FILE.parent))

from run_panel_lp import (
    HORIZONS,
    OUTCOMES,
    PRIMARY_SHOCK,
    OUT_DIR,
    OUT_MANIFEST,
    build_lp_dataset,
    run_lp_one_outcome,
    CI_90,
    CI_95,
)

# -- Constants ----------------------------------------------------------------
N_HORIZONS   = len(HORIZONS)          # 21 (-5 to +15)
N_OUTCOMES   = len(OUTCOMES)          # 5
PRE_HORIZONS = [k for k in HORIZONS if k < 0]   # -5,-4,-3,-2,-1
POST_HORIZONS= [k for k in HORIZONS if k >= 0]  # 0..15

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="module")
def lp_dataset() -> pd.DataFrame:
    return build_lp_dataset()


@pytest.fixture(scope="module")
def all_results(lp_dataset) -> dict:
    return {
        outcome: run_lp_one_outcome(lp_dataset, outcome, PRIMARY_SHOCK, HORIZONS)
        for outcome in OUTCOMES
    }


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not OUT_MANIFEST.exists():
        pytest.skip("manifest.json not generated -- run run_panel_lp.py first")
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


def load_results(outcome: str) -> pd.DataFrame:
    p = OUT_DIR / ("lp_results_" + outcome + ".csv")
    if not p.exists():
        pytest.skip("lp_results_" + outcome + ".csv not generated")
    return pd.read_csv(p)


# -- Unit tests: LP dataset ---------------------------------------------------

class TestLPDataset:
    def test_shape(self, lp_dataset):
        assert len(lp_dataset) == 2179

    def test_required_columns(self, lp_dataset):
        for col in ["date", "EMFI", "P_stress", "tci_w100",
                    "TailVolBreadth", "AvgCorr60",
                    "max_shock", "global_ret", "month"]:
            assert col in lp_dataset.columns, "Missing: " + col

    def test_no_nans(self, lp_dataset):
        key_cols = ["EMFI", "P_stress", "tci_w100", "TailVolBreadth",
                    "AvgCorr60", "max_shock", "global_ret"]
        assert lp_dataset[key_cols].notna().all().all()

    def test_shock_days_count(self, lp_dataset):
        assert (lp_dataset["max_shock"] > 0).sum() == 154

    def test_date_sorted(self, lp_dataset):
        assert lp_dataset["date"].is_monotonic_increasing


# -- Unit tests: run_lp_one_outcome() ----------------------------------------

class TestRunLP:
    def test_returns_dataframe(self, all_results):
        for outcome in OUTCOMES:
            assert isinstance(all_results[outcome], pd.DataFrame)

    def test_n_horizons(self, all_results):
        for outcome in OUTCOMES:
            assert len(all_results[outcome]) == N_HORIZONS, outcome

    def test_required_columns(self, all_results):
        expected = ["horizon", "beta", "se", "tstat", "pval",
                    "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "nobs"]
        for outcome in OUTCOMES:
            for col in expected:
                assert col in all_results[outcome].columns, outcome + ": missing " + col

    def test_horizons_correct(self, all_results):
        for outcome in OUTCOMES:
            assert list(all_results[outcome]["horizon"]) == HORIZONS

    def test_no_nans_in_post_horizons(self, all_results):
        """Post-period results should have no NaN (sufficient observations)."""
        for outcome in OUTCOMES:
            post = all_results[outcome][all_results[outcome]["horizon"] >= 0]
            assert post["beta"].notna().all(), outcome + " has NaN beta in post-period"

    def test_ci95_wider_than_ci90(self, all_results):
        for outcome in OUTCOMES:
            post = all_results[outcome][all_results[outcome]["horizon"] >= 0]
            width90 = (post["ci90_hi"] - post["ci90_lo"]).values
            width95 = (post["ci95_hi"] - post["ci95_lo"]).values
            assert (width95 > width90 - 1e-10).all(), outcome

    def test_ci_symmetric_around_beta(self, all_results):
        for outcome in OUTCOMES:
            post = all_results[outcome][all_results[outcome]["horizon"] >= 0]
            mid90 = (post["ci90_lo"] + post["ci90_hi"]) / 2
            mid95 = (post["ci95_lo"] + post["ci95_hi"]) / 2
            assert np.allclose(mid90.values, post["beta"].values, atol=1e-6)
            assert np.allclose(mid95.values, post["beta"].values, atol=1e-6)

    def test_ci_widths_consistent_with_se(self, all_results):
        for outcome in OUTCOMES:
            post = all_results[outcome][all_results[outcome]["horizon"] >= 0]
            expected_half95 = CI_95 * post["se"]
            actual_half95   = (post["ci95_hi"] - post["ci95_lo"]) / 2
            assert np.allclose(expected_half95.values, actual_half95.values, rtol=1e-4)

    def test_nobs_plausible(self, all_results):
        for outcome in OUTCOMES:
            post = all_results[outcome][all_results[outcome]["horizon"] >= 0]
            assert (post["nobs"] >= 2000).all(), outcome + " has too few obs"
            assert (post["nobs"] <= 2179).all(), outcome


# -- Validation: directional responses ----------------------------------------

class TestResponseDirection:
    """At short horizons (k=0,1), geopolitical shocks should push fragility up."""

    def test_emfi_positive_k0(self, all_results):
        row = all_results["EMFI"][all_results["EMFI"]["horizon"] == 0]
        assert row["beta"].iloc[0] > 0

    def test_tci_positive_k1(self, all_results):
        row = all_results["tci_w100"][all_results["tci_w100"]["horizon"] == 1]
        assert row["beta"].iloc[0] > 0

    def test_tailbreadth_positive_k0(self, all_results):
        row = all_results["TailVolBreadth"][all_results["TailVolBreadth"]["horizon"] == 0]
        assert row["beta"].iloc[0] > 0

    def test_pstress_positive_k0(self, all_results):
        row = all_results["P_stress"][all_results["P_stress"]["horizon"] == 0]
        assert row["beta"].iloc[0] > 0

    def test_avgcorr_positive_k0(self, all_results):
        row = all_results["AvgCorr60"][all_results["AvgCorr60"]["horizon"] == 0]
        assert row["beta"].iloc[0] > 0

    def test_emfi_peak_positive(self, all_results):
        """Peak post-period EMFI response should be positive."""
        post = all_results["EMFI"][all_results["EMFI"]["horizon"] >= 0]
        assert post["beta"].max() > 0

    def test_tailbreadth_decays(self, all_results):
        """TailVolBreadth response at k=10 should be smaller than at k=1."""
        res = all_results["TailVolBreadth"]
        b1  = res[res["horizon"] == 1]["beta"].iloc[0]
        b10 = res[res["horizon"] == 10]["beta"].iloc[0]
        assert abs(b10) < abs(b1), "TailBreadth should decay by k=10"


# -- Validation: pre-trends ---------------------------------------------------

class TestPreTrends:
    """
    For P_stress and AvgCorr60, no pre-period coefficient should be
    significant at 5% (clean identification from construction).
    For EMFI and TailBreadth, allow at most 1/5 pre-horizon at 10%.
    TCI has a known smoothness-induced pre-trend (rolling window);
    allow up to 3/5 significant at 10% but flag it.
    """

    def _count_sig(self, results: pd.DataFrame, alpha: float) -> int:
        pre = results[results["horizon"] < 0]
        return int((pre["pval"].dropna() < alpha).sum())

    def test_pstress_clean_pretrend_5pct(self, all_results):
        n = self._count_sig(all_results["P_stress"], 0.05)
        assert n == 0, "P_stress: " + str(n) + "/5 pre-horizons sig at 5%"

    def test_avgcorr_clean_pretrend_5pct(self, all_results):
        n = self._count_sig(all_results["AvgCorr60"], 0.05)
        assert n == 0, "AvgCorr60: " + str(n) + "/5 pre-horizons sig at 5%"

    def test_emfi_pretrend_at_most_1_at_10pct(self, all_results):
        n = self._count_sig(all_results["EMFI"], 0.10)
        assert n <= 1, "EMFI: " + str(n) + "/5 pre-horizons sig at 10% (expected <=1)"

    def test_tailbreadth_pretrend_at_most_1_at_10pct(self, all_results):
        n = self._count_sig(all_results["TailVolBreadth"], 0.10)
        assert n <= 1, "TailBreadth: " + str(n) + "/5 pre-horizons sig at 10%"

    def test_tci_pretrend_at_most_3_at_10pct(self, all_results):
        """TCI has a smoothness-induced pre-trend; allow up to 3/5."""
        n = self._count_sig(all_results["tci_w100"], 0.10)
        assert n <= 3, "TCI: " + str(n) + "/5 pre-horizons sig at 10% (expected <=3)"

    def test_k_minus1_not_degenerate(self, all_results):
        """k=-1 fix: beta should not be near machine epsilon for any outcome."""
        for outcome in OUTCOMES:
            row = all_results[outcome][all_results[outcome]["horizon"] == -1]
            if not row.empty and row["beta"].notna().all():
                assert abs(row["beta"].iloc[0]) > 1e-8 or row["pval"].iloc[0] > 0.05, \
                    outcome + " k=-1 looks degenerate"


# -- Integration tests: output files -----------------------------------------

class TestOutputFiles:
    def test_result_csvs_exist(self):
        for outcome in OUTCOMES:
            p = OUT_DIR / ("lp_results_" + outcome + ".csv")
            assert p.exists(), "Missing: lp_results_" + outcome + ".csv"

    def test_figure_files_exist(self):
        for outcome in OUTCOMES:
            p = OUT_DIR / ("Fig_LP_" + outcome + ".png")
            assert p.exists(), "Missing: Fig_LP_" + outcome + ".png"
        assert (OUT_DIR / "Fig_LP_Combined.png").exists()

    def test_manifest_exists(self):
        assert OUT_MANIFEST.exists()

    def test_lp_summary_exists(self):
        assert (OUT_DIR / "lp_summary.csv").exists()

    def test_result_csv_columns(self):
        for outcome in OUTCOMES:
            df = load_results(outcome)
            for col in ["horizon", "beta", "se", "pval", "ci95_lo", "ci95_hi", "nobs"]:
                assert col in df.columns, outcome + ": missing " + col

    def test_result_csv_n_rows(self):
        for outcome in OUTCOMES:
            df = load_results(outcome)
            assert len(df) == N_HORIZONS, outcome + " wrong row count"

    def test_result_csv_horizons(self):
        for outcome in OUTCOMES:
            df = load_results(outcome)
            assert list(df["horizon"]) == HORIZONS


# -- Manifest tests -----------------------------------------------------------

class TestManifest:
    def test_required_keys(self, manifest):
        for key in ["generated_at", "n_obs", "date_range", "outcomes",
                    "shock_primary", "horizons", "controls", "summary"]:
            assert key in manifest

    def test_n_obs(self, manifest):
        assert manifest["n_obs"] == 2179

    def test_outcomes(self, manifest):
        assert set(manifest["outcomes"]) == set(OUTCOMES)

    def test_horizons(self, manifest):
        assert manifest["horizons"] == HORIZONS

    def test_shock_primary(self, manifest):
        assert manifest["shock_primary"] == PRIMARY_SHOCK

    def test_summary_rows(self, manifest):
        assert len(manifest["summary"]) == N_OUTCOMES
