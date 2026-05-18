"""
Tests for SP05 -- HMM Market-Implied Stress Regimes
=====================================================
Run from Paper_GFJ root:
  pytest subprojects/05_stress_regimes/test_hmm_regimes.py -v

Tests verify correctness of hmm_daily.csv and companion files
AFTER build_hmm_regimes.py has been run.  They also exercise the
build_hmm_regimes() builder function directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# -- Path bootstrap -----------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]
sys.path.insert(0, str(THIS_FILE.parent))

from build_hmm_regimes import (
    COMPONENT_COLS,
    N_RESTARTS,
    N_STATES,
    OUT_CSV,
    OUT_MANIFEST,
    OUT_PARAMS,
    STATE_CALM,
    STATE_ELEVATED,
    STATE_STRESS,
    build_hmm_regimes,
)

# -- Constants ----------------------------------------------------------------
IN_EMFI = GFJ_ROOT / "results" / "emfi" / "emfi_daily.csv"

EMFI_START = pd.Timestamp("2017-05-19")
EMFI_END   = pd.Timestamp("2025-10-15")
N_EMFI_DATES = 2179

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="module")
def emfi_df() -> pd.DataFrame:
    if not IN_EMFI.exists():
        pytest.skip("emfi_daily.csv not found -- run SP04 first")
    df = pd.read_csv(IN_EMFI, parse_dates=["date"])
    return df


@pytest.fixture(scope="module")
def hmm_out() -> pd.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip("hmm_daily.csv not generated -- run build_hmm_regimes.py first")
    return pd.read_csv(OUT_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not OUT_MANIFEST.exists():
        pytest.skip("manifest.json not generated")
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hmm_params() -> dict:
    if not OUT_PARAMS.exists():
        pytest.skip("hmm_params.json not generated")
    return json.loads(OUT_PARAMS.read_text(encoding="utf-8"))


# -- Unit tests: build_hmm_regimes() -----------------------------------------

class TestBuildHMMRegimes:
    """Smoke-test the builder function on real EMFI data."""

    @pytest.fixture(scope="class")
    def built(self, emfi_df):
        result, model = build_hmm_regimes(emfi_df, n_restarts=5)
        return result, model

    def test_returns_dataframe(self, built):
        result, model = built
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_3state(self, built):
        result, _ = built
        for col in ["date", "EMFI", "regime", "P_calm", "P_elevated", "P_stress"]:
            assert col in result.columns, "Missing column: " + col

    def test_date_is_datetime(self, built):
        result, _ = built
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_n_rows_matches_input(self, built, emfi_df):
        result, _ = built
        assert len(result) == len(emfi_df)

    def test_regime_values_in_range(self, built):
        result, _ = built
        assert set(result["regime"].unique()).issubset({0, 1, 2})

    def test_all_three_states_present(self, built):
        result, _ = built
        assert len(result["regime"].unique()) == 3

    def test_prob_sums_to_one(self, built):
        result, _ = built
        row_sums = result["P_calm"] + result["P_elevated"] + result["P_stress"]
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_probs_in_unit_interval(self, built):
        result, _ = built
        for col in ["P_calm", "P_elevated", "P_stress"]:
            assert (result[col] >= -1e-10).all(), col + " has values < 0"
            assert (result[col] <= 1.0 + 1e-10).all(), col + " has values > 1"

    def test_state_ordering_by_emfi(self, built):
        """State 0 (calm) should have lower mean EMFI than state 2 (stress)."""
        result, _ = built
        mean_emfi_by_state = result.groupby("regime")["EMFI"].mean()
        assert mean_emfi_by_state[STATE_CALM] < mean_emfi_by_state[STATE_ELEVATED]
        assert mean_emfi_by_state[STATE_ELEVATED] < mean_emfi_by_state[STATE_STRESS]

    def test_transition_matrix_rows_sum_to_one(self, built):
        _, model = built
        row_sums = model.transmat_.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_no_nans_in_result(self, built):
        result, _ = built
        assert not result[["P_calm", "P_elevated", "P_stress", "regime"]].isna().any().any()


# -- Integration tests: output CSV -------------------------------------------

class TestHMMCSV:
    def test_columns(self, hmm_out):
        for col in ["date", "EMFI", "regime", "P_calm", "P_elevated", "P_stress"]:
            assert col in hmm_out.columns, "Missing column: " + col

    def test_n_rows(self, hmm_out):
        assert len(hmm_out) == N_EMFI_DATES

    def test_date_range(self, hmm_out):
        assert hmm_out["date"].min() == EMFI_START
        assert hmm_out["date"].max() == EMFI_END

    def test_no_duplicate_dates(self, hmm_out):
        assert not hmm_out["date"].duplicated().any()

    def test_regime_values_valid(self, hmm_out):
        assert set(hmm_out["regime"].unique()).issubset({0, 1, 2})

    def test_prob_sums_to_one(self, hmm_out):
        row_sums = hmm_out["P_calm"] + hmm_out["P_elevated"] + hmm_out["P_stress"]
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_probs_nonnegative(self, hmm_out):
        assert (hmm_out["P_stress"] >= 0).all()
        assert (hmm_out["P_elevated"] >= 0).all()
        assert (hmm_out["P_calm"] >= 0).all()

    def test_no_nans(self, hmm_out):
        assert hmm_out[["P_calm", "P_elevated", "P_stress"]].notna().all().all()

    def test_all_three_states_present(self, hmm_out):
        assert len(hmm_out["regime"].unique()) == 3


# -- Validation tests --------------------------------------------------------

class TestHMMValidation:
    """Face-validity checks against known major events."""

    def test_calm_state_is_largest(self, hmm_out):
        """Calm state should have the most days (markets are calm most of the time)."""
        counts = hmm_out["regime"].value_counts()
        assert counts[STATE_CALM] > counts[STATE_ELEVATED]
        assert counts[STATE_CALM] > counts[STATE_STRESS]

    def test_systemic_stress_state_has_plausible_count(self, hmm_out):
        """Systemic stress should be between 50 and 700 days (2.3% – 32% of sample)."""
        n_stress = (hmm_out["regime"] == STATE_STRESS).sum()
        assert 50 <= n_stress <= 700, "Systemic stress count outside plausible range: " + str(n_stress)

    def test_pstress_high_on_covid_peak(self, hmm_out):
        """P_stress should be very high on 2020-03-12 (COVID crash peak)."""
        t = pd.Timestamp("2020-03-12")
        val = hmm_out.set_index("date").loc[t, "P_stress"]
        assert val >= 0.90, "Expected P_stress >= 0.90 on COVID peak, got " + str(val)

    def test_pstress_high_on_ukraine(self, hmm_out):
        """P_stress should be high on 2022-02-24 (Ukraine invasion)."""
        t = pd.Timestamp("2022-02-24")
        val = hmm_out.set_index("date").loc[t, "P_stress"]
        assert val >= 0.70, "Expected P_stress >= 0.70 on Ukraine invasion, got " + str(val)

    def test_regime_stress_on_covid_peak(self, hmm_out):
        """Viterbi regime on COVID peak should be STATE_STRESS (=2)."""
        t = pd.Timestamp("2020-03-12")
        reg = hmm_out.set_index("date").loc[t, "regime"]
        assert reg == STATE_STRESS, "Expected regime=2 on COVID peak, got " + str(reg)

    def test_pstress_near_zero_in_calm_2019(self, hmm_out):
        """During mid-2019 (calm period), P_stress should be very low on most days."""
        calm_period = hmm_out[
            (hmm_out["date"] >= "2019-01-01") & (hmm_out["date"] <= "2019-10-01")
        ]
        low_stress_frac = (calm_period["P_stress"] < 0.2).mean()
        assert low_stress_frac >= 0.80, "Expected >=80%% of 2019 days with P_stress<0.2, got " + str(low_stress_frac)

    def test_state_ordering_emfi(self, hmm_out):
        """Mean EMFI should be strictly increasing from state 0 to state 2."""
        means = hmm_out.groupby("regime")["EMFI"].mean()
        assert means[0] < means[1] < means[2]

    def test_calm_state_persistence_plausible(self, hmm_params):
        """Calm state should have highest persistence (diagonal >= 0.5)."""
        transmat = np.array(hmm_params["transmat"])
        calm_persistence = transmat[STATE_CALM, STATE_CALM]
        assert calm_persistence >= 0.50, "Calm persistence < 0.50: " + str(calm_persistence)

    def test_hamas_trading_day_pstress(self, hmm_out):
        """Hamas attack (2023-10-07, a Sunday) -- nearest trading day should exist."""
        # 2023-10-09 is the first trading day after the weekend attack
        t = pd.Timestamp("2023-10-09")
        if t in hmm_out["date"].values:
            val = hmm_out.set_index("date").loc[t, "P_stress"]
            # No strong prior; just check it is in [0, 1]
            assert 0.0 <= val <= 1.0
        else:
            pytest.skip("2023-10-09 not in hmm sample")


# -- Parameter tests ---------------------------------------------------------

class TestHMMParams:
    def test_required_keys(self, hmm_params):
        for key in ["n_states", "transmat", "means", "covars", "startprob",
                    "component_cols", "converged"]:
            assert key in hmm_params

    def test_n_states(self, hmm_params):
        assert hmm_params["n_states"] == N_STATES

    def test_transmat_shape(self, hmm_params):
        tm = np.array(hmm_params["transmat"])
        assert tm.shape == (N_STATES, N_STATES)

    def test_transmat_rows_sum_to_one(self, hmm_params):
        tm = np.array(hmm_params["transmat"])
        assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-6)

    def test_means_shape(self, hmm_params):
        means = np.array(hmm_params["means"])
        assert means.shape == (N_STATES, len(COMPONENT_COLS))

    def test_component_cols(self, hmm_params):
        assert hmm_params["component_cols"] == COMPONENT_COLS

    def test_startprob_sums_to_one(self, hmm_params):
        sp = np.array(hmm_params["startprob"])
        assert abs(sp.sum() - 1.0) < 1e-6


# -- Manifest tests ----------------------------------------------------------

class TestManifest:
    def test_required_keys(self, manifest):
        for key in ["generated_at", "n_dates", "date_range", "n_states",
                    "component_cols", "state_counts", "state_persistence",
                    "event_pstress", "outputs"]:
            assert key in manifest

    def test_n_dates(self, manifest):
        assert manifest["n_dates"] == N_EMFI_DATES

    def test_date_range(self, manifest):
        assert manifest["date_range"]["start"] == str(EMFI_START.date())
        assert manifest["date_range"]["end"]   == str(EMFI_END.date())

    def test_n_states(self, manifest):
        assert manifest["n_states"] == N_STATES

    def test_state_counts_sum(self, manifest):
        sc = manifest["state_counts"]
        assert sc["calm"] + sc["elevated"] + sc["stress"] == N_EMFI_DATES

    def test_covid_peak_pstress_in_manifest(self, manifest):
        assert manifest["event_pstress"]["covid_peak"] is not None
        assert manifest["event_pstress"]["covid_peak"] >= 0.90

    def test_ukraine_pstress_in_manifest(self, manifest):
        assert manifest["event_pstress"]["ukraine"] is not None
        assert manifest["event_pstress"]["ukraine"] >= 0.70

    def test_hamas_pstress_null_in_manifest(self, manifest):
        """Hamas 2023-10-07 is a Sunday so falls outside the trading-day sample."""
        assert manifest["event_pstress"]["hamas"] is None

    def test_outputs_exist(self, manifest):
        for key, path_str in manifest["outputs"].items():
            p = Path(path_str)
            assert p.exists(), "Output file missing: " + key + " -> " + path_str
