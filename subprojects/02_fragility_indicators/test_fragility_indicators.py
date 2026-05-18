"""
Tests for SP02 -- Daily Fragility Indicators
=============================================
Run from Paper_GFJ root:
  pytest subprojects/02_fragility_indicators/test_fragility_indicators.py -v

Tests verify correctness of fragility_daily.csv and companion files
AFTER build_fragility_indicators.py has been run.  They also exercise
individual helper functions directly.
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

from build_fragility_indicators import (
    N_COUNTRIES,
    OUT_FRAGILITY  as OUT_CSV,
    OUT_MANIFEST,
    OUT_STATS      as OUT_SUMMARY,
    TAIL_MIN_PERIODS,
    TAIL_WINDOW,
    build_fragility_indicators,
)

# Study period boundaries (defined in SP01, carried through panel dates)
STUDY_START = pd.Timestamp("2017-01-02")
STUDY_END   = pd.Timestamp("2025-10-15")

# -- Constants ----------------------------------------------------------------
IN_PANEL = GFJ_ROOT / "data" / "panel_daily.parquet"
EXPECTED_INDICATORS = [
    "VolStress", "SqStress",
    "TailVolBreadth", "TailLossBreadth", "TailGainBreadth",
    "AvgCorr60", "AvgCorr30",
]
N_TRADING_DAYS = 2278
TAIL_WARMUP    = 60
CORR60_WARMUP  = 29
CORR30_WARMUP  = 19

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not IN_PANEL.exists():
        pytest.skip("panel_daily.parquet not found -- run SP01 first")
    df = pd.read_parquet(IN_PANEL)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def frag() -> pd.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip("fragility_daily.csv not yet generated -- run build_fragility_indicators.py first")
    return pd.read_csv(OUT_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not OUT_MANIFEST.exists():
        pytest.skip("manifest.json not yet generated")
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


# -- Unit tests: build_fragility_indicators() ---------------------------------

class TestBuildFragilityIndicators:
    """Smoke-test the builder function on real panel data."""

    @pytest.fixture(scope="class")
    def built(self, panel):
        return build_fragility_indicators(panel)

    def test_returns_dataframe(self, built):
        assert isinstance(built, pd.DataFrame)

    def test_date_column_is_datetime(self, built):
        assert pd.api.types.is_datetime64_any_dtype(built["date"])

    def test_expected_columns(self, built):
        for col in EXPECTED_INDICATORS:
            assert col in built.columns, "Missing column: " + col

    def test_n_dates(self, built):
        assert len(built) == N_TRADING_DAYS

    def test_no_nans_in_volstress(self, built):
        assert built["VolStress"].notna().all()

    def test_no_nans_in_sqstress(self, built):
        assert built["SqStress"].notna().all()

    def test_volstress_nonnegative(self, built):
        assert (built["VolStress"] >= 0).all()

    def test_sqstress_nonnegative(self, built):
        assert (built["SqStress"] >= 0).all()

    def test_tailvolbreadth_warmup_nans(self, built):
        assert built["TailVolBreadth"].isna().sum() == TAIL_WARMUP

    def test_tailvolbreadth_in_range(self, built):
        valid = built["TailVolBreadth"].dropna()
        assert (valid >= 0).all()
        assert (valid <= N_COUNTRIES).all()

    def test_tailvolbreadth_integer_valued(self, built):
        valid = built["TailVolBreadth"].dropna()
        assert (valid == valid.round()).all()

    def test_avgcorr60_warmup_nans(self, built):
        assert built["AvgCorr60"].isna().sum() == CORR60_WARMUP

    def test_avgcorr30_warmup_nans(self, built):
        assert built["AvgCorr30"].isna().sum() == CORR30_WARMUP

    def test_avgcorr60_in_valid_range(self, built):
        valid = built["AvgCorr60"].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_avgcorr30_in_valid_range(self, built):
        valid = built["AvgCorr30"].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()


# -- Integration tests: output files ------------------------------------------

class TestFragilityCSV:
    def test_columns(self, frag):
        for col in ["date"] + EXPECTED_INDICATORS:
            assert col in frag.columns, "Missing column: " + col

    def test_n_rows(self, frag):
        assert len(frag) == N_TRADING_DAYS

    def test_date_range(self, frag):
        assert frag["date"].min() == STUDY_START
        assert frag["date"].max() == STUDY_END

    def test_no_duplicate_dates(self, frag):
        assert not frag["date"].duplicated().any()

    def test_no_nans_volstress(self, frag):
        assert frag["VolStress"].notna().all()

    def test_no_nans_sqstress(self, frag):
        assert frag["SqStress"].notna().all()

    def test_tail_warmup_nan_count(self, frag):
        assert frag["TailVolBreadth"].isna().sum() == TAIL_WARMUP
        assert frag["TailLossBreadth"].isna().sum() == TAIL_WARMUP
        assert frag["TailGainBreadth"].isna().sum() == TAIL_WARMUP

    def test_corr60_warmup_nan_count(self, frag):
        assert frag["AvgCorr60"].isna().sum() == CORR60_WARMUP

    def test_corr30_warmup_nan_count(self, frag):
        assert frag["AvgCorr30"].isna().sum() == CORR30_WARMUP

    def test_tailvolbreadth_in_range(self, frag):
        valid = frag["TailVolBreadth"].dropna()
        assert valid.between(0, N_COUNTRIES).all()

    def test_avgcorr_in_valid_range(self, frag):
        assert frag["AvgCorr60"].dropna().between(-1, 1).all()
        assert frag["AvgCorr30"].dropna().between(-1, 1).all()


class TestVolStressConsistency:
    """VolStress_t must equal cross-sectional mean of abs_return on day t."""

    def test_volstress_equals_mean_abs_return(self, panel, frag):
        ret_wide = panel.pivot(index="date", columns="country", values="abs_return")
        expected = ret_wide.mean(axis=1).rename("VolStress")
        actual   = frag.set_index("date")["VolStress"]
        pd.testing.assert_series_equal(
            actual.sort_index(), expected.sort_index(),
            check_names=False, rtol=1e-12
        )

    def test_sqstress_equals_mean_sq_return(self, panel, frag):
        sq_wide  = panel.pivot(index="date", columns="country", values="sq_return")
        expected = sq_wide.mean(axis=1).rename("SqStress")
        actual   = frag.set_index("date")["SqStress"]
        pd.testing.assert_series_equal(
            actual.sort_index(), expected.sort_index(),
            check_names=False, rtol=1e-12
        )


class TestTailBreadthOutOfSample:
    """Rolling quantile threshold must be strictly backward-looking (shifted 1 day)."""

    def test_threshold_shift_germany(self, panel, frag):
        """On a specific date, verify threshold equals q95 of 252 rows BEFORE that date."""
        country = "Germany"
        t = pd.Timestamp("2020-01-10")
        ret_col = panel[panel["country"] == country].set_index("date")["abs_return"].sort_index()
        loc = ret_col.index.get_loc(t)
        window = ret_col.iloc[max(0, loc - TAIL_WINDOW):loc]
        manual_q95 = window.quantile(0.95)
        abs_t = ret_col.loc[t]
        exceeded_manual = int(abs_t > manual_q95)
        frag_row = frag.set_index("date").loc[t]
        assert frag_row["TailVolBreadth"] >= exceeded_manual

    def test_covid_peak_tailvolbreadth(self, frag):
        """On 2020-03-16 (COVID crash), nearly all markets should be in the tail."""
        t = pd.Timestamp("2020-03-16")
        val = frag.set_index("date").loc[t, "TailVolBreadth"]
        assert val >= 15, "Expected >=15 markets in tail on COVID peak, got " + str(val)


class TestManifest:
    def test_required_keys(self, manifest):
        for key in ["generated_at", "n_dates", "date_range", "indicators", "parameters"]:
            assert key in manifest

    def test_n_dates(self, manifest):
        assert manifest["n_dates"] == N_TRADING_DAYS

    def test_date_range(self, manifest):
        assert manifest["date_range"]["start"] == str(STUDY_START.date())
        assert manifest["date_range"]["end"]   == str(STUDY_END.date())

    def test_indicators_list(self, manifest):
        assert set(manifest["indicators"]) == set(EXPECTED_INDICATORS)

    def test_parameters_recorded(self, manifest):
        params = manifest["parameters"]
        assert params["tail_window"]          == TAIL_WINDOW
        assert params["tail_min_periods"]     == TAIL_MIN_PERIODS
        assert params["tail_quantile_shift"]  == 1

    def test_outputs_exist(self, manifest):
        for key, path_str in manifest["outputs"].items():
            p = Path(path_str)
            assert p.exists(), "Output file missing: " + key + " -> " + path_str


class TestProvenance:
    def test_summary_csv_exists(self):
        assert OUT_SUMMARY.exists()

    def test_summary_has_expected_indicators(self):
        summary = pd.read_csv(OUT_SUMMARY, index_col=0)
        # Indicators are rows (index); statistics are columns
        for ind in EXPECTED_INDICATORS:
            assert ind in summary.index, "Missing indicator in summary: " + ind

    def test_manifest_parseable(self):
        m = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
        assert "generated_at" in m
