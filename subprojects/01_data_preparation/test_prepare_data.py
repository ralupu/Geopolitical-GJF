"""
Tests for SP01 — Data Preparation
==================================
Run from Paper_GFJ root:
  pytest subprojects/01_data_preparation/test_prepare_data.py -v

Tests verify the correctness of panel_daily.parquet and companion files
AFTER prepare_data.py has been run. They also test the individual helper
functions directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]
sys.path.insert(0, str(THIS_FILE.parent))

from prepare_data import (
    N_COUNTRIES,
    OUT_AGG_SHOCKS,
    OUT_CLUSTERS,
    OUT_EVENTS,
    OUT_MANIFEST,
    OUT_PANEL,
    OUT_SUMMARY,
    STUDY_END,
    STUDY_START,
    build_aggregate_shocks,
    build_long_panel,
    build_study_period,
    load_clusters,
    load_events,
    load_returns,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not OUT_PANEL.exists():
        pytest.skip("panel_daily.parquet not yet generated — run prepare_data.py first")
    return pd.read_parquet(OUT_PANEL)


@pytest.fixture(scope="module")
def agg() -> pd.DataFrame:
    if not OUT_AGG_SHOCKS.exists():
        pytest.skip("aggregate_shocks.csv not yet generated")
    return pd.read_csv(OUT_AGG_SHOCKS, parse_dates=["date"])


@pytest.fixture(scope="module")
def events_out() -> pd.DataFrame:
    if not OUT_EVENTS.exists():
        pytest.skip("shocks_events.csv not yet generated")
    return pd.read_csv(OUT_EVENTS, parse_dates=["date"])


@pytest.fixture(scope="module")
def clusters_out() -> pd.DataFrame:
    if not OUT_CLUSTERS.exists():
        pytest.skip("clusters.csv not yet generated")
    return pd.read_csv(OUT_CLUSTERS)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not OUT_MANIFEST.exists():
        pytest.skip("manifest.json not yet generated")
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


# ── Unit tests: helper functions ─────────────────────────────────────────────

class TestLoadReturns:
    def test_shape(self):
        rets = load_returns()
        assert rets.shape[1] == N_COUNTRIES, f"Expected {N_COUNTRIES} countries"

    def test_no_nans(self):
        rets = load_returns()
        # Some zero-return days are allowed (non-trading days preserved as 0)
        # but no NaNs after diff+dropna
        assert not rets.isnull().values.any(), "NaN values in returns"

    def test_date_range(self):
        rets = load_returns()
        # After log-diff, first date must be at or after 2017-01-02
        assert rets.index.min() >= pd.Timestamp("2017-01-02")
        assert rets.index.max() == pd.Timestamp("2025-10-15")


class TestLoadEvents:
    def test_required_columns(self):
        ev = load_events()
        required = {"date", "country", "shock_intensity", "q_value", "p_value"}
        assert required.issubset(set(ev.columns))

    def test_s_formula(self):
        """S = -log(q_value) exactly."""
        ev = load_events()
        computed = -np.log(ev["q_value"])
        np.testing.assert_allclose(ev["shock_intensity"], computed, rtol=1e-10)

    def test_s_nonnegative(self):
        ev = load_events()
        assert (ev["shock_intensity"] >= 0).all()

    def test_no_duplicate_country_date(self):
        ev = load_events()
        assert not ev.duplicated(["date", "country"]).any()

    def test_all_countries_present(self):
        ev = load_events()
        clusters = load_clusters()
        all_countries = set(clusters["country"].str.strip())
        event_countries = set(ev["country"].str.strip())
        assert event_countries.issubset(all_countries), \
            f"Unknown countries in events: {event_countries - all_countries}"


class TestLoadClusters:
    def test_n_countries(self):
        cl = load_clusters()
        assert len(cl) == N_COUNTRIES

    def test_cluster_values(self):
        cl = load_clusters()
        assert set(cl["cluster"].unique()).issubset({1, 2, 3})

    def test_no_duplicate_countries(self):
        cl = load_clusters()
        assert not cl["country"].duplicated().any()


class TestBuildStudyPeriod:
    def test_returns_datetimeindex(self):
        rets = load_returns()
        sp = build_study_period(rets, set(rets.index))  # trivial case
        assert isinstance(sp, pd.DatetimeIndex)

    def test_excludes_dates_not_in_conflict(self):
        rets = load_returns()
        # Remove a few dates from conflict set
        conflict_set = set(rets.index) - {rets.index[10], rets.index[20]}
        sp = build_study_period(rets, conflict_set)
        assert rets.index[10] not in sp
        assert rets.index[20] not in sp

    def test_within_study_window(self):
        rets = load_returns()
        sp = build_study_period(rets, set(rets.index))
        assert sp.min() >= STUDY_START
        assert sp.max() <= STUDY_END


# ── Integration tests: output files ─────────────────────────────────────────

class TestPanel:
    EXPECTED_COLS = {
        "date", "country", "return", "abs_return", "sq_return",
        "shock_intensity", "cluster",
    }

    def test_columns(self, panel):
        assert self.EXPECTED_COLS.issubset(set(panel.columns))

    def test_n_countries(self, panel):
        assert panel["country"].nunique() == N_COUNTRIES

    def test_all_expected_countries(self, panel):
        expected = {
            "Austria", "Belgium", "Bulgaria", "Croatia", "Finland", "France",
            "Germany", "Greece", "Hungary", "Ireland", "Italy", "Netherlands",
            "Norway", "Poland", "Portugal", "Romania", "Spain", "Sweden",
            "United Kingdom",
        }
        assert set(panel["country"].unique()) == expected

    def test_date_range(self, panel):
        panel["date"] = pd.to_datetime(panel["date"])
        assert panel["date"].min() == STUDY_START
        assert panel["date"].max() == STUDY_END

    def test_no_nans_key_columns(self, panel):
        for col in ["date", "country", "return", "abs_return", "sq_return",
                    "shock_intensity", "cluster"]:
            assert panel[col].notna().all(), f"NaN in column: {col}"

    def test_abs_return_consistency(self, panel):
        """abs_return must equal |return| for every row."""
        np.testing.assert_allclose(
            panel["abs_return"].values,
            panel["return"].abs().values,
            rtol=1e-12,
        )

    def test_sq_return_consistency(self, panel):
        """sq_return must equal return² for every row."""
        np.testing.assert_allclose(
            panel["sq_return"].values,
            (panel["return"] ** 2).values,
            rtol=1e-12,
        )

    def test_shock_intensity_nonnegative(self, panel):
        assert (panel["shock_intensity"] >= 0).all()

    def test_cluster_range(self, panel):
        assert panel["cluster"].between(1, 3).all()

    def test_panel_balanced(self, panel):
        """Every date must have exactly N_COUNTRIES rows."""
        panel["date"] = pd.to_datetime(panel["date"])
        counts = panel.groupby("date")["country"].count()
        assert (counts == N_COUNTRIES).all(), \
            f"Unbalanced dates: {counts[counts != N_COUNTRIES]}"

    def test_n_shock_events_matches_events_file(self, panel, events_out):
        """Rows with shock_intensity > 0 must equal len(events_out) after alignment."""
        n_shocked = (panel["shock_intensity"] > 0).sum()
        assert n_shocked == len(events_out), \
            f"Panel shock rows ({n_shocked}) ≠ events file ({len(events_out)})"

    def test_shock_s_formula(self, panel, events_out):
        """For every aligned event, panel S must equal -log(q_value) from events file."""
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        events_out = events_out.copy()
        events_out["date"] = pd.to_datetime(events_out["date"])

        shocked = panel[panel["shock_intensity"] > 0].copy()
        # Match on aligned date (which is the trading day after forward-fill)
        merged = shocked.merge(
            events_out[["date", "country", "q_value"]],
            on=["date", "country"],
            how="inner",
        )
        assert len(merged) == len(shocked), \
            f"Not all shocked panel rows found in events file: {len(merged)} vs {len(shocked)}"
        expected_s = -np.log(merged["q_value"])
        np.testing.assert_allclose(
            merged["shock_intensity"].values,
            expected_s.values,
            rtol=1e-10,
        )

    def test_all_shock_dates_are_trading_days(self, panel, events_out):
        """All event dates in the output CSV must be trading days (in the panel)."""
        panel_dates = set(pd.to_datetime(panel["date"]).unique())
        event_dates = set(pd.to_datetime(events_out["date"]))
        assert event_dates.issubset(panel_dates), \
            f"Event dates not in panel: {event_dates - panel_dates}"

    def test_original_date_column_exists(self, events_out):
        """events CSV must record original_date for traceability."""
        assert "original_date" in events_out.columns


class TestAggregateShocks:
    def test_columns(self, agg):
        required = {"date", "max_shock", "breadth_shock", "sum_shock", "avg_shock_pos"}
        assert required.issubset(set(agg.columns))

    def test_all_nonnegative(self, agg):
        for col in ["max_shock", "breadth_shock", "sum_shock", "avg_shock_pos"]:
            assert (agg[col] >= 0).all(), f"Negative values in {col}"

    def test_max_shock_le_sum(self, agg):
        """max_shock can't exceed sum_shock (which sums all countries)."""
        assert (agg["max_shock"] <= agg["sum_shock"] + 1e-10).all()

    def test_breadth_le_n_countries(self, agg):
        assert (agg["breadth_shock"] <= N_COUNTRIES).all()

    def test_max_shock_consistent_with_panel(self, panel, agg):
        """max_shock_t == max S across countries on that day."""
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        agg = agg.copy()
        agg["date"] = pd.to_datetime(agg["date"])

        panel_max = panel.groupby("date")["shock_intensity"].max().reset_index()
        panel_max.columns = ["date", "max_shock_panel"]
        merged = agg.merge(panel_max, on="date")
        np.testing.assert_allclose(
            merged["max_shock"].values,
            merged["max_shock_panel"].values,
            rtol=1e-10,
        )

    def test_n_shock_days(self, agg, manifest):
        """Days with max_shock > 0 must match manifest."""
        n = int((agg["max_shock"] > 0).sum())
        assert n == manifest["shock_series"]["n_shock_days"]


class TestClusters:
    def test_n_rows(self, clusters_out):
        assert len(clusters_out) == N_COUNTRIES

    def test_cluster_values(self, clusters_out):
        assert set(clusters_out["cluster"].unique()).issubset({1, 2, 3})


class TestManifest:
    def test_study_period(self, manifest):
        assert manifest["study_period"]["n_countries"] == N_COUNTRIES
        assert manifest["study_period"]["start"] == str(STUDY_START.date())
        assert manifest["study_period"]["end"] == str(STUDY_END.date())

    def test_n_events_raw(self, manifest):
        """278 events before alignment — a fixed property of the BIR paper shock series."""
        assert manifest["shock_series"]["n_events_raw"] == 278

    def test_n_events_le_raw(self, manifest):
        """After alignment, collisions can only reduce the count."""
        assert manifest["shock_series"]["n_events"] <= manifest["shock_series"]["n_events_raw"]

    def test_forward_fill_count(self, manifest):
        """Most events should be forward-filled (181 are on weekends in the raw series)."""
        assert manifest["shock_series"]["n_events_forward_filled"] >= 150

    def test_sources_recorded(self, manifest):
        assert "prices" in manifest["sources"]
        assert "events" in manifest["sources"]


class TestProvenance:
    def test_summary_exists(self):
        assert OUT_SUMMARY.exists()

    def test_manifest_parseable(self):
        m = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
        assert "generated_at" in m
        assert "study_period" in m
