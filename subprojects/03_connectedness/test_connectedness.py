"""
Tests for SP03 -- Volatility Connectedness (TCI)
=================================================
Run from Paper_GFJ root:
  pytest subprojects/03_connectedness/test_connectedness.py -v

Tests verify correctness of tci_daily.csv and companion files
AFTER build_connectedness.py has been run.  They also test the
core GFEVD computation directly.
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

from build_connectedness import (
    H_HORIZON,
    N_COUNTRIES,
    OUT_DIR_CONN,
    OUT_MANIFEST,
    OUT_SUMMARY,
    OUT_TCI,
    PRIMARY_W,
    STUDY_END,
    STUDY_START,
    VAR_MAX_P,
    WINDOWS,
    compute_gfevd,
    load_vol_panel,
    rolling_connectedness,
)

# -- Constants ----------------------------------------------------------------
IN_PANEL = GFJ_ROOT / "data" / "panel_daily.parquet"
TCI_COLS = [f"tci_w{W}" for W in WINDOWS]

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="module")
def vol_wide():
    if not IN_PANEL.exists():
        pytest.skip("panel_daily.parquet not found -- run SP01 first")
    return load_vol_panel()


@pytest.fixture(scope="module")
def tci():
    if not OUT_TCI.exists():
        pytest.skip("tci_daily.csv not yet generated -- run build_connectedness.py first")
    return pd.read_csv(OUT_TCI, parse_dates=["date"])


@pytest.fixture(scope="module")
def dc():
    if not OUT_DIR_CONN.exists():
        pytest.skip("directional_connectedness.csv not yet generated")
    return pd.read_csv(OUT_DIR_CONN, parse_dates=["date"])


@pytest.fixture(scope="module")
def manifest():
    if not OUT_MANIFEST.exists():
        pytest.skip("manifest.json not yet generated")
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


# -- Unit tests: compute_gfevd ------------------------------------------------

class TestComputeGFEVD:
    """Test the GFEVD matrix directly on synthetic data."""

    @pytest.fixture(scope="class")
    def theta_tilde(self):
        rng = np.random.default_rng(42)
        # Generate a simple AR(1)-like panel for testing
        K, W = 5, 80
        data = rng.standard_normal((W, K))
        theta_tilde, p = compute_gfevd(data, H=5, max_p=2)
        return theta_tilde, p

    def test_returns_matrix(self, theta_tilde):
        mat, _ = theta_tilde
        assert isinstance(mat, np.ndarray)

    def test_shape(self, theta_tilde):
        mat, _ = theta_tilde
        assert mat.shape == (5, 5)

    def test_rows_sum_to_one(self, theta_tilde):
        mat, _ = theta_tilde
        np.testing.assert_allclose(mat.sum(axis=1), np.ones(5), atol=1e-10)

    def test_all_nonneg(self, theta_tilde):
        mat, _ = theta_tilde
        assert (mat >= 0).all()

    def test_diagonal_positive(self, theta_tilde):
        mat, _ = theta_tilde
        # Self-contribution should be strictly positive
        assert (np.diag(mat) > 0).all()

    def test_var_order_valid(self, theta_tilde):
        _, p = theta_tilde
        assert 1 <= p <= VAR_MAX_P

    def test_tci_in_range(self, theta_tilde):
        mat, _ = theta_tilde
        K = mat.shape[0]
        tci = (mat.sum() - np.trace(mat)) / K * 100
        assert 0 <= tci <= 100

    def test_real_data_gfevd(self, vol_wide):
        """On the real 19-country panel, GFEVD should give sensible TCI."""
        W = 100
        chunk = vol_wide.values[:W]
        theta, p = compute_gfevd(chunk, H=H_HORIZON, max_p=VAR_MAX_P)
        K = N_COUNTRIES
        assert theta.shape == (K, K)
        np.testing.assert_allclose(theta.sum(axis=1), np.ones(K), atol=1e-10)
        tci = (theta.sum() - np.trace(theta)) / K * 100
        # European equity markets should show high connectedness (>30%)
        assert tci > 30, "TCI unexpectedly low for European markets"
        assert tci < 100


class TestRollingConnectedness:
    """Test rolling_connectedness on a small synthetic window."""

    def test_output_length(self, vol_wide):
        W = 100
        tci_s, dir_df = rolling_connectedness(vol_wide.iloc[:120], W=W)
        assert len(tci_s) == 120 - W + 1  # 21 windows

    def test_tci_in_range(self, vol_wide):
        W = 100
        tci_s, _ = rolling_connectedness(vol_wide.iloc[:150], W=W)
        assert (tci_s >= 0).all()
        assert (tci_s <= 100).all()

    def test_dir_df_columns(self, vol_wide):
        W = 60
        _, dir_df = rolling_connectedness(vol_wide.iloc[:80], W=W)
        for col in ["date", "country", "from_spill", "to_spill", "net_spill"]:
            assert col in dir_df.columns


# -- Integration tests: output files ------------------------------------------

class TestTCIFile:
    def test_columns(self, tci):
        assert "date" in tci.columns
        for col in TCI_COLS:
            assert col in tci.columns, "Missing column: " + col

    def test_no_duplicate_dates(self, tci):
        assert not tci["date"].duplicated().any()

    def test_tci_primary_in_range(self, tci):
        col = f"tci_w{PRIMARY_W}"
        valid = tci[col].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_tci_all_windows_in_range(self, tci):
        for col in TCI_COLS:
            valid = tci[col].dropna()
            assert (valid >= 0).all(), "Negative TCI in " + col
            assert (valid <= 100).all(), "TCI >100 in " + col

    def test_warm_up_nan_pattern(self, tci):
        """Shorter windows should have more non-NaN values."""
        counts = {W: tci[f"tci_w{W}"].notna().sum() for W in WINDOWS}
        sorted_windows = sorted(WINDOWS)
        for i in range(len(sorted_windows) - 1):
            w_short = sorted_windows[i]
            w_long  = sorted_windows[i + 1]
            assert counts[w_short] >= counts[w_long], (
                "Shorter window should have >= valid values: "
                "W=%d (%d) vs W=%d (%d)" % (w_short, counts[w_short], w_long, counts[w_long])
            )

    def test_covid_spike(self, tci):
        """TCI should be high during COVID crash (March-April 2020)."""
        covid = tci[
            (tci["date"] >= "2020-03-01") & (tci["date"] <= "2020-04-30")
        ][f"tci_w{PRIMARY_W}"].dropna()
        assert not covid.empty, "No TCI data in COVID window"
        assert covid.mean() > 80, (
            "Expected TCI > 80 during COVID crash, got %.2f" % covid.mean()
        )

    def test_ukraine_elevated(self, tci):
        """TCI should be elevated during Ukraine crisis (Feb-Mar 2022)."""
        ukraine = tci[
            (tci["date"] >= "2022-02-24") & (tci["date"] <= "2022-05-31")
        ][f"tci_w{PRIMARY_W}"].dropna()
        assert not ukraine.empty
        # TCI should be above its overall mean during the crisis
        overall_mean = tci[f"tci_w{PRIMARY_W}"].dropna().mean()
        assert ukraine.mean() > overall_mean * 0.9, (
            "Ukraine TCI (%.2f) not elevated vs overall mean (%.2f)"
            % (ukraine.mean(), overall_mean)
        )


class TestDirectionalConnectedness:
    def test_columns(self, dc):
        for col in ["date", "country", "from_spill", "to_spill", "net_spill", "window"]:
            assert col in dc.columns, "Missing column: " + col

    def test_all_countries_present(self, dc):
        assert dc["country"].nunique() == N_COUNTRIES

    def test_window_column(self, dc):
        assert (dc["window"] == PRIMARY_W).all()

    def test_from_to_nonneg(self, dc):
        assert (dc["from_spill"] >= 0).all()
        assert (dc["to_spill"] >= 0).all()

    def test_net_equals_to_minus_from(self, dc):
        diff = (dc["net_spill"] - (dc["to_spill"] - dc["from_spill"])).abs()
        assert diff.max() < 1e-8

    def test_from_spill_le_100(self, dc):
        # FROM = share of FEV explained by others; normalized rows sum to 1 -> FROM < 100
        assert (dc["from_spill"] < 100).all()

    def test_covid_from_elevated(self, dc):
        """During COVID, FROM spillover should be very high (most FEV from other markets)."""
        covid = dc[
            (dc["date"] >= "2020-03-01") & (dc["date"] <= "2020-04-30")
        ]
        # At least 10 of 19 countries should have FROM > 80% during COVID
        high_from = (covid.groupby("date")["from_spill"].apply(lambda x: (x > 70).sum()))
        assert high_from.mean() >= 10, (
            "Expected most countries with FROM>70 during COVID, got avg %.1f" % high_from.mean()
        )


class TestManifestFile:
    def test_required_keys(self, manifest):
        for key in ["generated_at", "parameters", "tci_primary", "date_range", "outputs"]:
            assert key in manifest

    def test_parameters(self, manifest):
        p = manifest["parameters"]
        assert p["h_horizon"] == H_HORIZON
        assert p["primary_w"] == PRIMARY_W
        assert p["var_max_p"] == VAR_MAX_P
        assert set(p["windows"]) == set(WINDOWS)

    def test_tci_primary_stats(self, manifest):
        tp = manifest["tci_primary"]
        assert tp["window"] == PRIMARY_W
        # European equity markets have high connectedness
        assert tp["mean"] > 50
        assert tp["max"] <= 100
        assert tp["min"] >= 0

    def test_outputs_exist(self, manifest):
        for key, path_str in manifest["outputs"].items():
            p = Path(path_str)
            assert p.exists(), "Output missing: " + key + " -> " + path_str


class TestProvenance:
    def test_summary_exists(self):
        assert OUT_SUMMARY.exists()

    def test_summary_rows(self):
        summary = pd.read_csv(OUT_SUMMARY, index_col=0)
        # Each TCI window should have a row
        for W in WINDOWS:
            assert f"tci_w{W}" in summary.index, "Missing window in summary: tci_w" + str(W)

    def test_manifest_parseable(self):
        m = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
        assert "generated_at" in m
