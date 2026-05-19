"""
Tests for SP11: TCI Window Robustness
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GFJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR  = GFJ_ROOT / "results" / "tci_robustness"
WINDOWS  = [60, 100, 150, 200, 250]


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest():
    p = OUT_DIR / "manifest.json"
    assert p.exists(), "manifest.json not found — run SP11 first"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sensitivity():
    return pd.read_csv(OUT_DIR / "sensitivity_table.csv")


@pytest.fixture(scope="module")
def tci_w250():
    return pd.read_csv(OUT_DIR / "tci_w250.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def network():
    return pd.read_csv(OUT_DIR / "network_metrics.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def emfi_variants():
    return pd.read_csv(OUT_DIR / "emfi_variants.csv", parse_dates=["date"])


# -------------------------------------------------------------------------
# R3.T1 — W=250 TCI output
# -------------------------------------------------------------------------

class TestTciW250:

    def test_file_exists(self):
        assert (OUT_DIR / "tci_w250.csv").exists()

    def test_shape_reasonable(self, tci_w250):
        """W=250 TCI should have roughly 2029 rows (2278 - 250 + 1)."""
        assert 1800 <= len(tci_w250) <= 2100, f"Unexpected row count: {len(tci_w250)}"

    def test_columns(self, tci_w250):
        assert "date" in tci_w250.columns
        assert "tci_w250" in tci_w250.columns

    def test_tci_values_plausible(self, tci_w250):
        """TCI should be strictly between 0 and 100."""
        vals = tci_w250["tci_w250"].dropna()
        assert vals.min() > 0, f"Min TCI={vals.min():.2f}"
        assert vals.max() < 100, f"Max TCI={vals.max():.2f}"

    def test_tci_mean_similar_to_existing(self, tci_w250):
        """W=250 mean TCI should be broadly similar to W=100 (~65–80%)."""
        mean = tci_w250["tci_w250"].mean()
        assert 55 <= mean <= 90, f"W=250 mean TCI={mean:.2f} out of expected range"

    def test_no_all_nan(self, tci_w250):
        pct_valid = tci_w250["tci_w250"].notna().mean()
        assert pct_valid > 0.9, f"Only {pct_valid:.1%} non-NaN TCI values"


# -------------------------------------------------------------------------
# R3.T2 — Network metrics
# -------------------------------------------------------------------------

class TestNetworkMetrics:

    def test_file_exists(self):
        assert (OUT_DIR / "network_metrics.csv").exists()

    def test_columns(self, network):
        assert "lambda1_scaled" in network.columns
        assert "density" in network.columns

    def test_lambda1_positive(self, network):
        """First eigenvalue should be strictly positive."""
        valid = network["lambda1_scaled"].dropna()
        assert (valid > 0).all(), "Some lambda1_scaled values are non-positive"

    def test_lambda1_bounded(self, network):
        """lambda1/K should be in (0,1] for a correlation matrix."""
        valid = network["lambda1_scaled"].dropna()
        assert valid.max() <= 1.05, f"Max lambda1_scaled={valid.max():.3f} > 1"

    def test_density_in_unit_interval(self, network):
        """Density (fraction) should be in [0,1]."""
        valid = network["density"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1

    def test_density_reasonable(self, network):
        """Mean density should be non-trivially positive (markets co-move)."""
        mean_d = network["density"].mean()
        assert mean_d > 0.1, f"Mean density={mean_d:.3f} suspiciously low"

    def test_spike_at_covid(self, network):
        """Both metrics should spike around COVID (Feb-Apr 2020)."""
        covid = network[
            (network["date"] >= "2020-02-01") & (network["date"] <= "2020-05-31")
        ]
        non_covid = network[
            network["date"] < "2020-01-01"
        ]
        assert covid["lambda1_scaled"].mean() > non_covid["lambda1_scaled"].mean(), \
            "Expected higher lambda1 during COVID than pre-COVID"


# -------------------------------------------------------------------------
# R3.T3 — EMFI variants
# -------------------------------------------------------------------------

class TestEmfiVariants:

    def test_file_exists(self):
        assert (OUT_DIR / "emfi_variants.csv").exists()

    def test_all_window_columns_present(self, emfi_variants):
        for W in WINDOWS:
            assert f"EMFI_w{W}" in emfi_variants.columns, f"Missing EMFI_w{W}"

    def test_emfi_standardised_approximately(self, emfi_variants):
        """Each EMFI variant should be approximately mean-0, std-1."""
        for W in WINDOWS:
            col = f"EMFI_w{W}"
            vals = emfi_variants[col].dropna()
            if len(vals) < 50:
                continue
            assert abs(vals.mean()) < 0.2,  f"W={W} EMFI mean={vals.mean():.3f}"
            assert 0.5 < vals.std() < 2.0, f"W={W} EMFI std={vals.std():.3f}"

    def test_emfi_variants_correlated(self, emfi_variants):
        """Different TCI windows should give highly correlated EMFI (>0.8)."""
        col100 = emfi_variants["EMFI_w100"].dropna()
        for W in [60, 150, 200, 250]:
            col = emfi_variants[f"EMFI_w{W}"].dropna()
            # Align on common index
            merged = pd.concat([col100, col], axis=1).dropna()
            if len(merged) < 50:
                continue
            corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
            assert corr > 0.80, \
                f"EMFI_w{W} corr with EMFI_w100 = {corr:.3f} (expected >0.80)"


# -------------------------------------------------------------------------
# R3.T4 — Sensitivity table
# -------------------------------------------------------------------------

class TestSensitivityTable:

    def test_file_exists(self):
        assert (OUT_DIR / "sensitivity_table.csv").exists()

    def test_all_windows_present(self, sensitivity):
        present = set(sensitivity["window"].astype(int))
        for W in WINDOWS:
            assert W in present, f"Window W={W} missing from sensitivity table"

    def test_beta_k0_all_positive(self, sensitivity):
        """β_k0 should be positive for all TCI windows (geopolitical shocks raise fragility)."""
        for _, row in sensitivity.iterrows():
            assert row["lp_beta_k0"] > 0, \
                f"W={int(row['window'])}: β_k0={row['lp_beta_k0']:.4f} not positive"

    def test_theta_k0_all_positive(self, sensitivity):
        """θ_k0 should be positive (state amplification persists across windows)."""
        for _, row in sensitivity.iterrows():
            assert row["slp_theta_k0"] > 0, \
                f"W={int(row['window'])}: θ_k0={row['slp_theta_k0']:.4f} not positive"

    def test_ratio_all_above_one(self, sensitivity):
        """Amplification ratio (fragile/calm) should exceed 1 for all windows."""
        for _, row in sensitivity.iterrows():
            r = row["ratio"]
            if pd.isna(r):
                continue
            assert r > 1.0, \
                f"W={int(row['window'])}: ratio={r:.2f} not > 1 (no amplification)"

    def test_ratio_substantial(self, sensitivity):
        """At least some window should show ratio > 3× (meaningful amplification)."""
        max_ratio = sensitivity["ratio"].dropna().max()
        assert max_ratio > 3.0, f"Max ratio={max_ratio:.2f} — state amplification too weak"

    def test_beta_k0_stable_across_windows(self, sensitivity):
        """β_k0 should not vary more than 3× across windows (robustness)."""
        betas = sensitivity["lp_beta_k0"].dropna()
        ratio_range = betas.max() / betas.min()
        assert ratio_range < 3.0, \
            f"β_k0 range ratio={ratio_range:.2f} — estimate unstable across TCI windows"

    def test_manifest_summary_consistent(self, manifest, sensitivity):
        """Manifest summary should match sensitivity_table.csv."""
        for _, row in sensitivity.iterrows():
            W_str = str(int(row["window"]))
            if W_str in manifest.get("sensitivity_summary", {}):
                m_beta = manifest["sensitivity_summary"][W_str]["lp_beta_k0"]
                assert abs(m_beta - row["lp_beta_k0"]) < 0.001, \
                    f"W={W_str}: manifest β_k0={m_beta} != csv {row['lp_beta_k0']}"


# -------------------------------------------------------------------------
# R3.T5 — Output files and figures
# -------------------------------------------------------------------------

class TestOutputFiles:

    def test_all_files_present(self):
        expected = [
            "tci_w250.csv",
            "network_metrics.csv",
            "emfi_variants.csv",
            "sensitivity_table.csv",
            "Fig_TCI_WindowSensitivity.png",
            "Fig_NetworkMetrics.png",
            "manifest.json",
        ]
        for fname in expected:
            assert (OUT_DIR / fname).exists(), f"Missing: {fname}"

    def test_manifest_fields(self, manifest):
        for key in ["windows", "var_p_w250", "h_horizon", "n_sensitivity_rows",
                    "sensitivity_summary"]:
            assert key in manifest, f"Missing manifest field: {key}"

    def test_manifest_windows(self, manifest):
        assert set(manifest["windows"]) == set(WINDOWS)

    def test_manifest_row_count(self, manifest):
        assert manifest["n_sensitivity_rows"] == len(WINDOWS)
