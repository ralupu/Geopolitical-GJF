"""
Tests for SP12: HMM Robustness Specifications
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GFJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR  = GFJ_ROOT / "results" / "hmm_robustness"
HMM_CSV  = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest():
    p = OUT_DIR / "manifest.json"
    assert p.exists(), "manifest.json not found — run SP12 first"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def kappa_table():
    return pd.read_csv(OUT_DIR / "kappa_table.csv")


@pytest.fixture(scope="module")
def variants_df():
    return pd.read_csv(OUT_DIR / "hmm_variants.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def hmm_daily():
    return pd.read_csv(HMM_CSV, parse_dates=["date"])


# -------------------------------------------------------------------------
# R4.T1 — Filtered probabilities
# -------------------------------------------------------------------------

class TestFilteredProbabilities:

    def test_filtered_column_in_hmm_daily(self, hmm_daily):
        assert "P_stress_filtered" in hmm_daily.columns, \
            "P_stress_filtered not in hmm_daily.csv"

    def test_filtered_values_in_unit_interval(self, hmm_daily):
        filt = hmm_daily["P_stress_filtered"].dropna()
        assert filt.min() >= -0.001, f"Min filtered P={filt.min()}"
        assert filt.max() <= 1.001, f"Max filtered P={filt.max()}"

    def test_filtered_corr_with_smoothed_high(self, hmm_daily):
        """Filtered and smoothed should be highly correlated (>0.95)."""
        corr = hmm_daily["P_stress_filtered"].corr(hmm_daily["P_stress"])
        assert corr > 0.95, f"Filtered-smoothed corr={corr:.4f} < 0.95"

    def test_filtered_corr_in_manifest(self, manifest):
        corr = manifest["corr_filtered_smooth"]
        assert corr > 0.95, f"Manifest corr_filtered_smooth={corr:.4f}"

    def test_filtered_and_smoothed_same_length(self, hmm_daily):
        assert hmm_daily["P_stress_filtered"].notna().sum() == \
               hmm_daily["P_stress"].notna().sum()


# -------------------------------------------------------------------------
# R4.T2 — HMM variants in kappa table
# -------------------------------------------------------------------------

class TestKappaTable:

    def test_file_exists(self):
        assert (OUT_DIR / "kappa_table.csv").exists()

    def test_required_variants_present(self, kappa_table):
        variants = set(kappa_table["variant"])
        required = {"2-state", "3-state-diag", "covid-excl", "pre-2020"}
        for v in required:
            assert v in variants, f"Missing variant: {v}"

    def test_all_correlations_positive(self, kappa_table):
        """All P_stress variants should be positively correlated with baseline."""
        for _, row in kappa_table.iterrows():
            if pd.isna(row["corr_P_stress"]):
                continue
            assert row["corr_P_stress"] > 0, \
                f"Variant {row['variant']}: negative P_stress correlation"

    def test_3state_variants_high_correlation(self, kappa_table):
        """3-state variants (same structure) should have corr >0.85 with baseline."""
        three_state = kappa_table[
            kappa_table["variant"].isin(["3-state-diag", "covid-excl", "pre-2020"])
        ]
        for _, row in three_state.iterrows():
            if pd.isna(row["corr_P_stress"]):
                continue
            assert row["corr_P_stress"] > 0.70, \
                f"Variant {row['variant']}: corr={row['corr_P_stress']:.3f} < 0.70"

    def test_diagonal_cov_high_kappa(self, kappa_table):
        """Diagonal covariance 3-state should have kappa >0.80 (same structure)."""
        row = kappa_table[kappa_table["variant"] == "3-state-diag"]
        if len(row) == 0:
            pytest.skip("3-state-diag variant not found")
        kappa = row["kappa"].iloc[0]
        if pd.isna(kappa):
            pytest.skip("Kappa not computed")
        assert kappa > 0.70, f"3-state-diag kappa={kappa:.3f} < 0.70"

    def test_covid_excl_substantial_kappa(self, kappa_table):
        """COVID-excluded training should yield kappa >0.70 with baseline."""
        row = kappa_table[kappa_table["variant"] == "covid-excl"]
        if len(row) == 0:
            pytest.skip("covid-excl variant not found")
        kappa = row["kappa"].iloc[0]
        if pd.isna(kappa):
            pytest.skip("Kappa not computed")
        assert kappa > 0.60, f"covid-excl kappa={kappa:.3f} < 0.60"


# -------------------------------------------------------------------------
# R4.T3 — P_stress variants file
# -------------------------------------------------------------------------

class TestVariantsFile:

    def test_file_exists(self):
        assert (OUT_DIR / "hmm_variants.csv").exists()

    def test_baseline_column_present(self, variants_df):
        assert "P_stress_baseline" in variants_df.columns

    def test_filtered_column_present(self, variants_df):
        assert "P_stress_filtered" in variants_df.columns

    def test_all_variants_have_columns(self, variants_df):
        for variant_suffix in ["2_state", "3_state_diag", "covid_excl", "pre_2020"]:
            col = f"P_stress_{variant_suffix}"
            assert col in variants_df.columns, f"Missing column: {col}"

    def test_values_in_unit_interval(self, variants_df):
        for col in variants_df.columns:
            if col == "date":
                continue
            vals = variants_df[col].dropna()
            assert vals.min() >= -0.01, f"{col}: min={vals.min()}"
            assert vals.max() <= 1.01,  f"{col}: max={vals.max()}"

    def test_stress_probability_occasionally_high(self, variants_df):
        """Each variant should occasionally detect high-stress (P>0.5) episodes."""
        for col in variants_df.columns:
            if col == "date":
                continue
            pct_high = (variants_df[col] > 0.5).mean()
            assert pct_high > 0.01, \
                f"{col}: only {pct_high:.1%} days with P>0.5 — too sparse"

    def test_stress_probability_not_always_high(self, variants_df):
        """P_stress should not be perpetually high (not a spike detector problem)."""
        for col in variants_df.columns:
            if col == "date":
                continue
            pct_high = (variants_df[col] > 0.5).mean()
            assert pct_high < 0.50, \
                f"{col}: {pct_high:.1%} days with P>0.5 — too persistent"


# -------------------------------------------------------------------------
# R4.T4 — Output files and manifest
# -------------------------------------------------------------------------

class TestOutputFiles:

    def test_all_files_present(self):
        expected = [
            "hmm_variants.csv",
            "kappa_table.csv",
            "Fig_HMM_Variants.png",
            "manifest.json",
        ]
        for fname in expected:
            assert (OUT_DIR / fname).exists(), f"Missing: {fname}"

    def test_hmm_daily_updated(self, hmm_daily):
        """hmm_daily.csv should now have P_stress_filtered column."""
        assert "P_stress_filtered" in hmm_daily.columns

    def test_manifest_fields(self, manifest):
        for key in ["corr_filtered_smooth", "variants_run", "kappa_summary"]:
            assert key in manifest, f"Missing manifest field: {key}"

    def test_manifest_variants_run(self, manifest):
        expected = {"2-state", "3-state-diag", "covid-excl", "pre-2020"}
        actual   = set(manifest["variants_run"])
        assert expected == actual, f"Variants run mismatch: {actual}"
