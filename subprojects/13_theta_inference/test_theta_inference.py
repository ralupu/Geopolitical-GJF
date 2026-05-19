"""
Tests for SP13: Formal Inference for theta (State-Amplification Parameter)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GFJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR  = GFJ_ROOT / "results" / "theta_inference"
OUTCOMES = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest():
    p = OUT_DIR / "manifest.json"
    assert p.exists(), "manifest.json not found — run SP13 first"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bootstrap_table():
    return pd.read_csv(OUT_DIR / "bootstrap_ci_table.csv")


@pytest.fixture(scope="module")
def emfi_df():
    return pd.read_csv(OUT_DIR / "theta_inference_EMFI.csv")


def load_outcome_df(outcome: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"theta_inference_{outcome}.csv")


# -------------------------------------------------------------------------
# R5.T1 — Per-horizon theta inference files
# -------------------------------------------------------------------------

class TestPerHorizonFiles:

    def test_all_files_exist(self):
        for outcome in OUTCOMES:
            p = OUT_DIR / f"theta_inference_{outcome}.csv"
            assert p.exists(), f"Missing: theta_inference_{outcome}.csv"

    def test_correct_number_of_horizons(self, emfi_df):
        """Should have exactly 21 rows: k = -5 to +15."""
        assert len(emfi_df) == 21, f"Expected 21 rows, got {len(emfi_df)}"

    def test_horizon_range(self, emfi_df):
        assert emfi_df["horizon"].min() == -5
        assert emfi_df["horizon"].max() == 15

    def test_required_columns_present(self, emfi_df):
        required = [
            "horizon", "beta", "beta_se", "theta", "theta_se",
            "theta_tstat", "theta_pval", "cov_bt", "nobs",
        ]
        for col in required:
            assert col in emfi_df.columns, f"Missing column: {col}"

    def test_irf_columns_present(self, emfi_df):
        for p_tag in ["00", "05", "09"]:
            for prefix in ["irf_p", "se_irf_p", "ci95lo_irf_p", "ci95hi_irf_p"]:
                col = f"{prefix}{p_tag}"
                assert col in emfi_df.columns, f"Missing column: {col}"

    def test_pvalues_in_unit_interval(self, emfi_df):
        pvals = emfi_df["theta_pval"].dropna()
        assert (pvals >= 0).all() and (pvals <= 1).all(), "p-values outside [0,1]"

    def test_se_positive(self, emfi_df):
        """All SE values should be positive."""
        for col in ["beta_se", "theta_se"]:
            vals = emfi_df[col].dropna()
            assert (vals > 0).all(), f"Non-positive SE in {col}"

    def test_tstat_consistent_with_se(self, emfi_df):
        """t-stat should equal theta / theta_se."""
        for _, row in emfi_df.dropna(subset=["theta_tstat"]).iterrows():
            expected = row["theta"] / row["theta_se"]
            assert abs(row["theta_tstat"] - expected) < 0.01, \
                f"k={row['horizon']}: t-stat={row['theta_tstat']:.4f}, expected={expected:.4f}"

    def test_delta_method_se_consistent(self, emfi_df):
        """SE[IRF(p=0.9)] should match delta-method formula."""
        for _, row in emfi_df.dropna(subset=["se_irf_p09"]).iterrows():
            var_b = row["beta_se"] ** 2
            var_t = row["theta_se"] ** 2
            cov   = row["cov_bt"]
            p     = 0.9
            expected_se = np.sqrt(max(var_b + p**2 * var_t + 2*p*cov, 0))
            assert abs(row["se_irf_p09"] - expected_se) < 0.001, \
                f"k={row['horizon']}: delta SE mismatch"

    def test_ci_consistent_with_se(self, emfi_df):
        """CI bounds should be IRF +/- 1.96 * SE."""
        for _, row in emfi_df.dropna(subset=["irf_p09", "se_irf_p09"]).iterrows():
            irf = row["irf_p09"]
            se  = row["se_irf_p09"]
            expected_lo = irf - 1.96 * se
            expected_hi = irf + 1.96 * se
            assert abs(row["ci95lo_irf_p09"] - expected_lo) < 0.001
            assert abs(row["ci95hi_irf_p09"] - expected_hi) < 0.001

    def test_nobs_reasonable(self, emfi_df):
        """Observation count should be reasonable."""
        nobs = emfi_df["nobs"].dropna()
        assert (nobs > 1000).all(), "Too few observations"
        assert (nobs < 3000).all(), "Too many observations"


# -------------------------------------------------------------------------
# R5.T2 — k=0 theta values (key structural finding)
# -------------------------------------------------------------------------

class TestKzeroTheta:

    def test_emfi_theta_positive(self, emfi_df):
        """EMFI theta at k=0 should be positive (state amplification)."""
        k0 = emfi_df[emfi_df["horizon"] == 0].iloc[0]
        assert k0["theta"] > 0, f"EMFI theta_k0={k0['theta']:.4f} not positive"

    def test_emfi_theta_meaningful(self, emfi_df):
        """EMFI theta at k=0 should be substantially above zero (>0.5)."""
        k0 = emfi_df[emfi_df["horizon"] == 0].iloc[0]
        assert k0["theta"] > 0.5, f"EMFI theta_k0={k0['theta']:.4f} too small"

    def test_emfi_amplification_above_3x(self, bootstrap_table):
        """Amplification ratio for EMFI should exceed 3x."""
        row = bootstrap_table[bootstrap_table["outcome"] == "EMFI"].iloc[0]
        assert row["amplification"] > 3.0, \
            f"EMFI amplification={row['amplification']:.1f}x (expected >3x)"

    def test_tci_theta_positive(self):
        """TCI theta at k=0 should be positive."""
        df = load_outcome_df("tci_w100")
        k0 = df[df["horizon"] == 0].iloc[0]
        assert k0["theta"] > 0, f"TCI theta_k0={k0['theta']:.4f} not positive"

    def test_emfi_irf_p9_substantially_larger(self, emfi_df):
        """IRF(p=0.9) should be substantially larger than IRF(p=0.0)."""
        k0 = emfi_df[emfi_df["horizon"] == 0].iloc[0]
        assert k0["irf_p09"] > k0["irf_p00"] * 2, \
            f"IRF(0.9)={k0['irf_p09']:.4f} not >2x IRF(0.0)={k0['irf_p00']:.4f}"

    def test_pre_shock_horizons_theta_near_zero(self, emfi_df):
        """Pre-shock horizons (k=-5..-1) should have mean |theta| < 2."""
        pre = emfi_df[emfi_df["horizon"] < 0]["theta"].abs().mean()
        assert pre < 2.0, f"Pre-shock mean |theta|={pre:.4f} too large (anticipation?)"


# -------------------------------------------------------------------------
# R5.T3 — Bootstrap CI table
# -------------------------------------------------------------------------

class TestBootstrapTable:

    def test_file_exists(self):
        assert (OUT_DIR / "bootstrap_ci_table.csv").exists()

    def test_all_outcomes_present(self, bootstrap_table):
        present = set(bootstrap_table["outcome"])
        for o in OUTCOMES:
            assert o in present, f"Missing outcome: {o}"

    def test_bootstrap_ci_columns_present(self, bootstrap_table):
        for col in ["theta_boot_se", "theta_boot_ci95lo", "theta_boot_ci95hi",
                    "irf_p9_boot_se", "irf_p9_boot_ci95lo", "irf_p9_boot_ci95hi",
                    "n_boot_valid"]:
            assert col in bootstrap_table.columns, f"Missing column: {col}"

    def test_bootstrap_n_valid_high(self, bootstrap_table):
        """At least 950 of 1000 bootstrap reps should succeed."""
        for _, row in bootstrap_table.iterrows():
            assert row["n_boot_valid"] >= 950, \
                f"{row['outcome']}: only {row['n_boot_valid']} valid bootstrap reps"

    def test_bootstrap_ci_ordered(self, bootstrap_table):
        """CI lo should be below CI hi."""
        for _, row in bootstrap_table.iterrows():
            assert row["theta_boot_ci95lo"] < row["theta_boot_ci95hi"], \
                f"{row['outcome']}: bootstrap CI inverted"
            assert row["irf_p9_boot_ci95lo"] < row["irf_p9_boot_ci95hi"], \
                f"{row['outcome']}: IRF bootstrap CI inverted"

    def test_emfi_boot_theta_consistent_with_analytic(self, bootstrap_table):
        """Bootstrap SE should be in same ballpark as analytic SE (within 3x)."""
        row = bootstrap_table[bootstrap_table["outcome"] == "EMFI"].iloc[0]
        analytic_se = row["theta_se"]
        boot_se     = row["theta_boot_se"]
        ratio = boot_se / analytic_se if analytic_se > 0 else np.nan
        assert 0.3 < ratio < 3.0, \
            f"EMFI: boot_SE/analytic_SE={ratio:.2f} (expected 0.3–3.0)"

    def test_amplification_ratio_present(self, bootstrap_table):
        """Amplification ratio column should exist and be positive for EMFI."""
        assert "amplification" in bootstrap_table.columns
        row = bootstrap_table[bootstrap_table["outcome"] == "EMFI"].iloc[0]
        assert row["amplification"] > 0, "EMFI amplification should be positive"


# -------------------------------------------------------------------------
# R5.T4 — Output files and manifest
# -------------------------------------------------------------------------

class TestOutputFiles:

    def test_all_files_present(self):
        expected = [
            "bootstrap_ci_table.csv",
            "Fig_ThetaInference.png",
            "manifest.json",
        ] + [f"theta_inference_{o}.csv" for o in OUTCOMES]
        for fname in expected:
            assert (OUT_DIR / fname).exists(), f"Missing: {fname}"

    def test_manifest_fields(self, manifest):
        for key in ["run_utc", "spec", "shock", "horizons", "B_reps",
                    "block_len", "outcomes", "emfi_k0",
                    "all_theta_positive", "emfi_theta_sig05",
                    "n_sig05_outcomes", "n_sig10_outcomes"]:
            assert key in manifest, f"Missing manifest field: {key}"

    def test_manifest_emfi_k0(self, manifest):
        emfi = manifest["emfi_k0"]
        assert emfi["theta"] > 0, "Manifest EMFI theta_k0 not positive"
        assert emfi["theta_se"] > 0, "Manifest EMFI SE(theta) not positive"
        assert 0 <= emfi["theta_pval"] <= 1, "Manifest p-value out of [0,1]"

    def test_manifest_outcomes_complete(self, manifest):
        assert set(manifest["outcomes"]) == set(OUTCOMES)

    def test_manifest_bootstrap_params(self, manifest):
        assert manifest["B_reps"] == 1000
        assert manifest["block_len"] == 20

    def test_manifest_spec_correct(self, manifest):
        assert "covid" in manifest["spec"].lower(), \
            f"Unexpected spec: {manifest['spec']}"
