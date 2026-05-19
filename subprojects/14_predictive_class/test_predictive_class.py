"""
Tests for SP14: Predictive Event Classification
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GFJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR  = GFJ_ROOT / "results" / "predictive_class"


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest():
    p = OUT_DIR / "manifest.json"
    assert p.exists(), "manifest.json not found — run SP14 first"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def logit_table():
    return pd.read_csv(OUT_DIR / "logit_table.csv")


@pytest.fixture(scope="module")
def probit_table():
    return pd.read_csv(OUT_DIR / "probit_table.csv")


@pytest.fixture(scope="module")
def loocv():
    return pd.read_csv(OUT_DIR / "loocv_results.csv")


# -------------------------------------------------------------------------
# R6.T1 — Output files
# -------------------------------------------------------------------------

class TestOutputFiles:

    def test_all_files_exist(self):
        expected = [
            "logit_table.csv",
            "probit_table.csv",
            "loocv_results.csv",
            "Fig_ROC_Predictive.png",
            "Fig_Tree_Predictive.png",
            "manifest.json",
        ]
        for fname in expected:
            assert (OUT_DIR / fname).exists(), f"Missing: {fname}"


# -------------------------------------------------------------------------
# R6.T2 — Sample properties
# -------------------------------------------------------------------------

class TestSampleProperties:

    def test_sample_size(self, manifest):
        assert manifest["n_obs"] == 154, f"Expected 154 obs, got {manifest['n_obs']}"

    def test_systemic_count(self, manifest):
        assert manifest["n_systemic"] == 42, \
            f"Expected 42 systemic, got {manifest['n_systemic']}"

    def test_non_systemic_count(self, manifest):
        assert manifest["n_non_systemic"] == 112, \
            f"Expected 112 non-systemic, got {manifest['n_non_systemic']}"

    def test_features_present(self, manifest):
        expected = {"EMFI_lag", "TCI_lag", "P_stress_lag", "max_shock", "n_countries"}
        assert set(manifest["features"]) == expected


# -------------------------------------------------------------------------
# R6.T3 — Logit coefficient table
# -------------------------------------------------------------------------

class TestLogitTable:

    def test_correct_rows(self, logit_table):
        """Should have 6 rows: const + 5 features."""
        assert len(logit_table) == 6, f"Expected 6 rows, got {len(logit_table)}"

    def test_required_columns(self, logit_table):
        for col in ["variable", "coef_std", "se_std", "z_stat",
                    "p_value", "ci95lo", "ci95hi", "odds_ratio"]:
            assert col in logit_table.columns, f"Missing column: {col}"

    def test_emfi_positive_significant(self, logit_table):
        """EMFI_lag coefficient should be positive and p < 0.01."""
        row = logit_table[logit_table["variable"] == "EMFI_lag"].iloc[0]
        assert row["coef_std"] > 0, f"EMFI_lag coef={row['coef_std']:.4f} not positive"
        assert row["p_value"] < 0.01, f"EMFI_lag p={row['p_value']:.4f} not < 0.01"

    def test_emfi_odds_ratio_meaningful(self, logit_table):
        """EMFI odds ratio should be substantially above 1 (>2)."""
        row = logit_table[logit_table["variable"] == "EMFI_lag"].iloc[0]
        assert row["odds_ratio"] > 2.0, \
            f"EMFI_lag OR={row['odds_ratio']:.3f} not >2"

    def test_pvalues_in_unit_interval(self, logit_table):
        pvals = logit_table["p_value"]
        assert (pvals >= 0).all() and (pvals <= 1).all()

    def test_se_positive(self, logit_table):
        assert (logit_table["se_std"] > 0).all()

    def test_ci_ordered(self, logit_table):
        assert (logit_table["ci95lo"] < logit_table["ci95hi"]).all()

    def test_odds_ratio_consistent(self, logit_table):
        """Odds ratios should be exp(coef_std) for non-const rows."""
        non_const = logit_table[logit_table["variable"] != "const"]
        for _, row in non_const.iterrows():
            expected_or = np.exp(row["coef_std"])
            assert abs(row["odds_ratio"] - expected_or) < 0.01, \
                f"{row['variable']}: OR={row['odds_ratio']:.4f} vs exp(coef)={expected_or:.4f}"


# -------------------------------------------------------------------------
# R6.T4 — Probit robustness
# -------------------------------------------------------------------------

class TestProbitTable:

    def test_correct_rows(self, probit_table):
        assert len(probit_table) == 6

    def test_emfi_positive(self, probit_table):
        row = probit_table[probit_table["variable"] == "EMFI_lag"].iloc[0]
        assert row["coef_std"] > 0, f"Probit EMFI_lag coef not positive"

    def test_sign_consistency_with_logit(self, logit_table, probit_table):
        """Logit and probit should agree on sign for all variables."""
        non_const = logit_table[logit_table["variable"] != "const"]
        for _, row_l in non_const.iterrows():
            v = row_l["variable"]
            row_p = probit_table[probit_table["variable"] == v].iloc[0]
            sign_l = np.sign(row_l["coef_std"])
            sign_p = np.sign(row_p["coef_std"])
            # Allow exception for near-zero coefficients
            if abs(row_l["coef_std"]) < 0.05:
                continue
            assert sign_l == sign_p, \
                f"{v}: logit sign={sign_l}, probit sign={sign_p} (disagreement)"


# -------------------------------------------------------------------------
# R6.T5 — LOO-CV performance
# -------------------------------------------------------------------------

class TestLOOCV:

    def test_loocv_file_shape(self, loocv):
        assert len(loocv) == 154, f"Expected 154 rows, got {len(loocv)}"

    def test_loocv_prob_columns(self, loocv):
        assert "prob_logit" in loocv.columns
        assert "prob_probit" in loocv.columns

    def test_probs_in_unit_interval(self, loocv):
        for col in ["prob_logit", "prob_probit"]:
            vals = loocv[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), \
                f"{col}: values outside [0,1]"

    def test_auc_above_random(self, manifest):
        """AUC should be meaningfully above 0.5."""
        auc_l = manifest["loocv_auc_logit"]
        auc_p = manifest["loocv_auc_probit"]
        assert auc_l > 0.60, f"Logit AUC={auc_l:.4f} not > 0.60"
        assert auc_p > 0.60, f"Probit AUC={auc_p:.4f} not > 0.60"

    def test_auc_consistent_logit_probit(self, manifest):
        """Logit and probit AUCs should be close (within 0.05)."""
        auc_l = manifest["loocv_auc_logit"]
        auc_p = manifest["loocv_auc_probit"]
        assert abs(auc_l - auc_p) < 0.05, \
            f"AUC gap too large: logit={auc_l:.4f} probit={auc_p:.4f}"

    def test_brier_score_reasonable(self, manifest):
        """Brier score for a base rate of 42/154=0.27 is 0.197; model should beat it."""
        base_rate = 42 / 154
        null_brier = base_rate * (1 - base_rate)
        model_brier = manifest["brier_score"]
        assert model_brier < null_brier, \
            f"Brier={model_brier:.4f} not better than null ({null_brier:.4f})"


# -------------------------------------------------------------------------
# R6.T6 — Manifest fields
# -------------------------------------------------------------------------

class TestManifestFields:

    def test_required_keys(self, manifest):
        for key in ["run_utc", "n_obs", "n_systemic", "n_non_systemic",
                    "features", "loocv_auc_logit", "loocv_auc_probit",
                    "brier_score", "optimal_threshold", "confusion_matrix",
                    "logit_pstress_coef", "logit_pstress_pval",
                    "logit_pseudo_r2", "spotlight"]:
            assert key in manifest, f"Missing manifest key: {key}"

    def test_pseudo_r2_positive(self, manifest):
        assert manifest["logit_pseudo_r2"] > 0

    def test_confusion_matrix_fields(self, manifest):
        cm = manifest["confusion_matrix"]
        for field in ["sensitivity", "specificity", "accuracy", "tp", "tn", "fp", "fn"]:
            assert field in cm, f"Missing CM field: {field}"

    def test_confusion_matrix_counts_sum(self, manifest):
        cm = manifest["confusion_matrix"]
        total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
        assert total == 154, f"CM counts sum to {total}, expected 154"

    def test_ukraine_high_prob(self, manifest):
        """Ukraine invasion should have high predicted probability."""
        spot = manifest["spotlight"]
        ukraine_key = [k for k in spot if "Ukraine" in k]
        assert ukraine_key, "Ukraine event not in spotlight"
        assert spot[ukraine_key[0]]["prob"] > 0.4, \
            f"Ukraine predicted prob={spot[ukraine_key[0]]['prob']:.3f} — expected >0.4"
