"""
Tests for SP10: COVID Reclassification
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GFJ_ROOT = Path(__file__).resolve().parents[2]
OUT_LP  = GFJ_ROOT / "results" / "panel_lp_nocovid"
OUT_SLP = GFJ_ROOT / "results" / "state_lp_nocovid"
OUT_COV = GFJ_ROOT / "results" / "covid_reclassify"


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest_lp():
    p = OUT_LP / "manifest.json"
    assert p.exists(), "panel_lp_nocovid/manifest.json not found — run SP10 first"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def manifest_slp():
    p = OUT_SLP / "manifest.json"
    assert p.exists(), "state_lp_nocovid/manifest.json not found — run SP10 first"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def lp_emfi():
    return pd.read_csv(OUT_LP / "lp_results_EMFI.csv")


@pytest.fixture(scope="module")
def smooth_emfi():
    return pd.read_csv(OUT_SLP / "state_lp_smooth_EMFI.csv")


@pytest.fixture(scope="module")
def taxonomy_nocovid():
    return pd.read_csv(OUT_COV / "event_taxonomy_nocovid.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def comparison():
    return pd.read_csv(OUT_COV / "comparison_table.csv")


# -------------------------------------------------------------------------
# R2.T1 — COVID shock days zeroed out (not removed from sample)
# -------------------------------------------------------------------------

class TestCovidReclassification:

    def test_n_covid_days_zeroed(self, manifest_lp):
        """Exactly 16 COVID-period shock days should be zeroed."""
        assert manifest_lp["n_covid_shock_days_zeroed"] == 16

    def test_remaining_shock_days(self, manifest_lp):
        """147 or 138 non-COVID shock days remain (depends on LP sample start)."""
        # 163 total - 16 COVID = 147; dataset starts May 2017, may drop a few more
        assert 130 <= manifest_lp["n_remaining_shock_days"] <= 155

    def test_sample_size_unchanged(self, manifest_lp):
        """Full observation count is preserved — outcomes not dropped."""
        assert manifest_lp["n_obs"] >= 2100  # should be ~2179

    def test_taxonomy_has_covid_flag(self, taxonomy_nocovid):
        """event_taxonomy_nocovid.csv has covid_reclassified and lp_treatment columns."""
        assert "covid_reclassified" in taxonomy_nocovid.columns
        assert "lp_treatment" in taxonomy_nocovid.columns

    def test_taxonomy_16_covid_events(self, taxonomy_nocovid):
        """Exactly 16 events flagged as covid_reclassified."""
        n = taxonomy_nocovid["covid_reclassified"].sum()
        assert n == 16, f"Expected 16 COVID events, got {n}"

    def test_covid_events_not_lp_treatment(self, taxonomy_nocovid):
        """COVID-flagged events must have lp_treatment=False."""
        covid_rows = taxonomy_nocovid[taxonomy_nocovid["covid_reclassified"]]
        assert (~covid_rows["lp_treatment"]).all()

    def test_non_covid_events_are_lp_treatment(self, taxonomy_nocovid):
        """Non-COVID events should have lp_treatment=True."""
        non_covid = taxonomy_nocovid[~taxonomy_nocovid["covid_reclassified"]]
        assert non_covid["lp_treatment"].all()


# -------------------------------------------------------------------------
# R2.T2 — Panel LP results (no-COVID shock)
# -------------------------------------------------------------------------

class TestPanelLP:

    def test_lp_results_files_exist(self):
        """All five outcome LP files must be present."""
        outcomes = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
        for out in outcomes:
            assert (OUT_LP / f"lp_results_{out}.csv").exists(), \
                f"Missing lp_results_{out}.csv"

    def test_lp_horizons_complete(self, lp_emfi):
        """Horizons -5 to +15 must all be present."""
        expected = set(range(-5, 16))
        assert set(lp_emfi["horizon"]) == expected

    def test_lp_emfi_beta_k0_positive(self, lp_emfi):
        """EMFI β_k0 should be positive (geopolitical shocks raise fragility)."""
        k0 = lp_emfi.loc[lp_emfi["horizon"] == 0, "beta"].iloc[0]
        assert k0 > 0, f"Expected positive β_k0, got {k0:.4f}"

    def test_lp_emfi_beta_k0_plausible(self, lp_emfi):
        """EMFI β_k0 should be smaller than baseline (COVID excluded) but not zero."""
        k0 = lp_emfi.loc[lp_emfi["horizon"] == 0, "beta"].iloc[0]
        assert 0.05 < k0 < 1.5, f"β_k0={k0:.4f} outside plausible range"

    def test_comparison_emfi_beta_k0_lower(self, comparison):
        """No-COVID β_k0 for EMFI should be lower than full baseline."""
        row = comparison.loc[
            (comparison["outcome"] == "EMFI") &
            (comparison["metric"] == "LP_beta_k0")
        ]
        assert len(row) == 1
        assert row["nocovid"].iloc[0] < row["baseline"].iloc[0], \
            "COVID-excl EMFI β_k0 should be ≤ baseline (COVID raised fragility on those days)"

    def test_lp_figure_created(self):
        assert (OUT_LP / "Fig_LP_Combined_NoCovid.png").exists()

    def test_lp_nobs_reasonable(self, lp_emfi):
        """Observations at k=0 should be close to full sample (>2000)."""
        k0_nobs = lp_emfi.loc[lp_emfi["horizon"] == 0, "nobs"].iloc[0]
        assert k0_nobs > 2000, f"k=0 nobs={k0_nobs}, expected >2000"


# -------------------------------------------------------------------------
# R2.T3 — State LP (smooth) results
# -------------------------------------------------------------------------

class TestStateLPSmooth:

    def test_smooth_files_exist(self):
        outcomes = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
        for out in outcomes:
            assert (OUT_SLP / f"state_lp_smooth_{out}.csv").exists()
            assert (OUT_SLP / f"state_lp_binary_{out}.csv").exists()

    def test_theta_k0_positive_emfi(self, smooth_emfi):
        """State LP θ_k0 for EMFI should be positive — fragile markets amplify."""
        k0 = smooth_emfi.loc[smooth_emfi["horizon"] == 0, "theta"].iloc[0]
        assert k0 > 0, f"Expected positive θ_k0, got {k0:.4f}"

    def test_theta_survives_covid_excl(self, comparison):
        """State LP θ_k0 after COVID exclusion should remain substantially positive."""
        row = comparison.loc[
            (comparison["outcome"] == "EMFI") &
            (comparison["metric"] == "StateLPsmooth_theta_k0")
        ]
        if len(row) == 0:
            pytest.skip("State LP comparison not available")
        nc_theta = row["nocovid"].iloc[0]
        assert nc_theta > 0.5, \
            f"θ_k0={nc_theta:.4f} — state-dependence collapsed after COVID exclusion"

    def test_irf_evaluated_at_three_points(self, smooth_emfi):
        """IRF columns for p=0.0, p=0.5, p=0.9 must be present."""
        post = smooth_emfi[smooth_emfi["horizon"] == 0]
        assert "irf_p0"  in post.columns
        assert "irf_p5"  in post.columns
        assert "irf_p9"  in post.columns

    def test_irf_ordering_fragile_gt_calm(self, smooth_emfi):
        """At k=0, IRF at p=0.9 should be greater than at p=0.0 (amplification)."""
        k0 = smooth_emfi[smooth_emfi["horizon"] == 0]
        irf_calm   = k0["irf_p0"].iloc[0]
        irf_stress = k0["irf_p9"].iloc[0]
        assert irf_stress > irf_calm, \
            f"Expected irf_p9 > irf_p0 at k=0; got {irf_stress:.3f} vs {irf_calm:.3f}"

    def test_state_lp_figure_created(self):
        assert (OUT_SLP / "Fig_StateLPSmooth_NoCovid.png").exists()


# -------------------------------------------------------------------------
# R2.T4 — Output files and manifests
# -------------------------------------------------------------------------

class TestOutputFiles:

    def test_manifest_lp_fields(self, manifest_lp):
        for key in ["n_covid_shock_days_zeroed", "n_remaining_shock_days",
                    "n_obs", "date_range", "lp_summary"]:
            assert key in manifest_lp, f"Missing manifest field: {key}"

    def test_manifest_slp_fields(self, manifest_slp):
        for key in ["hf_threshold_emfi", "hf_pctile", "n_remaining_shock_days"]:
            assert key in manifest_slp

    def test_comparison_table_has_all_outcomes(self, comparison):
        outcomes = {"EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"}
        present = set(comparison["outcome"].unique())
        assert outcomes.issubset(present), f"Missing outcomes: {outcomes - present}"

    def test_summary_csv_exists(self):
        assert (OUT_LP / "lp_summary_nocovid.csv").exists()
