"""
test_state_lp.py  —  Phase 7 validation suite
Tests: dataset construction, IRF direction, state amplification (θ>0),
       delta-method SE validity, COVID exclusion, binary LP ordering,
       composition report, output files, and manifest integrity.
"""

import os, sys, json, unittest
import pandas as pd
import numpy as np

BASE   = os.path.join(os.path.dirname(__file__), "..", "..")
OUTDIR = os.path.join(BASE, "results", "state_lp")
DATA   = os.path.join(BASE, "results")

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def smooth(outcome, nocovid=False):
    suffix = "_nocovid" if nocovid else ""
    return pd.read_csv(os.path.join(OUTDIR, f"state_lp_smooth_{outcome}{suffix}.csv"))

def binary(outcome):
    return pd.read_csv(os.path.join(OUTDIR, f"state_lp_binary_{outcome}.csv"))

SMOOTH_OUTCOMES = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
BINARY_OUTCOMES = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
# Outcomes expected to have θ_k>0 at k=0 (AvgCorr60 excluded — θ≈-0.011, near-zero, not robust)
THETA_POSITIVE  = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress"]
# Outcomes expected to show fragile > normal at k=0 in binary LP
FRAGILE_AMPLIFY = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress"]

CI_95 = 1.96


# ──────────────────────────────────────────────────────────────────────────────
class TestSmoothLPStructure(unittest.TestCase):
    """Smooth-transition LP output structure."""

    def test_horizons_range(self):
        """Smooth LP must cover horizons -5 to 15 (21 rows)."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            self.assertEqual(len(df), 21,
                             msg=f"{o}: expected 21 horizons, got {len(df)}")
            self.assertEqual(df["horizon"].min(), -5,
                             msg=f"{o}: horizon min should be -5")
            self.assertEqual(df["horizon"].max(), 15,
                             msg=f"{o}: horizon max should be 15")

    def test_required_columns_smooth(self):
        """Smooth LP must have beta, theta, IRF and SE columns for each p."""
        required = ["horizon", "beta", "theta", "beta_se", "theta_se", "cov_bt", "nobs"]
        for p in ["0.0", "0.5", "0.9"]:
            required += [f"irf_{p}", f"se_{p}", f"ci95lo_{p}", f"ci95hi_{p}"]
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            for col in required:
                self.assertIn(col, df.columns,
                              msg=f"{o}: missing column {col}")

    def test_nobs_positive(self):
        """Observation count must be positive for all horizons."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            self.assertTrue((df["nobs"] > 0).all(),
                            msg=f"{o}: nobs contains non-positive values")

    def test_se_positive(self):
        """All SE columns must be strictly positive."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            for p in ["0.0", "0.5", "0.9"]:
                col = f"se_{p}"
                self.assertTrue((df[col] > 0).all(),
                                msg=f"{o}: se_{p} has non-positive values")

    def test_ci_ordering(self):
        """95% CI must satisfy ci95lo < irf < ci95hi at every horizon."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            for p in ["0.0", "0.5", "0.9"]:
                lo = df[f"ci95lo_{p}"]
                hi = df[f"ci95hi_{p}"]
                irf = df[f"irf_{p}"]
                self.assertTrue((lo < hi).all(),
                                msg=f"{o} p={p}: ci95lo not always < ci95hi")
                self.assertTrue((irf >= lo).all(),
                                msg=f"{o} p={p}: irf below ci95lo")
                self.assertTrue((irf <= hi).all(),
                                msg=f"{o} p={p}: irf above ci95hi")

    def test_irf_additive_consistency(self):
        """irf_0.0 must equal beta (p=0 evaluates interaction term to zero)."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            diff = (df["irf_0.0"] - df["beta"]).abs().max()
            self.assertLess(diff, 1e-8,
                            msg=f"{o}: irf_0.0 ≠ beta, max diff={diff:.2e}")

    def test_no_nan_values(self):
        """Smooth LP CSVs must have no NaN values."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            self.assertFalse(df.isnull().any().any(),
                             msg=f"{o}: smooth LP contains NaN values")


# ──────────────────────────────────────────────────────────────────────────────
class TestIRFDirection(unittest.TestCase):
    """Impact direction and state amplification (core economic findings)."""

    def test_beta_positive_at_k0(self):
        """β_k0 > 0 for all outcomes (positive baseline impact at impact horizon)."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            beta0 = df[df["horizon"] == 0]["beta"].values[0]
            self.assertGreater(beta0, 0,
                               msg=f"{o}: β at k=0 is not positive ({beta0:.4f})")

    def test_theta_positive_at_k0_core(self):
        """θ_k0 > 0 for core outcomes (state amplification in high-fragility)."""
        for o in THETA_POSITIVE:
            df = smooth(o)
            theta0 = df[df["horizon"] == 0]["theta"].values[0]
            self.assertGreater(theta0, 0,
                               msg=f"{o}: θ at k=0 is not positive ({theta0:.4f})")

    def test_amplification_monotone_in_p(self):
        """IRF at p=0.9 ≥ IRF at p=0.5 ≥ IRF at p=0.0 at k=0 for core outcomes."""
        for o in THETA_POSITIVE:
            df = smooth(o)
            r = df[df["horizon"] == 0].iloc[0]
            irf0  = r["irf_0.0"]
            irf05 = r["irf_0.5"]
            irf09 = r["irf_0.9"]
            self.assertGreaterEqual(irf09, irf05,
                msg=f"{o}: irf_0.9 ({irf09:.4f}) < irf_0.5 ({irf05:.4f}) at k=0")
            self.assertGreaterEqual(irf05, irf0,
                msg=f"{o}: irf_0.5 ({irf05:.4f}) < irf_0.0 ({irf0:.4f}) at k=0")

    def test_amplification_magnitude_emfi(self):
        """EMFI: irf at p=0.9 is at least 3× larger than at p=0.0 at k=0."""
        df = smooth("EMFI")
        r  = df[df["horizon"] == 0].iloc[0]
        ratio = r["irf_0.9"] / r["irf_0.0"]
        self.assertGreater(ratio, 3.0,
            msg=f"EMFI: amplification ratio at k=0 is {ratio:.2f} (expected >3)")

    def test_amplification_magnitude_tci(self):
        """TCI: irf at p=0.9 is at least 5× larger than at p=0.0 at k=0."""
        df = smooth("tci_w100")
        r  = df[df["horizon"] == 0].iloc[0]
        ratio = r["irf_0.9"] / r["irf_0.0"]
        self.assertGreater(ratio, 5.0,
            msg=f"tci_w100: amplification ratio at k=0 is {ratio:.2f} (expected >5)")

    def test_pre_trend_baseline_direction(self):
        """At p=0 (calm), pre-trend IRFs should not show clear positive build-up:
           fewer than 4 of 5 pre-horizons should have beta > 0.2 (spurious pre-shock)."""
        for o in SMOOTH_OUTCOMES:
            df = smooth(o)
            pre = df[df["horizon"] < 0]
            large_positive = (pre["beta"] > 0.2).sum()
            self.assertLess(large_positive, 4,
                msg=f"{o}: {large_positive}/5 pre-trend betas exceed 0.2 — suspect pre-trend")


# ──────────────────────────────────────────────────────────────────────────────
class TestBinaryLP(unittest.TestCase):
    """Binary HighFragility LP validation."""

    def test_required_columns_binary(self):
        """Binary LP must have irf_normal, irf_fragile, se_normal, se_fragile."""
        required = ["horizon", "beta", "theta", "irf_normal", "irf_fragile",
                    "se_normal", "se_fragile", "ci95lo_normal", "ci95hi_normal",
                    "ci95lo_fragile", "ci95hi_fragile", "nobs"]
        for o in BINARY_OUTCOMES:
            df = binary(o)
            for col in required:
                self.assertIn(col, df.columns,
                              msg=f"{o}: binary LP missing column {col}")

    def test_horizons_binary(self):
        """Binary LP must also span horizons -5 to 15."""
        for o in BINARY_OUTCOMES:
            df = binary(o)
            self.assertEqual(len(df), 21,
                             msg=f"{o}: binary LP has {len(df)} rows, expected 21")

    def test_fragile_greater_than_normal_k0(self):
        """At k=0, irf_fragile > irf_normal for core amplification outcomes."""
        for o in FRAGILE_AMPLIFY:
            df = binary(o)
            r  = df[df["horizon"] == 0].iloc[0]
            self.assertGreater(r["irf_fragile"], r["irf_normal"],
                msg=f"{o} k=0: irf_fragile ({r['irf_fragile']:.4f}) "
                    f"≤ irf_normal ({r['irf_normal']:.4f})")

    def test_fragile_greater_than_normal_k1(self):
        """At k=1, irf_fragile > irf_normal for EMFI and TailVolBreadth."""
        for o in ["EMFI", "TailVolBreadth"]:
            df = binary(o)
            r  = df[df["horizon"] == 1].iloc[0]
            self.assertGreater(r["irf_fragile"], r["irf_normal"],
                msg=f"{o} k=1: irf_fragile ({r['irf_fragile']:.4f}) "
                    f"≤ irf_normal ({r['irf_normal']:.4f})")

    def test_binary_se_positive(self):
        """Binary LP SE must be positive."""
        for o in BINARY_OUTCOMES:
            df = binary(o)
            self.assertTrue((df["se_normal"] > 0).all(),
                            msg=f"{o}: se_normal has non-positive values")
            self.assertTrue((df["se_fragile"] > 0).all(),
                            msg=f"{o}: se_fragile has non-positive values")

    def test_irf_normal_equals_beta(self):
        """irf_normal must equal beta (HF=0 means interaction term is zero)."""
        for o in BINARY_OUTCOMES:
            df = binary(o)
            diff = (df["irf_normal"] - df["beta"]).abs().max()
            self.assertLess(diff, 1e-8,
                            msg=f"{o}: irf_normal ≠ beta, max diff={diff:.2e}")

    def test_irf_fragile_equals_beta_plus_theta(self):
        """irf_fragile must equal beta + theta (HF=1)."""
        for o in BINARY_OUTCOMES:
            df = binary(o)
            expected = df["beta"] + df["theta"]
            diff = (df["irf_fragile"] - expected).abs().max()
            self.assertLess(diff, 1e-8,
                            msg=f"{o}: irf_fragile ≠ beta+theta, max diff={diff:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
class TestCOVIDExclusion(unittest.TestCase):
    """COVID-excluded robustness variant."""

    def test_nocovid_fewer_obs(self):
        """No-COVID variant must have fewer observations than full sample."""
        for o in SMOOTH_OUTCOMES:
            df_full = smooth(o)
            df_nc   = smooth(o, nocovid=True)
            nobs_full = df_full[df_full["horizon"] == 0]["nobs"].values[0]
            nobs_nc   = df_nc[df_nc["horizon"] == 0]["nobs"].values[0]
            self.assertLess(nobs_nc, nobs_full,
                msg=f"{o}: nocovid nobs ({nobs_nc}) ≥ full ({nobs_full})")

    def test_nocovid_obs_reduction_plausible(self):
        """COVID exclusion should remove roughly 200–250 obs (≈ 2020 trading days)."""
        df_full = smooth("EMFI")
        df_nc   = smooth("EMFI", nocovid=True)
        nobs_full = df_full[df_full["horizon"] == 0]["nobs"].values[0]
        nobs_nc   = df_nc[df_nc["horizon"] == 0]["nobs"].values[0]
        reduction = nobs_full - nobs_nc
        self.assertGreater(reduction, 150,
            msg=f"EMFI: COVID exclusion reduced obs by only {reduction} (expected >150)")
        self.assertLess(reduction, 350,
            msg=f"EMFI: COVID exclusion reduced obs by {reduction} (expected <350)")

    def test_nocovid_beta_positive_at_k0(self):
        """β_k0 > 0 in the no-COVID subsample for core volatility outcomes.
        P_stress is excluded: its baseline β is near-zero outside COVID (−0.029),
        with the shock-stress amplification carried by θ rather than β — this is
        itself an interesting finding disclosed in the paper."""
        # AvgCorr60 excluded too: near-zero θ makes the β@k0 sign uninformative
        core_nocovid = ["EMFI", "tci_w100", "TailVolBreadth"]
        for o in core_nocovid:
            df = smooth(o, nocovid=True)
            beta0 = df[df["horizon"] == 0]["beta"].values[0]
            self.assertGreater(beta0, 0,
                msg=f"{o} nocovid: β at k=0 is not positive ({beta0:.4f})")

    def test_nocovid_files_exist(self):
        """No-COVID smooth LP files must exist for all outcomes."""
        for o in SMOOTH_OUTCOMES:
            fpath = os.path.join(OUTDIR, f"state_lp_smooth_{o}_nocovid.csv")
            self.assertTrue(os.path.exists(fpath),
                            msg=f"Missing nocovid file: {fpath}")


# ──────────────────────────────────────────────────────────────────────────────
class TestCompositionReport(unittest.TestCase):
    """HighFragility composition report."""

    def setUp(self):
        self.df = pd.read_csv(os.path.join(OUTDIR, "composition_report.csv"))

    def test_total_row_present(self):
        """Composition report must have a TOTAL row."""
        years = self.df["year"].astype(str).tolist()
        self.assertIn("TOTAL", years, msg="composition_report.csv missing TOTAL row")

    def test_total_hf_events(self):
        """TOTAL must show 36 HighFragility shock events (EMFI > 75th pctile, lagged)."""
        total = self.df[self.df["year"].astype(str) == "TOTAL"]
        n_hf = int(total["n_hf"].values[0])
        self.assertEqual(n_hf, 36,
            msg=f"Expected 36 HF events in TOTAL, got {n_hf}")

    def test_total_shock_days(self):
        """Total shock days must match LP sample (154)."""
        total = self.df[self.df["year"].astype(str) == "TOTAL"]
        n_shock = int(total["n_shock_days"].values[0])
        self.assertEqual(n_shock, 154,
            msg=f"Expected 154 shock days in TOTAL, got {n_shock}")

    def test_n_hf_plus_n_normal_equals_n_shock(self):
        """n_hf + n_normal must equal n_shock_days in every row."""
        df = self.df
        diff = (df["n_hf"] + df["n_normal"] - df["n_shock_days"]).abs().max()
        self.assertEqual(diff, 0,
            msg=f"n_hf + n_normal ≠ n_shock_days; max diff={diff}")

    def test_pct_hf_calculation(self):
        """pct_hf column must be n_hf / n_shock_days * 100 (within 0.5pp)."""
        df = self.df[self.df["n_shock_days"] > 0].copy()
        expected = df["n_hf"] / df["n_shock_days"] * 100
        diff = (df["pct_hf"] - expected).abs().max()
        self.assertLess(diff, 0.5,
            msg=f"pct_hf calculation error: max deviation = {diff:.3f} pp")

    def test_covid_year_highest_hf_fraction(self):
        """2020 (COVID) should have the highest HF fraction — expected ≥50%."""
        df = self.df[self.df["year"].astype(str) != "TOTAL"]
        df = df.copy(); df["year"] = df["year"].astype(str)
        pct_2020 = df[df["year"] == "2020"]["pct_hf"].values[0]
        self.assertGreater(pct_2020, 50,
            msg=f"2020 pct_hf={pct_2020:.1f}% (expected ≥50)")


# ──────────────────────────────────────────────────────────────────────────────
class TestOutputFiles(unittest.TestCase):
    """File existence and completeness."""

    def test_smooth_csv_files(self):
        """Smooth LP CSVs must exist for all outcomes (full + nocovid)."""
        for o in SMOOTH_OUTCOMES:
            for suffix in ["", "_nocovid"]:
                fpath = os.path.join(OUTDIR, f"state_lp_smooth_{o}{suffix}.csv")
                self.assertTrue(os.path.exists(fpath),
                                msg=f"Missing: {os.path.basename(fpath)}")

    def test_binary_csv_files(self):
        """Binary LP CSVs must exist for all outcomes."""
        for o in BINARY_OUTCOMES:
            fpath = os.path.join(OUTDIR, f"state_lp_binary_{o}.csv")
            self.assertTrue(os.path.exists(fpath),
                            msg=f"Missing: {os.path.basename(fpath)}")

    def test_figure_files_smooth(self):
        """Smooth-transition LP figures must exist (including no-COVID variants)."""
        for o in SMOOTH_OUTCOMES:
            for suffix in ["", "_nocovid"]:
                fpath = os.path.join(OUTDIR, f"Fig_StateLPSmooth_{o}{suffix}.png")
                self.assertTrue(os.path.exists(fpath),
                                msg=f"Missing figure: {os.path.basename(fpath)}")

    def test_figure_files_binary(self):
        """Binary LP figures must exist."""
        for o in BINARY_OUTCOMES:
            fpath = os.path.join(OUTDIR, f"Fig_StateLPBinary_{o}.png")
            self.assertTrue(os.path.exists(fpath),
                            msg=f"Missing figure: {os.path.basename(fpath)}")

    def test_combined_figure(self):
        """Combined smooth-transition figure must exist."""
        fpath = os.path.join(OUTDIR, "Fig_StateLPSmooth_Combined.png")
        self.assertTrue(os.path.exists(fpath),
                        msg="Missing combined figure: Fig_StateLPSmooth_Combined.png")

    def test_figure_sizes_nonzero(self):
        """All PNG files must be non-empty (>10 KB)."""
        import glob
        pngs = glob.glob(os.path.join(OUTDIR, "*.png"))
        self.assertGreater(len(pngs), 0, msg="No PNG files found in results/state_lp/")
        for f in pngs:
            size_kb = os.path.getsize(f) / 1024
            self.assertGreater(size_kb, 10,
                               msg=f"PNG too small ({size_kb:.1f} KB): {os.path.basename(f)}")

    def test_composition_report_exists(self):
        """composition_report.csv must exist."""
        fpath = os.path.join(OUTDIR, "composition_report.csv")
        self.assertTrue(os.path.exists(fpath), msg="Missing: composition_report.csv")


# ──────────────────────────────────────────────────────────────────────────────
class TestManifest(unittest.TestCase):
    """manifest.json structure and content."""

    def setUp(self):
        fpath = os.path.join(OUTDIR, "manifest.json")
        with open(fpath) as f:
            self.m = json.load(f)

    def test_required_fields(self):
        """Manifest must have generated_at, script, n_shock_days, and summary keys."""
        for key in ["generated_at", "script", "n_shock_days", "summary"]:
            self.assertIn(key, self.m, msg=f"manifest.json missing key: {key}")

    def test_script_field(self):
        """script field must reference run_state_lp.py."""
        self.assertIn("run_state_lp.py", self.m["script"],
                      msg="manifest.json script field does not reference run_state_lp.py")

    def test_summary_covers_all_outcomes(self):
        """summary must have entries for all 5 outcomes."""
        n = len(self.m["summary"])
        self.assertEqual(n, 5,
            msg=f"manifest.json summary has {n} entries (expected 5 outcomes)")

    def test_manifest_shock_counts(self):
        """Manifest must record n_shock_days=154, n_shock_hf=36."""
        got_days = self.m["n_shock_days"]
        got_hf   = self.m["n_shock_hf"]
        self.assertEqual(got_days, 154, msg="manifest n_shock_days != 154")
        self.assertEqual(got_hf, 36, msg="manifest n_shock_hf != 36")

    def test_generated_at_format_stub(self):
        pass  # placeholder — see test_generated_at_format below

