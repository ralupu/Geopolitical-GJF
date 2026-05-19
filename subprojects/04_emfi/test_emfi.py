"""
Tests for SP04 -- Composite EMFI
=================================
Run from Paper_GFJ root:
  pytest subprojects/04_emfi/test_emfi.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]
sys.path.insert(0, str(THIS_FILE.parent))

from build_emfi import (
    COMPONENTS,
    HIGH_FRAG_PCTILE,
    OUT_EMFI,
    OUT_LOADINGS,
    OUT_MANIFEST,
    OUT_SUMMARY,
    PRIMARY_W_TCI,
    build_emfi,
    load_inputs,
)

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="module")
def inputs():
    if not (GFJ_ROOT / "results" / "fragility" / "fragility_daily.csv").exists():
        pytest.skip("fragility_daily.csv not found -- run SP02 first")
    if not (GFJ_ROOT / "results" / "connectedness" / "tci_daily.csv").exists():
        pytest.skip("tci_daily.csv not found -- run SP03 first")
    return load_inputs()


@pytest.fixture(scope="module")
def emfi_results(inputs):
    return build_emfi(inputs)


@pytest.fixture(scope="module")
def emfi():
    if not OUT_EMFI.exists():
        pytest.skip("emfi_daily.csv not yet generated")
    return pd.read_csv(OUT_EMFI, parse_dates=["date"])


@pytest.fixture(scope="module")
def loadings():
    if not OUT_LOADINGS.exists():
        pytest.skip("pca_loadings.csv not yet generated")
    return pd.read_csv(OUT_LOADINGS)


@pytest.fixture(scope="module")
def manifest():
    if not OUT_MANIFEST.exists():
        pytest.skip("manifest.json not yet generated")
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


# -- Unit tests: build_emfi ---------------------------------------------------

class TestBuildEMFI:
    def test_emfi_df_columns(self, emfi_results):
        emfi_df, _, _ = emfi_results
        assert "EMFI" in emfi_df.columns
        assert "date" in emfi_df.columns
        for comp in COMPONENTS:
            assert comp + "_std" in emfi_df.columns

    def test_emfi_length_matches_input(self, inputs, emfi_results):
        emfi_df, _, _ = emfi_results
        assert len(emfi_df) == len(inputs)

    def test_std_components_standardised(self, emfi_results):
        """Each _std column should have mean ~0 and std ~1."""
        emfi_df, _, _ = emfi_results
        for comp in COMPONENTS:
            col = emfi_df[comp + "_std"]
            assert abs(col.mean()) < 1e-10, "Mean not zero for " + comp
            assert abs(col.std(ddof=1) - 1.0) < 1e-10, "Std not 1 for " + comp

    def test_pc1_variance_above_50(self, emfi_results):
        _, _, meta = emfi_results
        assert meta["pca"]["pc1_variance_pct"] >= 50, (
            "PC1 explains less than 50%: %.1f%%" % meta["pca"]["pc1_variance_pct"]
        )

    def test_all_loadings_positive(self, emfi_results):
        """After orientation, all loadings should be positive
        (all four indicators increase together during stress)."""
        _, loadings, _ = emfi_results
        for _, row in loadings.iterrows():
            assert row["loading"] > 0, "Negative loading for " + row["component"]

    def test_eigenvalues_descending(self, emfi_results):
        _, _, meta = emfi_results
        eigs = meta["pca"]["eigenvalues"]
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i + 1], "Eigenvalues not descending"

    def test_emfi_correlated_with_volstress(self, inputs, emfi_results):
        """EMFI must be positively correlated with VolStress."""
        emfi_df, _, _ = emfi_results
        corr = emfi_df["EMFI"].corr(inputs["VolStress"])
        assert corr > 0.7, "EMFI-VolStress corr unexpectedly low: %.4f" % corr

    def test_emfi_peaks_during_covid(self, inputs, emfi_results):
        """Max EMFI should fall during the COVID crash window."""
        emfi_df, _, _ = emfi_results
        peak_date = emfi_df.loc[emfi_df["EMFI"].idxmax(), "date"]
        assert pd.Timestamp("2020-02-01") <= peak_date <= pd.Timestamp("2020-04-30"), (
            "EMFI peak not in COVID window: %s" % peak_date.date()
        )

    def test_loadings_sum_of_squares(self, emfi_results):
        """PC1 loadings vector has unit norm (eigenvector property)."""
        _, loadings, _ = emfi_results
        ss = (loadings["loading"] ** 2).sum()
        assert abs(ss - 1.0) < 1e-6, "Loading vector not unit norm: %.6f" % ss


# -- Integration tests --------------------------------------------------------

class TestEMFIFile:
    def test_columns(self, emfi):
        assert "date" in emfi.columns
        assert "EMFI" in emfi.columns
        for comp in COMPONENTS:
            assert comp + "_std" in emfi.columns

    def test_no_duplicate_dates(self, emfi):
        assert not emfi["date"].duplicated().any()

    def test_no_nans(self, emfi):
        for col in ["EMFI"] + [c + "_std" for c in COMPONENTS]:
            assert emfi[col].notna().all(), "NaN in " + col

    def test_emfi_finite(self, emfi):
        assert np.isfinite(emfi["EMFI"].values).all()

    def test_covid_peak_emfi(self, emfi):
        covid = emfi[
            (emfi["date"] >= "2020-02-01") & (emfi["date"] <= "2020-04-30")
        ]["EMFI"]
        overall_max = emfi["EMFI"].max()
        assert covid.max() >= overall_max * 0.9, (
            "COVID EMFI (%.2f) not near overall max (%.2f)" % (covid.max(), overall_max)
        )

    def test_high_fragility_threshold(self, emfi):
        """75th percentile threshold should be positive (EMFI > 0 in top quartile)."""
        p75 = emfi["EMFI"].quantile(HIGH_FRAG_PCTILE / 100)
        assert p75 > 0


class TestLoadings:
    def test_columns(self, loadings):
        for col in ["component", "loading", "loading_sq", "pct_variance_contribution"]:
            assert col in loadings.columns

    def test_all_components_present(self, loadings):
        assert set(loadings["component"]) == set(COMPONENTS)

    def test_loadings_positive(self, loadings):
        assert (loadings["loading"] > 0).all()

    def test_loading_sq_consistent(self, loadings):
        diff = (loadings["loading_sq"] - loadings["loading"] ** 2).abs()
        assert diff.max() < 1e-6

    def test_loadings_unit_norm(self, loadings):
        ss = loadings["loading_sq"].sum()
        assert abs(ss - 1.0) < 1e-10


class TestManifest:
    def test_required_keys(self, manifest):
        for key in ["generated_at", "pca", "emfi_stats", "date_range",
                    "components", "parameters", "outputs"]:
            assert key in manifest

    def test_pc1_variance_recorded(self, manifest):
        assert manifest["pca"]["pc1_variance_pct"] >= 50

    def test_emfi_stats_range(self, manifest):
        s = manifest["emfi_stats"]
        assert s["min"] < 0
        assert s["max"] > 0
        assert s["max"] > s["p75_threshold"] > 0

    def test_outputs_exist(self, manifest):
        for key, path_str in manifest["outputs"].items():
            assert Path(path_str).exists(), "Missing: " + key

    def test_components_recorded(self, manifest):
        assert set(manifest["components"]) == set(COMPONENTS)


class TestProvenance:
    def test_summary_exists(self):
        assert OUT_SUMMARY.exists()

    def test_summary_has_emfi(self):
        summary = pd.read_csv(OUT_SUMMARY, index_col=0)
        assert "EMFI" in summary.index

    def test_manifest_parseable(self):
        m = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
        assert "generated_at" in m
