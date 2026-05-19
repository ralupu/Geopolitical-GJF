"""
SP12 -- HMM Robustness: Alternative Specifications and Filtered Probabilities
=============================================================================
Addresses R4: reviewer concern about HMM specification sensitivity and
filtered vs smoothed probabilities.

This script:
  1. Fits four HMM variants alongside the baseline 3-state Gaussian HMM:
       (a) 2-state HMM (Gaussian, full covariance)
       (b) 3-state HMM, diagonal covariance
       (c) COVID-excluded HMM: train on non-2020 data, apply to full sample
       (d) Pre-2020 HMM: train on 2017-01-01 to 2019-12-31, apply to full sample
  2. Computes agreement with baseline (Cohen's kappa, P_stress correlation)
  3. Implements forward-only filtered probabilities (versus smoothed posteriors)
  4. Adds P_stress_filtered to hmm_daily.csv
  5. Saves robustness summary to results/hmm_robustness/

Inputs (relative to Paper_GFJ root):
  results/emfi/emfi_daily.csv
  results/stress_regimes/hmm_daily.csv     (baseline)
  results/stress_regimes/hmm_params.json   (baseline model params)

Outputs (relative to Paper_GFJ root):
  results/hmm_robustness/hmm_variants.csv       (P_stress for all variants)
  results/hmm_robustness/kappa_table.csv
  results/hmm_robustness/Fig_HMM_Variants.png
  results/hmm_robustness/manifest.json
  results/stress_regimes/hmm_daily.csv          (updated: adds P_stress_filtered)

Run from Paper_GFJ root:
  python subprojects/12_hmm_robustness/run_hmm_robustness.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_EMFI   = GFJ_ROOT / "results" / "emfi"          / "emfi_daily.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"
IN_PARAMS = GFJ_ROOT / "results" / "stress_regimes" / "hmm_params.json"

OUT_DIR   = GFJ_ROOT / "results" / "hmm_robustness"
OUT_HMM   = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"  # updated in-place

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
COMPONENT_COLS = ["VolStress_std", "TailVolBreadth_std", "AvgCorr60_std", "tci_w100_std"]
N_RESTARTS     = 30
N_ITER         = 400
RANDOM_SEED    = 42
COVID_START    = "2020-01-01"
COVID_END      = "2020-12-31"
PRE_COVID_END  = "2019-12-31"

MAJOR_EVENTS = [
    (pd.Timestamp("2020-02-24"), "COVID"),
    (pd.Timestamp("2022-02-24"), "Ukraine"),
    (pd.Timestamp("2023-10-07"), "Hamas"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ===========================================================================
# Core HMM fitting
# ===========================================================================

def fit_hmm(X: np.ndarray, n_states: int, covariance_type: str = "full",
            n_restarts: int = N_RESTARTS, seed: int = RANDOM_SEED) -> GaussianHMM:
    """Fit a GaussianHMM with multiple random restarts; return best by log-likelihood."""
    rng = np.random.default_rng(seed)
    best_model, best_ll = None, -np.inf
    for _ in range(n_restarts):
        rs = int(rng.integers(0, 2**31))
        model = GaussianHMM(
            n_components=n_states, covariance_type=covariance_type,
            n_iter=N_ITER, tol=1e-4, random_state=rs, verbose=False,
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError(f"All restarts failed for {n_states}-state {covariance_type} HMM")
    return best_model


def order_states_by_emfi(model: GaussianHMM, emfi: np.ndarray) -> np.ndarray:
    """
    Reorder HMM states so that state 0 = calm (low mean EMFI),
    state n-1 = systemic stress (high mean EMFI).
    Returns array of state labels aligned to model state indices.
    """
    smoothed = model.predict_proba(emfi.reshape(-1, 1) if emfi.ndim == 1 else emfi)
    # Weighted mean EMFI per state
    state_mean_emfi = []
    for s in range(model.n_components):
        w = smoothed[:, s]
        state_mean_emfi.append(np.average(emfi, weights=w + 1e-10))
    order = np.argsort(state_mean_emfi)  # ascending: calm first
    return order  # order[i] = original state index of the i-th ordered state


def get_stress_prob(model: GaussianHMM, X: np.ndarray, emfi: np.ndarray) -> np.ndarray:
    """
    Get P_stress = probability of the highest-EMFI state (smoothed posterior).
    Returns array of shape (T,).
    """
    smoothed = model.predict_proba(X)           # (T, n_states)
    order    = order_states_by_emfi(model, emfi)
    stress_state_original = order[-1]           # original index of stress state
    return smoothed[:, stress_state_original]


def get_viterbi_regime(model: GaussianHMM, X: np.ndarray, emfi: np.ndarray) -> np.ndarray:
    """
    Get Viterbi path mapped to ordered regime labels (0=calm, ..., n-1=stress).
    """
    order    = order_states_by_emfi(model, emfi)
    # Invert: for each original state, what is the ordered label?
    inv_order = np.zeros(model.n_components, dtype=int)
    for ordered_label, orig_state in enumerate(order):
        inv_order[orig_state] = ordered_label
    viterbi_orig = model.predict(X)
    return inv_order[viterbi_orig]


# ===========================================================================
# Filtered probabilities (forward algorithm)
# ===========================================================================

def compute_filtered_probabilities(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute filtered probabilities P(s_t | x_1, ..., x_t) using forward algorithm.
    This uses only past and current observations (no future data), making it
    appropriate for real-time or policy applications.

    Returns array of shape (T, n_states).
    """
    T, K = X.shape
    n    = model.n_components
    pi   = model.startprob_          # (n,)
    A    = model.transmat_           # (n, n)

    # Compute log emission probabilities for all t, all states
    # GaussianHMM: manual computation using stored means and covariances
    log_emit = np.zeros((T, n))
    for s in range(n):
        mu  = model.means_[s]        # (K,)
        cov = model.covars_[s] if model.covariance_type == "full" else np.diag(model.covars_[s])
        diff = X - mu                # (T, K)
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            maha    = np.einsum("ti,ij,tj->t", diff, cov_inv, diff)
            log_emit[:, s] = (
                -0.5 * maha
                - 0.5 * K * np.log(2 * np.pi)
                - 0.5 * np.log(np.maximum(cov_det, 1e-300))
            )
        except np.linalg.LinAlgError:
            # Fallback: diagonal
            var  = np.diag(cov)
            log_emit[:, s] = -0.5 * np.sum(diff**2 / np.maximum(var, 1e-10), axis=1)

    # Forward algorithm (log-space for numerical stability)
    log_alpha = np.zeros((T, n))
    log_alpha[0] = np.log(np.maximum(pi, 1e-300)) + log_emit[0]

    for t in range(1, T):
        for s in range(n):
            log_alpha[t, s] = np.logaddexp.reduce(
                log_alpha[t-1] + np.log(np.maximum(A[:, s], 1e-300))
            ) + log_emit[t, s]

    # Normalise to get filtered probabilities
    filtered = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    filtered = filtered / filtered.sum(axis=1, keepdims=True)
    return filtered


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_total = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 65)
    log.info("SP12 — HMM Robustness: Alternative Specifications")
    log.info("=" * 65)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    log.info("\n[1/6] Loading data ...")
    emfi_df = pd.read_csv(IN_EMFI, parse_dates=["date"])
    hmm_base = pd.read_csv(IN_HMM, parse_dates=["date"])

    # Prepare feature matrix (all dates)
    df = emfi_df.dropna(subset=COMPONENT_COLS).copy().reset_index(drop=True)
    X  = df[COMPONENT_COLS].values.astype(float)
    emfi_vals = df["EMFI"].values
    dates     = df["date"].values

    log.info("  Full sample: %d obs (%s to %s)",
             len(df), df["date"].min().date(), df["date"].max().date())

    # Baseline P_stress (smoothed)
    base = hmm_base.set_index("date")["P_stress"]
    dates_pd = pd.DatetimeIndex(dates)

    # -----------------------------------------------------------------------
    # [2] Filtered probabilities on baseline model
    # -----------------------------------------------------------------------
    log.info("\n[2/6] Computing filtered probabilities on baseline HMM ...")

    # Reconstruct baseline model from saved parameters
    params = json.load(open(IN_PARAMS))
    n_base = params["n_states"]
    baseline_model = GaussianHMM(
        n_components=n_base, covariance_type="full",
        n_iter=1, random_state=42
    )
    # We need to fit at least once to initialise internals, then override
    baseline_model.fit(X[:50])    # dummy fit to initialise
    baseline_model.startprob_ = np.array(params["startprob"])
    baseline_model.transmat_  = np.array(params["transmat"])
    baseline_model.means_     = np.array(params["means"])
    baseline_model.covars_    = np.array(params["covars"])

    t0 = time.time()
    filtered_all = compute_filtered_probabilities(baseline_model, X)
    log.info("  Filtered proba computed in %.1fs", time.time() - t0)

    # Get stress state index (highest-EMFI state)
    order = order_states_by_emfi(baseline_model, emfi_vals)
    stress_orig = order[-1]
    P_stress_filtered = filtered_all[:, stress_orig]

    # Correlation with smoothed
    base_aligned = base.reindex(dates_pd)
    corr_filt_smooth = pd.Series(P_stress_filtered, index=dates_pd).corr(base_aligned)
    log.info("  Filtered vs smoothed P_stress correlation: %.4f", corr_filt_smooth)

    # Update hmm_daily.csv with filtered proba
    hmm_updated = hmm_base.copy()
    filt_series = pd.Series(P_stress_filtered, index=dates_pd, name="P_stress_filtered")
    hmm_updated = hmm_updated.set_index("date")
    hmm_updated["P_stress_filtered"] = filt_series
    hmm_updated = hmm_updated.reset_index()
    hmm_updated.to_csv(OUT_HMM, index=False)
    log.info("  Updated hmm_daily.csv with P_stress_filtered")

    # -----------------------------------------------------------------------
    # [3] Fit HMM variants
    # -----------------------------------------------------------------------
    log.info("\n[3/6] Fitting HMM variants ...")

    variants = {}

    # (a) 2-state HMM
    log.info("  (a) 2-state Gaussian (full cov) ...")
    model_2s = fit_hmm(X, n_states=2, covariance_type="full")
    p_stress_2s = get_stress_prob(model_2s, X, emfi_vals)
    regime_2s   = get_viterbi_regime(model_2s, X, emfi_vals)
    log.info("    LL=%.2f, P_stress mean=%.3f", model_2s.score(X), p_stress_2s.mean())
    variants["2-state"] = {"P_stress": p_stress_2s, "regime": regime_2s, "model": model_2s}

    # (b) 3-state diagonal covariance
    log.info("  (b) 3-state Gaussian (diagonal cov) ...")
    model_diag = fit_hmm(X, n_states=3, covariance_type="diag")
    p_stress_diag = get_stress_prob(model_diag, X, emfi_vals)
    regime_diag   = get_viterbi_regime(model_diag, X, emfi_vals)
    log.info("    LL=%.2f, P_stress mean=%.3f", model_diag.score(X), p_stress_diag.mean())
    variants["3-state-diag"] = {"P_stress": p_stress_diag, "regime": regime_diag, "model": model_diag}

    # (c) COVID-excluded HMM: train on non-2020 data, apply to full sample
    log.info("  (c) COVID-excluded HMM (train on non-2020) ...")
    non_covid_mask = ~(
        (df["date"] >= COVID_START) & (df["date"] <= COVID_END)
    )
    X_nc = X[non_covid_mask]
    emfi_nc = emfi_vals[non_covid_mask]
    model_nc = fit_hmm(X_nc, n_states=3, covariance_type="full")
    # Apply to full sample (model fitted on non-COVID data)
    p_stress_nc = get_stress_prob(model_nc, X, emfi_vals)
    regime_nc   = get_viterbi_regime(model_nc, X, emfi_vals)
    log.info("    Train obs=%d, LL=%.2f (on full sample)",
             X_nc.shape[0], model_nc.score(X))
    variants["covid-excl"] = {"P_stress": p_stress_nc, "regime": regime_nc, "model": model_nc}

    # (d) Pre-2020 HMM: train on 2017-2019, apply to full sample (out-of-sample)
    log.info("  (d) Pre-2020 HMM (train 2017-2019, apply all) ...")
    pre2020_mask = df["date"] <= PRE_COVID_END
    X_pre = X[pre2020_mask]
    emfi_pre = emfi_vals[pre2020_mask]
    if len(X_pre) < 50:
        log.warning("    Too few pre-2020 observations (%d), skipping", len(X_pre))
    else:
        model_pre = fit_hmm(X_pre, n_states=3, covariance_type="full")
        p_stress_pre = get_stress_prob(model_pre, X, emfi_vals)
        regime_pre   = get_viterbi_regime(model_pre, X, emfi_vals)
        log.info("    Train obs=%d, LL=%.2f (on full sample)",
                 X_pre.shape[0], model_pre.score(X))
        variants["pre-2020"] = {"P_stress": p_stress_pre, "regime": regime_pre, "model": model_pre}

    # -----------------------------------------------------------------------
    # [4] Agreement statistics
    # -----------------------------------------------------------------------
    log.info("\n[4/6] Computing agreement statistics ...")

    # Baseline Viterbi regime (0=calm, 1=elevated, 2=stress)
    baseline_regime = get_viterbi_regime(baseline_model, X, emfi_vals)
    # Baseline P_stress (smoothed, from hmm_daily.csv)
    base_p = base.reindex(dates_pd).values

    kappa_rows = []
    for name, v in variants.items():
        p_s   = v["P_stress"]
        r     = v["regime"]
        corr  = pd.Series(p_s).corr(pd.Series(base_p))

        # For kappa: need same number of states
        try:
            if v["model"].n_components == 3:
                kappa = cohen_kappa_score(baseline_regime, r)
            else:
                # 2-state: map to binary (stress vs not-stress)
                base_binary = (baseline_regime == 2).astype(int)
                r_binary    = (r == 1).astype(int)   # state 1 = stress in 2-state
                kappa = cohen_kappa_score(base_binary, r_binary)
        except Exception as e:
            kappa = np.nan
            log.warning("    Kappa failed for %s: %s", name, e)

        pct_stress = float((p_s > 0.5).mean())
        row = {
            "variant": name,
            "n_states": v["model"].n_components,
            "corr_P_stress": round(float(corr), 4),
            "kappa": round(float(kappa), 4) if not np.isnan(kappa) else np.nan,
            "pct_high_stress": round(pct_stress, 4),
        }
        kappa_rows.append(row)
        log.info("  %s: corr=%.3f, kappa=%.3f, pct_stress=%.1f%%",
                 name, corr, kappa if not np.isnan(kappa) else float("nan"),
                 pct_stress * 100)

    # Add filtered-vs-smoothed row
    kappa_rows.insert(0, {
        "variant": "baseline-filtered",
        "n_states": 3,
        "corr_P_stress": round(float(corr_filt_smooth), 4),
        "kappa": None,
        "pct_high_stress": round(float((P_stress_filtered > 0.5).mean()), 4),
    })

    kappa_df = pd.DataFrame(kappa_rows)
    kappa_df.to_csv(OUT_DIR / "kappa_table.csv", index=False)
    log.info("  Saved kappa_table.csv")

    # -----------------------------------------------------------------------
    # [5] Save all P_stress variants
    # -----------------------------------------------------------------------
    log.info("\n[5/6] Saving variant P_stress series ...")
    variants_df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    variants_df["P_stress_baseline"]  = base_p
    variants_df["P_stress_filtered"]  = P_stress_filtered
    for name, v in variants.items():
        col = f"P_stress_{name.replace('-','_').replace(' ','_')}"
        variants_df[col] = v["P_stress"]
    variants_df.to_csv(OUT_DIR / "hmm_variants.csv", index=False)
    log.info("  Saved hmm_variants.csv (%d rows, %d cols)",
             len(variants_df), len(variants_df.columns))

    # -----------------------------------------------------------------------
    # [6] Figure
    # -----------------------------------------------------------------------
    log.info("\n[6/6] Generating figure ...")

    cols_to_plot = {
        "P_stress_baseline":  ("Baseline (3-state, full)", "#1f77b4", 1.2),
        "P_stress_filtered":  ("Baseline (filtered)", "#aec7e8", 0.8),
    }
    for name, v in variants.items():
        col  = f"P_stress_{name.replace('-','_').replace(' ','_')}"
        clrs = {"2-state": "#d62728", "3-state-diag": "#2ca02c",
                "covid_excl": "#ff7f0e", "pre_2020": "#9467bd"}
        col_key = col.replace("P_stress_","")
        clr = clrs.get(col_key, "#8c564b")
        cols_to_plot[col] = (name, clr, 0.7)

    n_panels = len(cols_to_plot)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (col, (label, clr, lw)) in zip(axes, cols_to_plot.items()):
        if col in variants_df.columns:
            ax.plot(variants_df["date"], variants_df[col], color=clr, lw=lw)
            ax.set_ylabel("$P_{stress}$", fontsize=8)
            ax.set_title(label, fontsize=9)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            for ev_date, ev_label in MAJOR_EVENTS:
                ax.axvline(ev_date, color="black", lw=0.6, ls="--", alpha=0.5)
            ax.axhline(0.5, color="grey", lw=0.5, ls=":")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("HMM Variant Comparison: Systemic Stress Probability", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig_HMM_Variants.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved Fig_HMM_Variants.png")

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------
    manifest = {
        "script":               "SP12 run_hmm_robustness.py",
        "run_at":               datetime.now(timezone.utc).isoformat(),
        "corr_filtered_smooth": round(float(corr_filt_smooth), 4),
        "variants_run":         list(variants.keys()),
        "kappa_summary": {
            r["variant"]: {k: v for k, v in r.items() if k != "variant"}
            for r in kappa_rows
        },
        "elapsed_s": round(time.time() - t_total, 1),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  Saved manifest.json")

    log.info("\n✅ SP12 complete (%.1fs)", time.time() - t_total)
    log.info("   Outputs → %s", OUT_DIR)
    return manifest


if __name__ == "__main__":
    main()
