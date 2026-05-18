"""
SP05 -- HMM Market-Implied Stress Regimes
==========================================
Estimates a Gaussian-emission Hidden Markov Model on the four standardised
EMFI components to classify each trading day into latent market stress regimes
without any reference to geopolitical shock dates.

Inputs (relative to Paper_GFJ root):
  results/emfi/emfi_daily.csv          -- standardised components + EMFI

Outputs (relative to Paper_GFJ root):
  results/stress_regimes/hmm_daily.csv     -- P_stress, P_elevated, P_calm, regime, EMFI
  results/stress_regimes/hmm_params.json   -- fitted HMM parameters
  results/stress_regimes/Fig_HMM_Regimes.png
  results/stress_regimes/Fig_HMM_Validation.png
  results/stress_regimes/manifest.json

HMM specification
-----------------
  Input vector : X_t = [VolStress_std, TailVolBreadth_std, AvgCorr60_std, tci_w100_std]
  Emission     : Gaussian (full covariance)
  States       : 3 (primary); 2 (robustness, saved separately)
  Ordering     : states labelled post-hoc by ascending mean EMFI
                 State 0 = Calm, State 1 = Elevated, State 2 = Systemic stress
  Restarts     : N_RESTARTS random initialisations; best log-likelihood kept
  Circularity  : shock dates are NOT an input -- verified by design

Run from Paper_GFJ root:
  python subprojects/05_stress_regimes/build_hmm_regimes.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE  = Path(__file__).resolve()
GFJ_ROOT   = THIS_FILE.parents[2]

IN_EMFI    = GFJ_ROOT / "results" / "emfi" / "emfi_daily.csv"

OUT_DIR    = GFJ_ROOT / "results" / "stress_regimes"
OUT_CSV    = OUT_DIR / "hmm_daily.csv"
OUT_PARAMS = OUT_DIR / "hmm_params.json"
OUT_FIG_REGIMES    = OUT_DIR / "Fig_HMM_Regimes.png"
OUT_FIG_VALIDATION = OUT_DIR / "Fig_HMM_Validation.png"
OUT_MANIFEST       = OUT_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_STATES          = 3          # primary specification
N_RESTARTS        = 50         # random initialisations
N_ITER            = 500        # max EM iterations per restart
RANDOM_SEED       = 42
COMPONENT_COLS    = ["VolStress_std", "TailVolBreadth_std", "AvgCorr60_std", "tci_w100_std"]

# State label constants (assigned post-hoc by EMFI ordering)
STATE_CALM     = 0
STATE_ELEVATED = 1
STATE_STRESS   = 2

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


# ---------------------------------------------------------------------------
# Core: fit HMM with multiple random restarts
# ---------------------------------------------------------------------------

def fit_hmm(X: np.ndarray, n_states: int, n_restarts: int,
            n_iter: int, seed: int) -> GaussianHMM:
    """
    Fit a GaussianHMM with full covariance to X.
    Runs n_restarts random initialisations and keeps the best (highest log-likelihood).
    """
    rng = np.random.default_rng(seed)
    best_model = None
    best_ll    = -np.inf

    for restart in range(n_restarts):
        rs = int(rng.integers(0, 2**31))
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=1e-4,
            random_state=rs,
            verbose=False,
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll    = ll
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed to converge.")

    log.info("  Best log-likelihood: %.2f  (over %d restarts)", best_ll, n_restarts)
    return best_model


def order_states_by_emfi(model: GaussianHMM, X: np.ndarray, emfi: pd.Series) -> np.ndarray:
    """
    Return a permutation array that maps current state indices to
    [calm, elevated, systemic] ordered by ascending mean EMFI over Viterbi path.
    """
    viterbi_raw = model.predict(X)
    means = np.array([emfi[viterbi_raw == s].mean() for s in range(model.n_components)])
    perm  = np.argsort(means)          # perm[0] = current index of "calm", etc.
    return perm


def apply_permutation(model: GaussianHMM, perm: np.ndarray) -> GaussianHMM:
    """
    Reorder a fitted HMM's parameters according to perm (in-place, returns model).
    """
    model.startprob_  = model.startprob_[perm]
    model.transmat_   = model.transmat_[np.ix_(perm, perm)]
    model.means_      = model.means_[perm]
    model.covars_     = model.covars_[perm]
    return model


def build_hmm_regimes(emfi_df: pd.DataFrame, n_states: int = N_STATES,
                       n_restarts: int = N_RESTARTS,
                       n_iter: int = N_ITER,
                       seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, GaussianHMM]:
    """
    Fit HMM on the standardised EMFI components and return:
      - daily DataFrame with columns [date, EMFI, regime, P_calm, P_elevated, P_stress]
      - the fitted (state-ordered) GaussianHMM model
    """
    df = emfi_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[COMPONENT_COLS].values.astype(float)
    log.info("  HMM input matrix: %s", X.shape)

    log.info("  Fitting %d-state GaussianHMM (%d restarts)...", n_states, n_restarts)
    model = fit_hmm(X, n_states, n_restarts, n_iter, seed)
    log.info("  Converged: %s", model.monitor_.converged)

    # Order states by ascending mean EMFI
    perm  = order_states_by_emfi(model, X, df["EMFI"])
    log.info("  State ordering permutation (old -> new): %s", perm)
    model = apply_permutation(model, perm)

    # Viterbi path (regime labels after reordering)
    viterbi = model.predict(X)

    # Posterior probabilities
    logging.disable(logging.CRITICAL)   # suppress hmmlearn internal logs during predict_proba
    posteriors = model.predict_proba(X)
    logging.disable(logging.NOTSET)

    result = df[["date", "EMFI"]].copy()
    result["regime"] = viterbi

    if n_states == 3:
        result["P_calm"]     = posteriors[:, STATE_CALM]
        result["P_elevated"] = posteriors[:, STATE_ELEVATED]
        result["P_stress"]   = posteriors[:, STATE_STRESS]
    elif n_states == 2:
        result["P_calm"]   = posteriors[:, 0]
        result["P_stress"] = posteriors[:, 1]
    else:
        for s in range(n_states):
            result["P_state_" + str(s)] = posteriors[:, s]

    # Diagnostics
    for s in range(n_states):
        n_days = (viterbi == s).sum()
        log.info("  State %d: %d days (%.1f%%)", s, n_days, 100 * n_days / len(viterbi))

    return result, model


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

MAJOR_EVENTS = {
    "COVID crash":     ("2020-02-20", "2020-04-15"),
    "Ukraine invasion":("2022-02-24", "2022-03-15"),
    "Hamas-Israel":    ("2023-10-07", "2023-10-20"),
}

EVENT_LINES = {
    "COVID\npeak":   "2020-03-12",
    "Ukraine":       "2022-02-24",
    "Hamas":         "2023-10-07",
}

STATE_COLORS = {
    0: "#2ca02c",   # calm    — green
    1: "#ff7f0e",   # elevated — orange
    2: "#d62728",   # stress   — red
}
STATE_LABELS = {0: "Calm", 1: "Elevated", 2: "Systemic stress"}


def _shade_events(ax):
    for label, (start, end) in MAJOR_EVENTS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.10, color="steelblue", zorder=0)


def plot_hmm_regimes(result: pd.DataFrame, out_path: Path):
    dates = pd.to_datetime(result["date"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: EMFI coloured by regime
    ax = axes[0]
    for s in range(3):
        mask = result["regime"] == s
        ax.scatter(dates[mask], result.loc[mask, "EMFI"],
                   c=STATE_COLORS[s], s=3, label=STATE_LABELS[s], zorder=2)
    ax.plot(dates, result["EMFI"], color="grey", lw=0.4, alpha=0.4, zorder=1)
    for ev_label, ev_date in EVENT_LINES.items():
        ax.axvline(pd.Timestamp(ev_date), color="navy", lw=0.8, alpha=0.6)
        ax.text(pd.Timestamp(ev_date), ax.get_ylim()[1] * 0.85, ev_label,
                fontsize=6, ha="center", color="navy")
    _shade_events(ax)
    ax.set_ylabel("EMFI", fontsize=9)
    ax.set_title("EMFI coloured by HMM regime", fontsize=9)
    ax.legend(fontsize=7, markerscale=4, loc="upper left")

    # Panel 2: P_stress
    ax = axes[1]
    ax.fill_between(dates, result["P_stress"], color=STATE_COLORS[2], alpha=0.7, label="P(systemic stress)")
    ax.fill_between(dates, result["P_elevated"], color=STATE_COLORS[1], alpha=0.5, label="P(elevated)")
    ax.set_ylabel("Posterior prob.", fontsize=9)
    ax.set_title("HMM state probabilities", fontsize=9)
    ax.set_ylim(0, 1)
    _shade_events(ax)
    ax.legend(fontsize=7)

    # Panel 3: Regime as integer (Viterbi)
    ax = axes[2]
    regime_colors = [STATE_COLORS[r] for r in result["regime"]]
    ax.bar(dates, result["regime"] + 0.5, width=1.2,
           color=regime_colors, alpha=0.8)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Calm", "Elevated", "Systemic"], fontsize=8)
    ax.set_ylabel("Regime", fontsize=9)
    ax.set_title("Viterbi regime path", fontsize=9)
    _shade_events(ax)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=0)
    plt.suptitle("HMM Market-Implied Stress Regimes (2017-2025)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", out_path)


def plot_hmm_validation(result: pd.DataFrame, out_path: Path):
    """
    Overlay P_stress with known major events and shock intensity series.
    """
    dates = pd.to_datetime(result["date"])

    # Load aggregate shocks for overlay
    shocks_path = GFJ_ROOT / "data" / "aggregate_shocks.csv"
    has_shocks  = shocks_path.exists()

    n_panels = 3 if has_shocks else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

    ax = axes[0]
    ax.plot(dates, result["P_stress"], color=STATE_COLORS[2], lw=0.9, label="P(stress)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    _shade_events(ax)
    for ev_label, ev_date in EVENT_LINES.items():
        ax.axvline(pd.Timestamp(ev_date), color="navy", lw=0.9, alpha=0.7)
    ax.set_ylabel("P(systemic stress)", fontsize=9)
    ax.set_title("Probability of systemic stress with known events overlaid", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(dates, result["EMFI"], color="#1f77b4", lw=0.7, label="EMFI")
    _shade_events(ax)
    ax.set_ylabel("EMFI", fontsize=9)
    ax.set_title("EMFI (reference)", fontsize=9)
    ax.legend(fontsize=7)

    if has_shocks:
        shocks = pd.read_csv(shocks_path, parse_dates=["date"])
        ax = axes[2]
        ax.bar(pd.to_datetime(shocks["date"]), shocks["max_shock"],
               color="#ff7f0e", width=1, alpha=0.7, label="MaxShock")
        _shade_events(ax)
        ax.set_ylabel("S = -log(q)", fontsize=9)
        ax.set_title("Daily maximum geopolitical shock intensity", fontsize=9)
        ax.legend(fontsize=7)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=0)
    plt.suptitle("HMM Validation: Stress Probability vs. Known Events", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== SP05 HMM Market-Implied Stress Regimes ===")
    t0 = datetime.now()

    if not IN_EMFI.exists():
        raise FileNotFoundError(str(IN_EMFI) + " not found. Run SP04 first.")

    log.info("Loading EMFI components from %s", IN_EMFI)
    emfi_df = pd.read_csv(IN_EMFI, parse_dates=["date"])
    log.info("  Shape: %s, dates: %s to %s",
             emfi_df.shape,
             emfi_df["date"].min().date(),
             emfi_df["date"].max().date())

    log.info("Fitting primary 3-state HMM...")
    result, model = build_hmm_regimes(emfi_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Writing hmm_daily.csv (%d rows)...", len(result))
    result.to_csv(OUT_CSV, index=False)

    # --- Compute additional diagnostics for manifest ---
    n_calm     = int((result["regime"] == STATE_CALM).sum())
    n_elevated = int((result["regime"] == STATE_ELEVATED).sum())
    n_stress   = int((result["regime"] == STATE_STRESS).sum())

    # COVID concentration in systemic stress state
    covid_mask   = (result["date"] >= "2020-02-01") & (result["date"] <= "2020-06-30")
    stress_mask  = result["regime"] == STATE_STRESS
    covid_stress_days = int((covid_mask & stress_mask).sum())
    covid_share  = covid_stress_days / max(n_stress, 1)

    log.info("  Regime counts -- Calm: %d, Elevated: %d, Systemic: %d",
             n_calm, n_elevated, n_stress)
    log.info("  COVID share of systemic-stress days: %.1f%% (%d/%d)",
             100 * covid_share, covid_stress_days, n_stress)

    # Transition matrix
    transmat_list = model.transmat_.tolist()
    means_list    = model.means_.tolist()
    covars_list   = model.covars_.tolist()
    startprob_list = model.startprob_.tolist()

    log.info("  Transition matrix (rows=from, cols=to):")
    for i, row in enumerate(transmat_list):
        log.info("    State %d: %s", i, ["%.3f" % v for v in row])

    # State persistence (diagonal of transition matrix)
    persistence = [transmat_list[s][s] for s in range(N_STATES)]
    log.info("  State persistence (diagonal): calm=%.3f, elevated=%.3f, stress=%.3f",
             persistence[0], persistence[1], persistence[2])

    # Mean P_stress on known event dates
    event_dates = {
        "covid_peak": "2020-03-12",
        "ukraine":    "2022-02-24",
        "hamas":      "2023-10-07",
    }
    event_pstress = {}
    for ev, d in event_dates.items():
        ts = pd.Timestamp(d)
        row = result[result["date"] == ts]
        if not row.empty:
            event_pstress[ev] = float(row["P_stress"].iloc[0])
        else:
            event_pstress[ev] = None

    log.info("  P_stress on key event dates: %s", event_pstress)

    # --- Save HMM params ---
    hmm_params = {
        "n_states":      N_STATES,
        "state_labels":  {str(STATE_CALM): "calm", str(STATE_ELEVATED): "elevated",
                          str(STATE_STRESS): "systemic_stress"},
        "startprob":     startprob_list,
        "transmat":      transmat_list,
        "means":         means_list,
        "covars":        covars_list,
        "component_cols": COMPONENT_COLS,
        "converged":     bool(model.monitor_.converged),
        "n_restarts":    N_RESTARTS,
    }
    OUT_PARAMS.write_text(json.dumps(hmm_params, indent=2), encoding="utf-8")
    log.info("HMM params saved to %s", OUT_PARAMS)

    # --- Figures ---
    log.info("Plotting HMM regime timeline...")
    plot_hmm_regimes(result, OUT_FIG_REGIMES)

    log.info("Plotting HMM validation figure...")
    plot_hmm_validation(result, OUT_FIG_VALIDATION)

    # --- Manifest ---
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script":       str(THIS_FILE.relative_to(GFJ_ROOT)),
        "n_dates":      len(result),
        "date_range": {
            "start": str(pd.to_datetime(result["date"]).min().date()),
            "end":   str(pd.to_datetime(result["date"]).max().date()),
        },
        "n_states":       N_STATES,
        "n_restarts":     N_RESTARTS,
        "component_cols": COMPONENT_COLS,
        "state_counts": {
            "calm":     n_calm,
            "elevated": n_elevated,
            "stress":   n_stress,
        },
        "covid_share_of_stress_days": round(covid_share, 4),
        "state_persistence": {
            "calm":     round(persistence[0], 4),
            "elevated": round(persistence[1], 4),
            "stress":   round(persistence[2], 4),
        },
        "event_pstress": event_pstress,
        "outputs": {
            "hmm_daily":    str(OUT_CSV),
            "hmm_params":   str(OUT_PARAMS),
            "fig_regimes":  str(OUT_FIG_REGIMES),
            "fig_validation": str(OUT_FIG_VALIDATION),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=== SP05 complete in %.1fs ===", elapsed)
    log.info("Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
