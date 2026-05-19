"""
SP10 -- COVID Reclassification: Non-Geopolitical Baseline LP
=============================================================
COVID-19 (2020) is a systemic financial stress event but NOT a geopolitical
conflict shock. This script reclassifies all 2020 shock observations by
zeroing out max_shock on COVID-period days (2020-01-01 to 2020-12-31).

The outcomes (EMFI, TCI, P_stress, etc.) remain in the sample — COVID is
retained as a validation benchmark for EMFI/HMM. Only the LP treatment is
modified: S_t is set to 0 for 2020 days, so COVID days do not enter as
treated observations in the causal regressions.

Results saved separately from the original baseline; the COVID-excluded
results become the new PRIMARY baseline reported in the paper.

Inputs (relative to Paper_GFJ root):
  results/emfi/emfi_daily.csv
  results/stress_regimes/hmm_daily.csv
  results/connectedness/tci_daily.csv
  results/fragility/fragility_daily.csv
  data/aggregate_shocks.csv
  data/panel_daily.parquet
  results/event_classification/event_taxonomy.csv

Outputs (relative to Paper_GFJ root):
  results/panel_lp_nocovid/lp_results_{outcome}.csv
  results/panel_lp_nocovid/Fig_LP_Combined_NoCovid.png
  results/panel_lp_nocovid/manifest.json
  results/state_lp_nocovid/state_lp_smooth_{outcome}.csv
  results/state_lp_nocovid/state_lp_binary_{outcome}.csv
  results/state_lp_nocovid/Fig_StateLPSmooth_NoCovid.png
  results/state_lp_nocovid/manifest.json
  results/covid_reclassify/event_taxonomy_nocovid.csv
  results/covid_reclassify/comparison_table.csv
  results/covid_reclassify/manifest.json

Run from Paper_GFJ root:
  python subprojects/10_covid_reclassify/run_covid_reclassify.py
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
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_EMFI   = GFJ_ROOT / "results" / "emfi"            / "emfi_daily.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes"  / "hmm_daily.csv"
IN_TCI    = GFJ_ROOT / "results" / "connectedness"   / "tci_daily.csv"
IN_FRAG   = GFJ_ROOT / "results" / "fragility"       / "fragility_daily.csv"
IN_SHOCKS = GFJ_ROOT / "data"    / "aggregate_shocks.csv"
IN_PANEL  = GFJ_ROOT / "data"    / "panel_daily.parquet"
IN_TAXON  = GFJ_ROOT / "results" / "event_classification" / "event_taxonomy.csv"

OUT_LP    = GFJ_ROOT / "results" / "panel_lp_nocovid"
OUT_SLP   = GFJ_ROOT / "results" / "state_lp_nocovid"
OUT_COV   = GFJ_ROOT / "results" / "covid_reclassify"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
COVID_START = "2020-01-01"
COVID_END   = "2020-12-31"

HORIZONS      = list(range(-5, 16))
OUTCOMES      = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
PRIMARY_SHOCK = "max_shock_nocovid"
CI_90, CI_95  = 1.645, 1.960

HF_PCTILE = 0.75       # HighFragility threshold percentile
P_EVAL    = [0.0, 0.5, 0.9]  # smooth-transition evaluation points

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
# Data loading
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Build merged daily dataset with COVID-reclassified shock series."""
    emfi  = pd.read_csv(IN_EMFI,   parse_dates=["date"])[["date", "EMFI"]]
    hmm   = pd.read_csv(IN_HMM,    parse_dates=["date"])[["date", "P_stress"]]
    tci   = pd.read_csv(IN_TCI,    parse_dates=["date"])[["date", "tci_w100"]]
    frag  = pd.read_csv(IN_FRAG,   parse_dates=["date"])[["date", "TailVolBreadth", "AvgCorr60"]]
    shocks= pd.read_csv(IN_SHOCKS, parse_dates=["date"])

    df = (emfi
          .merge(hmm,   on="date")
          .merge(tci,   on="date")
          .merge(frag,  on="date")
          .merge(shocks[["date", "max_shock"]], on="date", how="left"))

    df["max_shock"] = df["max_shock"].fillna(0)

    # COVID reclassification: zero out shock for 2020 days
    covid_mask = (df["date"] >= COVID_START) & (df["date"] <= COVID_END)
    df["max_shock_nocovid"] = df["max_shock"].copy()
    df.loc[covid_mask, "max_shock_nocovid"] = 0.0

    n_covid_zeroed = ((df["max_shock"] > 0) & covid_mask).sum()
    n_remaining    = (df["max_shock_nocovid"] > 0).sum()
    log.info("COVID reclassification: %d shock days zeroed (2020)", n_covid_zeroed)
    log.info("Remaining shock days: %d (was %d)", n_remaining, (df["max_shock"] > 0).sum())

    # Global return control
    panel = pd.read_parquet(IN_PANEL)
    ret_wide = panel.pivot(index="date", columns="country", values="return")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    global_ret = ret_wide.mean(axis=1).rename("global_ret").reset_index()
    global_ret.columns = ["date", "global_ret"]
    df = df.merge(global_ret, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month

    # HighFragility binary indicator (75th pctile of EMFI, full sample)
    hf_threshold = df["EMFI"].quantile(HF_PCTILE)
    df["HF"] = (df["EMFI"] > hf_threshold).astype(float)
    log.info("HF threshold (p75 EMFI): %.4f", hf_threshold)

    log.info("Dataset: %d rows, %s to %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df, hf_threshold


# ---------------------------------------------------------------------------
# LP estimation (panel LP)
# ---------------------------------------------------------------------------

def run_lp(df: pd.DataFrame, outcome: str, shock_col: str,
           horizons: list[int]) -> pd.DataFrame:
    """Run LP for one outcome across all horizons. Returns result DataFrame."""
    rows = []
    for k in horizons:
        y_fwd   = df[outcome].shift(-k)
        lag_shift = 2 if k == -1 else 1
        y_lag   = df[outcome].shift(lag_shift)
        shock_s = df[shock_col]
        gret    = df["global_ret"]
        month_dummies = pd.get_dummies(df["month"], prefix="m",
                                       drop_first=True).astype(float)

        X = pd.concat([shock_s.rename("shock"), y_lag.rename("y_lag"),
                       gret, month_dummies], axis=1)
        X = add_constant(X, prepend=True, has_constant="add")

        valid = y_fwd.notna() & X.notna().all(axis=1)
        y_est = y_fwd[valid].values
        X_est = X[valid].values

        if len(y_est) < 50:
            rows.append({"horizon": k, "beta": np.nan, "se": np.nan,
                         "tstat": np.nan, "pval": np.nan,
                         "ci90_lo": np.nan, "ci90_hi": np.nan,
                         "ci95_lo": np.nan, "ci95_hi": np.nan,
                         "nobs": len(y_est)})
            continue

        nw_lags = max(1, 2 * (abs(k) + 1))
        try:
            model = OLS(y_est, X_est).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": nw_lags, "use_correction": True})
        except Exception as exc:
            log.warning("k=%d OLS failed: %s", k, exc)
            rows.append({"horizon": k, "beta": np.nan, "se": np.nan,
                         "tstat": np.nan, "pval": np.nan,
                         "ci90_lo": np.nan, "ci90_hi": np.nan,
                         "ci95_lo": np.nan, "ci95_hi": np.nan,
                         "nobs": len(y_est)})
            continue

        beta, se = model.params[1], model.bse[1]
        rows.append({
            "horizon":  k,
            "beta":     beta,
            "se":       se,
            "tstat":    model.tvalues[1],
            "pval":     model.pvalues[1],
            "ci90_lo":  beta - CI_90 * se,
            "ci90_hi":  beta + CI_90 * se,
            "ci95_lo":  beta - CI_95 * se,
            "ci95_hi":  beta + CI_95 * se,
            "nobs":     int(model.nobs),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# State LP estimation (smooth-transition + binary HF)
# ---------------------------------------------------------------------------

def run_state_lp_smooth(df: pd.DataFrame, outcome: str,
                         shock_col: str, horizons: list[int]) -> pd.DataFrame:
    """
    Smooth-transition LP:
      Y_{t+k} = α + β S_t + θ (S_t × P_{t-1}) + φ P_{t-1}
               + γ Y_{t-1} + δ r^global_t + μ_m + ε
    Returns rows with beta, theta, and IRF evaluated at p ∈ P_EVAL.
    """
    rows = []
    for k in horizons:
        y_fwd   = df[outcome].shift(-k)
        lag_shift = 2 if k == -1 else 1
        y_lag   = df[outcome].shift(lag_shift)
        s_t     = df[shock_col]
        p_lag   = df["P_stress"].shift(1)
        gret    = df["global_ret"]
        interact = s_t * p_lag
        month_dummies = pd.get_dummies(df["month"], prefix="m",
                                       drop_first=True).astype(float)

        X = pd.concat([
            s_t.rename("shock"),
            interact.rename("shock_x_p"),
            p_lag.rename("p_lag"),
            y_lag.rename("y_lag"),
            gret,
            month_dummies,
        ], axis=1)
        X = add_constant(X, prepend=True, has_constant="add")

        valid = y_fwd.notna() & X.notna().all(axis=1)
        y_est = y_fwd[valid].values
        X_est = X[valid].values

        if len(y_est) < 50:
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         **{f"irf_p{int(p*10)}": np.nan for p in P_EVAL},
                         "nobs": len(y_est)})
            continue

        nw_lags = max(1, 2 * (abs(k) + 1))
        try:
            model = OLS(y_est, X_est).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": nw_lags, "use_correction": True})
        except Exception as exc:
            log.warning("state_lp smooth k=%d failed: %s", k, exc)
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         **{f"irf_p{int(p*10)}": np.nan for p in P_EVAL},
                         "nobs": len(y_est)})
            continue

        beta  = model.params[1]   # shock coefficient
        theta = model.params[2]   # interaction coefficient
        vcov  = model.cov_params()

        row = {"horizon": k, "beta": beta, "theta": theta,
               "nobs": int(model.nobs)}

        # IRF at evaluation points with delta-method SE
        for p in P_EVAL:
            irf   = beta + p * theta
            se_irf = np.sqrt(vcov[1, 1] + p**2 * vcov[2, 2] + 2*p*vcov[1, 2])
            key   = f"irf_p{int(p*10)}"
            row[key]            = irf
            row[f"se_{key}"]    = se_irf
            row[f"ci95lo_{key}"] = irf - CI_95 * se_irf
            row[f"ci95hi_{key}"] = irf + CI_95 * se_irf

        rows.append(row)
    return pd.DataFrame(rows)


def run_state_lp_binary(df: pd.DataFrame, outcome: str,
                         shock_col: str, horizons: list[int]) -> pd.DataFrame:
    """
    Binary HighFragility LP:
      Y_{t+k} = α + β S_t + θ (S_t × HF_{t-1}) + φ HF_{t-1}
               + γ Y_{t-1} + δ r^global_t + μ_m + ε
    Returns rows with beta, theta, irf_normal (β), irf_fragile (β+θ).
    """
    rows = []
    for k in horizons:
        y_fwd   = df[outcome].shift(-k)
        lag_shift = 2 if k == -1 else 1
        y_lag   = df[outcome].shift(lag_shift)
        s_t     = df[shock_col]
        hf_lag  = df["HF"].shift(1)
        gret    = df["global_ret"]
        interact = s_t * hf_lag
        month_dummies = pd.get_dummies(df["month"], prefix="m",
                                       drop_first=True).astype(float)

        X = pd.concat([
            s_t.rename("shock"),
            interact.rename("shock_x_hf"),
            hf_lag.rename("hf_lag"),
            y_lag.rename("y_lag"),
            gret,
            month_dummies,
        ], axis=1)
        X = add_constant(X, prepend=True, has_constant="add")

        valid = y_fwd.notna() & X.notna().all(axis=1)
        y_est = y_fwd[valid].values
        X_est = X[valid].values

        if len(y_est) < 50:
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         "irf_normal": np.nan, "irf_fragile": np.nan,
                         "nobs": len(y_est)})
            continue

        nw_lags = max(1, 2 * (abs(k) + 1))
        try:
            model = OLS(y_est, X_est).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": nw_lags, "use_correction": True})
        except Exception as exc:
            log.warning("state_lp binary k=%d failed: %s", k, exc)
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         "irf_normal": np.nan, "irf_fragile": np.nan,
                         "nobs": len(y_est)})
            continue

        beta  = model.params[1]
        theta = model.params[2]
        rows.append({
            "horizon":    k,
            "beta":       beta,
            "theta":      theta,
            "irf_normal": beta,
            "irf_fragile": beta + theta,
            "se_beta":    model.bse[1],
            "se_theta":   model.bse[2],
            "pval_beta":  model.pvalues[1],
            "pval_theta": model.pvalues[2],
            "nobs":       int(model.nobs),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

OUTCOME_LABELS = {
    "EMFI":           "EMFI",
    "tci_w100":       "TCI (W=100)",
    "TailVolBreadth": "Tail Breadth",
    "P_stress":       "P(systemic stress)",
    "AvgCorr60":      "Avg. Pairwise Corr.",
}


def plot_lp_combined(all_results: dict[str, pd.DataFrame], out_path: Path,
                     title_suffix: str = ""):
    n = len(OUTCOMES)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for ax, outcome in zip(axes, OUTCOMES):
        res  = all_results[outcome]
        hor  = res["horizon"].values
        pre  = res[hor < 0]
        post = res[hor >= 0]

        if len(pre):
            ax.plot(pre["horizon"], pre["beta"], color="grey", lw=0.9, ls="--")
            ax.fill_between(pre["horizon"], pre["ci95_lo"], pre["ci95_hi"],
                            color="grey", alpha=0.10)

        ax.plot(post["horizon"], post["beta"], color="#1f77b4", lw=1.4,
                marker="o", ms=3)
        ax.fill_between(post["horizon"], post["ci95_lo"], post["ci95_hi"],
                        color="#1f77b4", alpha=0.15)
        ax.fill_between(post["horizon"], post["ci90_lo"], post["ci90_hi"],
                        color="#1f77b4", alpha=0.20)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        k0_beta = post.loc[post["horizon"] == 0, "beta"].values
        if len(k0_beta):
            ax.set_title(f"{OUTCOME_LABELS.get(outcome, outcome)}"
                         f"  β₀={k0_beta[0]:.3f}", fontsize=8)

    axes[-1].set_xlabel("Horizon k (trading days)", fontsize=9)
    axes[-1].set_xticks(range(-5, 16, 5))
    fig.suptitle(f"LP Impulse Responses — COVID-Excluded Shock{title_suffix}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("LP combined figure saved: %s", out_path)


def plot_state_lp_smooth(all_smooth: dict[str, pd.DataFrame], out_path: Path):
    n = len(OUTCOMES)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    colors = {0.0: "#2ca02c", 0.5: "#ff7f0e", 0.9: "#d62728"}
    labels = {0.0: "p=0.0 (calm)", 0.5: "p=0.5", 0.9: "p=0.9 (stressed)"}

    for ax, outcome in zip(axes, OUTCOMES):
        res  = all_smooth[outcome]
        post = res[res["horizon"] >= 0]

        for p in P_EVAL:
            key = f"irf_p{int(p*10)}"
            lo  = f"ci95lo_{key}"
            hi  = f"ci95hi_{key}"
            if key not in post.columns:
                continue
            ax.plot(post["horizon"], post[key], color=colors[p], lw=1.4,
                    marker="o", ms=2.5, label=labels[p])
            ax.fill_between(post["horizon"], post[lo], post[hi],
                            color=colors[p], alpha=0.10)

        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if outcome == OUTCOMES[0]:
            ax.legend(fontsize=7, ncol=3, loc="upper right")

    axes[-1].set_xlabel("Horizon k (trading days)", fontsize=9)
    axes[-1].set_xticks(range(-5, 16, 5))
    fig.suptitle("State-Dependent LP (Smooth Transition) — COVID-Excluded",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("State LP figure saved: %s", out_path)


# ---------------------------------------------------------------------------
# Event taxonomy update
# ---------------------------------------------------------------------------

def reclassify_taxonomy(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Load event_taxonomy.csv and add covid_reclassified flag.
    COVID-period shock days are flagged; their category is preserved for
    EMFI validation context but a note column is added.
    """
    taxon = pd.read_csv(IN_TAXON, parse_dates=["date"])
    covid_dates = set(
        df_dataset.loc[
            (df_dataset["date"] >= COVID_START) &
            (df_dataset["date"] <= COVID_END) &
            (df_dataset["max_shock"] > 0), "date"
        ].dt.date
    )
    taxon["covid_reclassified"] = taxon["date"].dt.date.isin(covid_dates)
    taxon["lp_treatment"] = ~taxon["covid_reclassified"]
    n_covid = taxon["covid_reclassified"].sum()
    log.info("Event taxonomy: %d COVID-period events flagged (lp_treatment=False)",
             n_covid)
    return taxon


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison(baseline_lp: dict, nocovid_lp: dict,
                     baseline_slp: dict, nocovid_slp: dict) -> pd.DataFrame:
    """
    Compare key LP and state LP statistics between full baseline and
    COVID-excluded specification.
    """
    rows = []
    for outcome in OUTCOMES:
        b_k0  = baseline_lp[outcome].loc[baseline_lp[outcome]["horizon"] == 0, "beta"].iloc[0]
        nc_k0 = nocovid_lp[outcome].loc[nocovid_lp[outcome]["horizon"] == 0, "beta"].iloc[0]
        rows.append({
            "outcome": outcome,
            "metric":  "LP_beta_k0",
            "baseline": round(b_k0, 4),
            "nocovid":  round(nc_k0, 4),
            "pct_change": round(100 * (nc_k0 - b_k0) / abs(b_k0) if b_k0 != 0 else np.nan, 1),
        })

    # State LP theta_k0 for EMFI (smooth)
    for spec, b_res, nc_res in [("smooth", baseline_slp, nocovid_slp)]:
        b_row  = b_res["EMFI"].loc[b_res["EMFI"]["horizon"] == 0]
        nc_row = nc_res["EMFI"].loc[nc_res["EMFI"]["horizon"] == 0]
        if len(b_row) and len(nc_row):
            b_th  = b_row["theta"].iloc[0]
            nc_th = nc_row["theta"].iloc[0]
            rows.append({
                "outcome": "EMFI",
                "metric":  f"StateLPsmooth_theta_k0",
                "baseline": round(b_th, 4),
                "nocovid":  round(nc_th, 4),
                "pct_change": round(100 * (nc_th - b_th) / abs(b_th) if b_th != 0 else np.nan, 1),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== SP10: COVID Reclassification ===")
    for d in [OUT_LP, OUT_SLP, OUT_COV]:
        d.mkdir(parents=True, exist_ok=True)

    # Build dataset
    df, hf_threshold = build_dataset()

    # ------------------------------------------------------------------ #
    # Panel LP — COVID excluded                                           #
    # ------------------------------------------------------------------ #
    log.info("--- Panel LP (COVID-excluded shock) ---")
    nocovid_lp: dict[str, pd.DataFrame] = {}
    for outcome in OUTCOMES:
        log.info("  outcome: %s", outcome)
        res = run_lp(df, outcome, PRIMARY_SHOCK, HORIZONS)
        res.to_csv(OUT_LP / f"lp_results_{outcome}.csv", index=False)
        nocovid_lp[outcome] = res

    plot_lp_combined(nocovid_lp, OUT_LP / "Fig_LP_Combined_NoCovid.png")

    # LP summary for manifest
    lp_summary = []
    for outcome, res in nocovid_lp.items():
        post = res[res["horizon"] >= 0]
        k0   = post[post["horizon"] == 0].iloc[0]
        k1   = post[post["horizon"] == 1].iloc[0]
        k5   = post[post["horizon"] == 5].iloc[0]
        peak = post.loc[post["beta"].abs().idxmax()]
        lp_summary.append({
            "outcome":      outcome,
            "beta_k0":      round(k0["beta"], 6),
            "beta_k1":      round(k1["beta"], 6),
            "beta_k5":      round(k5["beta"], 6),
            "peak_horizon": int(peak["horizon"]),
            "peak_beta":    round(peak["beta"], 6),
            "peak_pval":    round(peak["pval"], 6),
        })

    # ------------------------------------------------------------------ #
    # Load original baseline LP results for comparison                   #
    # ------------------------------------------------------------------ #
    baseline_lp: dict[str, pd.DataFrame] = {}
    orig_lp_dir = GFJ_ROOT / "results" / "panel_lp"
    for outcome in OUTCOMES:
        p = orig_lp_dir / f"lp_results_{outcome}.csv"
        if p.exists():
            baseline_lp[outcome] = pd.read_csv(p)

    # ------------------------------------------------------------------ #
    # State LP — COVID excluded                                           #
    # ------------------------------------------------------------------ #
    log.info("--- State LP (COVID-excluded shock) ---")
    nocovid_smooth: dict[str, pd.DataFrame] = {}
    nocovid_binary: dict[str, pd.DataFrame] = {}

    for outcome in OUTCOMES:
        log.info("  outcome: %s (smooth)", outcome)
        sm = run_state_lp_smooth(df, outcome, PRIMARY_SHOCK, HORIZONS)
        sm.to_csv(OUT_SLP / f"state_lp_smooth_{outcome}.csv", index=False)
        nocovid_smooth[outcome] = sm

        log.info("  outcome: %s (binary HF)", outcome)
        bi = run_state_lp_binary(df, outcome, PRIMARY_SHOCK, HORIZONS)
        bi.to_csv(OUT_SLP / f"state_lp_binary_{outcome}.csv", index=False)
        nocovid_binary[outcome] = bi

    plot_state_lp_smooth(nocovid_smooth,
                         OUT_SLP / "Fig_StateLPSmooth_NoCovid.png")

    # State LP summary
    slp_summary = []
    for outcome in OUTCOMES:
        sm_row = nocovid_smooth[outcome][nocovid_smooth[outcome]["horizon"] == 0]
        bi_row = nocovid_binary[outcome][nocovid_binary[outcome]["horizon"] == 0]
        if len(sm_row) and len(bi_row):
            slp_summary.append({
                "outcome":         outcome,
                "theta_k0_smooth": round(sm_row["theta"].iloc[0], 6),
                "theta_k1_smooth": round(
                    nocovid_smooth[outcome].loc[
                        nocovid_smooth[outcome]["horizon"] == 1, "theta"
                    ].iloc[0], 6),
                "theta_k0_binary": round(bi_row["theta"].iloc[0], 6),
            })

    # Load original baseline state LP for comparison
    baseline_slp: dict[str, pd.DataFrame] = {}
    orig_slp_dir = GFJ_ROOT / "results" / "state_lp"
    for outcome in OUTCOMES:
        p = orig_slp_dir / f"state_lp_smooth_{outcome}.csv"
        if p.exists():
            baseline_slp[outcome] = pd.read_csv(p)

    # ------------------------------------------------------------------ #
    # Event taxonomy reclassification                                     #
    # ------------------------------------------------------------------ #
    log.info("--- Event taxonomy reclassification ---")
    taxon_new = reclassify_taxonomy(df)
    taxon_new.to_csv(OUT_COV / "event_taxonomy_nocovid.csv", index=False)

    # ------------------------------------------------------------------ #
    # Comparison table                                                    #
    # ------------------------------------------------------------------ #
    if baseline_lp and baseline_slp:
        comp = build_comparison(baseline_lp, nocovid_lp,
                                baseline_slp, nocovid_smooth)
        comp.to_csv(OUT_COV / "comparison_table.csv", index=False)
        log.info("Comparison table saved")
        log.info("\n%s", comp.to_string(index=False))
    else:
        log.warning("Baseline LP results not found — skipping comparison")

    # ------------------------------------------------------------------ #
    # LP summary CSV                                                      #
    # ------------------------------------------------------------------ #
    pd.DataFrame(lp_summary).to_csv(OUT_LP / "lp_summary_nocovid.csv", index=False)

    # ------------------------------------------------------------------ #
    # Manifests                                                           #
    # ------------------------------------------------------------------ #
    now = datetime.now(timezone.utc).isoformat()
    n_covid_shock_days = int(((df["max_shock"] > 0) &
                              (df["date"] >= COVID_START) &
                              (df["date"] <= COVID_END)).sum())
    n_remaining_shock_days = int((df["max_shock_nocovid"] > 0).sum())

    manifest_lp = {
        "generated_at": now,
        "script": "subprojects/10_covid_reclassify/run_covid_reclassify.py",
        "covid_exclusion_window": {"start": COVID_START, "end": COVID_END},
        "n_covid_shock_days_zeroed": n_covid_shock_days,
        "n_remaining_shock_days": n_remaining_shock_days,
        "n_obs": int(df.shape[0]),
        "date_range": {"start": str(df["date"].min().date()),
                       "end": str(df["date"].max().date())},
        "lp_summary": lp_summary,
    }
    with open(OUT_LP / "manifest.json", "w") as f:
        json.dump(manifest_lp, f, indent=2)

    manifest_slp = {
        "generated_at": now,
        "script": "subprojects/10_covid_reclassify/run_covid_reclassify.py",
        "covid_exclusion_window": {"start": COVID_START, "end": COVID_END},
        "hf_threshold_emfi": round(hf_threshold, 4),
        "hf_pctile": HF_PCTILE,
        "n_remaining_shock_days": n_remaining_shock_days,
        "slp_summary": slp_summary,
    }
    with open(OUT_SLP / "manifest.json", "w") as f:
        json.dump(manifest_slp, f, indent=2)

    manifest_cov = {
        "generated_at": now,
        "script": "subprojects/10_covid_reclassify/run_covid_reclassify.py",
        "covid_window": {"start": COVID_START, "end": COVID_END},
        "n_covid_shock_days_reclassified": n_covid_shock_days,
        "n_lp_treatment_days": n_remaining_shock_days,
        "outputs": [
            "panel_lp_nocovid/",
            "state_lp_nocovid/",
            "covid_reclassify/event_taxonomy_nocovid.csv",
            "covid_reclassify/comparison_table.csv",
        ],
    }
    with open(OUT_COV / "manifest.json", "w") as f:
        json.dump(manifest_cov, f, indent=2)

    log.info("=== SP10 complete ===")
    log.info("Panel LP (no-COVID): results/panel_lp_nocovid/")
    log.info("State LP (no-COVID): results/state_lp_nocovid/")
    log.info("COVID reclassify:    results/covid_reclassify/")


if __name__ == "__main__":
    main()
