"""
SP06 -- Panel Local Projections
================================
Estimates Jordà (2005) local projections to measure the dynamic causal effect
of geopolitical shocks on five systemic-fragility outcomes.

Specification (time-series LP, aggregate daily outcomes):
    Y_{t+k} = α + β_k S_t + γ Y_{t-1} + δ r^global_t + μ_m + ε_{t+k}

  for k ∈ {-5, ..., +15} (pre-period k<0 used for pre-trend verification)

Shock: S_t = max_shock (daily maximum EVT+FDR shock intensity across 19 countries)
Controls: Y_{t-1} (lagged outcome), r^global_t (equal-weight daily return), μ_m (month FE)
Inference: Newey-West HAC, bandwidth = 2*(|k|+1) at each horizon k

Outcomes (five separate regressions):
  1. EMFI          -- primary composite fragility index
  2. tci_w100      -- total volatility connectedness (W=100)
  3. TailVolBreadth -- count of markets in |r| tail
  4. P_stress      -- HMM posterior probability of systemic-stress state
  5. AvgCorr60     -- rolling 60-day average pairwise correlation

Alternative shocks: AvgShock (mean across countries), BreadthShock (count of countries)

Inputs (relative to Paper_GFJ root):
  results/emfi/emfi_daily.csv
  results/stress_regimes/hmm_daily.csv
  results/connectedness/tci_daily.csv
  results/fragility/fragility_daily.csv
  data/aggregate_shocks.csv
  data/panel_daily.parquet

Outputs (relative to Paper_GFJ root):
  results/panel_lp/lp_results_{outcome}.csv     -- β_k, SE, CIs for each horizon
  results/panel_lp/Fig_LP_{outcome}.png         -- individual IRF figure
  results/panel_lp/Fig_LP_Combined.png          -- paper-ready multi-panel figure
  results/panel_lp/manifest.json

Run from Paper_GFJ root:
  python subprojects/06_panel_lp/run_panel_lp.py
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
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_EMFI    = GFJ_ROOT / "results" / "emfi"          / "emfi_daily.csv"
IN_HMM     = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"
IN_TCI     = GFJ_ROOT / "results" / "connectedness"  / "tci_daily.csv"
IN_FRAG    = GFJ_ROOT / "results" / "fragility"      / "fragility_daily.csv"
IN_SHOCKS  = GFJ_ROOT / "data"    / "aggregate_shocks.csv"
IN_PANEL   = GFJ_ROOT / "data"    / "panel_daily.parquet"

OUT_DIR    = GFJ_ROOT / "results" / "panel_lp"
OUT_COMBINED = OUT_DIR / "Fig_LP_Combined.png"
OUT_MANIFEST = OUT_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZONS      = list(range(-5, 16))    # k = -5 to +15 inclusive
PRE_HORIZONS  = [k for k in HORIZONS if k < 0]
POST_HORIZONS = [k for k in HORIZONS if k >= 0]

OUTCOMES = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
PRIMARY_SHOCK  = "max_shock"
ALT_SHOCKS     = ["avg_shock_pos", "breadth_shock"]

CI_90 = 1.645
CI_95 = 1.960

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
# Data construction
# ---------------------------------------------------------------------------

def build_lp_dataset() -> pd.DataFrame:
    """
    Merge all inputs into a single daily DataFrame for LP estimation.
    Returns df with columns: date, EMFI, tci_w100, TailVolBreadth, P_stress,
    AvgCorr60, max_shock, avg_shock_pos, breadth_shock, global_ret, month.
    """
    emfi  = pd.read_csv(IN_EMFI,   parse_dates=["date"])[["date", "EMFI"]]
    hmm   = pd.read_csv(IN_HMM,    parse_dates=["date"])[["date", "P_stress"]]
    tci   = pd.read_csv(IN_TCI,    parse_dates=["date"])[["date", "tci_w100"]]
    frag  = pd.read_csv(IN_FRAG,   parse_dates=["date"])[["date", "TailVolBreadth", "AvgCorr60"]]
    shocks= pd.read_csv(IN_SHOCKS, parse_dates=["date"])

    df = emfi.merge(hmm,  on="date") \
             .merge(tci,  on="date") \
             .merge(frag, on="date") \
             .merge(shocks[["date", "max_shock", "avg_shock_pos", "breadth_shock"]],
                    on="date", how="left")
    df["max_shock"]    = df["max_shock"].fillna(0)
    df["avg_shock_pos"]= df["avg_shock_pos"].fillna(0)
    df["breadth_shock"]= df["breadth_shock"].fillna(0)

    # Equal-weighted global return
    panel   = pd.read_parquet(IN_PANEL)
    ret_wide = panel.pivot(index="date", columns="country", values="return")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    global_ret = ret_wide.mean(axis=1).rename("global_ret").reset_index()
    global_ret.columns = ["date", "global_ret"]
    df = df.merge(global_ret, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month

    log.info("  LP dataset: %d rows, %d cols", len(df), len(df.columns))
    log.info("  Date range: %s to %s", df["date"].min().date(), df["date"].max().date())
    log.info("  Shock days: %d", (df["max_shock"] > 0).sum())
    return df


# ---------------------------------------------------------------------------
# LP estimation
# ---------------------------------------------------------------------------

def run_lp_one_outcome(
    df: pd.DataFrame,
    outcome: str,
    shock: str,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Run LP for a single outcome variable across all horizons.
    Returns DataFrame with columns: horizon, beta, se, tstat, pval,
    ci90_lo, ci90_hi, ci95_lo, ci95_hi, nobs.
    """
    rows = []

    for k in horizons:
        # Dependent variable: Y_{t+k}
        y_fwd = df[outcome].shift(-k)

        # Controls
        # For k=-1, LHS = Y_{t-1} = y_lag -- degenerate. Use Y_{t-2} as control.
        lag_shift = 2 if k == -1 else 1
        y_lag   = df[outcome].shift(lag_shift)       # Y_{t-1} (or Y_{t-2} when k=-1)
        shock_s = df[shock]                  # S_t
        gret    = df["global_ret"]           # r^global_t
        month   = df["month"]

        # Month dummies (drop January = 1)
        month_dummies = pd.get_dummies(month, prefix="m", drop_first=True).astype(float)

        # Assemble regressor matrix
        X = pd.concat([
            shock_s.rename(shock),
            y_lag.rename("y_lag"),
            gret,
            month_dummies,
        ], axis=1)
        X = add_constant(X, prepend=True, has_constant="add")

        # Drop rows with any NaN
        valid = y_fwd.notna() & X.notna().all(axis=1)
        y_est = y_fwd[valid].values
        X_est = X[valid].values

        if len(y_est) < 50:
            log.warning("    k=%d: only %d valid obs, skipping", k, len(y_est))
            rows.append({
                "horizon": k, "beta": np.nan, "se": np.nan,
                "tstat": np.nan, "pval": np.nan,
                "ci90_lo": np.nan, "ci90_hi": np.nan,
                "ci95_lo": np.nan, "ci95_hi": np.nan,
                "nobs": len(y_est),
            })
            continue

        # Newey-West bandwidth = 2*(|k|+1)
        nw_lags = max(1, 2 * (abs(k) + 1))

        try:
            model = OLS(y_est, X_est).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": nw_lags, "use_correction": True},
            )
        except Exception as exc:
            log.warning("    k=%d OLS failed: %s", k, exc)
            rows.append({"horizon": k, "beta": np.nan, "se": np.nan,
                         "tstat": np.nan, "pval": np.nan,
                         "ci90_lo": np.nan, "ci90_hi": np.nan,
                         "ci95_lo": np.nan, "ci95_hi": np.nan,
                         "nobs": len(y_est)})
            continue

        # shock is always the first non-constant regressor (column index 1)
        beta = model.params[1]
        se   = model.bse[1]
        tstat= model.tvalues[1]
        pval = model.pvalues[1]

        rows.append({
            "horizon":  k,
            "beta":     beta,
            "se":       se,
            "tstat":    tstat,
            "pval":     pval,
            "ci90_lo":  beta - CI_90 * se,
            "ci90_hi":  beta + CI_90 * se,
            "ci95_lo":  beta - CI_95 * se,
            "ci95_hi":  beta + CI_95 * se,
            "nobs":     int(model.nobs),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

OUTCOME_LABELS = {
    "EMFI":           "EMFI",
    "tci_w100":       "TCI (W=100, %)",
    "TailVolBreadth": "Tail Breadth (count)",
    "P_stress":       "P(systemic stress)",
    "AvgCorr60":      "Avg. Pairwise Corr.",
}


def plot_irf(results: pd.DataFrame, outcome: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))

    hor = results["horizon"].values
    pre_mask  = hor < 0
    post_mask = hor >= 0

    # Pre-period (dashed, grey)
    if pre_mask.any():
        pre = results[pre_mask]
        ax.plot(pre["horizon"], pre["beta"], color="grey", lw=1.0,
                ls="--", label="Pre-period (pre-trend check)")
        ax.fill_between(pre["horizon"], pre["ci95_lo"], pre["ci95_hi"],
                        color="grey", alpha=0.12)

    # Post-period (solid)
    post = results[post_mask]
    ax.plot(post["horizon"], post["beta"], color="#1f77b4", lw=1.5,
            marker="o", ms=3.5, label="Response (95% CI)")
    ax.fill_between(post["horizon"], post["ci95_lo"], post["ci95_hi"],
                    color="#1f77b4", alpha=0.15, label="_nolegend_")
    ax.fill_between(post["horizon"], post["ci90_lo"], post["ci90_hi"],
                    color="#1f77b4", alpha=0.20, label="_nolegend_")

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.axvline(0, color="black", lw=0.6, ls=":")

    ax.set_xlabel("Horizon k (trading days)", fontsize=9)
    ax.set_ylabel("Response of " + OUTCOME_LABELS.get(outcome, outcome), fontsize=9)
    ax.set_title("LP impulse response: geopolitical shock → " +
                 OUTCOME_LABELS.get(outcome, outcome), fontsize=9)
    ax.set_xticks(list(range(-5, 16, 5)))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


def plot_combined(all_results: dict[str, pd.DataFrame], out_path: Path):
    """5-panel figure — one row per outcome variable."""
    fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(12, 3.2 * len(OUTCOMES)),
                             sharex=True)

    for ax, outcome in zip(axes, OUTCOMES):
        results = all_results[outcome]
        hor = results["horizon"].values
        pre_mask  = hor < 0
        post_mask = hor >= 0

        if pre_mask.any():
            pre = results[pre_mask]
            ax.plot(pre["horizon"], pre["beta"], color="grey", lw=0.9, ls="--")
            ax.fill_between(pre["horizon"], pre["ci95_lo"], pre["ci95_hi"],
                            color="grey", alpha=0.10)

        post = results[post_mask]
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

        # Annotate peak β
        if post["beta"].notna().any():
            peak_k  = post.loc[post["beta"].abs().idxmax(), "horizon"]
            peak_b  = post.loc[post["beta"].abs().idxmax(), "beta"]
            ax.annotate("peak k=" + str(int(peak_k)),
                        xy=(peak_k, peak_b),
                        xytext=(peak_k + 1.5, peak_b),
                        fontsize=6.5, color="#1f77b4",
                        arrowprops=dict(arrowstyle="-", lw=0.5))

    axes[-1].set_xlabel("Horizon k (trading days)", fontsize=9)
    axes[-1].set_xticks(list(range(-5, 16, 5)))
    plt.suptitle("LP Impulse Responses: Geopolitical Shock → Systemic Fragility Indicators",
                 fontsize=10, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Combined figure saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== SP06 Panel Local Projections ===")
    t0 = datetime.now()

    log.info("Building LP dataset...")
    df = build_lp_dataset()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for outcome in OUTCOMES:
        log.info("--- Outcome: %s ---", outcome)

        # Primary shock
        results = run_lp_one_outcome(df, outcome, PRIMARY_SHOCK, HORIZONS)

        out_csv = OUT_DIR / ("lp_results_" + outcome + ".csv")
        results.to_csv(out_csv, index=False)
        log.info("  Results written to %s", out_csv)

        all_results[outcome] = results

        # Report β at key horizons
        for h in [0, 1, 3, 5, 10, 15]:
            row = results[results["horizon"] == h]
            if not row.empty and row["beta"].notna().all():
                log.info("  k=%2d: β=%.4f  SE=%.4f  t=%.2f  p=%.3f",
                         h, row["beta"].iloc[0], row["se"].iloc[0],
                         row["tstat"].iloc[0], row["pval"].iloc[0])

        # Check pre-trends (k < 0)
        pre = results[results["horizon"] < 0]
        n_sig_pre = (pre["pval"].dropna() < 0.10).sum()
        log.info("  Pre-trend check: %d/%d pre-horizons significant at 10%%",
                 n_sig_pre, len(pre))

        # Summary row
        post_max = results[results["horizon"] > 0]
        if post_max["beta"].notna().any():
            peak_idx = post_max["beta"].abs().idxmax()
            summary_rows.append({
                "outcome":        outcome,
                "beta_k0":        results.loc[results["horizon"] == 0, "beta"].iloc[0],
                "beta_k1":        results.loc[results["horizon"] == 1, "beta"].iloc[0] if 1 in results["horizon"].values else np.nan,
                "beta_k5":        results.loc[results["horizon"] == 5, "beta"].iloc[0] if 5 in results["horizon"].values else np.nan,
                "peak_horizon":   int(post_max.loc[peak_idx, "horizon"]),
                "peak_beta":      post_max.loc[peak_idx, "beta"],
                "peak_pval":      post_max.loc[peak_idx, "pval"],
                "n_sig_pre":      int(n_sig_pre),
            })

        # Individual figure
        plot_irf(results, outcome, OUT_DIR / ("Fig_LP_" + outcome + ".png"))

    # Combined figure
    log.info("Plotting combined figure...")
    plot_combined(all_results, OUT_COMBINED)

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "lp_summary.csv", index=False)
    log.info("Summary written.")

    # Manifest
    manifest = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "script":        str(THIS_FILE.relative_to(GFJ_ROOT)),
        "n_obs":         len(df),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end":   str(df["date"].max().date()),
        },
        "outcomes":      OUTCOMES,
        "shock_primary": PRIMARY_SHOCK,
        "horizons":      HORIZONS,
        "nw_bandwidth":  "2*(|k|+1)",
        "controls":      ["y_lag", "global_ret", "month_dummies"],
        "summary":       summary_df.to_dict(orient="records"),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=== SP06 complete in %.1fs ===", elapsed)
    log.info("Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
