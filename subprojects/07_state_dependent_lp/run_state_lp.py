"""
SP07 -- State-Dependent Local Projections
==========================================
Tests whether geopolitical shocks are more destabilising when they strike
already-fragile markets.  Two complementary specifications:

PRIMARY — Smooth-Transition LP (P_stress continuous interaction):
    Y_{t+k} = α + β_k S_t + θ_k (S_t × P_{t-1}) + φ_k P_{t-1}
             + γ_k Y_{t-1} + δ_k r^global_t + μ_m + ε_{t+k}

  where P_{t-1} = P_stress_{t-1} (lagged HMM posterior, continuous [0,1]).
  Evaluated at p ∈ {0.0, 0.5, 0.9} → three IRF lines.
  Standard errors propagated via the delta method:
    Var(β_k + p·θ_k) = Var(β_k) + p²Var(θ_k) + 2p·Cov(β_k, θ_k).

SECONDARY — Binary HighFragility LP:
    Y_{t+k} = α + β_k S_t + θ_k (S_t × HF_{t-1}) + φ_k HF_{t-1}
             + γ_k Y_{t-1} + δ_k r^global_t + μ_m + ε_{t+k}

  where HF_{t-1} = 1[EMFI_{t-1} > 75th pctile of full-sample EMFI].
  Two IRF lines: normal (β_k) and fragile (β_k + θ_k).

Primary outcome: EMFI. All five outcomes estimated for completeness.
Additional specification: COVID exclusion (drop 2020-03-01 to 2020-12-31).

Inputs (relative to Paper_GFJ root):
  results/emfi/emfi_daily.csv
  results/stress_regimes/hmm_daily.csv
  results/connectedness/tci_daily.csv
  results/fragility/fragility_daily.csv
  data/aggregate_shocks.csv
  data/panel_daily.parquet

Outputs (relative to Paper_GFJ root):
  results/state_lp/state_lp_smooth_{outcome}.csv  -- smooth-transition results
  results/state_lp/state_lp_binary_{outcome}.csv  -- binary HF results
  results/state_lp/Fig_StateLPSmooth_{outcome}.png
  results/state_lp/Fig_StateLPBinary_{outcome}.png
  results/state_lp/Fig_StateLPSmooth_Combined.png  -- paper Figure 3
  results/state_lp/composition_report.csv
  results/state_lp/manifest.json

Run from Paper_GFJ root:
  python subprojects/07_state_dependent_lp/run_state_lp.py
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

IN_EMFI   = GFJ_ROOT / "results" / "emfi"           / "emfi_daily.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"
IN_TCI    = GFJ_ROOT / "results" / "connectedness"  / "tci_daily.csv"
IN_FRAG   = GFJ_ROOT / "results" / "fragility"      / "fragility_daily.csv"
IN_SHOCKS = GFJ_ROOT / "data"    / "aggregate_shocks.csv"
IN_PANEL  = GFJ_ROOT / "data"    / "panel_daily.parquet"

OUT_DIR   = GFJ_ROOT / "results" / "state_lp"
OUT_MANIFEST = OUT_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZONS      = list(range(-5, 16))
POST_HORIZONS = [k for k in HORIZONS if k >= 0]
PRE_HORIZONS  = [k for k in HORIZONS if k < 0]

OUTCOMES      = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
PRIMARY_SHOCK = "max_shock"

HF_PCTILE     = 0.75          # HighFragility threshold percentile
P_EVAL_POINTS = [0.0, 0.5, 0.9]  # smooth-transition evaluation points
P_LABELS      = {0.0: "Calm (p=0.0)", 0.5: "Elevated (p=0.5)", 0.9: "Near-systemic (p=0.9)"}
P_COLORS      = {0.0: "#2ca02c", 0.5: "#ff7f0e", 0.9: "#d62728"}

CI_90 = 1.645
CI_95 = 1.960

COVID_EXCL_START = "2020-03-01"
COVID_EXCL_END   = "2020-12-31"

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

def build_state_lp_dataset() -> tuple[pd.DataFrame, float]:
    """
    Build the state-LP dataset and return (df, emfi_75th_pctile).
    Columns: date, EMFI, tci_w100, TailVolBreadth, P_stress, AvgCorr60,
             max_shock, global_ret, month,
             P_stress_lag, EMFI_lag, HF_lag (HighFragility dummy).
    """
    emfi  = pd.read_csv(IN_EMFI,   parse_dates=["date"])[["date", "EMFI"]]
    hmm   = pd.read_csv(IN_HMM,    parse_dates=["date"])[["date", "P_stress"]]
    tci   = pd.read_csv(IN_TCI,    parse_dates=["date"])[["date", "tci_w100"]]
    frag  = pd.read_csv(IN_FRAG,   parse_dates=["date"])[["date", "TailVolBreadth", "AvgCorr60"]]
    shocks= pd.read_csv(IN_SHOCKS, parse_dates=["date"])

    df = emfi.merge(hmm, on="date") \
             .merge(tci, on="date") \
             .merge(frag, on="date") \
             .merge(shocks[["date", "max_shock", "avg_shock_pos", "breadth_shock"]],
                    on="date", how="left")
    df["max_shock"]     = df["max_shock"].fillna(0)
    df["avg_shock_pos"] = df["avg_shock_pos"].fillna(0)
    df["breadth_shock"] = df["breadth_shock"].fillna(0)

    panel    = pd.read_parquet(IN_PANEL)
    ret_wide = panel.pivot(index="date", columns="country", values="return")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    global_ret = ret_wide.mean(axis=1).rename("global_ret").reset_index()
    global_ret.columns = ["date", "global_ret"]
    df = df.merge(global_ret, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month

    # HighFragility threshold: 75th pctile of full-sample EMFI
    emfi_75 = df["EMFI"].quantile(HF_PCTILE)

    # Lagged conditioning variables
    df["P_stress_lag"] = df["P_stress"].shift(1)
    df["EMFI_lag"]     = df["EMFI"].shift(1)
    df["HF_lag"]       = (df["EMFI_lag"] > emfi_75).astype(float)

    log.info("  State-LP dataset: %d rows, date %s to %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    log.info("  EMFI 75th pctile (HF threshold): %.4f", emfi_75)
    log.info("  HighFragility days (HF_lag=1): %d (%.1f%%)",
             int(df["HF_lag"].sum()), 100 * df["HF_lag"].mean())

    shock_mask = df["max_shock"] > 0
    shock_with_lag = df[shock_mask].dropna(subset=["P_stress_lag"])
    log.info("  Shock days with valid P_stress_lag: %d", len(shock_with_lag))
    log.info("  Shock days in HF: %d (%.1f%%)",
             int(shock_with_lag["HF_lag"].sum()),
             100 * shock_with_lag["HF_lag"].mean())

    return df, emfi_75


# ---------------------------------------------------------------------------
# Smooth-transition LP
# ---------------------------------------------------------------------------

def run_smooth_lp(
    df: pd.DataFrame,
    outcome: str,
    shock: str,
    horizons: list[int],
    label: str = "full",
) -> pd.DataFrame:
    """
    Smooth-transition LP: regress Y_{t+k} on S_t, S_t*P_{t-1}, P_{t-1},
    Y_{t-1}, global_ret, month dummies.

    Returns DataFrame with columns:
      horizon, beta (at p=0), theta, beta_se, theta_se, cov_bt,
      [for each p in P_EVAL_POINTS: irf_{p}, se_{p}, ci90lo_{p}, ci90hi_{p},
       ci95lo_{p}, ci95hi_{p}], nobs.
    """
    rows = []
    nw_lags_fn = lambda k: max(1, 2 * (abs(k) + 1))

    for k in horizons:
        y_fwd  = df[outcome].shift(-k)
        lag_sh = 2 if k == -1 else 1
        y_lag  = df[outcome].shift(lag_sh)
        shock_s = df[shock]
        plag    = df["P_stress_lag"]
        interact = shock_s * plag
        gret    = df["global_ret"]
        month_d = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(float)

        X = pd.concat([
            shock_s.rename(shock),
            interact.rename("interact"),
            plag.rename("P_lag"),
            y_lag.rename("y_lag"),
            gret,
            month_d,
        ], axis=1)
        X = add_constant(X, prepend=True, has_constant="add")

        valid = y_fwd.notna() & X.notna().all(axis=1)
        y_est = y_fwd[valid].values
        X_est = X[valid].values

        if len(y_est) < 50:
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         "nobs": len(y_est), "label": label})
            continue

        try:
            res = OLS(y_est, X_est).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": nw_lags_fn(k), "use_correction": True},
            )
        except Exception as exc:
            log.warning("  k=%d smooth-LP failed: %s", k, exc)
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         "nobs": len(y_est), "label": label})
            continue

        # β = params[1] (shock), θ = params[2] (interaction)
        beta  = res.params[1]
        theta = res.params[2]
        vcov  = res.cov_params()
        var_b = vcov[1, 1]
        var_t = vcov[2, 2]
        cov_bt= vcov[1, 2]

        row = {
            "horizon": k, "beta": beta, "theta": theta,
            "beta_se": np.sqrt(var_b), "theta_se": np.sqrt(var_t),
            "cov_bt":  cov_bt, "nobs": int(res.nobs), "label": label,
        }
        # IRF at each evaluation point p
        for p in P_EVAL_POINTS:
            irf  = beta + p * theta
            se_p = np.sqrt(var_b + p**2 * var_t + 2 * p * cov_bt)
            row["irf_"    + str(p)] = irf
            row["se_"     + str(p)] = se_p
            row["ci90lo_" + str(p)] = irf - CI_90 * se_p
            row["ci90hi_" + str(p)] = irf + CI_90 * se_p
            row["ci95lo_" + str(p)] = irf - CI_95 * se_p
            row["ci95hi_" + str(p)] = irf + CI_95 * se_p
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Binary HighFragility LP
# ---------------------------------------------------------------------------

def run_binary_lp(
    df: pd.DataFrame,
    outcome: str,
    shock: str,
    horizons: list[int],
    label: str = "full",
) -> pd.DataFrame:
    """
    Binary HighFragility LP: regress Y_{t+k} on S_t, S_t*HF_{t-1}, HF_{t-1},
    Y_{t-1}, global_ret, month dummies.

    Returns DataFrame with columns:
      horizon, beta, theta, beta_se, theta_se,
      irf_normal (=beta), se_normal, irf_fragile (=beta+theta), se_fragile,
      ci95lo_normal, ci95hi_normal, ci95lo_fragile, ci95hi_fragile, nobs.
    """
    rows = []
    nw_lags_fn = lambda k: max(1, 2 * (abs(k) + 1))

    for k in horizons:
        y_fwd    = df[outcome].shift(-k)
        lag_sh   = 2 if k == -1 else 1
        y_lag    = df[outcome].shift(lag_sh)
        shock_s  = df[shock]
        hf_lag   = df["HF_lag"]
        interact = shock_s * hf_lag
        gret     = df["global_ret"]
        month_d  = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(float)

        X = pd.concat([
            shock_s.rename(shock),
            interact.rename("interact"),
            hf_lag.rename("HF_lag"),
            y_lag.rename("y_lag"),
            gret,
            month_d,
        ], axis=1)
        X = add_constant(X, prepend=True, has_constant="add")

        valid = y_fwd.notna() & X.notna().all(axis=1)
        y_est = y_fwd[valid].values
        X_est = X[valid].values

        if len(y_est) < 50:
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         "nobs": len(y_est), "label": label})
            continue

        try:
            res = OLS(y_est, X_est).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": nw_lags_fn(k), "use_correction": True},
            )
        except Exception as exc:
            log.warning("  k=%d binary-LP failed: %s", k, exc)
            rows.append({"horizon": k, "beta": np.nan, "theta": np.nan,
                         "nobs": len(y_est), "label": label})
            continue

        beta  = res.params[1]
        theta = res.params[2]
        vcov  = res.cov_params()
        var_b = vcov[1, 1]
        var_t = vcov[2, 2]
        cov_bt= vcov[1, 2]

        irf_frag = beta + theta
        se_frag  = np.sqrt(var_b + var_t + 2 * cov_bt)
        se_norm  = np.sqrt(var_b)

        rows.append({
            "horizon":        k,
            "beta":           beta,
            "theta":          theta,
            "beta_se":        se_norm,
            "theta_se":       np.sqrt(var_t),
            "irf_normal":     beta,
            "se_normal":      se_norm,
            "ci95lo_normal":  beta - CI_95 * se_norm,
            "ci95hi_normal":  beta + CI_95 * se_norm,
            "irf_fragile":    irf_frag,
            "se_fragile":     se_frag,
            "ci95lo_fragile": irf_frag - CI_95 * se_frag,
            "ci95hi_fragile": irf_frag + CI_95 * se_frag,
            "nobs":           int(res.nobs),
            "label":          label,
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


def plot_smooth_irf(results: pd.DataFrame, outcome: str, out_path: Path,
                    title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(9, 4))
    hor = results["horizon"].values

    for p in P_EVAL_POINTS:
        irf_col   = "irf_"    + str(p)
        ci95lo    = "ci95lo_" + str(p)
        ci95hi    = "ci95hi_" + str(p)

        post_mask = hor >= 0
        pre_mask  = hor < 0

        if not results[post_mask][irf_col].notna().any():
            continue

        color = P_COLORS[p]
        label = P_LABELS[p]

        # Post-period
        ax.plot(hor[post_mask], results[post_mask][irf_col],
                color=color, lw=1.5, label=label)
        ax.fill_between(hor[post_mask],
                        results[post_mask][ci95lo],
                        results[post_mask][ci95hi],
                        color=color, alpha=0.12)

        # Pre-period (dashed, same color, no legend entry)
        if pre_mask.any():
            ax.plot(hor[pre_mask], results[pre_mask][irf_col],
                    color=color, lw=0.8, ls="--", alpha=0.6)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Horizon k (trading days)", fontsize=9)
    ax.set_ylabel("Response of " + OUTCOME_LABELS.get(outcome, outcome), fontsize=9)
    ax.set_title("State-dependent LP: " + OUTCOME_LABELS.get(outcome, outcome) +
                 title_suffix, fontsize=9)
    ax.set_xticks(list(range(-5, 16, 5)))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


def plot_binary_irf(results: pd.DataFrame, outcome: str, out_path: Path,
                    title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(9, 4))
    hor = results["horizon"].values
    post = results["horizon"] >= 0

    if results[post]["irf_normal"].notna().any():
        ax.plot(hor[post], results[post]["irf_normal"],
                color="#2ca02c", lw=1.5, label="Normal state (β_k)")
        ax.fill_between(hor[post], results[post]["ci95lo_normal"],
                        results[post]["ci95hi_normal"],
                        color="#2ca02c", alpha=0.12)

    if results[post]["irf_fragile"].notna().any():
        ax.plot(hor[post], results[post]["irf_fragile"],
                color="#d62728", lw=1.5, label="HighFragility state (β_k + θ_k)")
        ax.fill_between(hor[post], results[post]["ci95lo_fragile"],
                        results[post]["ci95hi_fragile"],
                        color="#d62728", alpha=0.12)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Horizon k (trading days)", fontsize=9)
    ax.set_ylabel("Response of " + OUTCOME_LABELS.get(outcome, outcome), fontsize=9)
    ax.set_title("Binary state LP: " + OUTCOME_LABELS.get(outcome, outcome) +
                 title_suffix, fontsize=9)
    ax.set_xticks(list(range(-5, 16, 5)))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


def plot_smooth_combined(all_smooth: dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(len(OUTCOMES), 1,
                             figsize=(12, 3.2 * len(OUTCOMES)), sharex=True)
    for ax, outcome in zip(axes, OUTCOMES):
        results = all_smooth[outcome]
        hor = results["horizon"].values
        post = hor >= 0
        for p in P_EVAL_POINTS:
            irf_c = "irf_"    + str(p)
            lo_c  = "ci95lo_" + str(p)
            hi_c  = "ci95hi_" + str(p)
            if not results[post][irf_c].notna().any():
                continue
            ax.plot(hor[post], results[post][irf_c],
                    color=P_COLORS[p], lw=1.3, label=P_LABELS[p])
            ax.fill_between(hor[post], results[post][lo_c], results[post][hi_c],
                            color=P_COLORS[p], alpha=0.10)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Horizon k (trading days)", fontsize=9)
    axes[-1].set_xticks(list(range(-5, 16, 5)))
    plt.suptitle(
        "State-Dependent LP: Geopolitical Shock × Market Fragility",
        fontsize=10, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Combined figure saved: %s", out_path)


# ---------------------------------------------------------------------------
# Composition report
# ---------------------------------------------------------------------------

def build_composition_report(df: pd.DataFrame, emfi_75: float) -> pd.DataFrame:
    shock_mask = df["max_shock"] > 0
    shock_df   = df[shock_mask].dropna(subset=["P_stress_lag", "HF_lag"])

    rows = []
    for year, grp in shock_df.groupby(shock_df["date"].dt.year):
        rows.append({
            "year":          year,
            "n_shock_days":  len(grp),
            "n_hf":          int(grp["HF_lag"].sum()),
            "n_normal":      int((grp["HF_lag"] == 0).sum()),
            "pct_hf":        round(100 * grp["HF_lag"].mean(), 1),
            "mean_pstress_lag": round(grp["P_stress_lag"].mean(), 4),
        })

    # Totals
    rows.append({
        "year":          "TOTAL",
        "n_shock_days":  len(shock_df),
        "n_hf":          int(shock_df["HF_lag"].sum()),
        "n_normal":      int((shock_df["HF_lag"] == 0).sum()),
        "pct_hf":        round(100 * shock_df["HF_lag"].mean(), 1),
        "mean_pstress_lag": round(shock_df["P_stress_lag"].mean(), 4),
    })

    comp = pd.DataFrame(rows)
    log.info("Composition report: %d shock days, %d in HF (%.1f%%)",
             len(shock_df),
             int(shock_df["HF_lag"].sum()),
             100 * shock_df["HF_lag"].mean())
    return comp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== SP07 State-Dependent Local Projections ===")
    t0 = datetime.now()

    log.info("Building state-LP dataset...")
    df, emfi_75 = build_state_lp_dataset()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Composition report
    comp = build_composition_report(df, emfi_75)
    comp.to_csv(OUT_DIR / "composition_report.csv", index=False)

    all_smooth  = {}
    all_binary  = {}
    summary_rows = []

    for outcome in OUTCOMES:
        log.info("--- Outcome: %s ---", outcome)

        # --- Full sample: smooth-transition ---
        smooth = run_smooth_lp(df, outcome, PRIMARY_SHOCK, HORIZONS, label="full")
        smooth.to_csv(OUT_DIR / ("state_lp_smooth_" + outcome + ".csv"), index=False)
        all_smooth[outcome] = smooth

        # --- Full sample: binary HF ---
        binary = run_binary_lp(df, outcome, PRIMARY_SHOCK, HORIZONS, label="full")
        binary.to_csv(OUT_DIR / ("state_lp_binary_" + outcome + ".csv"), index=False)
        all_binary[outcome] = binary

        # --- COVID-excluded: smooth-transition ---
        covid_mask = (df["date"] >= COVID_EXCL_START) & (df["date"] <= COVID_EXCL_END)
        df_nocovid = df[~covid_mask].reset_index(drop=True)
        smooth_nc = run_smooth_lp(df_nocovid, outcome, PRIMARY_SHOCK, HORIZONS,
                                  label="no_covid")
        smooth_nc.to_csv(
            OUT_DIR / ("state_lp_smooth_" + outcome + "_nocovid.csv"), index=False)

        # --- Figures ---
        plot_smooth_irf(smooth, outcome,
                        OUT_DIR / ("Fig_StateLPSmooth_" + outcome + ".png"))
        plot_binary_irf(binary, outcome,
                        OUT_DIR / ("Fig_StateLPBinary_" + outcome + ".png"))
        plot_smooth_irf(smooth_nc, outcome,
                        OUT_DIR / ("Fig_StateLPSmooth_" + outcome + "_nocovid.png"),
                        title_suffix=" (COVID excluded)")

        # Log key results
        for h in [0, 1, 3, 5, 10]:
            r = smooth[smooth["horizon"] == h]
            if not r.empty and r["beta"].notna().all():
                log.info(
                    "  [smooth] k=%2d: β=%.4f θ=%.4f | p=0.9 irf=%.4f se=%.4f",
                    h, r["beta"].iloc[0], r["theta"].iloc[0],
                    r["irf_0.9"].iloc[0], r["se_0.9"].iloc[0])

        # Peak θ
        post = smooth[smooth["horizon"] >= 0]
        if post["theta"].notna().any():
            peak_idx = post["theta"].abs().idxmax()
            peak_k   = int(post.loc[peak_idx, "horizon"])
            peak_theta = post.loc[peak_idx, "theta"]
            log.info("  Peak |θ|: k=%d θ=%.4f", peak_k, peak_theta)

        # Binary HF: θ at k=0,1
        for h in [0, 1, 5]:
            r = binary[binary["horizon"] == h]
            if not r.empty and r["theta"].notna().all():
                log.info(
                    "  [binary] k=%2d: β=%.4f θ=%.4f irf_fragile=%.4f",
                    h, r["beta"].iloc[0], r["theta"].iloc[0], r["irf_fragile"].iloc[0])

        # Summary
        summary_rows.append({
            "outcome": outcome,
            "theta_k0_smooth": smooth[smooth["horizon"]==0]["theta"].iloc[0] if 0 in smooth["horizon"].values else np.nan,
            "theta_k1_smooth": smooth[smooth["horizon"]==1]["theta"].iloc[0] if 1 in smooth["horizon"].values else np.nan,
            "theta_k0_binary": binary[binary["horizon"]==0]["theta"].iloc[0] if 0 in binary["horizon"].values else np.nan,
            "theta_k1_binary": binary[binary["horizon"]==1]["theta"].iloc[0] if 1 in binary["horizon"].values else np.nan,
        })

    # Combined smooth figure (paper Figure 3)
    log.info("Plotting combined smooth-transition figure...")
    plot_smooth_combined(all_smooth, OUT_DIR / "Fig_StateLPSmooth_Combined.png")

    # Manifest
    n_shock_hf = int(comp[comp["year"] == "TOTAL"]["n_hf"].iloc[0])
    n_shock    = int(comp[comp["year"] == "TOTAL"]["n_shock_days"].iloc[0])
    manifest = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "script":         str(THIS_FILE.relative_to(GFJ_ROOT)),
        "n_obs":          len(df),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end":   str(df["date"].max().date()),
        },
        "hf_threshold_emfi": round(emfi_75, 4),
        "hf_pctile":         HF_PCTILE,
        "n_shock_days":      n_shock,
        "n_shock_hf":        n_shock_hf,
        "n_shock_normal":    n_shock - n_shock_hf,
        "p_eval_points":     P_EVAL_POINTS,
        "outcomes":          OUTCOMES,
        "covid_excl":        f"{COVID_EXCL_START} to {COVID_EXCL_END}",
        "summary":           summary_rows,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=== SP07 complete in %.1fs ===", elapsed)
    log.info("Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
