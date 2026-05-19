"""
SP13 -- Formal Inference for θ (State-Amplification Parameter)
================================================================
Provides rigorous inference for the state-amplification coefficient θ_k
from the smooth-transition LP:

    Y_{t+k} = α + β_k S_t + θ_k (S_t × P_{t-1}) + φ_k P_{t-1}
             + γ_k Y_{t-1} + δ_k r^global_t + μ_m + ε_{t+k}

Two complementary approaches:
  1. Analytic (Newey-West HAC):  SE(θ_k), t-stat, p-value from OLS.bse
  2. Block bootstrap (B=1000, block_length=20): percentile 95% CI for
     θ_{k=0} and IRF(p=0.9, k=0) per outcome.

Delta-method SE for the total effect IRF(p):
    SE[IRF(p)] = sqrt(Var(β) + p²·Var(θ) + 2p·Cov(β,θ))

Specification: COVID-excluded (drop 2020-03-01 to 2020-12-31),
               shock = max_shock, state = P_stress_{t-1}.

Inputs (relative to Paper_GFJ root):
  results/emfi/emfi_daily.csv
  results/stress_regimes/hmm_daily.csv
  results/connectedness/tci_daily.csv
  results/fragility/fragility_daily.csv
  data/aggregate_shocks.csv
  data/panel_daily.parquet

Outputs (relative to Paper_GFJ root):
  results/theta_inference/theta_inference_{outcome}.csv   -- per-horizon table
  results/theta_inference/bootstrap_ci_table.csv          -- k=0 bootstrap CIs
  results/theta_inference/Fig_ThetaInference.png          -- forest plot
  results/theta_inference/manifest.json

Run from Paper_GFJ root:
  python subprojects/13_theta_inference/run_theta_inference.py
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
from scipy import stats
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

OUT_DIR   = GFJ_ROOT / "results" / "theta_inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZONS          = list(range(-5, 16))
OUTCOMES          = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
PRIMARY_SHOCK     = "max_shock_nocovid"  # zeroes COVID-period shocks, full sample

P_EVAL            = 0.9         # evaluation point for "near-systemic" IRF
P_EVAL_POINTS     = [0.0, 0.5, 0.9]
CI_95             = 1.960
HF_PCTILE         = 0.75

# Block bootstrap
B_REPS            = 1000
BLOCK_LEN         = 20
RNG_SEED          = 42

COVID_EXCL_START  = "2020-03-01"
COVID_EXCL_END    = "2020-12-31"

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

def build_dataset() -> pd.DataFrame:
    """
    Build the LP dataset (full sample); COVID exclusion applied separately.
    Columns: date, EMFI, tci_w100, TailVolBreadth, P_stress, AvgCorr60,
             max_shock, global_ret, month, P_stress_lag, EMFI_lag, HF_lag.
    """
    emfi   = pd.read_csv(IN_EMFI,   parse_dates=["date"])[["date", "EMFI"]]
    hmm    = pd.read_csv(IN_HMM,    parse_dates=["date"])[["date", "P_stress"]]
    tci    = pd.read_csv(IN_TCI,    parse_dates=["date"])[["date", "tci_w100"]]
    frag   = pd.read_csv(IN_FRAG,   parse_dates=["date"])[["date", "TailVolBreadth", "AvgCorr60"]]
    shocks = pd.read_csv(IN_SHOCKS, parse_dates=["date"])

    df = (emfi.merge(hmm, on="date")
              .merge(tci, on="date")
              .merge(frag, on="date")
              .merge(shocks[["date", "max_shock"]], on="date", how="left"))
    df["max_shock"] = df["max_shock"].fillna(0)

    # COVID-zeroed shock: max_shock_nocovid zeroes shocks in 2020 Jan–Dec
    # (full sample kept for estimation; COVID-period shock = 0)
    covid_mask = (df["date"] >= COVID_EXCL_START) & (df["date"] <= COVID_EXCL_END)
    df["max_shock_nocovid"] = df["max_shock"].copy()
    df.loc[covid_mask, "max_shock_nocovid"] = 0.0
    log.info("  COVID-zeroed shocks: %d days → 0  (was %d shock days total)",
             int((df["max_shock"] > 0) & covid_mask).sum() if False
             else int(((df["max_shock"] > 0) & covid_mask).sum()),
             int((df["max_shock"] > 0).sum()))

    panel    = pd.read_parquet(IN_PANEL)
    ret_wide = panel.pivot(index="date", columns="country", values="return")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    global_ret = ret_wide.mean(axis=1).rename("global_ret").reset_index()
    global_ret.columns = ["date", "global_ret"]
    df = df.merge(global_ret, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["month"]       = df["date"].dt.month
    df["P_stress_lag"] = df["P_stress"].shift(1)
    df["EMFI_lag"]    = df["EMFI"].shift(1)
    emfi_75           = df["EMFI"].quantile(HF_PCTILE)
    df["HF_lag"]      = (df["EMFI_lag"] > emfi_75).astype(float)

    log.info("  Dataset: %d rows, %s to %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


def apply_covid_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the COVID crisis window (2020-03-01 to 2020-12-31)."""
    mask = (df["date"] >= COVID_EXCL_START) & (df["date"] <= COVID_EXCL_END)
    out  = df[~mask].reset_index(drop=True)
    log.info("  COVID excl: dropped %d rows → %d remain", mask.sum(), len(out))
    return out


# ---------------------------------------------------------------------------
# Design-matrix builder (single horizon)
# ---------------------------------------------------------------------------

def _build_Xy(df: pd.DataFrame, outcome: str, shock: str, k: int):
    """
    Build (y, X, valid_mask) for a single horizon k.
    Column order: [const, shock, interact, P_lag, y_lag, global_ret, month dummies]
    Indices:       0       1      2         3      4      5           6+
    β = params[1], θ = params[2].
    """
    y_fwd   = df[outcome].shift(-k)
    lag_sh  = 2 if k == -1 else 1
    y_lag   = df[outcome].shift(lag_sh)
    shock_s = df[shock]
    plag    = df["P_stress_lag"]
    interact= shock_s * plag
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

    valid  = y_fwd.notna() & X.notna().all(axis=1)
    y_arr  = y_fwd[valid].values
    X_arr  = X[valid].values
    return y_arr, X_arr, valid


# ---------------------------------------------------------------------------
# Delta-method SE for IRF(p) = β + p·θ
# ---------------------------------------------------------------------------

def compute_irf_se(var_b: float, var_t: float, cov_bt: float, p: float) -> float:
    """Var(β + p·θ) = Var(β) + p²·Var(θ) + 2p·Cov(β,θ)."""
    return np.sqrt(max(var_b + p**2 * var_t + 2 * p * cov_bt, 0.0))


# ---------------------------------------------------------------------------
# Single-horizon analytic regression
# ---------------------------------------------------------------------------

def run_horizon(
    df: pd.DataFrame,
    outcome: str,
    shock: str,
    k: int,
) -> dict:
    """
    Run one horizon regression; return dict with beta, theta, SE, p-val,
    and delta-method IRF SEs.
    """
    nw_lags = max(1, 2 * (abs(k) + 1))
    y_arr, X_arr, _ = _build_Xy(df, outcome, shock, k)

    if len(y_arr) < 50:
        return {"horizon": k, "nobs": len(y_arr), "converged": False}

    try:
        res = OLS(y_arr, X_arr).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": nw_lags, "use_correction": True},
        )
    except Exception as exc:
        log.warning("  k=%d %s failed: %s", k, outcome, exc)
        return {"horizon": k, "nobs": len(y_arr), "converged": False}

    beta  = float(res.params[1])
    theta = float(res.params[2])
    vcov  = res.cov_params()
    var_b = float(vcov[1, 1])
    var_t = float(vcov[2, 2])
    cov_bt= float(vcov[1, 2])

    # HAC t-stat and p-value for θ
    theta_se   = float(np.sqrt(max(var_t, 0.0)))
    theta_tstat= float(theta / theta_se) if theta_se > 0 else np.nan
    theta_pval = float(2 * stats.t.sf(abs(theta_tstat), df=max(res.nobs - X_arr.shape[1], 1)))

    row = {
        "horizon":     k,
        "beta":        beta,
        "beta_se":     float(np.sqrt(max(var_b, 0.0))),
        "theta":       theta,
        "theta_se":    theta_se,
        "theta_tstat": theta_tstat,
        "theta_pval":  theta_pval,
        "cov_bt":      cov_bt,
        "nobs":        int(res.nobs),
        "converged":   True,
    }

    for p in P_EVAL_POINTS:
        irf  = beta + p * theta
        se_p = compute_irf_se(var_b, var_t, cov_bt, p)
        tag  = str(p).replace(".", "")  # "00", "05", "09"
        row[f"irf_p{tag}"]       = irf
        row[f"se_irf_p{tag}"]    = se_p
        row[f"ci95lo_irf_p{tag}"] = irf - CI_95 * se_p
        row[f"ci95hi_irf_p{tag}"] = irf + CI_95 * se_p

    return row


def run_smooth_lp_full(
    df: pd.DataFrame,
    outcome: str,
    shock: str,
) -> pd.DataFrame:
    """Run all horizons; return DataFrame with full inference output."""
    rows = [run_horizon(df, outcome, shock, k) for k in HORIZONS]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def _circular_block_resample(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Circular block bootstrap indices.
    Returns index array of length n drawn from blocks of length block_len.
    """
    n_blocks = int(np.ceil(n / block_len))
    starts   = rng.integers(0, n, size=n_blocks)
    idx      = np.concatenate([
        (s + np.arange(block_len)) % n for s in starts
    ])
    return idx[:n]


def block_bootstrap_k0(
    df: pd.DataFrame,
    outcome: str,
    shock: str,
    k: int = 0,
    b_reps: int = B_REPS,
    block_len: int = BLOCK_LEN,
    seed: int = RNG_SEED,
) -> dict:
    """
    Block bootstrap for θ_{k} and IRF(p=0.9, k).

    Returns dict:
      theta_boot_se, theta_boot_ci95lo, theta_boot_ci95hi,
      irf_p9_boot_se, irf_p9_boot_ci95lo, irf_p9_boot_ci95hi,
      n_boot_valid
    """
    rng = np.random.default_rng(seed)
    y_arr, X_arr, _ = _build_Xy(df, outcome, shock, k)
    n = len(y_arr)

    if n < 50:
        return {"theta_boot_se": np.nan, "theta_boot_ci95lo": np.nan,
                "theta_boot_ci95hi": np.nan, "irf_p9_boot_se": np.nan,
                "irf_p9_boot_ci95lo": np.nan, "irf_p9_boot_ci95hi": np.nan,
                "n_boot_valid": 0}

    theta_boots = []
    irf_p9_boots = []

    for _ in range(b_reps):
        idx   = _circular_block_resample(n, block_len, rng)
        y_b   = y_arr[idx]
        X_b   = X_arr[idx]

        try:
            # OLS only (no HAC) for speed — bootstrap captures serial correlation
            coef = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
            beta_b  = float(coef[1])
            theta_b = float(coef[2])
            theta_boots.append(theta_b)
            irf_p9_boots.append(beta_b + P_EVAL * theta_b)
        except Exception:
            continue

    theta_arr  = np.array(theta_boots)
    irf_p9_arr = np.array(irf_p9_boots)

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        if len(arr) < 10:
            return np.nan, np.nan, np.nan
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        se = float(arr.std(ddof=1))
        return se, lo, hi

    t_se, t_lo, t_hi     = _ci(theta_arr)
    i_se, i_lo, i_hi     = _ci(irf_p9_arr)

    return {
        "theta_boot_se":       t_se,
        "theta_boot_ci95lo":   t_lo,
        "theta_boot_ci95hi":   t_hi,
        "irf_p9_boot_se":      i_se,
        "irf_p9_boot_ci95lo":  i_lo,
        "irf_p9_boot_ci95hi":  i_hi,
        "n_boot_valid":        len(theta_arr),
    }


# ---------------------------------------------------------------------------
# Summary table for Table 3 (k=0 across outcomes)
# ---------------------------------------------------------------------------

def build_table3_row(
    analytic_row: dict,
    boot_dict: dict,
    outcome: str,
) -> dict:
    """Build a Table-3 summary row for a given outcome."""
    row = {"outcome": outcome}
    for key in ["beta", "beta_se", "theta", "theta_se", "theta_tstat",
                "theta_pval", "nobs"]:
        row[key] = analytic_row.get(key, np.nan)

    p_tag = str(P_EVAL).replace(".", "")
    row["irf_p9"]          = analytic_row.get(f"irf_p{p_tag}", np.nan)
    row["se_irf_p9"]       = analytic_row.get(f"se_irf_p{p_tag}", np.nan)
    row["ci95lo_irf_p9"]   = analytic_row.get(f"ci95lo_irf_p{p_tag}", np.nan)
    row["ci95hi_irf_p9"]   = analytic_row.get(f"ci95hi_irf_p{p_tag}", np.nan)
    row["amplification"]   = (row["irf_p9"] / row["beta"]
                              if row["beta"] and abs(row["beta"]) > 1e-10 else np.nan)

    # Bootstrap CIs
    row.update(boot_dict)

    return row


# ---------------------------------------------------------------------------
# Plotting: forest plot of θ at k=0 across outcomes
# ---------------------------------------------------------------------------

def plot_theta_forest(table3_df: pd.DataFrame, out_path: Path) -> None:
    """
    Forest plot: θ_{k=0} analytic CI (horizontal bars) + bootstrap CI (whiskers).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = table3_df["outcome"].tolist()
    y_pos    = np.arange(len(outcomes))

    for i, row in table3_df.iterrows():
        theta   = row["theta"]
        se_t    = row["theta_se"]
        ci_lo   = theta - CI_95 * se_t
        ci_hi   = theta + CI_95 * se_t
        b_lo    = row["theta_boot_ci95lo"]
        b_hi    = row["theta_boot_ci95hi"]

        # Analytic 95% CI
        ax.plot([ci_lo, ci_hi], [y_pos[i], y_pos[i]],
                lw=2, color="#2166ac", solid_capstyle="round")
        # Bootstrap CI (dashed)
        if not np.isnan(b_lo):
            ax.plot([b_lo, b_hi], [y_pos[i] + 0.15, y_pos[i] + 0.15],
                    lw=2, color="#d6604d", linestyle="--", solid_capstyle="round",
                    label="Bootstrap 95% CI" if i == 0 else None)
        # Point estimate
        ax.scatter([theta], [y_pos[i]], color="#2166ac", s=50, zorder=5)
        # Significance star
        pval = row["theta_pval"]
        star = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        if star:
            ax.text(ci_hi + 0.05, y_pos[i], star, va="center", fontsize=9, color="#1a1a1a")

    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(outcomes, fontsize=10)
    ax.set_xlabel(r"$\hat{\theta}_{k=0}$ (state-amplification coefficient)", fontsize=10)
    ax.set_title(r"State Amplification $\theta_{k=0}$: Analytic HAC CI vs Block Bootstrap CI",
                 fontsize=10)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#2166ac", lw=2, label="Analytic 95% CI (HAC)"),
        Line2D([0], [0], color="#d6604d", lw=2, linestyle="--", label="Bootstrap 95% CI"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved forest plot → %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== SP13 Theta Inference ===")
    t0 = datetime.now(timezone.utc)

    # ---- Data ----
    log.info("Building dataset …")
    df_full = build_dataset()   # full sample, max_shock_nocovid already added

    # ---- Per-outcome full LP ----
    table3_rows = []
    for outcome in OUTCOMES:
        log.info("  Running LP: %s", outcome)
        lp_df = run_smooth_lp_full(df_full, outcome, PRIMARY_SHOCK)

        # Save per-horizon table
        out_csv = OUT_DIR / f"theta_inference_{outcome}.csv"
        lp_df.to_csv(out_csv, index=False)
        log.info("    Saved %s (%d rows)", out_csv.name, len(lp_df))

        # Get k=0 row
        mask_k0 = lp_df["horizon"] == 0
        if mask_k0.any():
            row_k0 = lp_df[mask_k0].iloc[0].to_dict()
            # Block bootstrap at k=0
            log.info("    Block bootstrap (B=%d) …", B_REPS)
            boot = block_bootstrap_k0(df_full, outcome, PRIMARY_SHOCK, k=0)
            t3_row = build_table3_row(row_k0, boot, outcome)
            table3_rows.append(t3_row)
            log.info(
                "    θ=%.4f  SE=%.4f  p=%.4f  boot_CI=[%.4f, %.4f]",
                t3_row["theta"], t3_row["theta_se"], t3_row["theta_pval"],
                t3_row["theta_boot_ci95lo"], t3_row["theta_boot_ci95hi"],
            )

    # ---- Bootstrap summary table ----
    table3_df = pd.DataFrame(table3_rows)
    boot_path  = OUT_DIR / "bootstrap_ci_table.csv"
    table3_df.to_csv(boot_path, index=False)
    log.info("Saved bootstrap CI table → %s", boot_path.name)

    # ---- Forest plot ----
    fig_path = OUT_DIR / "Fig_ThetaInference.png"
    plot_theta_forest(table3_df, fig_path)

    # ---- Manifest ----
    emfi_row = table3_df[table3_df["outcome"] == "EMFI"].iloc[0]
    manifest = {
        "run_utc":          t0.isoformat(),
        "spec":             "smooth_transition_lp_covid_zeroed",
        "shock":            PRIMARY_SHOCK,
        "horizons":         HORIZONS,
        "B_reps":           B_REPS,
        "block_len":        BLOCK_LEN,
        "outcomes":         OUTCOMES,
        "covid_excl":       f"{COVID_EXCL_START} to {COVID_EXCL_END}",
        "emfi_k0": {
            "beta":              float(emfi_row["beta"]),
            "theta":             float(emfi_row["theta"]),
            "theta_se":          float(emfi_row["theta_se"]),
            "theta_pval":        float(emfi_row["theta_pval"]),
            "irf_p9":            float(emfi_row["irf_p9"]),
            "se_irf_p9":         float(emfi_row["se_irf_p9"]),
            "theta_boot_ci95lo": float(emfi_row["theta_boot_ci95lo"]),
            "theta_boot_ci95hi": float(emfi_row["theta_boot_ci95hi"]),
        },
        "all_theta_positive": bool((table3_df["theta"] > 0).all()),
        "emfi_theta_sig05":   bool(emfi_row["theta_pval"] < 0.05),
        "n_sig05_outcomes":   int((table3_df["theta_pval"] < 0.05).sum()),
        "n_sig10_outcomes":   int((table3_df["theta_pval"] < 0.10).sum()),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved manifest.json")

    # ---- Summary print ----
    log.info("--- Table 3 Summary (k=0, COVID-zeroed) ---")
    log.info("%-16s  %8s  %8s  %8s  %8s  %8s  %8s", "outcome",
             "beta", "theta", "SE(th)", "p(th)", "IRF(p=0.9)", "Amp")
    for _, r in table3_df.iterrows():
        amp = r["amplification"]
        log.info("%-16s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %7.1fx",
                 r["outcome"], r["beta"], r["theta"],
                 r["theta_se"], r["theta_pval"],
                 r["irf_p9"], amp if not np.isnan(amp) else 0)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("SP13 complete in %.1f s", elapsed)


if __name__ == "__main__":
    main()
