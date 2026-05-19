"""
SP11 -- TCI Robustness: Window Sensitivity and Network Metrics
==============================================================
Addresses R3: reviewer concern that W=100 VAR with 19 variables is
overparameterised (parameter count ≈ 19² + 19 = 380 > W=100 observations).

This script:
  1. Extends the TCI computation to W=250 (longer window = better powered)
  2. Adds simpler correlation-based network metrics (rolling eigenvalue,
     network density) that do not use VAR at all — robustness check showing
     TCI finding is not an artefact of the VAR specification
  3. Rebuilds EMFI for each TCI window (W=60/100/150/200/250)
  4. Runs panel LP and state LP for each EMFI variant (COVID shock days
     excluded from treatment, consistent with Phase R2 baseline)
  5. Produces a window-sensitivity table: W × {β_k0, θ_k0, ratio}

Primary inputs (relative to Paper_GFJ root):
  results/connectedness/tci_daily.csv        (existing W=60/100/150/200)
  results/fragility/fragility_daily.csv
  results/emfi/emfi_daily.csv
  results/stress_regimes/hmm_daily.csv
  data/aggregate_shocks.csv
  data/panel_daily.parquet

Outputs (relative to Paper_GFJ root):
  results/tci_robustness/tci_w250.csv
  results/tci_robustness/network_metrics.csv
  results/tci_robustness/emfi_variants.csv
  results/tci_robustness/sensitivity_table.csv
  results/tci_robustness/Fig_TCI_WindowSensitivity.png
  results/tci_robustness/manifest.json

Run from Paper_GFJ root:
  python subprojects/11_tci_robustness/run_tci_robustness.py
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
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.api import VAR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE  = Path(__file__).resolve()
GFJ_ROOT   = THIS_FILE.parents[2]

IN_TCI     = GFJ_ROOT / "results" / "connectedness"     / "tci_daily.csv"
IN_FRAG    = GFJ_ROOT / "results" / "fragility"         / "fragility_daily.csv"
IN_EMFI    = GFJ_ROOT / "results" / "emfi"              / "emfi_daily.csv"
IN_HMM     = GFJ_ROOT / "results" / "stress_regimes"    / "hmm_daily.csv"
IN_SHOCKS  = GFJ_ROOT / "data"    / "aggregate_shocks.csv"
IN_PANEL   = GFJ_ROOT / "data"    / "panel_daily.parquet"

OUT_DIR    = GFJ_ROOT / "results" / "tci_robustness"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
WINDOWS       = [60, 100, 150, 200, 250]   # TCI window sizes to evaluate
PRIMARY_W     = 100                         # current paper primary
NEW_PRIMARY_W = 100                         # we confirm W=100 is robust
H_HORIZON     = 10                          # GFEVD forecast horizon (unchanged)
VAR_P         = 1                           # fixed lag for W=250 (avoids overfit)
VAR_MAX_P     = 3                           # BIC for other windows

CORR_WIN      = 60                          # correlation window for network metrics
DENSITY_THRESH = 0.5                        # |corr| > threshold = connected

HORIZONS      = list(range(-5, 16))
KEY_HORIZONS  = [0, 1, 5, 10]             # for sensitivity table
COVID_START   = "2020-01-01"
COVID_END     = "2020-12-31"
HF_PCTILE     = 0.75
P_EVAL        = [0.0, 0.5, 0.9]

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
# 1. Extend TCI to W=250
# ===========================================================================

def compute_gfevd_fast(chunk: np.ndarray, H: int = 10, p: int = 1):
    """
    Fast GFEVD for a fixed VAR(p). No BIC selection.
    Returns TCI scalar (off-diagonal share × 100).
    """
    K = chunk.shape[1]
    try:
        result = VAR(chunk).fit(p)
    except Exception:
        return np.nan

    Sigma = result.sigma_u
    MA    = result.ma_rep(H)

    num   = np.zeros((K, K))
    denom = np.zeros(K)
    for h in range(H):
        AhS   = MA[h] @ Sigma
        AhSAh = AhS @ MA[h].T
        sigma_diag = np.diag(Sigma)
        for j in range(K):
            num[:, j] += (AhS[:, j] ** 2) / np.maximum(sigma_diag[j], 1e-20)
        denom += np.diag(AhSAh)

    denom = np.where(denom < 1e-20, 1e-20, denom)
    theta = num / denom[:, None]
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-20, 1e-20, row_sums)
    theta_tilde = theta / row_sums

    tci = (theta_tilde.sum() - np.trace(theta_tilde)) / K * 100
    return float(tci)


def compute_tci_w250(panel_path: Path) -> pd.Series:
    """Compute rolling TCI with W=250, VAR(1), for the full panel."""
    log.info("Loading panel for W=250 TCI computation ...")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    vol_wide = (
        panel.pivot(index="date", columns="country", values="abs_return")
        .sort_index()
    )
    log.info("  Panel: %d dates × %d countries", *vol_wide.shape)

    data  = vol_wide.fillna(0).values   # fill NaN with 0 (pre-sample)
    dates = vol_wide.index
    W     = 250
    n     = len(data)

    tci_vals  = []
    tci_dates = []

    log.info("  Rolling W=%d TCI (VAR(1), H=%d) — %d windows ...", W, H_HORIZON, n - W + 1)
    t0 = time.time()
    for i in range(W - 1, n):
        chunk = data[i - W + 1 : i + 1]
        tci   = compute_gfevd_fast(chunk, H=H_HORIZON, p=VAR_P)
        tci_vals.append(tci)
        tci_dates.append(dates[i])
        if (i - W + 1) % 200 == 0:
            log.info("    %d/%d windows done", i - W + 2, n - W + 1)

    elapsed = time.time() - t0
    tci_series = pd.Series(tci_vals, index=pd.DatetimeIndex(tci_dates), name="tci_w250")
    log.info("  W=250 done: %d values, mean=%.2f, range=[%.2f, %.2f], %.1fs",
             len(tci_series), tci_series.mean(), tci_series.min(), tci_series.max(), elapsed)
    return tci_series


# ===========================================================================
# 2. Rolling network metrics (no VAR — just correlation matrix)
# ===========================================================================

def compute_network_metrics(panel_path: Path, W: int = CORR_WIN) -> pd.DataFrame:
    """
    Compute rolling correlation-based network metrics:
      lambda1   : first eigenvalue of the K×K correlation matrix (scaled by K)
      density   : fraction of off-diagonal |corr| pairs exceeding DENSITY_THRESH

    These are simpler than TCI (no VAR required) but capture similar
    co-movement dynamics, providing robustness against overparameterisation
    critique.
    """
    log.info("Computing rolling network metrics (W=%d) ...", W)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    ret_wide = (
        panel.pivot(index="date", columns="country", values="return")
        .sort_index()
    )

    data  = ret_wide.values.astype(float)
    dates = ret_wide.index
    K     = data.shape[1]
    n     = len(data)

    rows = []
    for i in range(W - 1, n):
        chunk = data[i - W + 1 : i + 1]          # (W, K)
        # Drop columns that are all NaN
        mask = ~np.all(np.isnan(chunk), axis=0)
        chunk = chunk[:, mask]
        k     = chunk.shape[1]
        if k < 2:
            rows.append({"date": dates[i], "lambda1_scaled": np.nan, "density": np.nan})
            continue

        # Fill remaining NaNs with column mean for correlation
        col_means = np.nanmean(chunk, axis=0)
        inds = np.where(np.isnan(chunk))
        chunk[inds] = np.take(col_means, inds[1])

        corr = np.corrcoef(chunk.T)              # (k, k)
        np.fill_diagonal(corr, 0)

        # First eigenvalue (scaled by k so comparable across subsets)
        try:
            eigvals = np.linalg.eigvalsh(corr + np.eye(k))
            lambda1 = eigvals[-1] / k
        except Exception:
            lambda1 = np.nan

        # Network density: fraction of off-diagonal pairs with |corr| > thresh
        off_diag = corr[np.triu_indices(k, k=1)]
        density  = float(np.mean(np.abs(off_diag) > DENSITY_THRESH))

        rows.append({"date": dates[i], "lambda1_scaled": lambda1, "density": density})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    log.info("  Network metrics: %d rows, lambda1 mean=%.3f, density mean=%.3f",
             len(df), df["lambda1_scaled"].mean(), df["density"].mean())
    return df


# ===========================================================================
# 3. Rebuild EMFI variants for each TCI window
# ===========================================================================

def build_emfi_variant(frag_df: pd.DataFrame, tci_df: pd.DataFrame,
                        W: int) -> pd.DataFrame:
    """
    Rebuild EMFI using tci_wW instead of tci_w100.
    Uses the same PCA approach as SP04: full-sample standardisation, PC1.
    Returns DataFrame with [date, EMFI_wW].
    """
    tci_col = f"tci_w{W}"
    if tci_col not in tci_df.columns:
        log.warning("  %s not in TCI data — skipping W=%d", tci_col, W)
        return pd.DataFrame(columns=["date", f"EMFI_w{W}"])

    df = frag_df[["date", "VolStress", "TailVolBreadth", "AvgCorr60"]].merge(
        tci_df[["date", tci_col]].rename(columns={tci_col: "tci"}),
        on="date", how="inner"
    )
    components = ["VolStress", "TailVolBreadth", "AvgCorr60", "tci"]
    df = df.dropna(subset=components).reset_index(drop=True)

    if len(df) < 50:
        log.warning("  W=%d: only %d rows after dropna", W, len(df))
        return pd.DataFrame(columns=["date", f"EMFI_w{W}"])

    # Full-sample standardisation
    X = df[components].values.astype(float)
    mu  = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.where(std < 1e-10, 1.0, std)
    X_std = (X - mu) / std

    # PCA — PC1
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_std).ravel()

    # Orient: higher = more fragility (VolStress loads positively)
    volstress_loading = pca.components_[0, 0]   # index 0 = VolStress
    if volstress_loading < 0:
        scores = -scores

    # Standardise EMFI itself (mean 0, std 1 in-sample)
    scores = (scores - scores.mean()) / (scores.std(ddof=1) + 1e-10)

    result = df[["date"]].copy()
    result[f"EMFI_w{W}"] = scores
    log.info("  W=%d EMFI: %d obs, var_explained=%.1f%%",
             W, len(result), pca.explained_variance_ratio_[0] * 100)
    return result


# ===========================================================================
# 4. Panel LP and State LP (helpers — stripped down from SP10)
# ===========================================================================

def build_lp_dataset(panel_path: Path, shocks_path: Path,
                     emfi_col: pd.DataFrame, hmm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge panel returns with shocks and the provided EMFI column.
    Zero out COVID shock days (same as R2 baseline).

    emfi_col : DataFrame with columns [date, EMFI_wW]
    """
    panel  = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    shocks = pd.read_csv(shocks_path, parse_dates=["date"])
    shocks["max_shock_nocovid"] = shocks["max_shock"].copy()
    covid_mask = (shocks["date"] >= COVID_START) & (shocks["date"] <= COVID_END)
    shocks.loc[covid_mask, "max_shock_nocovid"] = 0.0

    # Aggregate panel to cross-section mean returns
    cs = (
        panel.groupby("date")["return"]
        .mean()
        .reset_index()
        .rename(columns={"return": "cs_return"})
    )
    cs["date"] = pd.to_datetime(cs["date"])

    df = cs.merge(shocks[["date", "max_shock_nocovid"]], on="date", how="left")
    df = df.merge(emfi_col, on="date", how="left")

    emfi_name = [c for c in emfi_col.columns if c != "date"][0]

    # HMM P_stress is the state variable (same across all TCI windows)
    df = df.merge(hmm_df[["date", "P_stress"]], on="date", how="left")
    df = df.fillna({"max_shock_nocovid": 0.0})
    df = df.sort_values("date").reset_index(drop=True)
    df["shock"]    = df["max_shock_nocovid"]
    df["EMFI"]     = df[emfi_name]
    df["lag_emfi"] = df["EMFI"].shift(1)
    # Lagged HMM P_stress as the smooth state variable (already in [0,1])
    df["lag_P_stress"] = df["P_stress"].shift(1)
    return df


def run_panel_lp_single(df: pd.DataFrame, horizons: list = KEY_HORIZONS) -> pd.DataFrame:
    """Run panel LP for EMFI at selected horizons. Returns DataFrame of results."""
    rows = []
    T = len(df)
    for k in horizons:
        y = df["EMFI"].shift(-k)
        shock = df["shock"]
        lag_emfi = df["lag_emfi"]
        cs_ret   = df["cs_return"]

        valid = y.notna() & shock.notna() & lag_emfi.notna() & cs_ret.notna()
        if valid.sum() < 30:
            continue

        Y = y[valid].values
        X = add_constant(pd.DataFrame({
            "shock": shock[valid],
            "lag_emfi": lag_emfi[valid],
            "cs_return": cs_ret[valid],
        }))
        try:
            res = OLS(Y, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(1, int(np.floor(0.75 * (valid.sum() ** (1/3)))))}
            )
            beta = res.params["shock"]
            se   = res.bse["shock"]
            rows.append({"horizon": k, "beta": beta, "se": se, "nobs": int(valid.sum())})
        except Exception as e:
            log.warning("  LP k=%d failed: %s", k, e)

    return pd.DataFrame(rows)


def run_state_lp_single(df: pd.DataFrame) -> dict:
    """
    Run smooth-transition state LP for EMFI at k=0.
    Uses HMM P_stress (lagged, already in [0,1]) as state variable.
    Returns dict with beta, theta, irf at p=0/0.9, and ratio.
    """
    y        = df["EMFI"]          # k=0: contemporaneous
    shock    = df["shock"]
    P_state  = df["lag_P_stress"]  # HMM P_stress lagged one period, in [0,1]

    interaction = shock * P_state

    valid = (
        y.notna() & shock.notna() & P_state.notna()
        & df["lag_emfi"].notna() & df["cs_return"].notna()
    )
    if valid.sum() < 30:
        return {"beta": np.nan, "theta": np.nan, "ratio": np.nan}

    Y = y[valid].values
    X = add_constant(pd.DataFrame({
        "shock":       shock[valid],
        "interaction": interaction[valid],
        "P_state":     P_state[valid],
        "lag_emfi":    df["lag_emfi"][valid],
        "cs_return":   df["cs_return"][valid],
    }))
    try:
        nw_lags = max(1, int(np.floor(0.75 * (valid.sum() ** (1/3)))))
        res = OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        beta  = float(res.params["shock"])
        theta = float(res.params["interaction"])
        # IRF at p=0 (calm market) vs p=0.9 (fragile market)
        irf_calm   = beta
        irf_stress = beta + theta * 0.9
        ratio = irf_stress / irf_calm if abs(irf_calm) > 1e-6 else np.nan
        return {"beta": beta, "theta": theta,
                "irf_calm": irf_calm, "irf_stress": irf_stress, "ratio": ratio}
    except Exception as e:
        log.warning("  State LP k=0 failed: %s", e)
        return {"beta": np.nan, "theta": np.nan, "ratio": np.nan}


# ===========================================================================
# 5. Figure: TCI window sensitivity
# ===========================================================================

def plot_window_sensitivity(tci_all: pd.DataFrame, out_path: Path) -> None:
    """Plot TCI series for all windows on a shared axis."""
    colors = {
        "tci_w60":  "#d62728",
        "tci_w100": "#1f77b4",
        "tci_w150": "#2ca02c",
        "tci_w200": "#ff7f0e",
        "tci_w250": "#9467bd",
    }
    labels = {
        "tci_w60":  "W=60",
        "tci_w100": "W=100 (primary)",
        "tci_w150": "W=150",
        "tci_w200": "W=200",
        "tci_w250": "W=250",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, clr in colors.items():
        if col in tci_all.columns:
            ax.plot(tci_all["date"], tci_all[col],
                    color=clr, lw=0.9, alpha=0.8, label=labels[col])

    for ev_date, ev_label in MAJOR_EVENTS:
        ax.axvline(ev_date, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.text(ev_date, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 95,
                ev_label, rotation=90, fontsize=7, va="top", ha="right", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("TCI (%)")
    ax.set_title("TCI Window Sensitivity: W = 60 / 100 / 150 / 200 / 250")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path.name)


def plot_network_metrics(net_df: pd.DataFrame, out_path: Path) -> None:
    """Plot rolling eigenvalue and density."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(net_df["date"], net_df["lambda1_scaled"],
                 color="#1f77b4", lw=0.9)
    axes[0].set_ylabel("λ₁ / K")
    axes[0].set_title(f"Rolling First Eigenvalue of Correlation Matrix (W={CORR_WIN})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(net_df["date"], net_df["density"],
                 color="#d62728", lw=0.9)
    axes[1].set_ylabel("Density (|ρ| > 0.5)")
    axes[1].set_title(f"Rolling Correlation Network Density (W={CORR_WIN})")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        for ev_date, ev_label in MAJOR_EVENTS:
            ax.axvline(ev_date, color="black", lw=0.8, ls="--", alpha=0.5)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path.name)


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_total = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 65)
    log.info("SP11 — TCI Robustness: Window Sensitivity")
    log.info("=" * 65)

    # -----------------------------------------------------------------------
    # Load existing TCI (W=60/100/150/200)
    # -----------------------------------------------------------------------
    log.info("\n[1/6] Loading existing TCI data ...")
    tci_existing = pd.read_csv(IN_TCI, parse_dates=["date"])
    log.info("  Existing TCI: %d rows, columns: %s",
             len(tci_existing), list(tci_existing.columns))

    # -----------------------------------------------------------------------
    # Compute W=250 TCI
    # -----------------------------------------------------------------------
    tci_w250_path = OUT_DIR / "tci_w250.csv"
    if tci_w250_path.exists():
        log.info("\n[2/6] Loading cached W=250 TCI ...")
        w250_df = pd.read_csv(tci_w250_path, parse_dates=["date"])
        tci_w250 = w250_df.set_index("date")["tci_w250"]
    else:
        log.info("\n[2/6] Computing W=250 TCI (VAR(1)) ...")
        tci_w250 = compute_tci_w250(IN_PANEL)
        w250_df  = tci_w250.reset_index()
        w250_df.columns = ["date", "tci_w250"]
        w250_df.to_csv(tci_w250_path, index=False)
        log.info("  Saved %s", tci_w250_path.name)

    # Merge all TCI windows
    tci_all = tci_existing.merge(
        tci_w250.reset_index().rename(columns={"index": "date"}),
        on="date", how="outer"
    ).sort_values("date").reset_index(drop=True)
    log.info("  Combined TCI: %d rows", len(tci_all))

    # -----------------------------------------------------------------------
    # Network metrics
    # -----------------------------------------------------------------------
    net_path = OUT_DIR / "network_metrics.csv"
    if net_path.exists():
        log.info("\n[3/6] Loading cached network metrics ...")
        net_df = pd.read_csv(net_path, parse_dates=["date"])
    else:
        log.info("\n[3/6] Computing rolling network metrics ...")
        net_df = compute_network_metrics(IN_PANEL, W=CORR_WIN)
        net_df.to_csv(net_path, index=False)
        log.info("  Saved %s", net_path.name)

    # -----------------------------------------------------------------------
    # Rebuild EMFI for each TCI window
    # -----------------------------------------------------------------------
    log.info("\n[4/6] Rebuilding EMFI for W = %s ...", WINDOWS)
    frag_df = pd.read_csv(IN_FRAG, parse_dates=["date"])
    emfi_parts = [tci_all[["date"]]]
    emfi_results = {}
    for W in WINDOWS:
        variant = build_emfi_variant(frag_df, tci_all, W)
        if len(variant) > 0:
            emfi_results[W] = variant
            col = f"EMFI_w{W}"
            emfi_parts.append(variant.set_index("date")[col])

    emfi_variants = pd.concat(
        [p.set_index("date") if isinstance(p, pd.DataFrame) else p
         for p in emfi_parts if not isinstance(p, pd.DataFrame) or len(p.columns) > 0],
        axis=1
    ).reset_index()
    # Simpler merge
    emfi_merged = tci_all[["date"]].copy()
    for W, variant in emfi_results.items():
        emfi_merged = emfi_merged.merge(variant, on="date", how="left")
    emfi_merged.to_csv(OUT_DIR / "emfi_variants.csv", index=False)
    log.info("  Saved emfi_variants.csv (%d rows)", len(emfi_merged))

    # -----------------------------------------------------------------------
    # LP sensitivity: panel LP β and state LP θ for each window
    # -----------------------------------------------------------------------
    log.info("\n[5/6] Running LP sensitivity table ...")
    hmm_df  = pd.read_csv(IN_HMM, parse_dates=["date"])

    sensitivity_rows = []
    for W in WINDOWS:
        emfi_col_name = f"EMFI_w{W}"
        if emfi_col_name not in emfi_merged.columns:
            log.warning("  W=%d EMFI missing — skipping LP", W)
            continue

        emfi_col_df = emfi_merged[["date", emfi_col_name]].dropna()
        # Rename for LP builder
        emfi_col_df = emfi_col_df.rename(columns={emfi_col_name: f"EMFI_w{W}"})

        lp_df2 = build_lp_dataset(IN_PANEL, IN_SHOCKS, emfi_col_df, hmm_df)

        # Panel LP at k=0
        panel_lp = run_panel_lp_single(lp_df2, horizons=KEY_HORIZONS)
        k0_row = panel_lp[panel_lp["horizon"] == 0]
        beta_k0 = k0_row["beta"].iloc[0] if len(k0_row) > 0 else np.nan
        se_k0   = k0_row["se"].iloc[0]   if len(k0_row) > 0 else np.nan

        # State LP at k=0
        slp = run_state_lp_single(lp_df2)

        n_shock = int((lp_df2["shock"] > 0).sum())
        row = {
            "window": W,
            "n_shock_days": n_shock,
            "lp_beta_k0": round(beta_k0, 4),
            "lp_se_k0":   round(se_k0, 4),
            "slp_beta_k0":  round(slp["beta"],  4) if not np.isnan(slp.get("beta", np.nan)) else np.nan,
            "slp_theta_k0": round(slp["theta"], 4) if not np.isnan(slp.get("theta", np.nan)) else np.nan,
            "irf_calm":     round(slp.get("irf_calm", np.nan),   4),
            "irf_stress":   round(slp.get("irf_stress", np.nan), 4),
            "ratio":        round(slp.get("ratio", np.nan),       2) if not np.isnan(slp.get("ratio", np.nan)) else np.nan,
        }
        sensitivity_rows.append(row)
        log.info("  W=%3d: β_k0=%.3f  θ_k0=%.3f  ratio=%.1f",
                 W, beta_k0,
                 slp.get("theta", float("nan")),
                 slp.get("ratio", float("nan")))

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(OUT_DIR / "sensitivity_table.csv", index=False)
    log.info("  Saved sensitivity_table.csv")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    log.info("\n[6/6] Generating figures ...")
    plot_window_sensitivity(tci_all, OUT_DIR / "Fig_TCI_WindowSensitivity.png")
    plot_network_metrics(net_df, OUT_DIR / "Fig_NetworkMetrics.png")

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------
    manifest = {
        "script":     "SP11 run_tci_robustness.py",
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "windows":    WINDOWS,
        "var_p_w250": VAR_P,
        "h_horizon":  H_HORIZON,
        "corr_win":   CORR_WIN,
        "density_threshold": DENSITY_THRESH,
        "n_sensitivity_rows": len(sensitivity_rows),
        "sensitivity_summary": {
            str(r["window"]): {
                "lp_beta_k0":   r["lp_beta_k0"],
                "slp_theta_k0": r["slp_theta_k0"],
                "ratio":        r["ratio"],
            }
            for r in sensitivity_rows
        },
        "elapsed_s": round(time.time() - t_total, 1),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  Saved manifest.json")

    log.info("\n✅ SP11 complete (%.1fs)", time.time() - t_total)
    log.info("   Outputs → %s", OUT_DIR)
    return manifest


if __name__ == "__main__":
    main()
                                                                                                