"""
SP03 -- Volatility Connectedness (TCI)
======================================
Constructs a daily Total Connectedness Index (TCI) using the rolling
Diebold-Yilmaz (2012) generalized forecast-error variance decomposition
(GFEVD) on the 19-country absolute-return panel.

Method
------
For each rolling window of W trading days:
1. Fit VAR(p) on the 19-dimensional vector of abs returns.
   p is BIC-selected from {1, 2, 3}; default fallback = 1.
2. Compute the Pesaran-Shin (1998) generalized FEVD at H-step horizon.
3. Row-sum normalize to obtain theta_tilde (rows sum to 1).
4. TCI = (sum of off-diagonal theta_tilde) / K * 100.
5. FROM_i = sum_{j!=i} theta_tilde_{ij} * 100  (how much i's FEV is
   explained by other countries).
   TO_j   = sum_{i!=j} theta_tilde_{ij} * 100  (how much j contributes
   to other countries' FEV).
   NET_j  = TO_j - FROM_j.

Primary window: W=100, H=10.
Robustness windows: W=60, W=150, W=200.

Usage
-----
  python subprojects/03_connectedness/build_connectedness.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- Paths --------------------------------------------------------------------
THIS_FILE  = Path(__file__).resolve()
GFJ_ROOT   = THIS_FILE.parents[2]

IN_PANEL   = GFJ_ROOT / "data" / "panel_daily.parquet"
OUT_DIR    = GFJ_ROOT / "results" / "connectedness"
OUT_TCI    = OUT_DIR / "tci_daily.csv"
OUT_DIR_CONN = OUT_DIR / "directional_connectedness.csv"
OUT_FIG_TCI  = OUT_DIR / "Fig_TCI_Timeline.png"
OUT_FIG_NET  = OUT_DIR / "Fig_Network_MajorEvents.png"
OUT_SUMMARY  = OUT_DIR / "summary_stats.csv"
OUT_MANIFEST = OUT_DIR / "manifest.json"

# -- Parameters ---------------------------------------------------------------
STUDY_START = pd.Timestamp("2017-01-02")
STUDY_END   = pd.Timestamp("2025-10-15")
N_COUNTRIES = 19
H_HORIZON   = 10          # GFEVD forecast horizon
VAR_MAX_P   = 3           # BIC search range: p in {1, 2, ..., VAR_MAX_P}
WINDOWS     = [60, 100, 150, 200]   # rolling window sizes (trading days)
PRIMARY_W   = 100         # primary specification for paper figures

# Major events for plot overlays (date, label)
MAJOR_EVENTS = [
    (pd.Timestamp("2020-02-24"), "COVID"),
    (pd.Timestamp("2022-02-24"), "Ukraine"),
    (pd.Timestamp("2023-10-07"), "Hamas-Israel"),
]


# -- Core functions -----------------------------------------------------------

def load_vol_panel() -> pd.DataFrame:
    """Load panel and pivot to wide absolute-return matrix."""
    panel = pd.read_parquet(IN_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    vol_wide = (
        panel.pivot(index="date", columns="country", values="abs_return")
        .sort_index()
    )
    log.info("  Vol panel: %d dates x %d countries", *vol_wide.shape)
    return vol_wide


def compute_gfevd(
    chunk: np.ndarray,
    H: int = H_HORIZON,
    max_p: int = VAR_MAX_P,
) -> Tuple[np.ndarray, int]:
    """
    Compute normalized GFEVD (Pesaran-Shin 1998) for one window.

    Parameters
    ----------
    chunk : (W, K) array of abs returns
    H     : forecast horizon
    max_p : maximum VAR order to consider (BIC selection)

    Returns
    -------
    theta_tilde : (K, K) normalized GFEVD; rows sum to 1.
                  theta_tilde[i, j] = fraction of i's H-step FEV
                  explained by shock j.
    p_selected  : VAR order that was chosen.
    """
    K = chunk.shape[1]

    # BIC order selection
    try:
        sel = VAR(chunk).select_order(max_p)
        p = max(1, sel.bic)   # bic returns the optimal lag (int)
    except Exception:
        p = 1
    p = min(int(p), max_p)

    # Fit VAR(p)
    try:
        result = VAR(chunk).fit(p)
    except Exception:
        result = VAR(chunk).fit(1)
        p = 1

    Sigma = result.sigma_u      # (K, K) residual covariance
    MA    = result.ma_rep(H)    # (H+1, K, K) MA coefficients; MA[0] = I

    # Unnormalized GFEVD
    # theta_ij = sigma_jj^{-1} * sum_{h=0}^{H-1} ((A_h Sigma)[i,j])^2
    #           / sum_{h=0}^{H-1} (A_h Sigma A_h')[i,i]
    num   = np.zeros((K, K))
    denom = np.zeros(K)
    for h in range(H):
        AhS   = MA[h] @ Sigma           # (K, K)
        AhSAh = AhS @ MA[h].T           # (K, K)
        sigma_diag = np.diag(Sigma)     # (K,)
        for j in range(K):
            num[:, j] += (AhS[:, j] ** 2) / sigma_diag[j]
        denom += np.diag(AhSAh)

    # Guard against near-zero denom (numerical edge case)
    denom = np.where(denom < 1e-20, 1e-20, denom)
    theta = num / denom[:, None]

    # Row-sum normalize (Diebold-Yilmaz 2012 eq. 8)
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-20, 1e-20, row_sums)
    theta_tilde = theta / row_sums

    return theta_tilde, p


def rolling_connectedness(
    vol_wide: pd.DataFrame,
    W: int,
    H: int = H_HORIZON,
    max_p: int = VAR_MAX_P,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute rolling TCI and directional spillovers for window size W.

    Returns
    -------
    tci_series : pd.Series indexed by date (last date in each window)
    dir_df     : pd.DataFrame with columns [date, country, from_spill,
                 to_spill, net_spill]
    """
    data      = vol_wide.values
    dates     = vol_wide.index
    countries = vol_wide.columns.tolist()
    K         = len(countries)
    n         = len(data)

    tci_vals  = []
    tci_dates = []
    dir_rows  = []

    for i in range(W - 1, n):
        chunk = data[i - W + 1 : i + 1]   # (W, K)
        theta_tilde, _ = compute_gfevd(chunk, H=H, max_p=max_p)

        tci = (theta_tilde.sum() - np.trace(theta_tilde)) / K * 100
        tci_vals.append(tci)
        tci_dates.append(dates[i])

        # Directional spillovers (%)
        from_spill = (theta_tilde.sum(axis=1) - np.diag(theta_tilde)) * 100
        to_spill   = (theta_tilde.sum(axis=0) - np.diag(theta_tilde)) * 100
        net_spill  = to_spill - from_spill

        for k, country in enumerate(countries):
            dir_rows.append({
                "date":       dates[i],
                "country":    country,
                "from_spill": from_spill[k],
                "to_spill":   to_spill[k],
                "net_spill":  net_spill[k],
            })

    tci_series = pd.Series(tci_vals, index=pd.DatetimeIndex(tci_dates), name=f"tci_w{W}")
    dir_df     = pd.DataFrame(dir_rows)
    dir_df["date"] = pd.to_datetime(dir_df["date"])
    return tci_series, dir_df


def build_all_windows(
    vol_wide: pd.DataFrame,
    windows: List[int] = WINDOWS,
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Run rolling connectedness for all window sizes.

    Returns
    -------
    tci_df   : wide DataFrame [date, tci_w60, tci_w100, tci_w150, tci_w200]
    dir_dict : dict {W: dir_df}
    """
    tci_parts = []
    dir_dict  = {}

    for W in windows:
        log.info("  Computing rolling GFEVD: W=%d ...", W)
        t0 = time.time()
        tci_s, dir_df = rolling_connectedness(vol_wide, W=W)
        elapsed = time.time() - t0
        log.info(
            "    W=%d: %d values, TCI mean=%.2f, range=[%.2f, %.2f], %.1fs",
            W, len(tci_s), tci_s.mean(), tci_s.min(), tci_s.max(), elapsed,
        )
        tci_parts.append(tci_s)
        dir_dict[W] = dir_df

    tci_df = pd.concat(tci_parts, axis=1).reset_index().rename(columns={"index": "date"})
    tci_df["date"] = pd.to_datetime(tci_df["date"])
    return tci_df, dir_dict


def plot_tci_timeline(tci_df: pd.DataFrame, out_path: Path) -> None:
    """2-panel figure: primary TCI and comparison across all window sizes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Volatility Connectedness Index (TCI) -- European Equity Markets", fontsize=11)

    primary_col = f"tci_w{PRIMARY_W}"
    colors_all  = {"tci_w60": "#d62728", "tci_w100": "#1f77b4",
                   "tci_w150": "#2ca02c", "tci_w200": "#ff7f0e"}

    # -- Panel 1: primary TCI with event shading
    ax1 = axes[0]
    dates = tci_df["date"]
    ax1.plot(dates, tci_df[primary_col], color="#1f77b4", lw=1.2, label=f"TCI (W={PRIMARY_W})")
    ax1.fill_between(dates, tci_df[primary_col], alpha=0.15, color="#1f77b4")
    for ev_date, ev_label in MAJOR_EVENTS:
        if ev_date >= dates.min() and ev_date <= dates.max():
            ax1.axvline(ev_date, color="red", lw=0.9, ls="--", alpha=0.7)
            ax1.text(ev_date, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 80,
                     ev_label, rotation=90, fontsize=7, va="top", ha="right", color="red")
    ax1.set_ylabel("TCI (%)")
    ax1.set_title(f"Primary specification: W={PRIMARY_W}, H={H_HORIZON}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # -- Panel 2: all window sizes
    ax2 = axes[1]
    for col, clr in colors_all.items():
        if col in tci_df.columns:
            w_label = col.replace("tci_w", "W=")
            ax2.plot(tci_df["date"], tci_df[col], color=clr, lw=0.9,
                     alpha=0.85, label=w_label)
    for ev_date, ev_label in MAJOR_EVENTS:
        if ev_date >= dates.min() and ev_date <= dates.max():
            ax2.axvline(ev_date, color="red", lw=0.9, ls="--", alpha=0.7)
    ax2.set_ylabel("TCI (%)")
    ax2.set_title("Robustness: all window specifications")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


def plot_network_events(
    vol_wide: pd.DataFrame,
    dir_dict: Dict[int, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    Bar chart of TO/FROM spillovers for each country around key events.
    Uses W=100 directional data.
    Compares: pre-Ukraine (2021), peak-Ukraine (2022), calm (2023).
    """
    dir_df   = dir_dict[PRIMARY_W]
    countries = sorted(vol_wide.columns.tolist())
    K        = len(countries)

    # Average TO spillovers over 3-month windows
    periods = {
        "Pre-Ukraine\n(2021 avg)":       ("2021-01-01", "2021-12-31"),
        "Ukraine peak\n(Mar 2022)":       ("2022-02-24", "2022-05-31"),
        "Calm\n(2023 avg)":              ("2023-01-01", "2023-12-31"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Country TO-Spillovers Across Market Regimes (W=100)", fontsize=11)

    for ax, (label, (start, end)) in zip(axes, periods.items()):
        mask = (dir_df["date"] >= start) & (dir_df["date"] <= end)
        sub  = dir_df[mask].groupby("country")[["to_spill", "from_spill"]].mean()
        sub  = sub.reindex(countries).fillna(0)
        x    = np.arange(K)
        ax.bar(x, sub["to_spill"],   width=0.4, label="TO",   color="#1f77b4", alpha=0.8)
        ax.bar(x + 0.4, sub["from_spill"], width=0.4, label="FROM", color="#ff7f0e", alpha=0.8)
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels([c[:3] for c in countries], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Spillover (%)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


def write_outputs(
    tci_df: pd.DataFrame,
    dir_dict: Dict[int, pd.DataFrame],
    vol_wide: pd.DataFrame,
) -> None:
    """Write all outputs to results/connectedness/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # TCI time series (wide)
    tci_df.to_csv(OUT_TCI, index=False)
    log.info("  TCI written: %s (%d rows)", OUT_TCI.name, len(tci_df))

    # Directional connectedness (primary window, long format)
    dir_primary = dir_dict[PRIMARY_W].copy()
    dir_primary["window"] = PRIMARY_W
    dir_primary.to_csv(OUT_DIR_CONN, index=False)
    log.info("  Directional written: %s (%d rows)", OUT_DIR_CONN.name, len(dir_primary))

    # Summary statistics
    tci_cols = [c for c in tci_df.columns if c.startswith("tci_")]
    summary  = tci_df[tci_cols].describe().T
    summary.to_csv(OUT_SUMMARY)
    log.info("  Summary stats written")

    # Figures
    log.info("  Plotting TCI timeline...")
    plot_tci_timeline(tci_df, OUT_FIG_TCI)
    log.info("  Plotting network/spillover figure...")
    plot_network_events(vol_wide, dir_dict, OUT_FIG_NET)

    # Manifest
    primary_tci = tci_df[f"tci_w{PRIMARY_W}"].dropna()
    manifest = {
        "generated_at":    pd.Timestamp.now(tz="UTC").isoformat(),
        "script":          str(THIS_FILE.relative_to(GFJ_ROOT)),
        "date_range":      {
            "start": str(tci_df["date"].min().date()),
            "end":   str(tci_df["date"].max().date()),
        },
        "parameters": {
            "h_horizon":  H_HORIZON,
            "var_max_p":  VAR_MAX_P,
            "primary_w":  PRIMARY_W,
            "windows":    WINDOWS,
        },
        "tci_primary": {
            "window":  PRIMARY_W,
            "n_dates": int(primary_tci.notna().sum()),
            "mean":    float(primary_tci.mean()),
            "std":     float(primary_tci.std()),
            "min":     float(primary_tci.min()),
            "max":     float(primary_tci.max()),
        },
        "outputs": {
            "tci_daily":            str(OUT_TCI),
            "directional":          str(OUT_DIR_CONN),
            "summary_stats":        str(OUT_SUMMARY),
            "fig_tci_timeline":     str(OUT_FIG_TCI),
            "fig_network_events":   str(OUT_FIG_NET),
        },
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.info("  Manifest written")


# -- Main entry point ---------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log.info("=== SP03 Volatility Connectedness (TCI) ===")

    log.info("Loading vol panel...")
    vol_wide = load_vol_panel()

    log.info("Computing rolling GFEVD for all window specs...")
    tci_df, dir_dict = build_all_windows(vol_wide)

    log.info("Writing outputs...")
    write_outputs(tci_df, dir_dict, vol_wide)

    elapsed = time.time() - t_start
    primary_tci = tci_df[f"tci_w{PRIMARY_W}"].dropna()
    log.info("=== SP03 complete in %.1fs ===", elapsed)
    log.info(
        "TCI (W=%d): mean=%.2f  std=%.2f  range=[%.2f, %.2f]",
        PRIMARY_W, primary_tci.mean(), primary_tci.std(),
        primary_tci.min(), primary_tci.max(),
    )
    log.info("Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
