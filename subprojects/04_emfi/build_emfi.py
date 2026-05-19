"""
SP04 -- Composite European Market Fragility Index (EMFI)
=========================================================
Combines the four daily fragility series from SP02 and SP03 into a single
composite index using PCA. The EMFI is the paper's primary outcome variable.

Components
----------
  VolStress      (SP02) -- average absolute return
  TailVolBreadth (SP02) -- tail co-exceedance count (rolling q95, 252-day)
  AvgCorr60      (SP02) -- rolling average pairwise correlation (60-day)
  TCI_w100       (SP03) -- total connectedness index (W=100)

Method
------
1. Merge SP02 and SP03 outputs on date; drop warm-up NaN rows.
2. Standardise each component to mean 0, std 1 using the *full-sample*
   mean and std (not rolling -- EMFI is an outcome variable, not a detector).
3. PCA on the (T x 4) standardised matrix; extract PC1.
4. Orient PC1 so that higher values = more fragility
   (flip sign if VolStress has negative correlation with PC1).
5. Report variance explained and factor loadings.

Usage
-----
  python subprojects/04_emfi/build_emfi.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- Paths --------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_FRAGILITY = GFJ_ROOT / "results" / "fragility"  / "fragility_daily.csv"
IN_TCI       = GFJ_ROOT / "results" / "connectedness" / "tci_daily.csv"

OUT_DIR      = GFJ_ROOT / "results" / "emfi"
OUT_EMFI     = OUT_DIR / "emfi_daily.csv"
OUT_LOADINGS = OUT_DIR / "pca_loadings.csv"
OUT_SUMMARY  = OUT_DIR / "summary_stats.csv"
OUT_FIG_TL   = OUT_DIR / "Fig_EMFI_Timeline.png"
OUT_FIG_COMP = OUT_DIR / "Fig_EMFI_Components.png"
OUT_MANIFEST = OUT_DIR / "manifest.json"

# -- Parameters ---------------------------------------------------------------
STUDY_START   = pd.Timestamp("2017-01-02")
STUDY_END     = pd.Timestamp("2025-10-15")
COMPONENTS    = ["VolStress", "TailVolBreadth", "AvgCorr60", "tci_w100"]
PRIMARY_W_TCI = 100

# Threshold for defining HighFragility state (for summary reporting)
HIGH_FRAG_PCTILE = 75

# Major events
MAJOR_EVENTS = [
    (pd.Timestamp("2020-02-24"), "COVID"),
    (pd.Timestamp("2022-02-24"), "Ukraine"),
    (pd.Timestamp("2023-10-07"), "Hamas-Israel"),
]

STRESS_SHADING = [
    ("COVID crash",      "2020-02-20", "2020-04-30"),
    ("Ukraine invasion", "2022-02-24", "2022-06-30"),
]


# -- Core functions -----------------------------------------------------------

def load_inputs() -> pd.DataFrame:
    """Merge fragility indicators and TCI into one aligned DataFrame."""
    frag = pd.read_csv(IN_FRAGILITY, parse_dates=["date"])
    tci  = pd.read_csv(IN_TCI, parse_dates=["date"])

    df = frag[["date"] + [c for c in COMPONENTS if c != "tci_w100"]].merge(
        tci[["date", f"tci_w{PRIMARY_W_TCI}"]].rename(
            columns={f"tci_w{PRIMARY_W_TCI}": "tci_w100"}
        ),
        on="date",
        how="inner",
    )
    log.info("  Merged input: %d rows, date range %s to %s",
             len(df), df["date"].min().date(), df["date"].max().date())

    n_before = len(df)
    df = df.dropna(subset=COMPONENTS)
    log.info("  After dropping NaN warm-up rows: %d (dropped %d)", len(df), n_before - len(df))
    return df.reset_index(drop=True)


def build_emfi(df: pd.DataFrame):
    """
    Fit PCA on the four components and return the EMFI series plus metadata.

    Returns
    -------
    emfi_df    : DataFrame [date, EMFI, VolStress_std, TailVolBreadth_std,
                             AvgCorr60_std, tci_w100_std]
    loadings   : DataFrame [component, loading, loading_sq, pct_variance_component]
    meta       : dict with PCA statistics
    """
    X = df[COMPONENTS].values.astype(float)

    # Full-sample standardisation
    mu  = X.mean(axis=0)
    sig = X.std(axis=0, ddof=1)
    X_std = (X - mu) / sig

    # PCA via eigen-decomposition of covariance matrix
    cov_mat  = np.cov(X_std.T)                        # (4, 4)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    # eigh returns ascending order -- reverse to descending
    eigvals  = eigvals[::-1]
    eigvecs  = eigvecs[:, ::-1]

    # PC1 scores
    pc1_raw = X_std @ eigvecs[:, 0]

    # Orient: flip if VolStress anti-correlated with PC1
    vol_idx = COMPONENTS.index("VolStress")
    if np.corrcoef(X_std[:, vol_idx], pc1_raw)[0, 1] < 0:
        eigvecs[:, 0] = -eigvecs[:, 0]
        pc1_raw       = -pc1_raw
        log.info("  PC1 sign flipped to align with fragility direction")

    var_explained = eigvals / eigvals.sum() * 100
    log.info("  Variance explained: PC1=%.1f%%  PC2=%.1f%%  PC3=%.1f%%  PC4=%.1f%%",
             *var_explained)
    log.info("  PC1 loadings: %s", dict(zip(COMPONENTS, eigvecs[:, 0].round(4))))

    # Build output DataFrame
    emfi_df = df[["date"]].copy()
    emfi_df["EMFI"] = pc1_raw
    for i, comp in enumerate(COMPONENTS):
        emfi_df[comp + "_std"] = X_std[:, i]

    # Loadings table
    loadings_data = []
    for i, comp in enumerate(COMPONENTS):
        lv = eigvecs[i, 0]
        loadings_data.append({
            "component":                comp,
            "loading":                  round(float(lv), 6),
            "loading_sq":               round(float(lv ** 2), 6),
            "pct_variance_contribution": round(float(lv ** 2) * 100, 2),
        })
    loadings = pd.DataFrame(loadings_data)

    meta = {
        "n_obs":               int(len(df)),
        "date_range":          {
            "start": str(df["date"].min().date()),
            "end":   str(df["date"].max().date()),
        },
        "standardisation": {
            comp: {"mean": float(mu[i]), "std": float(sig[i])}
            for i, comp in enumerate(COMPONENTS)
        },
        "pca": {
            "eigenvalues":       [round(float(v), 6) for v in eigvals],
            "variance_explained": [round(float(v), 2) for v in var_explained],
            "pc1_variance_pct":  round(float(var_explained[0]), 2),
            "sign_flipped":      bool(eigvecs[vol_idx, 0] > 0),  # True = kept positive
        },
        "emfi_stats": {
            "mean":  round(float(pc1_raw.mean()), 6),
            "std":   round(float(pc1_raw.std(ddof=1)), 6),
            "min":   round(float(pc1_raw.min()), 6),
            "max":   round(float(pc1_raw.max()), 6),
            "p75_threshold": round(float(np.percentile(pc1_raw, HIGH_FRAG_PCTILE)), 6),
        },
    }
    return emfi_df, loadings, meta


# -- Figures ------------------------------------------------------------------

def plot_emfi_timeline(emfi_df: pd.DataFrame, out_path: Path) -> None:
    """EMFI over time with stress-episode shading and event lines."""
    fig, ax = plt.subplots(figsize=(13, 5))

    dates = emfi_df["date"]
    emfi  = emfi_df["EMFI"]

    # Stress episode shading
    for label, start, end in STRESS_SHADING:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.12, color="red", zorder=0, label=label)

    # EMFI series
    ax.plot(dates, emfi, color="#1f77b4", lw=1.0, label="EMFI (PC1)", zorder=2)
    ax.fill_between(dates, emfi, 0, where=(emfi > 0), alpha=0.12, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)

    # High-fragility threshold (75th pctile)
    p75 = emfi.quantile(0.75)
    ax.axhline(p75, color="#d62728", lw=0.8, ls=":", alpha=0.7,
               label=f"75th pctile ({p75:.2f})")

    # Event lines
    for ev_date, ev_label in MAJOR_EVENTS:
        if ev_date >= dates.min() and ev_date <= dates.max():
            ax.axvline(ev_date, color="darkred", lw=1.0, ls="--", alpha=0.8)
            ylim = ax.get_ylim()
            ax.text(ev_date, emfi.max() * 0.85, ev_label,
                    rotation=90, fontsize=8, va="top", ha="right", color="darkred")

    ax.set_ylabel("EMFI (standardised PC1 score)", fontsize=10)
    ax.set_title("European Market Fragility Index (EMFI)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


def plot_emfi_components(emfi_df: pd.DataFrame, out_path: Path) -> None:
    """5-panel figure: EMFI + each standardised component."""
    comp_cols  = [c + "_std" for c in COMPONENTS]
    comp_labels = {
        "VolStress_std":      "VolStress (std)",
        "TailVolBreadth_std": "TailVolBreadth (std)",
        "AvgCorr60_std":      "AvgCorr60 (std)",
        "tci_w100_std":       "TCI W=100 (std)",
    }
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("EMFI and Component Series (standardised)", fontsize=11)
    dates = emfi_df["date"]

    # Panel 0: EMFI
    axes[0].plot(dates, emfi_df["EMFI"], color=colors[0], lw=1.0, label="EMFI")
    axes[0].axhline(0, color="black", lw=0.4, ls="--", alpha=0.5)
    axes[0].set_ylabel("EMFI", fontsize=8)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for idx, (col, label) in enumerate(comp_labels.items()):
        ax = axes[idx + 1]
        ax.plot(dates, emfi_df[col], color=colors[idx + 1], lw=0.8, label=label)
        ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.5)
        ax.set_ylabel(label.split()[0], fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for ev_date, _ in MAJOR_EVENTS:
            if ev_date >= dates.min() and ev_date <= dates.max():
                ax.axvline(ev_date, color="red", lw=0.7, ls="--", alpha=0.6)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure saved: %s", out_path)


# -- Write outputs ------------------------------------------------------------

def write_outputs(emfi_df, loadings, meta):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emfi_df.to_csv(OUT_EMFI, index=False)
    log.info("  EMFI written: %s (%d rows)", OUT_EMFI.name, len(emfi_df))

    loadings.to_csv(OUT_LOADINGS, index=False)
    log.info("  Loadings written: %s", OUT_LOADINGS.name)

    summary = emfi_df[["EMFI"] + [c + "_std" for c in COMPONENTS]].describe().T
    summary.to_csv(OUT_SUMMARY)
    log.info("  Summary stats written")

    log.info("  Plotting EMFI timeline...")
    plot_emfi_timeline(emfi_df, OUT_FIG_TL)

    log.info("  Plotting component breakdown...")
    plot_emfi_components(emfi_df, OUT_FIG_COMP)

    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "script":       str(THIS_FILE.relative_to(GFJ_ROOT)),
        "components":   COMPONENTS,
        "parameters": {
            "primary_w_tci":      PRIMARY_W_TCI,
            "high_frag_pctile":   HIGH_FRAG_PCTILE,
            "standardisation":    "full-sample mean/std (ddof=1)",
        },
        "pca":       meta["pca"],
        "emfi_stats": meta["emfi_stats"],
        "date_range": meta["date_range"],
        "standardisation": meta["standardisation"],
        "outputs": {
            "emfi_daily":    str(OUT_EMFI),
            "pca_loadings":  str(OUT_LOADINGS),
            "summary_stats": str(OUT_SUMMARY),
            "fig_timeline":  str(OUT_FIG_TL),
            "fig_components": str(OUT_FIG_COMP),
        },
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.info("  Manifest written")


# -- Main ---------------------------------------------------------------------

def main():
    t0 = time.time()
    log.info("=== SP04 Composite EMFI ===")

    log.info("Loading and merging inputs...")
    df = load_inputs()

    log.info("Running PCA...")
    emfi_df, loadings, meta = build_emfi(df)

    log.info("Writing outputs...")
    write_outputs(emfi_df, loadings, meta)

    log.info("=== SP04 complete in %.1fs ===", time.time() - t0)
    log.info(
        "EMFI: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
        meta["emfi_stats"]["mean"], meta["emfi_stats"]["std"],
        meta["emfi_stats"]["min"],  meta["emfi_stats"]["max"],
    )
    log.info("PC1 variance explained: %.1f%%", meta["pca"]["pc1_variance_pct"])
    log.info("75th pctile threshold: %.4f", meta["emfi_stats"]["p75_threshold"])


if __name__ == "__main__":
    main()
