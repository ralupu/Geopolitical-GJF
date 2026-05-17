"""
SP02 -- Daily Fragility Indicators
====================================
Constructs daily cross-sectional market fragility measures from the
19-country European equity return panel built in SP01.

Outputs (relative to Paper_GFJ root):
  results/fragility/fragility_daily.csv   -- all indicators, one row per date
  results/fragility/summary_stats.csv     -- descriptive statistics
  results/fragility/Fig_FragilityTimeline.png

Indicators
----------
  VolStress_t       = mean_i |r_{i,t}|                    (no rolling)
  SqStress_t        = mean_i r_{i,t}^2                    (no rolling)
  TailVolBreadth_t  = #{i : |r_{i,t}| > Q_{i,95,t-1}}    (rolling 252-day threshold)
  TailLossBreadth_t = #{i : r_{i,t} < Q_{i,5,t-1}}       (rolling 252-day threshold)
  TailGainBreadth_t = #{i : r_{i,t} > Q_{i,95,t-1}}      (rolling 252-day threshold)
  AvgCorr60_t       = mean pairwise Pearson corr (60-day rolling)
  AvgCorr30_t       = mean pairwise Pearson corr (30-day rolling)

Rolling quantile design: threshold at day t is the quantile computed over the
252-day window [t-252, t-1] (shift by 1 day -- strictly out-of-sample).
min_periods=60 -- first 59 days have NaN tail breadth.

Run from Paper_GFJ root:
  python subprojects/02_fragility_indicators/build_fragility_indicators.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE  = Path(__file__).resolve()
GFJ_ROOT   = THIS_FILE.parents[2]

IN_PANEL       = GFJ_ROOT / "data" / "panel_daily.parquet"
IN_AGG_SHOCKS  = GFJ_ROOT / "data" / "aggregate_shocks.csv"

OUT_DIR        = GFJ_ROOT / "results" / "fragility"
OUT_FRAGILITY  = OUT_DIR / "fragility_daily.csv"
OUT_STATS      = OUT_DIR / "summary_stats.csv"
OUT_FIGURE     = OUT_DIR / "Fig_FragilityTimeline.png"
OUT_MANIFEST   = OUT_DIR / "manifest.json"
# Aliases for test compatibility
OUT_CSV        = OUT_FRAGILITY
OUT_SUMMARY    = OUT_STATS

# Rolling window parameters
CORR_WINDOW_60  = 60
CORR_WINDOW_30  = 30
CORR_MIN_PERIODS_60 = 30
CORR_MIN_PERIODS_30 = 20
TAIL_WINDOW     = 252   # 1 trading year
TAIL_MIN_PERIODS = 60

N_COUNTRIES = 19

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
# Core computation
# ---------------------------------------------------------------------------

def _rolling_avg_corr(returns_wide: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """
    Compute rolling average pairwise Pearson correlation across all N*(N-1)/2 country pairs.

    Uses pandas rolling().corr() internally (vectorised C implementation).

    Parameters
    ----------
    returns_wide : (T, N) DataFrame, index=date, columns=countries
    window       : rolling window length in trading days
    min_periods  : minimum non-NaN observations required

    Returns
    -------
    pd.Series, index=date, values=average pairwise correlation
    """
    countries = sorted(returns_wide.columns.tolist())
    rc = returns_wide.rolling(window=window, min_periods=min_periods).corr()
    # rc has MultiIndex (date, country_i) x country_j

    # Extract upper-triangle pairs and average per date
    pair_series = []
    for i, c1 in enumerate(countries):
        for c2 in countries[i + 1:]:
            s = rc.loc[(slice(None), c1), c2]
            s.index = s.index.droplevel(1)  # drop country level, keep date
            pair_series.append(s)

    if not pair_series:
        return pd.Series(np.nan, index=returns_wide.index)

    avg = pd.concat(pair_series, axis=1).mean(axis=1)
    avg.name = None
    return avg


def build_fragility_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all daily fragility indicators from the long-format return panel.

    Parameters
    ----------
    panel : output of SP01 (columns: date, country, return, abs_return, sq_return, ...)

    Returns
    -------
    pd.DataFrame with index=date and one column per indicator.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # Pivot to wide format
    ret_w = panel.pivot(index="date", columns="country", values="return")
    abs_w = panel.pivot(index="date", columns="country", values="abs_return")
    sq_w  = panel.pivot(index="date", columns="country", values="sq_return")
    ret_w.columns.name  = None
    abs_w.columns.name  = None
    sq_w.columns.name   = None

    log.info("  Wide returns: %s, range %s to %s",
             ret_w.shape, ret_w.index.min().date(), ret_w.index.max().date())

    result = pd.DataFrame(index=ret_w.index)

    # ------------------------------------------------------------------
    # 1. Level-based stress (no rolling -- available for full sample)
    # ------------------------------------------------------------------
    result["VolStress"]  = abs_w.mean(axis=1)
    result["SqStress"]   = sq_w.mean(axis=1)
    log.info("  VolStress: min=%.5f  mean=%.5f  max=%.5f",
             result["VolStress"].min(), result["VolStress"].mean(), result["VolStress"].max())

    # ------------------------------------------------------------------
    # 2. Tail co-exceedance breadth (rolling 252-day quantile thresholds)
    #    Threshold at day t is based on [t-252, t-1] -- strictly backward.
    # ------------------------------------------------------------------
    log.info("  Computing rolling tail quantile thresholds (window=%d, min_periods=%d)...",
             TAIL_WINDOW, TAIL_MIN_PERIODS)

    # Shift by 1: quantile up to yesterday, compared against today's return
    q95_abs = (
        abs_w.rolling(TAIL_WINDOW, min_periods=TAIL_MIN_PERIODS)
             .quantile(0.95)
             .shift(1)
    )
    q05_ret = (
        ret_w.rolling(TAIL_WINDOW, min_periods=TAIL_MIN_PERIODS)
             .quantile(0.05)
             .shift(1)
    )
    q95_ret = (
        ret_w.rolling(TAIL_WINDOW, min_periods=TAIL_MIN_PERIODS)
             .quantile(0.95)
             .shift(1)
    )

    # Count countries exceeding threshold on each day
    # Only count on days where threshold is available (not NaN)
    result["TailVolBreadth"]  = (abs_w > q95_abs).sum(axis=1).where(q95_abs.notna().any(axis=1))
    result["TailLossBreadth"] = (ret_w  < q05_ret).sum(axis=1).where(q05_ret.notna().any(axis=1))
    result["TailGainBreadth"] = (ret_w  > q95_ret).sum(axis=1).where(q95_ret.notna().any(axis=1))

    n_nan_tail = result["TailVolBreadth"].isna().sum()
    log.info("  TailVolBreadth: %d NaN dates (warm-up), %d valid",
             n_nan_tail, result["TailVolBreadth"].notna().sum())
    log.info("  TailVolBreadth: mean=%.2f  max=%d",
             result["TailVolBreadth"].mean(), int(result["TailVolBreadth"].max()))

    # ------------------------------------------------------------------
    # 3. Rolling average pairwise correlation
    # ------------------------------------------------------------------
    log.info("  Computing rolling average pairwise correlation (window=%d)...", CORR_WINDOW_60)
    result["AvgCorr60"] = _rolling_avg_corr(ret_w, CORR_WINDOW_60, CORR_MIN_PERIODS_60)
    log.info("  AvgCorr60: mean=%.3f  min=%.3f  max=%.3f",
             result["AvgCorr60"].mean(), result["AvgCorr60"].min(), result["AvgCorr60"].max())

    log.info("  Computing rolling average pairwise correlation (window=%d)...", CORR_WINDOW_30)
    result["AvgCorr30"] = _rolling_avg_corr(ret_w, CORR_WINDOW_30, CORR_MIN_PERIODS_30)
    log.info("  AvgCorr30: mean=%.3f  min=%.3f  max=%.3f",
             result["AvgCorr30"].mean(), result["AvgCorr30"].min(), result["AvgCorr30"].max())

    result = result.reset_index().rename(columns={"index": "date"})
    result["date"] = pd.to_datetime(result["date"])
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_fragility_timeline(
    fragility: pd.DataFrame,
    agg_shocks: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Plot all fragility indicators over time with major shock events overlaid.

    Produces a 4-panel figure:
      1. VolStress (+ SqStress on secondary axis)
      2. TailVolBreadth and TailLossBreadth
      3. AvgCorr60 (with AvgCorr30 as lighter line)
      4. Max shock intensity (MaxShock_t)
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    dates   = pd.to_datetime(fragility["date"])
    shock_dates = pd.to_datetime(agg_shocks.loc[agg_shocks["max_shock"] > 0, "date"])

    # Shared: shade major event periods
    major_events = {
        "COVID crash": ("2020-02-20", "2020-04-15"),
        "Ukraine invasion": ("2022-02-24", "2022-03-15"),
        "Hamas-Israel": ("2023-10-07", "2023-10-20"),
    }

    def _shade_events(ax):
        for label, (start, end) in major_events.items():
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.12, color="red", zorder=0)

    # Panel 1: VolStress
    ax = axes[0]
    ax.plot(dates, fragility["VolStress"], color="#1f77b4", lw=0.8, label="VolStress")
    ax.set_ylabel("VolStress", fontsize=9)
    ax.set_title("Average absolute return (VolStress)", fontsize=9)
    _shade_events(ax)
    ax.legend(fontsize=7, loc="upper left")

    # Panel 2: Tail Breadth
    ax = axes[1]
    ax.plot(dates, fragility["TailVolBreadth"],  color="#2ca02c", lw=0.8, label="TailVolBreadth")
    ax.plot(dates, fragility["TailLossBreadth"], color="#d62728", lw=0.8, alpha=0.7, label="TailLossBreadth")
    ax.set_ylabel("Count (0-19)", fontsize=9)
    ax.set_title("Tail co-exceedance breadth (TailVolBreadth, TailLossBreadth)", fontsize=9)
    ax.set_ylim(-0.5, N_COUNTRIES + 0.5)
    _shade_events(ax)
    ax.legend(fontsize=7, loc="upper left")

    # Panel 3: Average correlation
    ax = axes[2]
    ax.plot(dates, fragility["AvgCorr60"], color="#9467bd", lw=0.9, label="AvgCorr60")
    ax.plot(dates, fragility["AvgCorr30"], color="#c5b0d5", lw=0.7, alpha=0.8, label="AvgCorr30")
    ax.set_ylabel("Correlation", fontsize=9)
    ax.set_title("Rolling average pairwise correlation", fontsize=9)
    ax.set_ylim(-0.1, 1.0)
    _shade_events(ax)
    ax.legend(fontsize=7, loc="upper left")

    # Panel 4: Max shock intensity
    ax = axes[3]
    ax.bar(pd.to_datetime(agg_shocks["date"]), agg_shocks["max_shock"],
           color="#ff7f0e", width=1, alpha=0.7, label="MaxShock")
    ax.set_ylabel("S = -log(q)", fontsize=9)
    ax.set_title("Daily maximum shock intensity (MaxShock)", fontsize=9)
    _shade_events(ax)
    ax.legend(fontsize=7, loc="upper left")

    # x-axis formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=0)

    # Red shading legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="red", alpha=0.12, label="Major event period")]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=7)

    plt.suptitle("European Market Fragility Indicators (2017-2025)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== SP02 Daily Fragility Indicators ===")
    t0 = datetime.now()

    if not IN_PANEL.exists():
        raise FileNotFoundError(
            f"Panel not found: {IN_PANEL}\n"
            "Run subprojects/01_data_preparation/prepare_data.py first."
        )

    log.info("Loading panel from %s", IN_PANEL)
    panel = pd.read_parquet(IN_PANEL)
    log.info("  Panel shape: %s", panel.shape)

    log.info("Loading aggregate shocks from %s", IN_AGG_SHOCKS)
    agg_shocks = pd.read_csv(IN_AGG_SHOCKS, parse_dates=["date"])

    log.info("Building fragility indicators...")
    fragility = build_fragility_indicators(panel)
    log.info("Fragility DataFrame shape: %s", fragility.shape)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Writing fragility_daily.csv (%d rows)...", len(fragility))
    fragility.to_csv(OUT_FRAGILITY, index=False)

    log.info("Computing summary statistics...")
    indicator_cols = ["VolStress", "SqStress",
                      "TailVolBreadth", "TailLossBreadth", "TailGainBreadth",
                      "AvgCorr60", "AvgCorr30"]
    stats = fragility[indicator_cols].describe().T
    stats["skew"]     = fragility[indicator_cols].skew()
    stats["kurtosis"] = fragility[indicator_cols].kurt()
    stats.to_csv(OUT_STATS)
    log.info("Summary stats written to %s", OUT_STATS)

    log.info("Plotting fragility timeline...")
    plot_fragility_timeline(fragility, agg_shocks, OUT_FIGURE)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(THIS_FILE.relative_to(GFJ_ROOT)),
        "n_dates": len(fragility),
        "date_range": {
            "start": str(pd.to_datetime(fragility["date"]).min().date()),
            "end": str(pd.to_datetime(fragility["date"]).max().date()),
        },
        "indicators": indicator_cols,
        "parameters": {
            "tail_window": TAIL_WINDOW,
            "tail_min_periods": TAIL_MIN_PERIODS,
            "tail_quantile_shift": 1,
            "corr_window_60": CORR_WINDOW_60,
            "corr_window_30": CORR_WINDOW_30,
        },
        "outputs": {
            "fragility_daily": str(OUT_FRAGILITY),
            "summary_stats": str(OUT_STATS),
            "figure": str(OUT_FIGURE),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=== SP02 complete in %.1fs ===", elapsed)
    log.info("Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
