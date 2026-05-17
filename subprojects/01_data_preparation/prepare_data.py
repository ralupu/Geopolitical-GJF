"""
SP01 -- Data Preparation
========================
Builds the unified analysis-ready data artefacts for Paper_GFJ from the
parent GeopoliticalRisk repository.

Outputs (all relative to Paper_GFJ root):
  data/panel_daily.parquet      -- long-format return panel (date x country)
  data/shocks_events.csv        -- declustered shock events with S = -log(q)
  data/aggregate_shocks.csv     -- daily aggregate shock measures
  data/clusters.csv             -- country -> cluster mapping
  results/data_provenance/summary.md
  results/data_provenance/manifest.json

Design note -- weekend shock alignment
  The conflict intensity index runs on all calendar days (including weekends).
  Shock events can therefore fall on Saturdays or Sundays when equity markets
  are closed.  We forward-fill every non-trading-day event to the NEXT
  available trading day: economically, a shock detected on Saturday is first
  priced on Monday's open.  If two events for the same country land on the
  same next trading day after alignment (rare given the 10-day declustering
  gap), we keep the higher shock intensity.

Run from Paper_GFJ root:
  python subprojects/01_data_preparation/prepare_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]   # Paper_GFJ/
PARENT    = GFJ_ROOT.parent        # GeopoliticalRisk/ (parent repo)

SRC_PRICES   = PARENT / "data" / "stock_prices.xlsx"
SRC_CLUSTERS = PARENT / "data" / "clusters_input.csv"
SRC_CONFLICT = (
    PARENT / "data" / "gdelt_indices" / "conflict_intensity"
    / "20251222_paper_candidate" / "conflict_intensity_index_wide.csv"
)
_SHOCK_PREFIX = (
    "sp2_conflicts_diff_per_country_evt_fdr_q0.05_u0.95_gap10"
    "_eventsbyp0.005_paper_rawcov_v1"
)
SRC_EVENTS      = PARENT / "results" / "shocks" / f"{_SHOCK_PREFIX}_events.csv"
SRC_SHOCKS_LONG = PARENT / "results" / "shocks" / f"{_SHOCK_PREFIX}_shocks_long.csv"
SRC_MANIFEST    = PARENT / "results" / "shocks" / f"{_SHOCK_PREFIX}_manifest.json"

OUT_DATA       = GFJ_ROOT / "data"
OUT_PROVENANCE = GFJ_ROOT / "results" / "data_provenance"
OUT_PANEL      = OUT_DATA / "panel_daily.parquet"
OUT_EVENTS     = OUT_DATA / "shocks_events.csv"
OUT_AGG_SHOCKS = OUT_DATA / "aggregate_shocks.csv"
OUT_CLUSTERS   = OUT_DATA / "clusters.csv"
OUT_SUMMARY    = OUT_PROVENANCE / "summary.md"
OUT_MANIFEST   = OUT_PROVENANCE / "manifest.json"

STUDY_START = pd.Timestamp("2017-01-02")
STUDY_END   = pd.Timestamp("2025-10-15")
N_COUNTRIES = 19
N_EVENTS_RAW = 278  # known fixed count from BIR paper shock construction

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
# Helpers
# ---------------------------------------------------------------------------

def check_sources() -> None:
    """Raise FileNotFoundError if any required source file is missing."""
    required = {
        "stock_prices.xlsx": SRC_PRICES,
        "clusters_input.csv": SRC_CLUSTERS,
        "conflict_intensity_index_wide.csv": SRC_CONFLICT,
        "events.csv": SRC_EVENTS,
        "shocks_long.csv": SRC_SHOCKS_LONG,
    }
    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing source files: {missing}\n"
            f"Expected parent repo at: {PARENT}"
        )
    log.info("All source files present.")


def load_returns() -> pd.DataFrame:
    """
    Load daily equity prices and compute log returns.

    Returns wide-format DataFrame: index=date, columns=countries.
    First return date is 2017-01-02 (log-diff drops 2016-12-30).
    """
    log.info("Loading equity prices from %s", SRC_PRICES)
    prices = pd.read_excel(SRC_PRICES, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    log.info("  Prices shape: %s, range: %s to %s",
             prices.shape, prices.index.min().date(), prices.index.max().date())
    returns = np.log(prices).diff().dropna(how="all")
    log.info("  Returns shape: %s, range: %s to %s",
             returns.shape, returns.index.min().date(), returns.index.max().date())
    return returns


def load_conflict_dates() -> set:
    """Return the set of calendar dates present in the conflict index."""
    log.info("Loading conflict index date coverage from %s", SRC_CONFLICT)
    conflict = pd.read_csv(SRC_CONFLICT, index_col=0, parse_dates=True, usecols=[0])
    return set(conflict.index)


def load_clusters() -> pd.DataFrame:
    """Load country cluster assignments. Returns DataFrame[country, cluster]."""
    log.info("Loading cluster assignments from %s", SRC_CLUSTERS)
    clusters = pd.read_csv(SRC_CLUSTERS, encoding="utf-8-sig")
    clusters.columns = clusters.columns.str.strip().str.lower()
    assert set(clusters.columns) == {"country", "cluster"}, \
        f"Unexpected columns: {list(clusters.columns)}"
    assert len(clusters) == N_COUNTRIES, \
        f"Expected {N_COUNTRIES} countries, got {len(clusters)}"
    log.info("  %d countries loaded.", len(clusters))
    return clusters


def load_events() -> pd.DataFrame:
    """
    Load declustered shock events and compute S = -log(q_value).

    Returns DataFrame with columns:
      date, country, shock_size, p_value, q_value, threshold_u, shock_intensity
    Dates are still on the original calendar day (incl. weekends).
    """
    log.info("Loading shock events from %s", SRC_EVENTS)
    events = pd.read_csv(SRC_EVENTS, parse_dates=["Date"])
    events.columns = events.columns.str.strip()
    events = events.rename(columns={
        "Date": "date",
        "Country": "country",
        "ShockSize": "shock_size",
        "p_value": "p_value",
        "q_value": "q_value",
        "threshold_u": "threshold_u",
    })
    events["shock_intensity"] = -np.log(events["q_value"])

    assert (events["q_value"] > 0).all(), "q_value must be positive for log"
    assert (events["shock_intensity"] >= 0).all(), "S must be non-negative"
    assert not events.duplicated(["date", "country"]).any(), \
        "Duplicate (date, country) pairs in raw events"

    log.info("  %d declustered shock events loaded.", len(events))
    log.info("  S range: %.4f to %.4f (mean %.4f)",
             events["shock_intensity"].min(),
             events["shock_intensity"].max(),
             events["shock_intensity"].mean())
    return events


def build_study_period(returns: pd.DataFrame, conflict_dates: set) -> pd.DatetimeIndex:
    """
    Return trading days present in BOTH returns and conflict index,
    within [STUDY_START, STUDY_END].
    """
    ret_dates = set(returns.index)
    common = sorted(
        d for d in (ret_dates & conflict_dates)
        if STUDY_START <= d <= STUDY_END
    )
    study = pd.DatetimeIndex(common)

    dropped_in_window = [
        d for d in (ret_dates - conflict_dates)
        if STUDY_START <= d <= STUDY_END
    ]
    log.info("Study period: %s to %s (%d days)",
             study.min().date(), study.max().date(), len(study))
    log.info("  Return dates dropped (not in conflict index): %d", len(dropped_in_window))
    if dropped_in_window:
        log.info("  Dropped: %s", [str(d.date()) for d in sorted(dropped_in_window)])
    return study


def align_events_to_trading_days(
    events: pd.DataFrame,
    study_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Forward-fill weekend/holiday shock events to the next trading day.

    Rules:
      1. Trading-day events: kept as-is.
      2. Non-trading-day events: shifted to the next trading day in study_dates.
      3. Events shifted to the same (trading_date, country): keep max S.
      4. Events with no next trading day in study_dates: dropped.

    Adds 'original_date' column recording the raw calendar date.
    """
    trading_set  = set(study_dates)
    sorted_dates = sorted(study_dates)

    def _next_trading(d):
        for t in sorted_dates:
            if t > d:
                return t
        return None

    events = events.copy()
    events["original_date"] = events["date"]

    mask_off = ~events["date"].isin(trading_set)
    n_off = int(mask_off.sum())
    if n_off:
        events.loc[mask_off, "date"] = events.loc[mask_off, "date"].apply(_next_trading)
        log.info("  Forward-filled %d weekend/holiday events to next trading day.", n_off)

    n_before = len(events)
    events = events.dropna(subset=["date"])
    events = events[events["date"].isin(trading_set)].copy()
    n_dropped = n_before - len(events)
    if n_dropped:
        log.info("  Dropped %d events (no trading day within study period).", n_dropped)

    # Resolve collisions: same (trading_date, country) -> keep highest S
    events = (
        events
        .sort_values("shock_intensity", ascending=False)
        .drop_duplicates(subset=["date", "country"], keep="first")
        .sort_values(["date", "country"])
        .reset_index(drop=True)
    )

    n_on  = int((events["original_date"] == events["date"]).sum())
    n_fwd = int((events["original_date"] != events["date"]).sum())
    log.info(
        "  Events after alignment: %d total (%d on original trading day, %d forward-filled).",
        len(events), n_on, n_fwd,
    )
    return events


def build_long_panel(
    returns: pd.DataFrame,
    study_dates: pd.DatetimeIndex,
    clusters: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the long-format daily panel.

    Columns: date, country, return, abs_return, sq_return,
             shock_intensity, cluster

    Shape: len(study_dates) x N_COUNTRIES rows.
    """
    log.info("Building long-format panel...")

    rets = returns.loc[returns.index.isin(study_dates)].copy()
    assert len(rets) == len(study_dates)

    rets.index.name = "date"
    panel = (
        rets.reset_index()
        .melt(id_vars="date", var_name="country", value_name="return")
    )
    panel["abs_return"] = panel["return"].abs()
    panel["sq_return"]  = panel["return"] ** 2

    cluster_map = clusters.set_index("country")["cluster"].to_dict()
    panel["cluster"] = panel["country"].map(cluster_map)
    if panel["cluster"].isna().any():
        bad = panel.loc[panel["cluster"].isna(), "country"].unique()
        raise ValueError(f"Countries without cluster: {bad}")

    shock_lookup = events[["date", "country", "shock_intensity"]].copy()
    panel = panel.merge(shock_lookup, on=["date", "country"], how="left")
    panel["shock_intensity"] = panel["shock_intensity"].fillna(0.0)

    panel = panel.sort_values(["date", "country"]).reset_index(drop=True)
    panel["cluster"] = panel["cluster"].astype(int)

    expected = len(study_dates) * N_COUNTRIES
    assert len(panel) == expected, f"Expected {expected} rows, got {len(panel)}"
    assert panel["return"].notna().all()
    assert (panel["shock_intensity"] >= 0).all()
    assert panel["cluster"].between(1, 3).all()

    n_shock = int((panel["shock_intensity"] > 0).sum())
    log.info("  Panel shape: %s", panel.shape)
    log.info("  Shock-event rows: %d (%.2f%% of panel)",
             n_shock, 100.0 * n_shock / len(panel))
    return panel


def build_aggregate_shocks(
    panel: pd.DataFrame,
    study_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Daily aggregate shock measures across 19 markets.

    Columns: date, max_shock, sum_shock, breadth_shock, avg_shock_pos
    """
    log.info("Building aggregate shock series...")

    agg = (
        panel.groupby("date")["shock_intensity"]
        .agg(
            max_shock="max",
            sum_shock="sum",
            breadth_shock=lambda x: int((x > 0).sum()),
        )
        .reset_index()
    )

    pos = panel[panel["shock_intensity"] > 0]
    avg_pos = pos.groupby("date")["shock_intensity"].mean().rename("avg_shock_pos")
    agg = agg.merge(avg_pos, on="date", how="left")
    agg["avg_shock_pos"] = agg["avg_shock_pos"].fillna(0.0)

    full = pd.DataFrame({"date": study_dates})
    agg = full.merge(agg, on="date", how="left").fillna(0.0)
    agg = agg.sort_values("date").reset_index(drop=True)

    n_shock_days = int((agg["max_shock"] > 0).sum())
    log.info("  Aggregate shock rows: %d", len(agg))
    log.info("  Days with any shock: %d", n_shock_days)
    log.info("  max_shock range: %.4f to %.4f",
             agg["max_shock"].min(), agg["max_shock"].max())
    return agg


def write_provenance_summary(
    panel: pd.DataFrame,
    agg: pd.DataFrame,
    events: pd.DataFrame,
    study_dates: pd.DatetimeIndex,
    dropped_dates_count: int,
) -> None:
    """Write summary.md and manifest.json to results/data_provenance/."""
    OUT_PROVENANCE.mkdir(parents=True, exist_ok=True)

    n_shock_days   = int((agg["max_shock"] > 0).sum())
    countries      = sorted(panel["country"].unique().tolist())
    cluster_counts = (
        panel.drop_duplicates("country")
        .groupby("cluster")["country"].count().to_dict()
    )
    n_on  = int((events["original_date"] == events["date"]).sum())
    n_fwd = int((events["original_date"] != events["date"]).sum())

    summary = (
        "# Data Provenance Summary -- SP01\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "## Study sample\n"
        f"- **Countries:** {N_COUNTRIES} European equity markets\n"
        f"- **Study period:** {study_dates.min().date()} to {study_dates.max().date()}\n"
        f"- **Trading days:** {len(study_dates)}\n"
        f"- **Dropped return dates** (no conflict index coverage): {dropped_dates_count}\n"
        f"- **Total panel rows:** {len(panel):,} "
        f"({N_COUNTRIES} countries x {len(study_dates)} days)\n\n"
        "## Countries and clusters\n"
        + "\n".join(
            f"  - Cluster {c}: {cluster_counts.get(c, '?')} countries"
            for c in sorted(cluster_counts)
        ) + "\n\n"
        f"Full list: {', '.join(countries)}\n\n"
        "## Return statistics\n"
        "| Statistic | Value |\n"
        "|-----------|-------|\n"
        f"| Mean return | {panel['return'].mean():.6f} |\n"
        f"| Std return  | {panel['return'].std():.6f} |\n"
        f"| Min return  | {panel['return'].min():.6f} |\n"
        f"| Max return  | {panel['return'].max():.6f} |\n"
        f"| NaN returns | {panel['return'].isna().sum()} |\n\n"
        "## Shock series (declustered EVT+FDR)\n"
        f"- **Source vintage:** 20251222_paper_candidate (locked -- same as BIR paper)\n"
        f"- **Raw events (before alignment):** {N_EVENTS_RAW}\n"
        f"- **Events on original trading day:** {n_on}\n"
        f"- **Events forward-filled from weekend/holiday:** {n_fwd}\n"
        f"- **Events after alignment (collisions removed):** {len(events)}\n"
        f"- **Days with >=1 country shocked:** {n_shock_days}\n"
        f"- **S = -log(q) range:** "
        f"{events['shock_intensity'].min():.4f} to {events['shock_intensity'].max():.4f}\n\n"
        "## Aggregate shock measures (daily)\n"
        "| Measure | Days > 0 | Max | Mean (excl. 0) |\n"
        "|---------|----------|-----|----------------|\n"
        f"| max_shock | {(agg['max_shock']>0).sum()} "
        f"| {agg['max_shock'].max():.4f} "
        f"| {agg.loc[agg['max_shock']>0,'max_shock'].mean():.4f} |\n"
        f"| breadth_shock | {(agg['breadth_shock']>0).sum()} "
        f"| {int(agg['breadth_shock'].max())} "
        f"| {agg.loc[agg['breadth_shock']>0,'breadth_shock'].mean():.2f} |\n"
    )

    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    log.info("Provenance summary written to %s", OUT_SUMMARY)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(THIS_FILE.relative_to(GFJ_ROOT)),
        "study_period": {
            "start": str(study_dates.min().date()),
            "end": str(study_dates.max().date()),
            "n_days": len(study_dates),
            "n_countries": N_COUNTRIES,
            "n_panel_rows": len(panel),
        },
        "shock_series": {
            "n_events_raw": N_EVENTS_RAW,
            "n_events": len(events),
            "n_events_on_trading_day": n_on,
            "n_events_forward_filled": n_fwd,
            "n_shock_days": n_shock_days,
            "s_min": float(events["shock_intensity"].min()),
            "s_max": float(events["shock_intensity"].max()),
            "s_mean": float(events["shock_intensity"].mean()),
        },
        "sources": {
            "prices": str(SRC_PRICES),
            "clusters": str(SRC_CLUSTERS),
            "conflict_index": str(SRC_CONFLICT),
            "events": str(SRC_EVENTS),
            "shocks_long": str(SRC_SHOCKS_LONG),
            "upstream_manifest": str(SRC_MANIFEST),
        },
        "outputs": {
            "panel_daily": str(OUT_PANEL),
            "shocks_events": str(OUT_EVENTS),
            "aggregate_shocks": str(OUT_AGG_SHOCKS),
            "clusters": str(OUT_CLUSTERS),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Manifest written to %s", OUT_MANIFEST)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== SP01 Data Preparation ===")
    t0 = datetime.now()

    check_sources()

    returns        = load_returns()
    conflict_dates = load_conflict_dates()
    clusters       = load_clusters()
    events         = load_events()

    study_dates    = build_study_period(returns, conflict_dates)
    dropped_count  = len([
        d for d in set(returns.index) - conflict_dates
        if STUDY_START <= d <= STUDY_END
    ])

    events         = align_events_to_trading_days(events, study_dates)
    panel          = build_long_panel(returns, study_dates, clusters, events)
    agg_shocks     = build_aggregate_shocks(panel, study_dates)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_PROVENANCE.mkdir(parents=True, exist_ok=True)

    log.info("Writing panel_daily.parquet (%d rows)...", len(panel))
    panel.to_parquet(OUT_PANEL, index=False, engine="pyarrow")

    log.info("Writing shocks_events.csv (%d rows)...", len(events))
    events.to_csv(OUT_EVENTS, index=False)

    log.info("Writing aggregate_shocks.csv (%d rows)...", len(agg_shocks))
    agg_shocks.to_csv(OUT_AGG_SHOCKS, index=False)

    log.info("Writing clusters.csv (%d rows)...", len(clusters))
    clusters.to_csv(OUT_CLUSTERS, index=False)

    write_provenance_summary(panel, agg_shocks, events, study_dates, dropped_count)

    elapsed = (datetime.now() - t0).total_seconds()
    log.info("=== SP01 complete in %.1fs ===", elapsed)
    log.info("Outputs written to %s", OUT_DATA)


if __name__ == "__main__":
    main()
