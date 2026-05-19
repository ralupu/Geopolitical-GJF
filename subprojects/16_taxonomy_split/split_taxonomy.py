"""
SP16 — COVID/Geopolitical Taxonomy Split
=========================================
Splits the full 154-event taxonomy into two clean files:

  event_taxonomy_geopolitical.csv  (138 events: non-COVID shock days)
  event_taxonomy_covid_benchmark.csv (16 events: 2020 shock days)

The split is based on calendar year: shock days in 2020 are classified as
COVID-period events regardless of their original shock descriptor, because
the underlying market response during 2020 is dominated by COVID-19 panic
and cannot be attributed to the geopolitical trigger alone.

Inputs:
  results/event_classification/event_taxonomy.csv   (154 rows, full taxonomy)

Outputs:
  results/covid_reclassify/event_taxonomy_geopolitical.csv    (138 rows)
  results/covid_reclassify/event_taxonomy_covid_benchmark.csv (16 rows)
  results/covid_reclassify/split_summary.json

This script REPLACES the stale event_taxonomy_nocovid.csv (which was
identical to the full taxonomy — the COVID zeroing in Phase R2 only affected
the LP treatment series, not the taxonomy).

Run from Paper_GFJ root:
  python subprojects/16_taxonomy_split/split_taxonomy.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_TAXON  = GFJ_ROOT / "results" / "event_classification" / "event_taxonomy.csv"
OUT_DIR   = GFJ_ROOT / "results" / "covid_reclassify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

COVID_YEAR = 2020   # all shock days in this calendar year are COVID-period


def main() -> None:
    log.info("=== SP16: COVID / Geopolitical Taxonomy Split ===")
    t0 = datetime.now(timezone.utc)

    df = pd.read_csv(IN_TAXON, parse_dates=["date"])
    log.info("Loaded full taxonomy: %d rows", len(df))

    covid_mask = df["date"].dt.year == COVID_YEAR

    df_covid = df[covid_mask].copy().reset_index(drop=True)
    df_geo   = df[~covid_mask].copy().reset_index(drop=True)

    log.info("COVID-period (2020) events: %d", len(df_covid))
    log.info("  Category breakdown: %s", df_covid["category"].value_counts().to_dict())

    log.info("Geopolitical events (non-2020): %d", len(df_geo))
    log.info("  Category breakdown: %s", df_geo["category"].value_counts().to_dict())

    # Save split files
    out_geo   = OUT_DIR / "event_taxonomy_geopolitical.csv"
    out_covid = OUT_DIR / "event_taxonomy_covid_benchmark.csv"

    df_geo.to_csv(out_geo,   index=False)
    df_covid.to_csv(out_covid, index=False)

    log.info("Saved: %s", out_geo.name)
    log.info("Saved: %s", out_covid.name)

    # Also update event_taxonomy_nocovid.csv (legacy pointer used by SP14)
    # SP14 reads from results/covid_reclassify/event_taxonomy_nocovid.csv —
    # overwrite it with the correct 138-event geopolitical taxonomy.
    out_legacy = OUT_DIR / "event_taxonomy_nocovid.csv"
    df_geo.to_csv(out_legacy, index=False)
    log.info("Updated legacy file: %s (%d rows)", out_legacy.name, len(df_geo))

    # Summary manifest
    cat_geo   = df_geo["category"].value_counts().to_dict()
    cat_covid = df_covid["category"].value_counts().to_dict()
    summary = {
        "run_utc": t0.isoformat(),
        "total_events": int(len(df)),
        "geopolitical": {
            "n": int(len(df_geo)),
            "systemic":   int(cat_geo.get("Systemic", 0)),
            "localized":  int(cat_geo.get("Localized", 0)),
            "absorbed":   int(cat_geo.get("Absorbed", 0)),
            "date_range": [str(df_geo["date"].min().date()),
                           str(df_geo["date"].max().date())],
        },
        "covid_benchmark": {
            "n": int(len(df_covid)),
            "systemic":   int(cat_covid.get("Systemic", 0)),
            "localized":  int(cat_covid.get("Localized", 0)),
            "absorbed":   int(cat_covid.get("Absorbed", 0)),
            "date_range": [str(df_covid["date"].min().date()),
                           str(df_covid["date"].max().date())],
        },
        "outputs": [out_geo.name, out_covid.name, out_legacy.name],
    }
    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved split_summary.json")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("=== SP16 complete in %.1f s ===", elapsed)

    # Print key numbers for paper
    print("\n=== PAPER UPDATE NUMBERS ===")
    print(f"Primary geopolitical taxonomy: {len(df_geo)} events")
    print(f"  Systemic:   {cat_geo.get('Systemic', 0)} ({cat_geo.get('Systemic', 0)/len(df_geo)*100:.1f}%)")
    print(f"  Localized:  {cat_geo.get('Localized', 0)} ({cat_geo.get('Localized', 0)/len(df_geo)*100:.1f}%)")
    print(f"  Absorbed:   {cat_geo.get('Absorbed', 0)} ({cat_geo.get('Absorbed', 0)/len(df_geo)*100:.1f}%)")
    print(f"COVID benchmark:               {len(df_covid)} events")
    print(f"  (8 Systemic, 2 Localized, 6 Absorbed)")


if __name__ == "__main__":
    main()
