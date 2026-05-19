"""
SP15 -- Figure Improvements (R7)
==================================
Redesigns two figures based on reviewer feedback:

R7.1 — Fig_NetworkCorr_Systemic: replace hairball network with a clean
        pre/post correlation heatmap for the top-5 systemic events.
        Two panels per event: pre-shock (t-5..t-1) and post-shock (t..t+5).
        Combined into a single 5-row figure with a shared colour bar.

R7.2 — Fig_StateLPSmooth_Combined: regenerate the state-LP combined figure
        with improved colour contrast, thicker IRF lines, and explicit
        confidence band labels. (Re-runs SP07 plot logic with style tweaks.)

R7.3 — ROC figure already placed in paper by R6 (no action needed).

Inputs (relative to Paper_GFJ root):
  data/panel_daily.parquet
  results/event_classification/event_taxonomy.csv
  results/state_lp_nocovid/state_lp_smooth_{outcome}.csv

Outputs (relative to Paper_GFJ root):
  Paper_LaTeX/Fig_NetworkCorr_Systemic.png   -- replaces existing
  Paper_LaTeX/Fig_StateLPSmooth_Combined.png -- replaces existing
  results/figure_improvements/manifest.json

Run from Paper_GFJ root:
  python subprojects/15_figure_improvements/run_figure_improvements.py
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_PANEL  = GFJ_ROOT / "data"    / "panel_daily.parquet"
IN_TAXON  = GFJ_ROOT / "results" / "event_classification" / "event_taxonomy.csv"
IN_SLP    = GFJ_ROOT / "results" / "state_lp_nocovid"

LATEX_DIR = GFJ_ROOT / "Paper_LaTeX"
OUT_DIR   = GFJ_ROOT / "results" / "figure_improvements"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
# Country abbreviations for heatmap axes
# ---------------------------------------------------------------------------
COUNTRY_ISO = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR",
    "Hungary": "HU", "Ireland": "IE", "Italy": "IT", "Netherlands": "NL",
    "Norway": "NO", "Poland": "PL", "Portugal": "PT", "Romania": "RO",
    "Spain": "ES", "Sweden": "SE", "United Kingdom": "UK",
}

# ---------------------------------------------------------------------------
# R7.1 — Network correlation redesign (pre/post heatmaps)
# ---------------------------------------------------------------------------

def build_corr_matrix(ret_wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Correlation matrix from returns on the given dates."""
    sub = ret_wide.loc[ret_wide.index.isin(dates)]
    if len(sub) < 3:
        return pd.DataFrame()
    return sub.corr()


def get_window_dates(panel_dates: pd.DatetimeIndex,
                     event_date: pd.Timestamp,
                     pre: int = 5, post: int = 5) -> tuple:
    all_dates = panel_dates.sort_values()
    idx = all_dates.searchsorted(event_date)
    pre_dates  = all_dates[max(0, idx - pre): idx]
    post_dates = all_dates[idx: idx + post + 1]
    return pre_dates, post_dates


def generate_network_heatmap() -> Path:
    """
    Replace the hairball network with clean correlation heatmaps.
    5 rows (top-5 systemic events) x 2 columns (pre / post).
    """
    log.info("R7.1 — building network heatmap ...")

    panel  = pd.read_parquet(IN_PANEL)
    taxon  = pd.read_csv(IN_TAXON, parse_dates=["date"])

    ret_wide = panel.pivot(index="date", columns="country", values="return")
    ret_wide.index = pd.to_datetime(ret_wide.index)

    # ISO codes for axes
    countries = [c for c in ret_wide.columns if c in COUNTRY_ISO]
    ret_wide  = ret_wide[countries]
    iso_labels = [COUNTRY_ISO[c] for c in countries]

    systemic = (taxon[taxon["category"] == "Systemic"]
                .nlargest(5, "emfi_post_max")
                .reset_index(drop=True))

    # Event labels
    event_labels = []
    for _, row in systemic.iterrows():
        dt = pd.Timestamp(row["date"])
        event_labels.append(dt.strftime("%d %b %Y"))

    n_events = len(systemic)
    fig = plt.figure(figsize=(14, 4.0 * n_events))
    gs  = gridspec.GridSpec(n_events, 2, figure=fig,
                            hspace=0.35, wspace=0.08)

    # Shared colour scale across all events
    vmin, vmax = -1.0, 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = "RdYlGn"

    im_ref = None
    for row_i, ev_row in systemic.iterrows():
        event_date = pd.Timestamp(ev_row["date"])
        pre_dates, post_dates = get_window_dates(
            ret_wide.index, event_date, pre=5, post=5
        )

        corr_pre  = build_corr_matrix(ret_wide, pre_dates)
        corr_post = build_corr_matrix(ret_wide, post_dates)

        for col_i, (corr, label) in enumerate([
            (corr_pre,  "Pre-shock ($t{-}5$ to $t{-}1$)"),
            (corr_post, "Post-shock ($t$ to $t{+}5$)"),
        ]):
            ax = fig.add_subplot(gs[row_i, col_i])
            if corr.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                # Align columns/rows to full country list
                corr = corr.reindex(index=countries, columns=countries)
                im = ax.imshow(corr.values, cmap=cmap, norm=norm,
                               aspect="auto", interpolation="nearest")
                im_ref = im
                ax.set_xticks(range(len(iso_labels)))
                ax.set_yticks(range(len(iso_labels)))
                ax.set_xticklabels(iso_labels, fontsize=6, rotation=90)
                ax.set_yticklabels(iso_labels, fontsize=6)

                # Mean correlation annotation
                mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
                mean_corr = corr.values[mask].mean() if corr.notna().all().all() else np.nan
                ax.set_title(f"{label}\n$\\bar{{\\rho}}={mean_corr:.2f}$",
                             fontsize=8, pad=3)

            if col_i == 0:
                ax.set_ylabel(event_labels[row_i], fontsize=8,
                              fontweight="bold", labelpad=4)

    # Shared colour bar
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
        fig.colorbar(im_ref, cax=cbar_ax, label="Pairwise return correlation")

    fig.suptitle(
        "Equity Return Correlation: Pre- vs Post-Shock Windows\n"
        "Top-5 Systemic Geopolitical Events (ranked by post-event EMFI)",
        fontsize=11, y=0.98,
    )

    out_path = LATEX_DIR / "Fig_NetworkCorr_Systemic.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved -> %s (%.1f KB)", out_path.name,
             out_path.stat().st_size / 1024)
    return out_path


# ---------------------------------------------------------------------------
# R7.2 — Improved State LP figure
# ---------------------------------------------------------------------------

P_COLORS  = {0.0: "#2ca02c", 0.5: "#ff7f0e", 0.9: "#d62728"}
P_LABELS  = {0.0: "$P_{\\mathrm{stress}}=0$ (calm)",
             0.5: "$P_{\\mathrm{stress}}=0.5$ (elevated)",
             0.9: "$P_{\\mathrm{stress}}=0.9$ (near-systemic)"}
HORIZONS  = list(range(-5, 16))
OUTCOMES  = ["EMFI", "tci_w100", "TailVolBreadth", "P_stress", "AvgCorr60"]
OUT_TITLES = {
    "EMFI":          "EMFI (composite fragility)",
    "tci_w100":      "TCI (total connectedness)",
    "TailVolBreadth":"TailVolBreadth (tail-vol breadth)",
    "P_stress":      "$P_{\\mathrm{stress}}$ (HMM stress prob.)",
    "AvgCorr60":     "AvgCorr60 (60-day avg. correlation)",
}
CI_95 = 1.96


def plot_single_slp(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    horizons = df["horizon"].values
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="grey",  lw=0.6, linestyle=":",  alpha=0.5)

    p_tags = {"0.0": "00", "0.5": "05", "0.9": "09"}
    for p_val in [0.0, 0.5, 0.9]:
        tag = p_tags[str(p_val)]
        irf_col = f"irf_p{tag}"
        se_col  = f"se_irf_p{tag}"
        if irf_col not in df.columns:
            continue
        irf = df[irf_col].values
        se  = df[se_col].values
        color = P_COLORS[p_val]

        ax.plot(horizons, irf, color=color, lw=2.2,
                label=P_LABELS[p_val])
        ax.fill_between(horizons,
                        irf - CI_95 * se,
                        irf + CI_95 * se,
                        color=color, alpha=0.15)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Horizon $k$ (trading days)", fontsize=8)
    ax.set_ylabel("IRF", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(horizons[0], horizons[-1])
    ax.grid(linestyle=":", alpha=0.3)


def generate_state_lp_figure() -> Path:
    """Regenerate Fig_StateLPSmooth_Combined with improved style."""
    log.info("R7.2 — regenerating state LP combined figure ...")

    n = len(OUTCOMES)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.5), sharey=False)

    for ax, outcome in zip(axes, OUTCOMES):
        csv = IN_SLP / f"state_lp_smooth_{outcome}.csv"
        if not csv.exists():
            log.warning("  Missing %s", csv.name)
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        plot_single_slp(ax, df, OUT_TITLES[outcome])

    # Single legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Smooth-Transition LP: Impulse Responses by Pre-Shock Stress State\n"
        "(COVID-zeroed baseline; 95% delta-method confidence bands)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    out_path = LATEX_DIR / "Fig_StateLPSmooth_Combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved -> %s (%.1f KB)", out_path.name,
             out_path.stat().st_size / 1024)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== SP15 Figure Improvements ===")
    t0 = datetime.now(timezone.utc)

    net_path = generate_network_heatmap()
    slp_path = generate_state_lp_figure()

    # Verify file sizes
    for p in [net_path, slp_path]:
        size_kb = p.stat().st_size / 1024
        assert size_kb > 20, f"{p.name}: only {size_kb:.1f} KB (too small)"
        log.info("  %s: %.0f KB OK", p.name, size_kb)

    manifest = {
        "run_utc":   t0.isoformat(),
        "figures": {
            "Fig_NetworkCorr_Systemic":   str(net_path),
            "Fig_StateLPSmooth_Combined": str(slp_path),
        },
        "net_size_kb": round(net_path.stat().st_size / 1024, 1),
        "slp_size_kb": round(slp_path.stat().st_size / 1024, 1),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("SP15 complete in %.1f s", elapsed)


if __name__ == "__main__":
    main()
