"""
SP08 — Geopolitical Event Classification
=========================================
Classifies each of the 154 shock days in the LP sample into one of three
categories based on post-shock market response:

  Systemic  : max(P_stress[t:t+5]) ≥ 0.5  AND  max(EMFI[t:t+5]) > q75_emfi
  Localized : not Systemic AND mean(EMFI[t+1:t+5]) > mean(EMFI[t-5:t-1])
  Absorbed  : neither (EMFI does not rise post-shock)

Also identifies "Market-only stress" episodes: days with P_stress > 0.5 that
are NOT within ±3 trading days of any shock event.

Produces:
  results/event_classification/event_taxonomy.csv
  results/event_classification/summary_by_category.csv
  results/event_classification/market_stress_episodes.csv
  results/event_classification/Fig_EventTimeline.png
  results/event_classification/Fig_NetworkCorr_Systemic.png
  results/event_classification/manifest.json
"""

from __future__ import annotations
import json, logging, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── paths ──────────────────────────────────────────────────────────────────────
GFJ_ROOT  = Path(__file__).resolve().parents[2]
IN_EMFI   = GFJ_ROOT / "results" / "emfi"          / "emfi_daily.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"
IN_TCI    = GFJ_ROOT / "results" / "connectedness"  / "tci_daily.csv"
IN_SHOCKS = GFJ_ROOT / "data"    / "aggregate_shocks.csv"
IN_EVENTS = GFJ_ROOT / "data"    / "shocks_events.csv"
IN_PANEL  = GFJ_ROOT / "data"    / "panel_daily.parquet"
OUTDIR    = GFJ_ROOT / "results" / "event_classification"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── classification thresholds ──────────────────────────────────────────────────
EMFI_Q75        = 0.548   # 75th pctile of EMFI (from Phase 4)
PSTRESS_SYSTEMIC= 0.50    # P_stress peak threshold for Systemic
PRE_WIN         = 5       # trading days before shock for pre-window
POST_WIN        = 5       # trading days after shock for post-window
STRESS_WINDOW   = 5       # [t, t+POST_WIN] for stress peak
NET_WIN         = 20      # trading days for correlation network window
MARKET_ONLY_GAP = 3       # gap (days) from nearest shock to count as market-only

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def window_mean(series: pd.Series, date_idx: pd.DatetimeIndex,
                t: pd.Timestamp, lo: int, hi: int) -> float:
    """Mean of series over trading-day offsets [lo, hi] from t (inclusive).
    lo < 0 means pre-event; hi > 0 means post-event. hi=0 includes t itself."""
    pos = date_idx.get_loc(t)
    sl  = slice(max(0, pos + lo), min(len(series), pos + hi + 1))
    vals = series.iloc[sl]
    return float(vals.mean()) if len(vals) > 0 else np.nan


def window_max(series: pd.Series, date_idx: pd.DatetimeIndex,
               t: pd.Timestamp, lo: int, hi: int) -> float:
    pos = date_idx.get_loc(t)
    sl  = slice(max(0, pos + lo), min(len(series), pos + hi + 1))
    vals = series.iloc[sl]
    return float(vals.max()) if len(vals) > 0 else np.nan


# ── load data ─────────────────────────────────────────────────────────────────

def load_data():
    emfi   = pd.read_csv(IN_EMFI,   parse_dates=["date"]).set_index("date")["EMFI"]
    hmm    = pd.read_csv(IN_HMM,    parse_dates=["date"]).set_index("date")["P_stress"]
    tci    = pd.read_csv(IN_TCI,    parse_dates=["date"]).set_index("date")["tci_w100"]
    shocks = pd.read_csv(IN_SHOCKS, parse_dates=["date"]).set_index("date")
    events = pd.read_csv(IN_EVENTS, parse_dates=["date"])

    # align on common trading calendar (EMFI is the reference)
    idx = emfi.index
    hmm = hmm.reindex(idx).ffill().fillna(0.0)
    tci = tci.reindex(idx).ffill()
    shocks = shocks.reindex(idx).fillna(0.0)

    # shock days: max_shock > 0 restricted to EMFI sample
    shock_days = shocks[shocks["max_shock"] > 0].index
    log.info(f"Shock days in EMFI sample: {len(shock_days)}")

    panel = pd.read_parquet(IN_PANEL)
    return emfi, hmm, tci, shocks, events, shock_days, idx, panel


# ── classify events ───────────────────────────────────────────────────────────

def classify_events(emfi, hmm, tci, shocks, events, shock_days, idx):
    log.info("Classifying shock events…")
    records = []

    # per-shock-day: which countries?
    ev_by_date = events.groupby("date")["country"].apply(list).to_dict()

    for t in shock_days:
        row = {}
        row["date"] = t

        # shock intensity
        row["max_shock"]   = shocks.at[t, "max_shock"]  if t in shocks.index else np.nan
        row["breadth"]     = shocks.at[t, "breadth_shock"] if t in shocks.index else np.nan
        row["countries"]   = "; ".join(sorted(ev_by_date.get(t, [])))
        row["n_countries"] = len(ev_by_date.get(t, []))

        # EMFI windows
        row["emfi_pre_mean"]  = window_mean(emfi, idx, t, -PRE_WIN, -1)
        row["emfi_post_mean"] = window_mean(emfi, idx, t,  1, POST_WIN)
        row["emfi_post_max"]  = window_max (emfi, idx, t,  0, POST_WIN)
        row["emfi_delta"]     = row["emfi_post_mean"] - row["emfi_pre_mean"]

        # P_stress windows
        row["p_stress_on_day"] = float(hmm.at[t]) if t in hmm.index else np.nan
        row["p_stress_peak"]   = window_max(hmm, idx, t, 0, STRESS_WINDOW)

        # TCI windows
        row["tci_pre_mean"]  = window_mean(tci, idx, t, -PRE_WIN, -1)
        row["tci_post_max"]  = window_max (tci, idx, t,  0, POST_WIN)

        # classify
        systemic  = (row["p_stress_peak"] >= PSTRESS_SYSTEMIC and
                     row["emfi_post_max"]  >  EMFI_Q75)
        localized = (not systemic and row["emfi_delta"] > 0)
        row["category"] = ("Systemic"  if systemic else
                           "Localized" if localized else
                           "Absorbed")

        records.append(row)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


# ── market-only stress ────────────────────────────────────────────────────────

def find_market_only_stress(hmm, shock_days, idx):
    """Days with P_stress > 0.5 and no shock within ±MARKET_ONLY_GAP trading days."""
    log.info("Identifying market-only stress episodes…")
    shock_set = set(shock_days)
    results = []
    dates = idx.tolist()
    shock_positions = {d: i for i, d in enumerate(dates) if d in shock_set}

    for i, d in enumerate(dates):
        if hmm.iloc[i] < 0.5:
            continue
        # check gap from nearest shock
        near = False
        for sd, sp in shock_positions.items():
            if abs(i - sp) <= MARKET_ONLY_GAP:
                near = True
                break
        if not near:
            results.append({"date": d, "P_stress": hmm.iloc[i]})

    df = pd.DataFrame(results)
    log.info(f"Market-only stress days: {len(df)}")
    return df


# ── summary by category ───────────────────────────────────────────────────────

def build_summary(taxonomy: pd.DataFrame) -> pd.DataFrame:
    cats = ["Systemic", "Localized", "Absorbed"]
    rows = []
    for cat in cats:
        sub = taxonomy[taxonomy["category"] == cat]
        rows.append({
            "category"        : cat,
            "n_events"        : len(sub),
            "pct_events"      : len(sub) / len(taxonomy) * 100,
            "mean_max_shock"  : sub["max_shock"].mean(),
            "mean_breadth"    : sub["breadth"].mean(),
            "mean_emfi_delta" : sub["emfi_delta"].mean(),
            "mean_p_stress_peak": sub["p_stress_peak"].mean(),
            "mean_tci_post_max" : sub["tci_post_max"].mean(),
        })
    return pd.DataFrame(rows)


# ── timeline figure ───────────────────────────────────────────────────────────

def plot_event_timeline(emfi, hmm, taxonomy, outdir):
    log.info("Plotting event timeline figure…")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1, ax2 = axes
    ax1.plot(emfi.index, emfi.values, color="#2d6a9f", lw=0.8, label="EMFI")
    ax1.axhline(EMFI_Q75, color="grey", lw=0.8, ls="--", label=f"q75={EMFI_Q75:.2f}")

    colors = {"Systemic": "#c0392b", "Localized": "#e67e22", "Absorbed": "#27ae60"}
    markers = {"Systemic": "^", "Localized": "o", "Absorbed": "s"}
    for cat in ["Systemic", "Localized", "Absorbed"]:
        sub = taxonomy[taxonomy["category"] == cat]
        ys  = emfi.reindex(sub["date"]).values
        ax1.scatter(sub["date"], ys, s=30 if cat == "Systemic" else 15,
                    c=colors[cat], marker=markers[cat], zorder=5,
                    label=f"{cat} (n={len(sub)})", alpha=0.85)

    ax1.set_ylabel("EMFI")
    ax1.legend(fontsize=7, ncol=4)
    ax1.set_title("Geopolitical Shock Classification: EMFI with Event Taxonomy")

    ax2.fill_between(hmm.index, 0, hmm.values, color="#8e44ad", alpha=0.35,
                     label="P(Systemic stress)")
    ax2.axhline(0.5, color="grey", lw=0.8, ls="--", label="0.5 threshold")
    ax2.set_ylabel("P(Systemic stress)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fpath = outdir / "Fig_EventTimeline.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {fpath}")
    return fpath


# ── correlation network figure ────────────────────────────────────────────────

def plot_network_figure(taxonomy, panel, outdir):
    """
    For the top systemic events (by P_stress peak), plot country-level
    return correlation networks in the pre- and post-event windows.
    Shows average across top-N systemic events.
    """
    log.info("Building correlation network figure…")

    systemic = (taxonomy[taxonomy["category"] == "Systemic"]
                .sort_values("emfi_post_max", ascending=False))
    N_TOP = min(5, len(systemic))
    if N_TOP == 0:
        log.warning("No systemic events — skipping network figure")
        return None

    top_events = systemic.head(N_TOP)
    log.info(f"Top {N_TOP} systemic events: {top_events['date'].dt.date.tolist()}")

    # build wide return matrix: date × country
    pivot = panel.pivot_table(index="date", columns="country", values="return")
    pivot.index = pd.to_datetime(pivot.index)
    dates = pivot.index
    countries = sorted(pivot.columns.tolist())

    # compute average pre/post correlation matrices
    pre_corrs  = []
    post_corrs = []
    for _, ev in top_events.iterrows():
        t   = pd.Timestamp(ev["date"])
        pos = int(dates.searchsorted(t, side="left"))
        pos = min(pos, len(dates)-1)
        pre_sl  = slice(max(0, pos - NET_WIN), pos)
        post_sl = slice(pos, min(len(dates), pos + NET_WIN + 1))
        pre_mat  = pivot.iloc[pre_sl][countries].corr().values
        post_mat = pivot.iloc[post_sl][countries].corr().values
        pre_corrs.append(pre_mat)
        post_corrs.append(post_mat)

    avg_pre  = np.nanmean(pre_corrs,  axis=0)
    avg_post = np.nanmean(post_corrs, axis=0)
    diff     = avg_post - avg_pre   # positive = higher correlation post-event

    n = len(countries)
    pos_layout = nx.circular_layout(nx.complete_graph(n))
    node_pos   = {countries[i]: pos_layout[i] for i in range(n)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [f"Pre-event (−{NET_WIN} to −1 days)",
              f"Post-event (0 to +{NET_WIN} days)",
              "Post − Pre (correlation change)"]
    mats   = [avg_pre, avg_post, diff]
    cmaps  = ["Blues", "Reds", "RdBu_r"]
    thresh = [0.50, 0.50, 0.05]   # min edge weight to display

    for ax, mat, title, cmap, thr in zip(axes, mats, titles, cmaps, thresh):
        G = nx.Graph()
        G.add_nodes_from(countries)
        for i in range(n):
            for j in range(i + 1, n):
                w = mat[i, j]
                if abs(w) >= thr:
                    G.add_edge(countries[i], countries[j], weight=w)

        edges  = G.edges(data=True)
        weights= [abs(d["weight"]) for _, _, d in edges]
        colors_e = []
        for _, _, d in G.edges(data=True):
            w = d["weight"]
            if cmap == "RdBu_r":
                colors_e.append("#c0392b" if w > 0 else "#2980b9")
            else:
                colors_e.append("#2d6a9f" if cmap == "Blues" else "#c0392b")

        nx.draw_networkx_nodes(G, node_pos, ax=ax, node_size=150,
                               node_color="#ecf0f1", edgecolors="#7f8c8d", linewidths=0.8)
        nx.draw_networkx_labels(G, node_pos, ax=ax, font_size=5)
        if weights:
            nx.draw_networkx_edges(G, node_pos, ax=ax,
                                   width=[w * 3 for w in weights],
                                   edge_color=colors_e, alpha=0.65)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Return correlation networks averaged over top-{N_TOP} systemic events\n"
        f"({', '.join(str(d.date()) for d in top_events['date'])})",
        fontsize=9)
    fig.tight_layout()
    fpath = outdir / "Fig_NetworkCorr_Systemic.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {fpath}")
    return fpath


# ── main ──────────────────────────────────────────────────────────────────────

def classify_events_main():
    t0 = time.time()
    log.info("=== SP08: Event Classification ===")

    emfi, hmm, tci, shocks, events, shock_days, idx, panel = load_data()

    taxonomy = classify_events(emfi, hmm, tci, shocks, events, shock_days, idx)
    log.info(f"Taxonomy: {taxonomy['category'].value_counts().to_dict()}")

    # verify empirical anchors
    for anchor_date, expected_cat, label in [
        ("2020-03-12", "Systemic",  "COVID peak"),
        ("2022-02-24", "Systemic",  "Ukraine invasion"),
        ("2023-10-09", "Localized", "Hamas attack"),
    ]:
        ts  = pd.Timestamp(anchor_date)
        row = taxonomy[taxonomy["date"] == ts]
        if row.empty:
            # try nearest shock day within 2 days
            near = [d for d in shock_days
                    if abs((d - ts).days) <= 10]
            if near:
                ts  = min(near, key=lambda d: abs((d - ts).days))
                row = taxonomy[taxonomy["date"] == ts]
        if not row.empty:
            cat = row.iloc[0]["category"]
            log.info(f"  {label} ({ts.date()}): {cat} "
                     f"[P_stress_peak={row.iloc[0]['p_stress_peak']:.3f}, "
                     f"emfi_post_max={row.iloc[0]['emfi_post_max']:.3f}]")
        else:
            log.warning(f"  {label} ({anchor_date}): NOT FOUND in shock days")

    market_only = find_market_only_stress(hmm, shock_days, idx)
    summary     = build_summary(taxonomy)

    log.info("Category summary:")
    log.info(f"\n{summary.to_string(index=False)}")

    # save CSVs
    taxonomy.to_csv(OUTDIR / "event_taxonomy.csv", index=False)
    summary.to_csv(OUTDIR / "summary_by_category.csv", index=False)
    market_only.to_csv(OUTDIR / "market_stress_episodes.csv", index=False)
    log.info("CSVs written")

    # figures
    fig1 = plot_event_timeline(emfi, hmm, taxonomy, OUTDIR)
    fig2 = plot_network_figure(taxonomy, panel, OUTDIR)

    # manifest
    systemic_rows = taxonomy[taxonomy["category"] == "Systemic"]
    manifest = {
        "generated_at" : datetime.now(timezone.utc).isoformat(),
        "script"       : "subprojects/08_event_classification/classify_events.py",
        "n_shock_days" : int(len(taxonomy)),
        "emfi_q75"     : EMFI_Q75,
        "p_stress_threshold": PSTRESS_SYSTEMIC,
        "category_counts": taxonomy["category"].value_counts().to_dict(),
        "market_only_stress_days": int(len(market_only)),
        "top_systemic_events": [
            {"date": str(r["date"].date()),
             "p_stress_peak": round(r["p_stress_peak"], 4),
             "emfi_post_max": round(r["emfi_post_max"], 4),
             "countries": r["countries"]}
            for _, r in systemic_rows.sort_values(
                "emfi_post_max", ascending=False).head(10).iterrows()
        ],
        "outputs": [str(p.name) for p in sorted(OUTDIR.glob("*"))
                    if p.name != ".gitkeep"],
    }
    with open(OUTDIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== SP08 complete in {elapsed:.1f}s ===")
    log.info(f"Outputs written to {OUTDIR}")


if __name__ == "__main__":
    classify_events_main()
