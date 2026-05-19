"""
SP15b — Figure 8: Robustness theta bar chart (COVID-zeroed variants)
=====================================================================
Re-runs the EMFI-construction robustness state LP theta checks with
COVID-zeroed max_shock (matching the primary specification), then regenerates
Fig_Robustness_StateLPTheta.png.

The original SP09 ran these checks on the FULL shock series (max_shock
non-zeroed on COVID days). This caused Figure 8's baseline to show 1.854
(full-sample theta) while Table 3 shows 1.439 (COVID-zeroed theta).

Fix:
- Baseline theta: read from results/theta_inference/manifest.json (1.439)
- Variant thetas (R03, R04, R05, R11, R12): re-estimated with max_shock=0
  on 2020-01-01 to 2020-12-31 (16 shock days zeroed)

Output:
  results/robustness/Fig_Robustness_StateLPTheta.png  (overwrite)
  Paper_LaTeX/Fig_Robustness_StateLPTheta.png         (copy to LaTeX)
  results/figure_improvements/theta_nocovid_variants.csv

Run from Paper_GFJ root:
  python subprojects/15_figure_improvements/fig8_theta_nocovid.py
"""
from __future__ import annotations

import json
import logging
import sys
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_EMFI   = GFJ_ROOT / "results" / "emfi"           / "emfi_daily.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes" / "hmm_daily.csv"
IN_TCI    = GFJ_ROOT / "results" / "connectedness"  / "tci_daily.csv"
IN_FRAG   = GFJ_ROOT / "results" / "fragility"      / "fragility_daily.csv"
IN_SHOCKS = GFJ_ROOT / "data"    / "aggregate_shocks.csv"
IN_PANEL  = GFJ_ROOT / "data"    / "panel_daily.parquet"
IN_THETA_MANIFEST = GFJ_ROOT / "results" / "theta_inference" / "manifest.json"

OUT_ROBUSTNESS = GFJ_ROOT / "results" / "robustness"
OUT_FIG_IMPROV = GFJ_ROOT / "results" / "figure_improvements"
LATEX_DIR      = GFJ_ROOT / "Paper_LaTeX"
OUT_FIG_IMPROV.mkdir(parents=True, exist_ok=True)

COVID_START = "2020-01-01"
COVID_END   = "2020-12-31"
KEY_HORIZONS = [0]   # only k=0 needed for the theta figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_base_data() -> pd.DataFrame:
    emfi   = pd.read_csv(IN_EMFI,   parse_dates=["date"])
    hmm    = pd.read_csv(IN_HMM,    parse_dates=["date"])[["date", "P_stress"]]
    tci    = pd.read_csv(IN_TCI,    parse_dates=["date"])
    frag   = pd.read_csv(IN_FRAG,   parse_dates=["date"])
    shocks = pd.read_csv(IN_SHOCKS, parse_dates=["date"])

    panel     = pd.read_parquet(IN_PANEL)
    ret_wide  = panel.pivot(index="date", columns="country", values="return")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    global_ret = ret_wide.mean(axis=1).rename("global_ret").reset_index()
    global_ret.columns = ["date", "global_ret"]

    df = (emfi
          .merge(hmm,        on="date", how="left")
          .merge(tci[["date","tci_w60","tci_w100","tci_w150"]], on="date", how="left")
          .merge(frag[["date","VolStress","TailVolBreadth","AvgCorr60"]], on="date", how="left")
          .merge(shocks[["date","max_shock","breadth_shock","avg_shock_pos"]], on="date", how="left")
          .merge(global_ret, on="date", how="left")
    )
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["max_shock"] = df["max_shock"].fillna(0.0)
    return df


def zero_covid_shocks(df: pd.DataFrame) -> pd.DataFrame:
    """Set max_shock = 0 on COVID-period dates (primary specification)."""
    df = df.copy()
    covid_mask = (df["date"] >= COVID_START) & (df["date"] <= COVID_END)
    df.loc[covid_mask, "max_shock"] = 0.0
    n_zeroed = covid_mask.sum()
    log.info("  COVID zeroing: %d rows → max_shock=0 on %s to %s", n_zeroed, COVID_START, COVID_END)
    return df


def build_alt_emfi(df: pd.DataFrame, components: list[str],
                   method: str = "pca") -> pd.Series:
    """Build alternative EMFI from given component columns."""
    sub = df[components].copy().fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    if method == "pca":
        pca = PCA(n_components=1)
        idx = pca.fit_transform(X)[:, 0]
        # flip sign so positive = more stress
        if np.corrcoef(idx, X[:, 0])[0, 1] < 0:
            idx = -idx
        return pd.Series(idx, index=df.index)
    elif method == "equal":
        return pd.Series(X.mean(axis=1), index=df.index)
    raise ValueError(f"Unknown method: {method}")


# ── State LP theta at k=0 ──────────────────────────────────────────────────────

def run_theta_k0(df: pd.DataFrame, outcome: str, shock: str = "max_shock") -> float:
    """Estimate smooth-transition LP at k=0 and return theta_k0."""
    k = 0
    p_stress = df["P_stress"].shift(1)
    y_fwd    = df[outcome].shift(-k)
    y_lag    = df[outcome].shift(1)
    shock_s  = df[shock]
    inter    = shock_s * p_stress
    gret     = df["global_ret"]
    month    = df["month"]

    month_dummies = pd.get_dummies(month, prefix="m", drop_first=True).astype(float)
    X = pd.concat([
        shock_s.rename(shock),
        inter.rename("inter"),
        p_stress.rename("P_stress_lag"),
        y_lag.rename("y_lag"),
        gret,
        month_dummies,
    ], axis=1)
    X = add_constant(X, prepend=True, has_constant="add")

    valid = y_fwd.notna() & X.notna().all(axis=1)
    y_est = y_fwd[valid].values
    X_est = X[valid].values

    model = OLS(y_est, X_est).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 2, "use_correction": True},
    )
    theta = float(model.params[2])
    log.info("    %s θ_k0 = %.4f (nobs=%d)", outcome, theta, int(model.nobs))
    return theta


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_theta_figure(checks: list[dict], baseline_theta: float, out_path: Path):
    labels = ["Baseline\n(4-comp PCA)"] + [c["label"] for c in checks]
    thetas = [baseline_theta]           + [c["theta"] for c in checks]
    colors = ["#2166ac", "#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#16a085"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(labels)), thetas,
                  color=colors[:len(labels)], alpha=0.85, edgecolor="k", linewidth=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(baseline_theta, color="#2166ac", lw=1.2, ls="--", alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\hat\theta_{k=0}$ (amplification coefficient)", fontsize=10)
    ax.set_title(
        r"State-dependent LP: amplification $\hat\theta_{k=0}$ across EMFI specifications" + "\n"
        r"(COVID-zeroed primary specification; $N=138$ geopolitical shock observations)",
        fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for bar, val in zip(bars, thetas):
        ypos = bar.get_height() + 0.04 if val >= 0 else bar.get_height() - 0.12
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=== SP15b: Figure 8 — Robustness Theta (COVID-zeroed) ===")

    # 1. Load baseline COVID-zeroed theta from theta_inference manifest
    with open(IN_THETA_MANIFEST) as f:
        ti_manifest = json.load(f)
    baseline_theta = ti_manifest["emfi_k0"]["theta"]
    log.info("Baseline theta from theta_inference: %.4f", baseline_theta)

    # 2. Load data and zero COVID shocks
    log.info("Loading base data...")
    df = load_base_data()
    df = zero_covid_shocks(df)

    # 3. Build alternative EMFI outcomes
    log.info("Building alternative EMFI outcomes...")
    df["EMFI_3comp"]   = build_alt_emfi(df, ["VolStress", "TailVolBreadth", "AvgCorr60"], "pca")
    df["AcuteEMFI"]    = build_alt_emfi(df, ["VolStress", "TailVolBreadth"], "pca")
    df["EqWtEMFI"]     = build_alt_emfi(df, ["VolStress", "TailVolBreadth", "AvgCorr60", "tci_w100"], "equal")

    # Build EMFI variants with alternative TCI windows
    for tci_col, suffix in [("tci_w60", "tci60"), ("tci_w150", "tci150")]:
        df[f"EMFI_{suffix}"] = build_alt_emfi(
            df, ["VolStress", "TailVolBreadth", "AvgCorr60", tci_col], "pca"
        )

    # 4. Run theta_k0 for each variant on COVID-zeroed shocks
    variants = [
        ("R03", "EMFI_3comp (no TCI)",   "EMFI_3comp"),
        ("R04", "AcuteEMFI (vol only)",  "AcuteEMFI"),
        ("R05", "Equal-weight EMFI",     "EqWtEMFI"),
        ("R11", "TCI $w=60$",            "EMFI_tci60"),
        ("R12", "TCI $w=150$",           "EMFI_tci150"),
    ]

    checks = []
    for rid, label, outcome in variants:
        log.info("Running %s (%s)...", rid, label)
        theta = run_theta_k0(df, outcome=outcome)
        checks.append({"id": rid, "label": label.replace("$", ""), "theta": theta})

    log.info("\n=== Variant thetas (COVID-zeroed) ===")
    log.info("Baseline: %.4f", baseline_theta)
    for c in checks:
        log.info("  %s: %.4f", c["id"], c["theta"])

    # 5. Save variant table
    variant_df = pd.DataFrame([{"check_id": "Baseline", "check_name": "4-comp PCA",
                                 "theta_k0_nocovid": baseline_theta}] +
                               [{"check_id": c["id"], "check_name": c["label"],
                                 "theta_k0_nocovid": c["theta"]} for c in checks])
    variant_path = OUT_FIG_IMPROV / "theta_nocovid_variants.csv"
    variant_df.to_csv(variant_path, index=False)
    log.info("Saved variant table: %s", variant_path)

    # 6. Regenerate figure
    archive_path = OUT_FIG_IMPROV / "Fig_Robustness_StateLPTheta_NoCovid.png"
    rob_path     = OUT_ROBUSTNESS  / "Fig_Robustness_StateLPTheta.png"
    latex_path   = LATEX_DIR       / "Fig_Robustness_StateLPTheta.png"

    plot_theta_figure(checks, baseline_theta, archive_path)
    shutil.copy2(archive_path, rob_path)
    shutil.copy2(archive_path, latex_path)
    log.info("Copied to robustness: %s", rob_path)
    log.info("Copied to LaTeX dir:  %s", latex_path)

    log.info("=== SP15b complete ===")


if __name__ == "__main__":
    main()
