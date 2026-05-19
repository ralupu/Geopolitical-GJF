"""
SP17 — Predictive Logit Improvements (R14)
===========================================
Extends SP14 with two additional validation strategies requested by the reviewer:

  R14.1 — Leave-One-Year-Out (LOYO) cross-validation
    For each year y in {2017..2025} (excl. 2020, which has no events):
      Train on all events not in year y; predict on events in year y.
    Aggregate held-out predictions across all years; compute AUC.
    This is stricter than LOO-CV because it preserves temporal clustering
    within geopolitical episodes.

  R14.2 — Quasi-Real-Time EMFI Robustness
    Replace full-sample standardised EMFI_lag with an expanding-window
    standardised version (mean and std computed on EMFI up to t−1).
    Refit the logit and rerun LOO-CV to verify predictive AUC does not
    collapse in the quasi-real-time setting.

Inputs (relative to Paper_GFJ root):
  results/covid_reclassify/event_taxonomy_nocovid.csv
  results/stress_regimes/hmm_daily.csv
  results/emfi/emfi_daily.csv
  results/connectedness/tci_daily.csv
  results/predictive_class/manifest.json   (baseline LOO-CV AUC)

Outputs:
  results/predictive_class/loyo_cv_results.csv
  results/predictive_class/qrt_logit_table.csv
  results/predictive_class/qrt_loocv_results.csv
  results/predictive_class/r14_manifest.json
  Paper_LaTeX/Fig_ROC_Predictive.png       (updated with LOYO curve)

Run from Paper_GFJ root:
  python subprojects/17_logit_improvements/run_logit_improvements.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smdisc

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_TAXON  = GFJ_ROOT / "results" / "covid_reclassify" / "event_taxonomy_nocovid.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes"   / "hmm_daily.csv"
IN_EMFI   = GFJ_ROOT / "results" / "emfi"             / "emfi_daily.csv"
IN_TCI    = GFJ_ROOT / "results" / "connectedness"    / "tci_daily.csv"
IN_BASE   = GFJ_ROOT / "results" / "predictive_class" / "manifest.json"

OUT_DIR   = GFJ_ROOT / "results" / "predictive_class"
LATEX_DIR = GFJ_ROOT / "Paper_LaTeX"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FEATURE_NAMES = ["EMFI_lag", "TCI_lag", "P_stress_lag", "max_shock", "n_countries"]


# ── Data construction ──────────────────────────────────────────────────────────

def build_dataset(emfi_col: str = "EMFI") -> pd.DataFrame:
    """Build feature-merged dataset. emfi_col selects which EMFI column to use."""
    taxon = pd.read_csv(IN_TAXON, parse_dates=["date"])
    hmm   = pd.read_csv(IN_HMM,   parse_dates=["date"])[["date", "P_stress"]]
    emfi  = pd.read_csv(IN_EMFI,  parse_dates=["date"])[["date", "EMFI"]]
    tci   = pd.read_csv(IN_TCI,   parse_dates=["date"])[["date", "tci_w100"]]

    daily = (emfi.merge(hmm, on="date")
                 .merge(tci, on="date")
                 .sort_values("date")
                 .reset_index(drop=True))

    # Quasi-real-time EMFI: expanding-window standardisation up to t−1
    emfi_vals = daily["EMFI"].values.copy()
    qrt_emfi  = np.full(len(daily), np.nan)
    for i in range(1, len(daily)):
        past = emfi_vals[:i]  # all data up to (not including) t
        mu, sigma = past.mean(), past.std()
        if sigma > 0:
            qrt_emfi[i] = (emfi_vals[i] - mu) / sigma
        else:
            qrt_emfi[i] = 0.0
    daily["EMFI_qrt"] = qrt_emfi

    daily["EMFI_lag"]     = daily["EMFI"].shift(1)
    daily["EMFI_qrt_lag"] = daily["EMFI_qrt"].shift(1)
    daily["TCI_lag"]      = daily["tci_w100"].shift(1)
    daily["P_stress_lag"] = daily["P_stress"].shift(1)

    lag_df = daily[["date", "EMFI_lag", "EMFI_qrt_lag", "TCI_lag", "P_stress_lag"]]
    df = taxon.merge(lag_df, on="date", how="left")
    df["systemic"] = (df["category"] == "Systemic").astype(int)
    df = df.dropna(subset=["EMFI_lag", "EMFI_qrt_lag", "TCI_lag",
                            "P_stress_lag", "max_shock", "n_countries"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year

    log.info("  Dataset: %d events, Systemic=%d, NonSystemic=%d",
             len(df), df["systemic"].sum(), (df["systemic"] == 0).sum())
    return df


def get_Xy(df: pd.DataFrame, emfi_feature: str = "EMFI_lag"):
    """Scale features; emfi_feature selects full-sample or QRT EMFI."""
    features = [emfi_feature, "TCI_lag", "P_stress_lag", "max_shock", "n_countries"]
    X = df[features].values
    y = df["systemic"].values
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    return X_scaled, y, X_mean, X_std, features


# ── Model helpers ──────────────────────────────────────────────────────────────

def fit_logit(X_sc: np.ndarray, y: np.ndarray):
    Xc = sm.add_constant(X_sc, has_constant="add")
    try:
        return smdisc.Logit(y, Xc).fit(disp=False, maxiter=200)
    except Exception:
        return smdisc.Logit(y, Xc).fit(method="bfgs", disp=False, maxiter=500)


def run_loocv(X_sc: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    loo  = LeaveOneOut()
    n    = len(y)
    prob = np.zeros(n)
    for train_idx, test_idx in loo.split(X_sc):
        X_tr, y_tr = X_sc[train_idx], y[train_idx]
        X_te       = X_sc[test_idx]
        Xc_tr = sm.add_constant(X_tr, has_constant="add")
        Xc_te = sm.add_constant(X_te, has_constant="add")
        try:
            m = smdisc.Logit(y_tr, Xc_tr).fit(disp=False, maxiter=200)
            prob[test_idx[0]] = float(m.predict(Xc_te)[0])
        except Exception:
            prob[test_idx[0]] = y_tr.mean()
    return prob, float(roc_auc_score(y, prob))


# ── R14.1: Leave-One-Year-Out CV ───────────────────────────────────────────────

def run_loyo_cv(df: pd.DataFrame, X_sc: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, pd.DataFrame]:
    """
    For each year, train on all other years, predict on held-out year.
    Returns (predicted_proba_array, AUC, per-year table).
    """
    years = sorted(df["year"].unique())
    prob  = np.zeros(len(y))
    rows  = []

    for yr in years:
        test_mask  = (df["year"] == yr).values
        train_mask = ~test_mask
        X_tr, y_tr = X_sc[train_mask], y[train_mask]
        X_te, y_te = X_sc[test_mask],  y[test_mask]
        n_te  = int(test_mask.sum())
        n_sys = int(y_te.sum())

        Xc_tr = sm.add_constant(X_tr, has_constant="add")
        Xc_te = sm.add_constant(X_te, has_constant="add")
        try:
            m     = smdisc.Logit(y_tr, Xc_tr).fit(disp=False, maxiter=200)
            preds = m.predict(Xc_te)
        except Exception:
            preds = np.full(n_te, y_tr.mean())

        prob[test_mask] = preds

        if n_sys > 0 and n_sys < n_te:
            yr_auc = float(roc_auc_score(y_te, preds))
        else:
            yr_auc = np.nan  # only one class in test set

        rows.append({
            "year": yr, "n_events": n_te, "n_systemic": n_sys,
            "loyo_auc": yr_auc,
        })
        log.info("  LOYO year=%d: n=%d, systemic=%d, AUC=%s",
                 yr, n_te, n_sys,
                 f"{yr_auc:.4f}" if not np.isnan(yr_auc) else "n/a (one class)")

    overall_auc = float(roc_auc_score(y, prob))
    log.info("  LOYO overall AUC: %.4f", overall_auc)
    return prob, overall_auc, pd.DataFrame(rows)


# ── R14.2: Quasi-real-time EMFI ───────────────────────────────────────────────

def run_qrt_analysis(df: pd.DataFrame):
    """
    Refit logit and LOO-CV using expanding-window standardised EMFI_lag
    instead of full-sample standardised EMFI_lag.
    """
    log.info("  Building QRT feature matrix ...")
    X_qrt, y, X_mean_qrt, X_std_qrt, feat_qrt = get_Xy(df, emfi_feature="EMFI_qrt_lag")

    log.info("  Fitting full-sample QRT logit ...")
    res_qrt = fit_logit(X_qrt, y)

    rows = []
    for i, name in enumerate(["const"] + feat_qrt):
        rows.append({
            "variable": name,
            "coef_std": float(res_qrt.params[i]),
            "se_std":   float(res_qrt.bse[i]),
            "p_value":  float(res_qrt.pvalues[i]),
            "odds_ratio": float(np.exp(res_qrt.params[i])) if name != "const" else np.nan,
        })
    qrt_tab = pd.DataFrame(rows)

    log.info("  Running LOO-CV on QRT logit ...")
    prob_qrt, auc_qrt = run_loocv(X_qrt, y)
    log.info("  QRT LOO-CV AUC: %.4f", auc_qrt)

    qrt_loocv_df = df[["date", "category", "systemic"]].copy()
    qrt_loocv_df["prob_qrt"] = prob_qrt

    # EMFI coefficient comparison
    emfi_std = qrt_tab[qrt_tab["variable"] == "EMFI_qrt_lag"].iloc[0]
    log.info("  QRT EMFI coef=%.4f, OR=%.3f, p=%.4f",
             emfi_std["coef_std"], emfi_std["odds_ratio"], emfi_std["p_value"])

    return qrt_tab, prob_qrt, auc_qrt, qrt_loocv_df


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_roc_enhanced(y, prob_loo, auc_loo, prob_loyo, auc_loyo,
                      prob_qrt, auc_qrt, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    configs = [
        (prob_loo,  auc_loo,  "LOO-CV (primary)", "#2166ac", "-",  2.0),
        (prob_loyo, auc_loyo, "LOYO-CV (stricter)", "#1a9641", "--", 2.0),
        (prob_qrt,  auc_qrt,  "QRT EMFI, LOO-CV", "#e08214", ":",  1.8),
    ]
    for prob, a, label, color, ls, lw in configs:
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                label=f"{label} (AUC={a:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title(
        "ROC Curve: Predicting Systemic Geopolitical Shocks\n"
        "(Primary LOO-CV, stricter LOYO-CV, and quasi-real-time EMFI)",
        fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ROC plot: %s", out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=== SP17: R14 Logit Improvements ===")
    t0 = datetime.now(timezone.utc)

    # Load baseline manifest
    with open(IN_BASE) as f:
        base = json.load(f)
    loo_auc_baseline = base["loocv_auc_logit"]
    log.info("Baseline LOO-CV AUC (SP14): %.4f", loo_auc_baseline)

    # Build dataset
    df = build_dataset()
    X_sc, y, X_mean, X_std, feats = get_Xy(df, emfi_feature="EMFI_lag")

    # Reload baseline LOO-CV probabilities for ROC
    loocv_base = pd.read_csv(OUT_DIR / "loocv_results.csv", parse_dates=["date"])
    # Align on same rows (should be identical after SP14 re-run on 138 events)
    prob_loo = loocv_base["prob_logit"].values[:len(y)]

    # ── R14.1: LOYO-CV ────────────────────────────────────────────────────────
    log.info("--- R14.1: Leave-One-Year-Out CV ---")
    prob_loyo, auc_loyo, loyo_table = run_loyo_cv(df, X_sc, y)
    loyo_table.to_csv(OUT_DIR / "loyo_cv_results.csv", index=False)
    log.info("Saved loyo_cv_results.csv")

    # ── R14.2: Quasi-real-time EMFI ───────────────────────────────────────────
    log.info("--- R14.2: Quasi-Real-Time EMFI Robustness ---")
    qrt_tab, prob_qrt, auc_qrt, qrt_loocv_df = run_qrt_analysis(df)
    qrt_tab.to_csv(OUT_DIR / "qrt_logit_table.csv", index=False)
    qrt_loocv_df.to_csv(OUT_DIR / "qrt_loocv_results.csv", index=False)
    log.info("Saved qrt_logit_table.csv and qrt_loocv_results.csv")

    # ── ROC figure ────────────────────────────────────────────────────────────
    fig_path   = LATEX_DIR / "Fig_ROC_Predictive.png"
    fig_archive = OUT_DIR / "Fig_ROC_Predictive_R14.png"
    plot_roc_enhanced(y, prob_loo, loo_auc_baseline,
                      prob_loyo, auc_loyo, prob_qrt, auc_qrt, fig_archive)
    import shutil
    shutil.copy2(fig_archive, fig_path)
    log.info("Copied ROC to LaTeX dir: %s", fig_path)

    # ── LOYO per-year table ────────────────────────────────────────────────────
    log.info("\n=== LOYO CV per-year table ===")
    log.info("%-6s  %-8s  %-9s  %-8s", "Year", "N_events", "N_systemic", "AUC")
    for _, row in loyo_table.iterrows():
        auc_str = f"{row['loyo_auc']:.4f}" if not pd.isna(row['loyo_auc']) else "n/a"
        log.info("%-6d  %-8d  %-9d  %-8s", int(row['year']),
                 int(row['n_events']), int(row['n_systemic']), auc_str)

    # ── QRT EMFI coefficient summary ──────────────────────────────────────────
    log.info("\n=== QRT logit coefficients ===")
    for _, row in qrt_tab.iterrows():
        stars = ("***" if row["p_value"] < 0.01
                 else "**" if row["p_value"] < 0.05
                 else "*" if row["p_value"] < 0.10 else "")
        log.info("  %-20s  coef=%7.4f  p=%6.4f  OR=%.3f %s",
                 row["variable"], row["coef_std"], row["p_value"],
                 row["odds_ratio"] if not np.isnan(row["odds_ratio"]) else 0,
                 stars)

    # ── Manifest ──────────────────────────────────────────────────────────────
    emfi_qrt_row = qrt_tab[qrt_tab["variable"] == "EMFI_qrt_lag"].iloc[0]
    r14_manifest = {
        "run_utc":              t0.isoformat(),
        "n_obs":                int(len(df)),
        "n_systemic":           int(y.sum()),
        "loocv_auc_baseline":   float(loo_auc_baseline),
        "loyo_auc_overall":     float(auc_loyo),
        "loyo_per_year":        loyo_table.to_dict(orient="records"),
        "qrt_loocv_auc":        float(auc_qrt),
        "qrt_emfi_coef":        float(emfi_qrt_row["coef_std"]),
        "qrt_emfi_or":          float(emfi_qrt_row["odds_ratio"]),
        "qrt_emfi_pval":        float(emfi_qrt_row["p_value"]),
    }
    with open(OUT_DIR / "r14_manifest.json", "w") as f:
        json.dump(r14_manifest, f, indent=2)
    log.info("Saved r14_manifest.json")

    log.info("\n=== SP17 SUMMARY ===")
    log.info("Baseline LOO-CV AUC:    %.4f", loo_auc_baseline)
    log.info("LOYO-CV AUC (stricter): %.4f", auc_loyo)
    log.info("QRT EMFI LOO-CV AUC:    %.4f", auc_qrt)
    log.info("QRT EMFI OR:            %.3f (p=%.4f)",
             emfi_qrt_row["odds_ratio"], emfi_qrt_row["p_value"])
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("=== SP17 complete in %.1f s ===", elapsed)


if __name__ == "__main__":
    main()
