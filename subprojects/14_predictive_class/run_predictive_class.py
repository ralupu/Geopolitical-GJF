"""
SP14 -- Predictive Event Classification
=========================================
Answers the question: which pre-shock market conditions predict whether a
geopolitical shock will become systemic?

Model: binary logit (primary) and probit (robustness)
  Pr(Systemic_i = 1) = F(alpha + beta1*EMFI_{t-1} + beta2*TCI_{t-1}
                           + beta3*P_stress_{t-1} + beta4*S_t
                           + beta5*breadth_t)

Outcome: Systemic = 1 (42 events); Absorbed + Localized = 0 (112 events)
Sample:  154 shock-days (COVID-zeroed taxonomy from SP10)

In addition to logit/probit:
  - Leave-one-out cross-validation AUC (unbiased performance estimate)
  - Confusion matrix at optimal threshold (Youden J)
  - CART classification tree (illustrative; max depth 3)
  - Predicted probabilities for selected high-profile events

Inputs (relative to Paper_GFJ root):
  results/covid_reclassify/event_taxonomy_nocovid.csv
  results/stress_regimes/hmm_daily.csv
  results/emfi/emfi_daily.csv
  results/connectedness/tci_daily.csv

Outputs (relative to Paper_GFJ root):
  results/predictive_class/logit_table.csv       -- coefficient table
  results/predictive_class/probit_table.csv      -- probit robustness
  results/predictive_class/loocv_results.csv     -- per-event predicted proba
  results/predictive_class/Fig_ROC_Predictive.png
  results/predictive_class/Fig_Tree_Predictive.png
  results/predictive_class/manifest.json

Run from Paper_GFJ root:
  python subprojects/14_predictive_class/run_predictive_class.py
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
from sklearn.metrics import (
    auc, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier, export_text
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smdisc

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_TAXON  = GFJ_ROOT / "results" / "covid_reclassify" / "event_taxonomy_nocovid.csv"
IN_HMM    = GFJ_ROOT / "results" / "stress_regimes"   / "hmm_daily.csv"
IN_EMFI   = GFJ_ROOT / "results" / "emfi"             / "emfi_daily.csv"
IN_TCI    = GFJ_ROOT / "results" / "connectedness"    / "tci_daily.csv"

OUT_DIR   = GFJ_ROOT / "results" / "predictive_class"
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
# Notable events for predicted-probability spotlight
# ---------------------------------------------------------------------------
SPOTLIGHT_DATES = {
    "COVID-19 onset (2020-02-24)":  "2020-02-24",
    "Ukraine invasion (2022-02-24)": "2022-02-24",
    "Hamas attack (2023-10-09)":     "2023-10-09",
}


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """
    Merge event taxonomy with lagged (t-1) EMFI, TCI, P_stress.
    Returns DataFrame with one row per shock day.
    """
    taxon = pd.read_csv(IN_TAXON, parse_dates=["date"])
    hmm   = pd.read_csv(IN_HMM,   parse_dates=["date"])[["date", "P_stress"]]
    emfi  = pd.read_csv(IN_EMFI,  parse_dates=["date"])[["date", "EMFI"]]
    tci   = pd.read_csv(IN_TCI,   parse_dates=["date"])[["date", "tci_w100"]]

    # Build daily panel for lagged merge
    daily = (emfi.merge(hmm,  on="date")
                 .merge(tci,  on="date")
                 .sort_values("date"))
    daily["EMFI_lag"]     = daily["EMFI"].shift(1)
    daily["TCI_lag"]      = daily["tci_w100"].shift(1)
    daily["P_stress_lag"] = daily["P_stress"].shift(1)
    lag_df = daily[["date", "EMFI_lag", "TCI_lag", "P_stress_lag"]]

    df = taxon.merge(lag_df, on="date", how="left")

    # Binary outcome
    df["systemic"] = (df["category"] == "Systemic").astype(int)

    # Predictors: use pre-existing taxonomy fields + freshly lagged values
    # emfi_pre_mean = 5-day pre-shock EMFI mean (from taxonomy)
    # tci_pre_mean  = 5-day pre-shock TCI mean (from taxonomy)
    # P_stress_lag  = HMM posterior on t-1
    # max_shock     = shock intensity
    # n_countries   = breadth of shock

    df = df.dropna(subset=["EMFI_lag", "TCI_lag", "P_stress_lag",
                            "max_shock", "n_countries"]).reset_index(drop=True)

    log.info("  Dataset: %d shock events, Systemic=%d, NonSystemic=%d",
             len(df), df["systemic"].sum(), (df["systemic"] == 0).sum())
    return df


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["EMFI_lag", "TCI_lag", "P_stress_lag", "max_shock", "n_countries"]
FEATURE_LABELS = {
    "EMFI_lag":      r"$\mathrm{EMFI}_{t-1}$",
    "TCI_lag":       r"$\mathrm{TCI}_{t-1}$",
    "P_stress_lag":  r"$\hat{P}_{\mathrm{stress},t-1}$",
    "max_shock":     r"$S_t$ (shock intensity)",
    "n_countries":   r"Breadth ($N$ countries)",
}


def get_Xy(df: pd.DataFrame):
    X = df[FEATURE_NAMES].values
    y = df["systemic"].values
    # Standardise for numerical stability (logit/probit)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    return X_scaled, y, X_mean, X_std


# ---------------------------------------------------------------------------
# Logit / Probit via statsmodels (full-sample)
# ---------------------------------------------------------------------------

def fit_binary_model(X_scaled: np.ndarray, y: np.ndarray,
                     model_type: str = "logit") -> sm.discrete.discrete_model.BinaryResultsWrapper:
    Xc = sm.add_constant(X_scaled)
    if model_type == "logit":
        model = smdisc.Logit(y, Xc)
    else:
        model = smdisc.Probit(y, Xc)
    try:
        result = model.fit(disp=False, maxiter=200)
    except Exception:
        result = model.fit(method="bfgs", disp=False, maxiter=500)
    return result


def extract_coef_table(result, feat_names: list[str], X_mean, X_std,
                       model_type: str) -> pd.DataFrame:
    """
    Build a tidy coefficient table with:
      - standardised coefficient (from model)
      - unstandardised coefficient (divided by X_std for comparability)
      - SE, z-stat, p-value, odds-ratio (logit only)
    """
    params = result.params
    bse    = result.bse
    pvals  = result.pvalues
    conf   = result.conf_int()

    rows = []
    feat_full = ["const"] + feat_names
    for i, name in enumerate(feat_full):
        coef_std = float(params[i])
        se_std   = float(bse[i])
        z        = float(coef_std / se_std) if se_std > 0 else np.nan
        p        = float(pvals[i])
        ci_lo    = float(conf[i, 0])
        ci_hi    = float(conf[i, 1])
        if name == "const":
            coef_raw, or_val = np.nan, np.nan
        else:
            idx = feat_names.index(name)
            std_i    = float(X_std[idx])
            coef_raw = coef_std / std_i  # back to raw scale
            or_val   = float(np.exp(coef_std)) if model_type == "logit" else np.nan
        rows.append({
            "variable":   name,
            "coef_std":   coef_std,
            "se_std":     se_std,
            "z_stat":     z,
            "p_value":    p,
            "ci95lo":     ci_lo,
            "ci95hi":     ci_hi,
            "coef_raw":   coef_raw,
            "odds_ratio": or_val,
            "model":      model_type,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LOO-CV
# ---------------------------------------------------------------------------

def run_loocv(X_scaled: np.ndarray, y: np.ndarray,
              model_type: str = "logit") -> tuple[np.ndarray, float]:
    """
    Leave-one-out cross-validation.
    Returns (predicted_probas, AUC).
    """
    loo  = LeaveOneOut()
    n    = len(y)
    prob = np.zeros(n)

    for train_idx, test_idx in loo.split(X_scaled):
        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_te       = X_scaled[test_idx]
        Xc_tr      = sm.add_constant(X_tr, has_constant="add")
        Xc_te      = sm.add_constant(X_te, has_constant="add")
        try:
            if model_type == "logit":
                m = smdisc.Logit(y_tr, Xc_tr).fit(disp=False, maxiter=200)
            else:
                m = smdisc.Probit(y_tr, Xc_tr).fit(disp=False, maxiter=200)
            prob[test_idx[0]] = float(m.predict(Xc_te)[0])
        except Exception:
            prob[test_idx[0]] = y_tr.mean()  # fallback: base rate

    loo_auc = roc_auc_score(y, prob)
    log.info("  LOO-CV AUC (%s): %.4f", model_type, loo_auc)
    return prob, loo_auc


# ---------------------------------------------------------------------------
# Optimal threshold (Youden J)
# ---------------------------------------------------------------------------

def optimal_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, dict]:
    fpr, tpr, thresholds = roc_curve(y, prob)
    j_stat = tpr - fpr
    ix     = j_stat.argmax()
    thresh = float(thresholds[ix])
    pred   = (prob >= thresh).astype(int)
    cm     = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "threshold":   thresh,
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision":   float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "accuracy":    float((tp + tn) / len(y)),
        "youden_j":    float(j_stat[ix]),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    return thresh, metrics


# ---------------------------------------------------------------------------
# CART classification tree
# ---------------------------------------------------------------------------

def fit_cart_tree(X_scaled: np.ndarray, y: np.ndarray,
                  feat_names: list[str]) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                  class_weight="balanced", random_state=42)
    tree.fit(X_scaled, y)
    rules = export_text(tree, feature_names=feat_names)
    log.info("  CART tree rules:\n%s", rules)
    return tree


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_roc(y: np.ndarray, prob_logit: np.ndarray, prob_probit: np.ndarray,
             auc_logit: float, auc_probit: float,
             thresh: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for prob, a, label, color, ls in [
        (prob_logit,  auc_logit,  "Logit",  "#2166ac", "-"),
        (prob_probit, auc_probit, "Probit", "#d6604d", "--"),
    ]:
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                label=f"{label} (AUC={a:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title("ROC Curve: Predicting Systemic Geopolitical Shocks\n"
                 "(Leave-One-Out Cross-Validation)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved ROC plot -> %s", out_path.name)


def plot_tree_importance(tree: DecisionTreeClassifier,
                         feat_names: list[str], out_path: Path) -> None:
    importances = tree.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2166ac" if i == order[0] else "#92c5de" for i in range(len(feat_names))]
    ax.bar(range(len(feat_names)),
           importances[order],
           color=[colors[i] for i in order])
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels([feat_names[i] for i in order], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Gini Importance", fontsize=10)
    ax.set_title("CART Tree: Feature Importance for Systemic Event Prediction", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved tree importance plot -> %s", out_path.name)


# ---------------------------------------------------------------------------
# Spotlight predictions
# ---------------------------------------------------------------------------

def compute_spotlight(df: pd.DataFrame, result_logit,
                      X_mean: np.ndarray, X_std: np.ndarray) -> dict:
    out = {}
    for label, date_str in SPOTLIGHT_DATES.items():
        row = df[df["date"] == date_str]
        if len(row) == 0:
            out[label] = None
            continue
        r = row.iloc[0]
        x_raw = np.array([r[f] for f in FEATURE_NAMES])
        x_sc  = (x_raw - X_mean) / X_std
        xc    = np.concatenate([[1.0], x_sc])
        eta   = float(result_logit.params @ xc)
        prob  = float(1 / (1 + np.exp(-eta)))
        out[label] = {
            "date":     date_str,
            "category": r["category"],
            "prob":     round(prob, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== SP14 Predictive Event Classification ===")
    t0 = datetime.now(timezone.utc)

    # ---- Data ----
    log.info("Building dataset ...")
    df = build_dataset()
    X_scaled, y, X_mean, X_std = get_Xy(df)

    # ---- Full-sample logit + probit ----
    log.info("Fitting logit ...")
    res_logit  = fit_binary_model(X_scaled, y, "logit")
    log.info("Fitting probit ...")
    res_probit = fit_binary_model(X_scaled, y, "probit")

    logit_tab  = extract_coef_table(res_logit,  FEATURE_NAMES, X_mean, X_std, "logit")
    probit_tab = extract_coef_table(res_probit, FEATURE_NAMES, X_mean, X_std, "probit")

    logit_tab.to_csv(OUT_DIR / "logit_table.csv",  index=False)
    probit_tab.to_csv(OUT_DIR / "probit_table.csv", index=False)
    log.info("Saved coefficient tables")

    # Print summary
    log.info("\n--- Logit coefficients (standardised) ---")
    for _, row in logit_tab.iterrows():
        stars = ("***" if row["p_value"] < 0.01
                 else "**" if row["p_value"] < 0.05
                 else "*" if row["p_value"] < 0.10 else "")
        log.info("  %-18s  coef=%7.4f  se=%6.4f  p=%6.4f  OR=%6.3f %s",
                 row["variable"], row["coef_std"], row["se_std"],
                 row["p_value"],
                 row["odds_ratio"] if not np.isnan(row["odds_ratio"]) else 0,
                 stars)

    # ---- LOO-CV ----
    log.info("Running LOO-CV logit (%d obs) ...", len(y))
    prob_logit,  auc_logit  = run_loocv(X_scaled, y, "logit")
    log.info("Running LOO-CV probit ...")
    prob_probit, auc_probit = run_loocv(X_scaled, y, "probit")

    # Save LOO-CV predictions
    loocv_df = df[["date", "category", "systemic"]].copy()
    loocv_df["prob_logit"]  = prob_logit
    loocv_df["prob_probit"] = prob_probit
    loocv_df.to_csv(OUT_DIR / "loocv_results.csv", index=False)

    thresh, cm_metrics = optimal_threshold(y, prob_logit)
    log.info("  Optimal threshold (Youden J=%.3f): %.3f",
             cm_metrics["youden_j"], thresh)
    log.info("  Sensitivity=%.3f  Specificity=%.3f  Accuracy=%.3f",
             cm_metrics["sensitivity"], cm_metrics["specificity"],
             cm_metrics["accuracy"])

    # Brier score
    brier = float(np.mean((prob_logit - y) ** 2))
    log.info("  Brier score (logit LOO): %.4f", brier)

    # ---- CART tree ----
    log.info("Fitting CART tree ...")
    tree = fit_cart_tree(X_scaled, y, FEATURE_NAMES)

    # ---- Spotlight ----
    spotlight = compute_spotlight(df, res_logit, X_mean, X_std)
    log.info("--- Spotlight predictions ---")
    for label, info in spotlight.items():
        if info:
            log.info("  %s: category=%s  Pr(Systemic)=%.3f",
                     label, info["category"], info["prob"])

    # ---- Figures ----
    plot_roc(y, prob_logit, prob_probit, auc_logit, auc_probit,
             thresh, OUT_DIR / "Fig_ROC_Predictive.png")
    plot_tree_importance(tree, FEATURE_NAMES, OUT_DIR / "Fig_Tree_Predictive.png")

    # ---- Manifest ----
    logit_k0 = logit_tab[logit_tab["variable"] == "P_stress_lag"].iloc[0]
    manifest = {
        "run_utc":          t0.isoformat(),
        "n_obs":            int(len(df)),
        "n_systemic":       int(y.sum()),
        "n_non_systemic":   int((y == 0).sum()),
        "features":         FEATURE_NAMES,
        "loocv_auc_logit":  float(auc_logit),
        "loocv_auc_probit": float(auc_probit),
        "brier_score":      float(brier),
        "optimal_threshold": float(thresh),
        "confusion_matrix": cm_metrics,
        "logit_pstress_coef": float(logit_k0["coef_std"]),
        "logit_pstress_pval": float(logit_k0["p_value"]),
        "logit_pseudo_r2":   float(res_logit.prsquared),
        "spotlight":         {k: v for k, v in spotlight.items() if v is not None},
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved manifest.json")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("SP14 complete in %.1f s", elapsed)
    log.info("LOO-CV AUC: logit=%.4f  probit=%.4f", auc_logit, auc_probit)
    log.info("Pseudo-R2 (McFadden): %.4f", res_logit.prsquared)


if __name__ == "__main__":
    main()
