"""
SP15a — Figure 4: 3-Panel State-Dependent LP (COVID-zeroed, full-page)
=======================================================================
Generates a publication-quality 3-panel figure showing the smooth-transition
state LP impulse responses for the three primary outcomes:
  Panel A: EMFI
  Panel B: TCI (tci_w100)
  Panel C: TailVolBreadth

Data source: results/theta_inference/ (COVID-zeroed primary specification,
  SP13 outputs with delta-method SEs and block-bootstrap CIs).

Figure specification (per reviewer R12.2 requirements):
  - 3 panels in a 1×3 horizontal layout at full-page width
  - 3 IRF lines per panel: P_stress = 0.0 (calm), 0.5 (elevated), 0.9 (systemic)
  - 95% confidence bands (shaded) around each line
  - Horizontal reference at 0
  - X-axis: horizons k = 0 to 15 (trading days post-shock)
  - 300 dpi, at least 14 × 5 inches

Output:
  Paper_LaTeX/Fig_StateLPSmooth_Combined.png  (overwrites old full-sample figure)
  results/figure_improvements/Fig_StateLPSmooth_3Panel_NoCovid.png (archive copy)

Run from Paper_GFJ root:
  python subprojects/15_figure_improvements/fig4_state_lp_3panel.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
GFJ_ROOT  = THIS_FILE.parents[2]

IN_DIR    = GFJ_ROOT / "results" / "theta_inference"
OUT_DIR   = GFJ_ROOT / "results" / "figure_improvements"
LATEX_DIR = GFJ_ROOT / "Paper_LaTeX"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
OUTCOMES = [
    ("EMFI",          "EMFI",           "A"),
    ("tci_w100",      "TCI ($w=100$)",  "B"),
    ("TailVolBreadth","TailVolBreadth", "C"),
]

P_CONFIGS = [
    # (p_value, column_prefix, label, color, linestyle, alpha_band)
    (0.0, "p00", "Calm ($p=0.0$)",     "#4e8bcd", "--",  0.15),
    (0.5, "p05", "Elevated ($p=0.5$)", "#f4a44d", "-",   0.15),
    (0.9, "p09", "Systemic ($p=0.9$)", "#c0392b", "-",   0.20),
]

HORIZONS_PLOT = list(range(0, 16))   # k = 0..15


def load_outcome(outcome: str) -> pd.DataFrame:
    path = IN_DIR / f"theta_inference_{outcome}.csv"
    df = pd.read_csv(path)
    # Keep only post-shock horizons for the figure body
    df = df[df["horizon"].isin(HORIZONS_PLOT)].copy()
    return df


def plot_3panel(fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, (outcome, ylabel, panel_label) in zip(axes, OUTCOMES):
        df = load_outcome(outcome)
        hors = df["horizon"].values

        ax.axhline(0, color="black", lw=0.8, zorder=1)
        ax.axvline(0, color="gray",  lw=0.5, ls=":", zorder=1)

        for p_val, p_sfx, p_label, color, ls, alpha_b in P_CONFIGS:
            irf_col  = f"irf_{p_sfx}"
            lo_col   = f"ci95lo_irf_{p_sfx}"
            hi_col   = f"ci95hi_irf_{p_sfx}"

            # Some column names in theta_inference use p00/p05/p09
            if irf_col not in df.columns:
                # Try alternative naming
                alt = {"p00": "p00", "p05": "p05", "p09": "p09"}
                irf_col = f"irf_{alt[p_sfx]}"
                lo_col  = f"ci95lo_irf_{alt[p_sfx]}"
                hi_col  = f"ci95hi_irf_{alt[p_sfx]}"

            if irf_col not in df.columns:
                log.warning("Column %s not found in %s; skipping", irf_col, outcome)
                continue

            irf = df[irf_col].values
            lo  = df[lo_col].values
            hi  = df[hi_col].values

            ax.plot(hors, irf, color=color, lw=2.0, linestyle=ls,
                    label=p_label, zorder=3)
            ax.fill_between(hors, lo, hi, color=color, alpha=alpha_b, zorder=2)

        ax.set_xlabel("Horizon $k$ (trading days)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-0.5, 15.5)
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
        ax.text(-0.05, 1.03, f"({panel_label})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom")

    # Shared legend below panels
    handles = [
        mpatches.Patch(color=c, label=lbl, linestyle=ls)
        for _, _, lbl, c, ls, _ in P_CONFIGS
    ]
    # Use line handles instead for clarity
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=c, lw=2, linestyle=ls, label=lbl)
        for _, _, lbl, c, ls, _ in P_CONFIGS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "State-dependent LP: impulse responses at three stress levels\n"
        "(COVID-zeroed primary specification; N = 138 geopolitical shock observations; "
        "95\\% delta-method CI shaded)",
        fontsize=10, y=1.01
    )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", fig_path)


def main() -> None:
    log.info("=== SP15a: Figure 4 — 3-Panel State LP ===")

    # Verify data exists
    for outcome, _, _ in OUTCOMES:
        p = IN_DIR / f"theta_inference_{outcome}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        df = pd.read_csv(p)
        log.info("  %s: %d horizons, columns: %s", outcome, len(df), list(df.columns)[:6])

    # Generate figure
    archive_path = OUT_DIR / "Fig_StateLPSmooth_3Panel_NoCovid.png"
    latex_path   = LATEX_DIR / "Fig_StateLPSmooth_Combined.png"

    plot_3panel(archive_path)

    # Copy to LaTeX dir (overwrite old full-sample combined figure)
    import shutil
    shutil.copy2(archive_path, latex_path)
    log.info("Copied to LaTeX dir: %s", latex_path)

    # Verify
    sz = latex_path.stat().st_size / 1024
    log.info("Output size: %.1f KB", sz)
    log.info("=== SP15a complete ===")


if __name__ == "__main__":
    main()
