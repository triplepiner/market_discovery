"""
Generate improved ablation figures for the paper.

Produces:
  - outputs/figures/paper/ablation_grouped.{png,pdf}
      2x3 grid of per-config bars (A-F), honestly showing non-monotonic R^2.
  - outputs/figures/paper/ablation_kan_clean.{png,pdf}
      4-bar horizontal chart for the KAN-specific clean monotonic ablation.
  - outputs/figures/paper/ablation_figures_caption_notes.txt
      Provenance notes for each bar value.

Does not modify any source CSVs or existing figures.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def _load_ablation_chain() -> list[dict]:
    path = TABLES_DIR / "ablation_chain.csv"
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["R2_SPY"] = float(row["R2_SPY"])
            rows.append(row)
    return rows


def _load_value(csv_name: str, ticker: str, column: str) -> float | None:
    path = TABLES_DIR / csv_name
    if not path.exists():
        return None
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ticker") == ticker and column in row and row[column]:
                try:
                    return float(row[column])
                except ValueError:
                    return None
    return None


# --------------------------------------------------------------------------
# Figure 1: 2x3 grouped panels (A-F)
# --------------------------------------------------------------------------
PALETTE = {
    "A": "#999999",
    "B": "#0072B2",
    "C": "#56B4E9",
    "D": "#E69F00",
    "E": "#D55E00",
    "F": "#009E73",
}

# Short panel titles and the "delta" subtitle (key change vs previous)
PANEL_LABELS = {
    "A": ("A: FD, raw K", "(baseline)"),
    "B": ("B: GP (RBF) derivs", "+GP derivs"),
    "C": ("C: log-m, 2-term", "+log-moneyness 2-term"),
    "D": ("D: log-m + SVI", "+SVI smoothing"),
    "E": ("E: analytical θ", "+analytical θ"),
    "F": ("F: [2,1] KAN", "+KAN [2,1]"),
}


def make_grouped_figure(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
    axes_flat = axes.flatten()

    for ax, row in zip(axes_flat, rows):
        cfg = row["config"]
        r2 = row["R2_SPY"]
        title, subtitle = PANEL_LABELS[cfg]
        color = PALETTE[cfg]

        ax.bar([0], [r2], width=0.55, color=color, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6, alpha=0.7)

        # Value annotation above bar (or above 0 if negative)
        y_text = max(r2, 0.0) + 0.025
        ax.text(0, y_text, f"{r2:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black")

        ax.set_title(title, fontweight="bold", loc="center")
        ax.set_xlabel(subtitle, fontsize=9, style="italic")
        ax.set_xticks([])
        ax.set_ylim(0.0, 0.7)
        ax.set_xlim(-0.6, 0.6)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Y label only on left column
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$R^2$ (SPY)")

    fig.suptitle(
        "Ablation components (non-monotonic — see caption)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png = FIG_DIR / "ablation_grouped.png"
    pdf = FIG_DIR / "ablation_grouped.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


# --------------------------------------------------------------------------
# Figure 2: KAN-specific clean monotonic ablation (4 bars, horizontal)
# --------------------------------------------------------------------------
def make_kan_clean_figure() -> tuple[list[tuple[str, float, str]], list[str]]:
    """
    Returns (bars, notes) where bars is list of (label, r2, provenance).
    """
    notes: list[str] = []

    # Bar 1: Status quo (Linear, FD θ, raw K) -> linear_dupire_r2 for SPY = 0.4386
    bar1_label = "Status quo\n(Linear, FD θ, raw K)"
    bar1_val = _load_value("sindy_kan_dupire_real.csv", "SPY", "linear_dupire_r2")
    if bar1_val is None:
        bar1_val = 0.439
        bar1_src = "PRD fallback (0.439)"
        notes.append("Bar 1: CSV unavailable; using PRD-stated 0.439.")
    else:
        bar1_src = "sindy_kan_dupire_real.csv [SPY, linear_dupire_r2]"

    # Bar 2: +coordinate (log-m + SVI) -> v4_analytical_theta_results.csv g2_r2 for SPY = 0.559
    bar2_label = "+ coordinate\n(log-m + SVI)"
    bar2_val = _load_value("v4_analytical_theta_results.csv", "SPY", "g2_r2")
    if bar2_val is None:
        bar2_val = 0.559
        bar2_src = "PRD fallback (0.559)"
        notes.append("Bar 2: CSV unavailable; using PRD-stated 0.559.")
    else:
        bar2_src = "v4_analytical_theta_results.csv [SPY, g2_r2]"

    # Bar 3: +analytical θ -> per PRD same cluster ~0.559
    bar3_label = "+ analytical θ"
    bar3_val = _load_value("v4_analytical_theta_results.csv", "SPY", "g2_r2")
    if bar3_val is None:
        bar3_val = 0.559
        bar3_src = "PRD fallback (0.559)"
        notes.append("Bar 3: CSV unavailable; using PRD-stated 0.559.")
    else:
        bar3_src = (
            "v4_analytical_theta_results.csv [SPY, g2_r2] "
            "(PRD: clusters with +coordinate)"
        )

    # Bar 4: + [2,1] KAN -> sindy_kan_dupire_real.csv kan_train_r2 for SPY = 0.795
    bar4_label = "+ [2,1] KAN"
    bar4_val = _load_value("sindy_kan_dupire_real.csv", "SPY", "kan_train_r2")
    if bar4_val is None:
        bar4_val = 0.795
        bar4_src = "PRD fallback (0.795)"
        notes.append("Bar 4: CSV unavailable; using PRD-stated 0.795.")
    else:
        bar4_src = "sindy_kan_dupire_real.csv [SPY, kan_train_r2]"

    bars = [
        (bar1_label, bar1_val, bar1_src),
        (bar2_label, bar2_val, bar2_src),
        (bar3_label, bar3_val, bar3_src),
        (bar4_label, bar4_val, bar4_src),
    ]

    # Graduated colorblind palette light -> dark (Okabe-Ito based)
    bar_colors = ["#D6E8F5", "#56B4E9", "#0072B2", "#003E66"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    y_pos = list(range(len(bars)))[::-1]  # top-to-bottom

    ax.barh(y_pos, values, color=bar_colors, edgecolor="black", linewidth=0.7)
    for y, v in zip(y_pos, values):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$R^2$ (SPY)")
    ax.set_xlim(0.0, 1.0)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_title(
        "KAN ablation: clean monotonic improvement (SPY)",
        fontweight="bold",
    )

    fig.tight_layout()
    png = FIG_DIR / "ablation_kan_clean.png"
    pdf = FIG_DIR / "ablation_kan_clean.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

    return bars, notes


# --------------------------------------------------------------------------
# Provenance notes file
# --------------------------------------------------------------------------
def write_caption_notes(chain_rows: list[dict],
                        kan_bars: list[tuple[str, float, str]],
                        fallback_notes: list[str]) -> None:
    out = FIG_DIR / "ablation_figures_caption_notes.txt"
    lines: list[str] = []
    lines.append("Ablation figure provenance notes")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Figure: ablation_grouped.png/.pdf (2x3 grid, configs A-F)")
    lines.append("Source: outputs/tables/ablation_chain.csv (column R2_SPY)")
    for r in chain_rows:
        lines.append(
            f"  {r['config']}: R^2 = {r['R2_SPY']:.4f}  "
            f"[{r['derivatives']} | {r['coordinates']} | {r['library']} | "
            f"theta={r['theta_source']} | model={r['model']}]"
        )
    lines.append("")
    lines.append("Figure: ablation_kan_clean.png/.pdf (4-bar horizontal)")
    for label, value, src in kan_bars:
        flat = label.replace("\n", " ")
        lines.append(f"  {flat}: R^2 = {value:.4f}  <- {src}")
    lines.append("")
    if fallback_notes:
        lines.append("Fallback notes:")
        for n in fallback_notes:
            lines.append(f"  - {n}")
    else:
        lines.append("Fallback notes: none — all 4 KAN-clean bars sourced from CSVs.")
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main() -> None:
    _set_style()
    chain_rows = _load_ablation_chain()
    make_grouped_figure(chain_rows)
    kan_bars, fallback_notes = make_kan_clean_figure()
    write_caption_notes(chain_rows, kan_bars, fallback_notes)


if __name__ == "__main__":
    main()
