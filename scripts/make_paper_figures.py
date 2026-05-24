"""
Generate Track C paper figures:
  - fig1_noise_robustness.{png,pdf}
  - fig2_ablation_bars.{png,pdf}
  - pipeline_diagram.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

np.random.seed(42)

ROOT = "/Users/makar/Desktop/black-scholes-project/bs-pde-discovery"
TABLES = os.path.join(ROOT, "outputs", "tables")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "paper")
os.makedirs(OUTDIR, exist_ok=True)

# ---- Global style ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "figure.dpi": 120,
})

PALETTE = {
    "blue":    "#0072B2",
    "orange":  "#D55E00",
    "green":   "#009E73",
    "magenta": "#CC79A7",
}


def style_axes(ax):
    """Major y-grid only, light gray; remove x grid; clean spines."""
    ax.grid(axis="y", which="major", color="#D0D0D0", alpha=0.5, linewidth=0.7)
    ax.grid(axis="x", which="both", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_fig(fig, basename):
    png = os.path.join(OUTDIR, basename + ".png")
    pdf = os.path.join(OUTDIR, basename + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


# =============================================================
# FIGURE 1 — Noise robustness (FD, SavGol, GP, Weak-SINDy)
# =============================================================
def figure1_noise_robustness():
    print("Figure 1: noise robustness")
    # FD and SavGol come from the unified comparison v2 (r2_clean column).
    main = pd.read_csv(os.path.join(TABLES, "all_methods_noise_comparison_v2.csv"))
    fd     = main[main["method"] == "fd"].sort_values("noise_pct")
    savgol = main[main["method"] == "savgol"].sort_values("noise_pct")

    gp = pd.read_csv(os.path.join(TABLES, "gp_noise_robustness.csv")).sort_values("noise_pct")
    weak = pd.read_csv(os.path.join(TABLES, "weak_sindy_noise_robustness.csv")).sort_values("noise_level")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Reference line at y=0
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="-", alpha=0.4)

    # Convert noise to percent for display (x in %)
    ax.plot(fd["noise_pct"] * 100, fd["r2_clean"],
            marker="o", color=PALETTE["blue"],
            label="Finite Differences")
    ax.plot(savgol["noise_pct"] * 100, savgol["r2_clean"],
            marker="s", color=PALETTE["orange"],
            label="Savitzky-Golay")
    ax.plot(gp["noise_pct"] * 100, gp["r2_clean"],
            marker="^", color=PALETTE["green"],
            label="Gaussian Process")
    ax.plot(weak["noise_level"] * 100, weak["r2_clean"],
            marker="D", color=PALETTE["magenta"],
            label="Weak-SINDy")

    ax.set_xlabel("Noise level (% of price)")
    ax.set_ylabel(r"$R^2$ on clean test set")
    ax.set_title("Noise robustness of derivative estimation methods")
    ax.set_ylim(-0.2, 1.05)
    ax.legend(loc="lower left", frameon=False)
    style_axes(ax)

    plt.tight_layout()
    save_fig(fig, "fig1_noise_robustness")
    plt.close(fig)

    return {
        "fd_n": len(fd),
        "savgol_n": len(savgol),
        "gp_n": len(gp),
        "weak_n": len(weak),
    }


# =============================================================
# FIGURE 2 — Ablation horizontal bar chart
# =============================================================
def figure2_ablation_bars():
    print("Figure 2: ablation bars")
    df = pd.read_csv(os.path.join(TABLES, "ablation_chain.csv"))

    labels = [
        "A: FD raw-K 5-term",
        "B: GP raw-K",
        "C: GP log-m 2-term",
        "D: + SVI",
        "E: + analytical $\\theta$",
        "F: + [2,1] KAN",
    ]
    values = df["R2_SPY"].tolist()

    # Reverse so F is on top
    y_labels = list(reversed(labels))
    y_vals = list(reversed(values))

    # Graduated blues: lightest for A, darkest for F
    cmap = mpl.cm.get_cmap("Blues")
    colors = [cmap(0.35 + 0.55 * i / 5) for i in range(6)]
    y_colors = list(reversed(colors))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    y_pos = np.arange(len(y_labels))
    bars = ax.barh(y_pos, y_vals, color=y_colors, edgecolor="white", linewidth=0.6)

    ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(r"$R^2$ on SPY")
    ax.set_title("Ablation: component contributions to SPY $R^2$")
    ax.set_xlim(0.0, max(y_vals) * 1.18)

    for bar, v in zip(bars, y_vals):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9)

    style_axes(ax)
    # For a horizontal bar chart we want vertical (x) gridlines only
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="major", color="#D0D0D0", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, "fig2_ablation_bars")
    plt.close(fig)

    return dict(zip(["A", "B", "C", "D", "E", "F"], values))


# =============================================================
# FIGURE 3 — Pipeline diagram
# =============================================================
def figure3_pipeline_diagram():
    print("Figure 3: pipeline diagram")
    stages = [
        ("Market Data\n(options chain)",                              PALETTE["blue"]),
        ("SVI Smoothing\n(per-expiration IV fit)",                    PALETTE["orange"]),
        ("Log-Moneyness Transform\n(k = log(K/F))",                   PALETTE["green"]),
        ("GP Derivative Estimation\n(analytical kernel derivatives)", PALETTE["magenta"]),
        ("SINDy Sparse Regression\n(identify active PDE terms)",      PALETTE["blue"]),
        ("KAN Nonlinear Diagnostics\n(learn operator on active terms)", PALETTE["orange"]),
        ("Validation\n(PINN + OoS + misspecification)",               PALETTE["green"]),
    ]

    fig, ax = plt.subplots(figsize=(4.0, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    n = len(stages)
    box_h = 0.095
    gap   = 0.045
    total = n * box_h + (n - 1) * gap
    y_top = 0.5 + total / 2.0

    box_x = 0.08
    box_w = 0.84

    centers = []
    for i, (text, color) in enumerate(stages):
        y_upper = y_top - i * (box_h + gap)
        y_lower = y_upper - box_h
        cy = (y_upper + y_lower) / 2.0
        centers.append((box_x + box_w / 2.0, y_upper, y_lower, cy))

        box = FancyBboxPatch(
            (box_x, y_lower), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(box)
        # Inner darker border for contrast
        ax.add_patch(FancyBboxPatch(
            (box_x, y_lower), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=color,
            facecolor="none",
        ))
        ax.text(box_x + box_w / 2.0, cy, text,
                ha="center", va="center", fontsize=9.5, color="black")

    # Arrows between successive boxes
    for i in range(n - 1):
        cx_top, _, y_lower_top, _ = centers[i]
        cx_bot, y_upper_bot, _, _ = centers[i + 1]
        arrow = FancyArrowPatch(
            (cx_top, y_lower_top - 0.002),
            (cx_bot, y_upper_bot + 0.002),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#555555",
        )
        ax.add_patch(arrow)

    plt.tight_layout()
    save_fig(fig, "pipeline_diagram")
    plt.close(fig)


def main():
    f1 = figure1_noise_robustness()
    f2 = figure2_ablation_bars()
    figure3_pipeline_diagram()

    print("\nSummary")
    print(" Figure 1 row counts:", f1)
    print(" Figure 2 ablation values:")
    for k, v in f2.items():
        print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
