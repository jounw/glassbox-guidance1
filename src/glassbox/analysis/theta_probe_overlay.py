

import matplotlib.pyplot as plt
import numpy as np

from glassbox.analysis.theta_probe_teacher_distilled import (
    MODEL_PATHS,
    SEEDS,
    consensus_curve,
    distilled_curve,
    load_equations,
    plot_wrapped,
    qlaw_curve,
    raw_teacher_curve,
    style_axes,
    theta_grid,
    EQUATIONS_CSV,
)

OUT_PATH = "outputs/theta_probe_overlay"
SEED_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
SEED_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
SEED_MARKERS = ["o", "s", "^", "D", "v"]


def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "legend.fontsize": 13,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
    })

    L = theta_grid()
    L_deg = np.degrees(L)
    eqs_by_teacher = load_equations(EQUATIONS_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)

    ax1 = axes[0]
    for seed, color, style, marker in zip(SEEDS, SEED_COLORS, SEED_STYLES, SEED_MARKERS):
        raw = raw_teacher_curve(MODEL_PATHS[seed], L)
        plot_wrapped(ax1, L_deg, raw, color=color, lw=1.5, ls=style, marker=marker, ms=2.8,
                     label=f"SAC seed{seed}", zorder=2)
    style_axes(ax1)
    ax1.set_ylabel(r"$\alpha$ (deg)")
    ax1.set_title("raw SAC teachers")
    ax1.legend(loc="upper left", framealpha=0.9, borderpad=0.4, handlelength=1.6)

    ax2 = axes[1]
    qlaw = qlaw_curve(L)
    consensus = consensus_curve(L)
    plot_wrapped(ax2, L_deg, qlaw, color="#2ca02c", lw=1.8, ls="-", marker="o", ms=3,
                 label="Q-law (baseline)", zorder=3)
    plot_wrapped(ax2, L_deg, consensus, color="#d62728", lw=1.8, ls="--", marker="^", ms=3,
                 label="consensus closed form", zorder=2)
    style_axes(ax2)
    ax2.set_ylabel(r"$\alpha$ (deg)")
    ax2.set_title("Q-law vs. consensus equation")
    ax2.legend(loc="upper left", framealpha=0.9, borderpad=0.4, handlelength=1.6)

    ax3 = axes[2]
    for seed, color, style, marker in zip(SEEDS, SEED_COLORS, SEED_STYLES, SEED_MARKERS):
        uT_str, uR_str = eqs_by_teacher[seed][0]  # PySR seed0 run
        curve = distilled_curve(uT_str, uR_str, L)
        plot_wrapped(ax3, L_deg, curve, color=color, lw=1.5, ls=style, marker=marker, ms=2.8,
                     label=f"SAC seed{seed}", zorder=2)
    style_axes(ax3)
    ax3.set_ylabel(r"$\alpha$ (deg)")
    ax3.set_title("distilled equations (PySR seed0)")
    ax3.legend(loc="upper left", framealpha=0.9, borderpad=0.4, handlelength=1.6)

    fig.tight_layout()

    fig.savefig(f"{OUT_PATH}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_PATH}.pdf", bbox_inches="tight")
    print(f"saved -> {OUT_PATH}.png / .pdf")


if __name__ == "__main__":
    main()
