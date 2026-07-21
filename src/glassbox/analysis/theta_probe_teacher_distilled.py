
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.ticker import MultipleLocator
from stable_baselines3 import SAC
from sympy import lambdify, symbols

from glassbox.baselines.qlaw import qlaw_policy

MODELS_DIR = "src/glassbox/models"
EQUATIONS_CSV = "outputs/distilled_equations.csv"
OUT_PATH = "outputs/theta_probe_teacher_distilled"

E0 = 0.03
DP0 = 0.0
TARGET_P = 2.0
THRUST_ACCEL = 0.001

SEEDS = [0, 1, 2, 3, 4]
MODEL_PATHS = {
    0: f"{MODELS_DIR}/Orbit_SAC_1.zip",
    1: f"{MODELS_DIR}/Orbit_SAC_seed1.zip",
    2: f"{MODELS_DIR}/Orbit_SAC_seed2.zip",
    3: f"{MODELS_DIR}/Orbit_SAC_seed3.zip",
    4: f"{MODELS_DIR}/Orbit_SAC_seed4.zip",
}

FEATURES = ["dp", "f", "g", "cosL", "sinL"]
PYSR_COLORS = ["#d95f02", "#7570b3", "#1b9e77"]  # seed0,1,2 distilled
PYSR_STYLES = ["--", "-.", ":"]
PYSR_MARKERS = ["s", "^", "D"]


def theta_grid(n=73):
    return np.linspace(0.0, 2 * np.pi, n)


def unwrap_deg(alpha_rad):
    """Unwrap a cyclic angle (rad) across the +-pi branch cut so the plotted
    curve is continuous, then convert to degrees."""
    return np.degrees(np.unwrap(alpha_rad))


def plot_wrapped(ax, x_deg, y_deg, **kwargs):
    ax.plot(x_deg, y_deg, **kwargs)


def raw_teacher_curve(model_path, L):
    model = SAC.load(model_path, device="cpu")
    p = TARGET_P + DP0
    f, g = E0 * np.cos(0.0), E0 * np.sin(0.0)
    alpha_rad = np.empty_like(L)
    for i, Li in enumerate(L):
        obs = np.array([p, f, g, np.cos(Li), np.sin(Li)], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        alpha_rad[i] = float(action[0]) * np.pi
    return unwrap_deg(alpha_rad)


def load_equations(csv_path):
    """label -> [(uT_str, uR_str), ...] grouped by SAC seed, in PySR-seed order."""
    by_teacher = {}
    with open(csv_path) as fp:
        for row in csv.DictReader(fp):
            m = re.match(r"SAC_seed(\d+)", row["label"])
            seed = int(m.group(1))
            by_teacher.setdefault(seed, []).append((row["uT_equation"], row["uR_equation"]))
    return by_teacher


def make_fn(eq_str):
    syms = symbols(FEATURES)
    expr = sympy.sympify(eq_str, locals=dict(zip(FEATURES, syms)))
    return lambdify(syms, expr, "numpy")


def distilled_curve(uT_str, uR_str, L):
    dp = np.full_like(L, DP0)
    f = np.full_like(L, E0 * np.cos(0.0))
    g = np.full_like(L, E0 * np.sin(0.0))
    cosL, sinL = np.cos(L), np.sin(L)
    fT, fR = make_fn(uT_str), make_fn(uR_str)
    uT = np.broadcast_to(fT(dp, f, g, cosL, sinL), L.shape).astype(float)
    uR = np.broadcast_to(fR(dp, f, g, cosL, sinL), L.shape).astype(float)
    return unwrap_deg(np.arctan2(uR, uT))


def qlaw_curve(L):
    alpha_rad = np.empty_like(L)
    p = TARGET_P + DP0
    f, g = E0 * np.cos(0.0), E0 * np.sin(0.0)
    for i, Li in enumerate(L):
        a = qlaw_policy(np.array([p, f, g, Li]), THRUST_ACCEL, TARGET_P, 0.0)
        alpha_rad[i] = float(a[0]) * np.pi
    return unwrap_deg(alpha_rad)


def consensus_curve(L):
    dp = DP0
    e = E0
    nu = L  # w = 0 => L = w + nu
    uT = -(0.75 * e * np.cos(nu) + 0.50 * dp) / (dp**2 + 0.088)
    uR = -4.65 * e * np.sin(nu)
    return unwrap_deg(np.arctan2(uR, uT))


def style_axes(ax):
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.yaxis.set_major_locator(MultipleLocator(90))
    ax.margins(y=0.08)
    ax.axhline(0, color="0.75", lw=0.6, ls=":", zorder=0)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlabel(r"$\theta$ (deg)")


def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "legend.fontsize": 12.5,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
    })

    L = theta_grid()
    L_deg = np.degrees(L)
    eqs_by_teacher = load_equations(EQUATIONS_CSV)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharey=False)
    axes = axes.ravel()

    for i, seed in enumerate(SEEDS):
        ax = axes[i]
        raw = raw_teacher_curve(MODEL_PATHS[seed], L)
        plot_wrapped(ax, L_deg, raw, color="#4c72b0", lw=1.6, ls="-", marker="o", ms=3,
                     label="teacher (raw NN)", zorder=3)
        for j, (uT_str, uR_str) in enumerate(eqs_by_teacher[seed]):
            curve = distilled_curve(uT_str, uR_str, L)
            plot_wrapped(ax, L_deg, curve, color=PYSR_COLORS[j], lw=1.4, ls=PYSR_STYLES[j],
                         marker=PYSR_MARKERS[j], ms=2.6,
                         alpha=0.9, label=f"distilled (PySR seed{j})", zorder=2)
        style_axes(ax)
        ax.set_ylabel(r"$\alpha$ (deg)")
        ax.set_title(f"SAC seed{seed}")
        ax.legend(loc="upper left", framealpha=0.9, borderpad=0.4, handlelength=1.6)

    ax6 = axes[5]
    qlaw = qlaw_curve(L)
    consensus = consensus_curve(L)
    plot_wrapped(ax6, L_deg, qlaw, color="#2ca02c", lw=1.8, ls="-", marker="o", ms=3,
                 label="Q-law (baseline)", zorder=3)
    plot_wrapped(ax6, L_deg, consensus, color="#d62728", lw=1.8, ls="--", marker="^", ms=3,
                 label="consensus closed form", zorder=2)
    style_axes(ax6)
    ax6.set_title("Q-law vs. consensus equation")
    ax6.legend(loc="upper left", framealpha=0.9, borderpad=0.4, handlelength=1.6)

    fig.tight_layout()

    fig.savefig(f"{OUT_PATH}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_PATH}.pdf", bbox_inches="tight")
    print(f"saved -> {OUT_PATH}.png / .pdf")


if __name__ == "__main__":
    main()
