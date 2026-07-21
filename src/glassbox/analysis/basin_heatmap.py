"""Closed-loop success-rate heatmap over fixed-gain controllers
    uT = k * dp
    uR = G0 * (-e sin(nu))
swept over a (k, G0) grid, 20 episodes/cell, plus the linearized
gain of the consensus closed-form equation marked as a reference point.

Produces outputs/basin_heatmap.{png,pdf}.
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from glassbox.env.orbit_env import OrbitEnv

OUT_PATH = "outputs/basin_heatmap"

N_GRID = 15
N_EPISODES = 20
MAX_STEPS = 5000

K_RANGE = (-15.0, 7.5)
G0_RANGE = (-7.5, 15.0)

# linearized gains of the consensus closed form at dp ~ 0:
#   uT = -(0.75 e cos nu + 0.50 dp) / (dp^2 + 0.088)  ->  d(uT)/d(dp)|_{dp=0} = -0.50/0.088
#   uR = -4.65 e sin nu
CONSENSUS_K = -0.50 / 0.088
CONSENSUS_G0 = 4.65


def controller(obs, k, G0):
    p, f, g, cosL, sinL = obs
    dp = p - 2.0
    e = np.sqrt(f**2 + g**2)
    w = np.arctan2(g, f)
    L = np.arctan2(sinL, cosL)
    nu = L - w
    uT = k * dp
    uR = G0 * (-e * np.sin(nu))
    alpha = np.arctan2(uR, uT)
    return np.array([alpha / np.pi], dtype=np.float32)


def run_cell(k, G0, n_episodes, max_steps):
    env = OrbitEnv(max_steps=max_steps)
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = controller(obs, k, G0)
            obs, _, terminated, truncated, _ = env.step(action)
        if terminated:
            successes += 1
    return successes


def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
    })

    k_vals = np.linspace(*K_RANGE, N_GRID)
    G0_vals = np.linspace(*G0_RANGE, N_GRID)

    grid = np.zeros((N_GRID, N_GRID))
    t_start = time.time()
    total_cells = N_GRID * N_GRID
    for i, k in enumerate(k_vals):
        for j, G0 in enumerate(G0_vals):
            t0 = time.time()
            grid[i, j] = run_cell(k, G0, N_EPISODES, MAX_STEPS)
            done = i * N_GRID + j + 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total_cells - done)
            print(
                f"[{done:3d}/{total_cells}] k={k:+7.3f} G0={G0:+7.3f} "
                f"-> {int(grid[i, j]):2d}/{N_EPISODES}  "
                f"({time.time() - t0:5.2f}s cell, {elapsed:6.1f}s elapsed, "
                f"eta {eta:6.1f}s)",
                flush=True,
            )
        row_str = " ".join(f"{int(v):2d}" for v in grid[i])
        print(f"row k={k:+7.3f} done: [{row_str}]", flush=True)

    np.savez(f"{OUT_PATH}_data.npz", k=k_vals, G0=G0_vals, successes=grid)

    dk = k_vals[1] - k_vals[0]
    dG0 = G0_vals[1] - G0_vals[0]
    extent = [G0_vals[0] - dG0 / 2, G0_vals[-1] + dG0 / 2,
              k_vals[0] - dk / 2, k_vals[-1] + dk / 2]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto",
                   cmap="gray_r", vmin=0, vmax=N_EPISODES)

    ax.axhline(0, color="black", lw=1.2, ls="--", zorder=2)
    ax.axvline(0, color="black", lw=1.2, ls="--", zorder=2)

    ax.plot(CONSENSUS_G0, CONSENSUS_K, marker="*", ms=20, color="#d62728",
            markeredgecolor="black", markeredgewidth=0.8, zorder=3)
    ax.annotate("consensus\nformula", (CONSENSUS_G0, CONSENSUS_K),
                xytext=(8, -8), textcoords="offset points",
                color="#d62728", fontsize=13, va="top")

    ax.set_xlabel(r"$G_0$  (radial gain)")
    ax.set_ylabel(r"$k$  (tangential gain)")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"successes / {N_EPISODES}")

    fig.tight_layout()
    fig.savefig(f"{OUT_PATH}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_PATH}.pdf", bbox_inches="tight")
    print(f"saved -> {OUT_PATH}.png / .pdf / _data.npz")


if __name__ == "__main__":
    main()
