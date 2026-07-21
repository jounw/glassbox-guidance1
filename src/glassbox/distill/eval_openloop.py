"""Stage 4 validation gate 1 (open-loop).

Combine a chosen uT expression and uR expression, reconstruct
alpha_hat = atan2(uR_hat, uT_hat), and measure wrap-aware angle error
on the held-out test set.

Usage:
    python eval_openloop.py results/pysr_uT_seed0 CU results/pysr_uR_seed0 CR
where CU / CR are the chosen complexity levels (pick at the Pareto elbow).

PASS criterion (pre-registered): median angle error < 5 deg.
Also reports p90 and per-e-bin breakdown.

Note: closed-loop gate (plugging the expression into OrbitEnv as a
controller, 200-seed protocol) is intentionally NOT here — that script
touches env stepping logic and is yours to write. This file only does
dataset-level arithmetic.
"""

import json
import sys

import numpy as np
import sympy
from sympy import lambdify, symbols

DATA_PATH = "src/glassbox/data/pysr_dataset.npz"
FEATURES = ["dp", "f", "g", "cosL", "sinL"]


def load_expr(run_dir: str, complexity: int):
    with open(f"{run_dir}/summary.json") as fp:
        s = json.load(fp)
    for row in s["pareto"]:
        if row["complexity"] == complexity:
            return row["equation"]
    raise SystemExit(f"complexity {complexity} not on Pareto front in {run_dir}")


def make_fn(eq_str: str):
    syms = symbols(FEATURES)
    expr = sympy.sympify(eq_str, locals=dict(zip(FEATURES, syms)))
    return lambdify(syms, expr, "numpy"), expr


def main():
    uT_dir, cT, uR_dir, cR = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])

    eq_T = load_expr(uT_dir, cT)
    eq_R = load_expr(uR_dir, cR)
    print(f"uT (c={cT}): {eq_T}")
    print(f"uR (c={cR}): {eq_R}")

    fT, _ = make_fn(eq_T)
    fR, _ = make_fn(eq_R)

    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X_test"]
    alpha_true = d["alpha_test"]
    cols = [X[:, i] for i in range(5)]

    uT_hat = np.broadcast_to(fT(*cols), alpha_true.shape).astype(float)
    uR_hat = np.broadcast_to(fR(*cols), alpha_true.shape).astype(float)
    alpha_hat = np.arctan2(uR_hat, uT_hat)

    err = np.abs(np.degrees(np.angle(np.exp(1j * (alpha_hat - alpha_true)))))
    med, p90 = np.median(err), np.percentile(err, 90)
    print(f"\nangle error: median {med:.2f} deg, p90 {p90:.2f} deg, "
          f"max {err.max():.2f} deg   (N={len(err)})")

    # per-e-bin breakdown
    e = np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2)
    edges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, np.inf]
    print(f"\n{'e bin':>14} {'N':>6} {'median':>8} {'p90':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e >= lo) & (e < hi)
        if m.sum() == 0:
            continue
        print(f"[{lo:.2f}, {hi:.2f}) {m.sum():>6} "
              f"{np.median(err[m]):>7.2f}° {np.percentile(err[m], 90):>7.2f}°")

    print(f"\nGATE 1 (median < 5 deg): {'PASS' if med < 5.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
