

import csv
import re

import numpy as np
import sympy
from sympy import lambdify, symbols

EQUATIONS_CSV = "outputs/distilled_equations.csv"
FEATURES = ["dp", "f", "g", "cosL", "sinL"]

E_PROBE = 0.03
H_DP = 1e-6          # central-difference step for k
N_NU = 256           # nu grid for Fourier projection
N_W = 8              # periapsis angles averaged over
N_L = 64             # L grid averaged over for k


def make_fn(eq_str):
    syms = symbols(FEATURES)
    loc = dict(zip(FEATURES, syms))
    loc["square"] = lambda x: x**2
    expr = sympy.sympify(eq_str, locals=loc)
    return lambdify(syms, expr, "numpy")


def eval_uT(fn, dp, f, g, L):
    L = np.asarray(L, dtype=float)
    out = fn(np.full_like(L, dp), np.full_like(L, f), np.full_like(L, g),
             np.cos(L), np.sin(L))
    return np.broadcast_to(out, L.shape).astype(float)


def tangential_gain_k(fn):
    """d(uT)/d(dp) at dp=0, e=0 (f=g=0), averaged over L."""
    L = np.linspace(0, 2 * np.pi, N_L, endpoint=False)
    up = eval_uT(fn, +H_DP, 0.0, 0.0, L)
    dn = eval_uT(fn, -H_DP, 0.0, 0.0, L)
    return float(np.mean((up - dn) / (2 * H_DP)))


def first_harmonic(fn, harmonic):
    """cos(nu) or sin(nu) Fourier coefficient of the expression at dp=0,
    e=E_PROBE, divided by e, averaged over periapsis angle w."""
    nu = np.linspace(0, 2 * np.pi, N_NU, endpoint=False)
    basis = np.cos(nu) if harmonic == "cos" else np.sin(nu)
    coeffs = []
    for w in np.linspace(0, 2 * np.pi, N_W, endpoint=False):
        f, g = E_PROBE * np.cos(w), E_PROBE * np.sin(w)
        L = w + nu
        u = eval_uT(fn, 0.0, f, g, L)
        coeffs.append(2.0 * np.mean(u * basis) / E_PROBE)
    return float(np.mean(coeffs)), float(np.std(coeffs))


def main():
    rows = list(csv.DictReader(open(EQUATIONS_CSV)))
    assert len(rows) == 15, f"expected 15 rows, got {len(rows)}"

    ks, Acs, G0s = [], [], []
    print(f"{'label':<45} {'k':>8} {'A_cos':>8} {'G0':>8}   (std over w)")
    for row in rows:
        fT = make_fn(row["uT_equation"])
        fR = make_fn(row["uR_equation"])

        k = tangential_gain_k(fT)
        A_cos, A_std = first_harmonic(fT, "cos")
        # uR ~ G0 * (-e sin nu)  ->  G0 = -(sin-harmonic coefficient)
        sin_coeff, s_std = first_harmonic(fR, "sin")
        G0 = -sin_coeff

        ks.append(k)
        Acs.append(A_cos)
        G0s.append(G0)
        print(f"{row['label']:<45} {k:>8.3f} {A_cos:>8.3f} {G0:>8.3f}   "
              f"(±{A_std:.3f}, ±{s_std:.3f})")

    ks, Acs, G0s = map(np.array, (ks, Acs, G0s))

    print("\n--- sign consistency ---")
    print(f"k < 0     : {int((ks < 0).sum())}/15")
    print(f"A_cos < 0 : {int((Acs < 0).sum())}/15")
    print(f"G0 > 0    : {int((G0s > 0).sum())}/15")

    print("\n--- medians (paper A.5: |k|=5.72, |A_cos|=8.50, G0=4.65) ---")
    print(f"median |k|     = {np.median(np.abs(ks)):.3f}")
    print(f"median |A_cos| = {np.median(np.abs(Acs)):.3f}")
    print(f"median G0      = {np.median(G0s):.3f}")

    print("\n--- spread (paper 3.2: k CV 7.5%, G0 CV 15.7%, A_cos CV 34%) ---")
    for name, v in [("k", np.abs(ks)), ("G0", np.abs(G0s)), ("A_cos", np.abs(Acs))]:
        print(f"{name:>6}: mean {v.mean():.3f}, CV {100*v.std()/v.mean():.1f}%")

    print(f"\nconsensus check: median|A_cos| * 0.088 = "
          f"{np.median(np.abs(Acs))*0.088:.3f} (Eq.6: 0.75), "
          f"median|k| * 0.088 = {np.median(np.abs(ks))*0.088:.3f} (Eq.6: 0.50)")


if __name__ == "__main__":
    main()
