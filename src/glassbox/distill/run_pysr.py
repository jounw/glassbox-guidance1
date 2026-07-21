"""Stage 4: symbolic distillation of the SAC policy via PySR.

Modes:
    python run_pysr.py toy            # install/compat sanity check (run FIRST)
    python run_pysr.py uT [seed]      # distill tangential component cos(alpha)
    python run_pysr.py uR [seed]      # distill radial component     sin(alpha)

Repeat-run protocol: run uT/uR each with seeds 0..2 (or 0..4) and compare
which structures dominate the Pareto front across runs.

Outputs (per run): results/pysr_{target}_seed{n}/
    - hall_of_fame.csv      (full Pareto front: complexity, loss, equation)
    - summary.json          (config + best-by-elbow candidates + test metrics)

Design locked in conversation (2026-07-15):
- features: raw-only [p, f, g, cosL, sinL]
- operators: +, -, *, / and square only (no trig — cosL/sinL are features;
  no sqrt to start — f^2+g^2 can express e^2; add sqrt only if the Pareto
  front visibly stalls, and log that decision)
- maxsize 25, angle reconstruction alpha_hat = atan2(uR_hat, uT_hat)
"""

import json
import sys
from pathlib import Path

import numpy as np

DATA_PATH = "src/glassbox/data/pysr_dataset.npz"
RESULTS_ROOT = Path("results")

PYSR_KWARGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    maxsize=25,
    niterations=400,
    populations=24,
    population_size=40,
    # keep expressions interpretable: no nested division
    constraints={"/": (-1, 5)},
    model_selection="best",
    progress=True,
)


def toy_check():
    """Verify PySR + Julia work at all (Python 3.14 risk). Target:
    y = 2*x0*x1 - 0.5*x2^2. PySR must recover it near-exactly."""
    from pysr import PySRRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(500, 3))
    y = 2.0 * X[:, 0] * X[:, 1] - 0.5 * X[:, 2] ** 2

    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        maxsize=15,
        niterations=40,
        progress=True,
    )
    model.fit(X, y)
    pred = model.predict(X)
    mse = float(np.mean((pred - y) ** 2))
    print(f"\ntoy MSE: {mse:.3e}  (expect ~1e-10 or better)")
    print("best equation:", model.get_best()["equation"])
    ok = mse < 1e-6
    print("TOY CHECK:", "PASS" if ok else "FAIL -- investigate before main runs")
    return ok


def wrapped_angle_err_deg(alpha_hat, alpha_true):
    d = np.angle(np.exp(1j * (alpha_hat - alpha_true)))
    return np.abs(np.degrees(d))


def run(target: str, seed: int):
    from pysr import PySRRegressor

    assert target in ("uT", "uR")
    d = np.load(DATA_PATH, allow_pickle=True)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train = d[f"{target}_train"]
    y_test = d[f"{target}_test"]
    feature_names = [str(s) for s in d["feature_names"]]

    outdir = RESULTS_ROOT / f"pysr_{target}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    model = PySRRegressor(
        **PYSR_KWARGS,
        random_state=seed,
        deterministic=True,
        parallelism="serial",  # required for deterministic runs
        output_directory=str(outdir),
    )
    model.fit(X_train, y_train, variable_names=feature_names)

    # full Pareto front
    eqs = model.equations_
    eqs.to_csv(outdir / "hall_of_fame.csv", index=False)

    # per-complexity test loss (component-level; angle-level eval is done
    # in eval_openloop.py once BOTH components exist)
    rows = []
    for _, row in eqs.iterrows():
        pred = model.predict(X_test, index=row.name)
        mse = float(np.mean((pred - y_test) ** 2))
        rows.append({
            "complexity": int(row["complexity"]),
            "train_loss": float(row["loss"]),
            "test_mse": mse,
            "equation": str(row["equation"]),
        })
        print(f"  c={row['complexity']:>2}  test_mse={mse:.3e}  {row['equation']}")

    summary = {
        "target": target,
        "seed": seed,
        "config": {k: str(v) for k, v in PYSR_KWARGS.items()},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pareto": rows,
    }
    with open(outdir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nsaved -> {outdir}/")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "toy"
    if mode == "toy":
        sys.exit(0 if toy_check() else 1)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(mode, seed)
