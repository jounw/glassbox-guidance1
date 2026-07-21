# src/glassbox/distill/run_pysr_alpha.py
"""Trial: regress alpha directly with a wrap-aware periodic loss, instead
of fitting uT=cos(alpha) and uR=sin(alpha) as two independent regressions.

Motivation (2026-07-15): the two-regression approach loses the unit-circle
constraint uT^2+uR^2=1. Near dp=0 the fitted (uT_hat, uR_hat) vector
magnitude collapses (median 0.14, min 0.007 vs 0.5-0.9 elsewhere), so
ordinary regression error in uT/uR gets amplified into large angle error
via atan2 -- this is exactly gate 1's worst region.

Fix tried here: fit alpha itself with elementwise_loss =
1 - cos(prediction - target), which is smooth, periodic, and treats
179 deg vs -179 deg as a 2-degree difference instead of a 358-degree one.
sin/cos are opened as unary operators since alpha must be able to express
its own periodicity.

This is a quick trial (reduced niterations) to check the approach is
viable before committing to a full run.
"""

import json
from pathlib import Path

import numpy as np

DATA_PATH = "src/glassbox/data/pysr_dataset.npz"
RESULTS_ROOT = Path("results")

PYSR_KWARGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "sin", "cos"],
    elementwise_loss="loss(prediction, target) = 1 - cos(prediction - target)",
    maxsize=25,
    niterations=150,   # trial: reduced from 400 for a quick feasibility check
    populations=24,
    population_size=40,
    constraints={"/": (-1, 5)},
    model_selection="best",
    progress=True,
)


def wrapped_angle_err_deg(alpha_hat, alpha_true):
    d = np.angle(np.exp(1j * (alpha_hat - alpha_true)))
    return np.abs(np.degrees(d))


def run(seed: int):
    from pysr import PySRRegressor

    d = np.load(DATA_PATH, allow_pickle=True)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["alpha_train"], d["alpha_test"]
    feature_names = [str(s) for s in d["feature_names"]]

    outdir = RESULTS_ROOT / f"pysr_alpha_trial_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    model = PySRRegressor(
        **PYSR_KWARGS,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        output_directory=str(outdir),
    )
    model.fit(X_train, y_train, variable_names=feature_names)

    eqs = model.equations_
    eqs.to_csv(outdir / "hall_of_fame.csv", index=False)

    rows = []
    for _, row in eqs.iterrows():
        pred = model.predict(X_test, index=row.name)
        ang_err = wrapped_angle_err_deg(pred, y_test)
        rows.append({
            "complexity": int(row["complexity"]),
            "train_loss": float(row["loss"]),
            "test_median_angle_err_deg": float(np.median(ang_err)),
            "test_p90_angle_err_deg": float(np.percentile(ang_err, 90)),
            "equation": str(row["equation"]),
        })
        print(f"  c={row['complexity']:>2}  median_err={np.median(ang_err):6.2f}deg  "
              f"p90={np.percentile(ang_err, 90):6.2f}deg  {row['equation']}")

    summary = {
        "target": "alpha (direct, periodic loss)",
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
    run(seed=0)
