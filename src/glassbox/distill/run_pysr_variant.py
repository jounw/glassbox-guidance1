# src/glassbox/distill/run_pysr_variant.py
"""Generalized run_pysr.py: fit uT or uR from an arbitrary dataset path,
for the cross-SAC reproducibility study.

Usage: python run_pysr_variant.py <data_path> <target: uT|uR> <seed> <output_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

PYSR_KWARGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    maxsize=25,
    niterations=400,
    populations=24,
    population_size=40,
    constraints={"/": (-1, 5)},
    model_selection="best",
    progress=True,
)


def run(data_path: str, target: str, seed: int, outdir: Path):
    from pysr import PySRRegressor

    assert target in ("uT", "uR")
    d = np.load(data_path, allow_pickle=True)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train = d[f"{target}_train"]
    y_test = d[f"{target}_test"]
    feature_names = [str(s) for s in d["feature_names"]]

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
        "data_path": data_path,
        "config": {k: str(v) for k, v in PYSR_KWARGS.items()},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pareto": rows,
    }
    with open(outdir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nsaved -> {outdir}/")


if __name__ == "__main__":
    data_path, target, seed, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    run(data_path, target, seed, outdir)
