# src/glassbox/distill/run_pysr_no_overshoot.py
"""Ablation: refit uT/uR after dropping all p > target (dp > 0) samples.

Motivation (2026-07-15): the seed0 c=16/c=20 equations use a dp^2 "gate"
term whose only job is to flip sign in the overshoot region (dp > 0.2).
Ablating that term to a constant left closed-loop success at 200/200 on
the standard seed range, because those episodes never actually visit
dp > 0.05 (see gate1_by_dp.py / p_trajectory analysis). This script asks
the complementary question: if the overshoot region is removed from the
FITTING data entirely (not just ablated post-hoc), does PySR even
propose a dp^2 term, or does a simpler cruise-only expression win the
Pareto front?

Train AND test are both restricted to dp <= 0 (p <= target_p) so the
comparison is apples-to-apples within the retained domain -- this run is
not meant to pass gate 1 on overshoot cases, by construction.
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
    constraints={"/": (-1, 5)},
    model_selection="best",
    progress=True,
)


def run(target: str, seed: int):
    from pysr import PySRRegressor

    assert target in ("uT", "uR")
    d = np.load(DATA_PATH, allow_pickle=True)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train = d[f"{target}_train"]
    y_test = d[f"{target}_test"]
    feature_names = [str(s) for s in d["feature_names"]]

    dp_train, dp_test = X_train[:, 0], X_test[:, 0]
    keep_train = dp_train <= 0
    keep_test = dp_test <= 0
    print(f"train: {keep_train.sum()}/{len(dp_train)} kept (dp<=0)")
    print(f"test:  {keep_test.sum()}/{len(dp_test)} kept (dp<=0)")
    X_train, y_train = X_train[keep_train], y_train[keep_train]
    X_test, y_test = X_test[keep_test], y_test[keep_test]

    outdir = RESULTS_ROOT / f"pysr_{target}_no_overshoot_seed{seed}"
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
        "note": "train/test restricted to dp<=0 (p<=target_p); overshoot excluded",
        "config": {k: str(v) for k, v in PYSR_KWARGS.items()},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pareto": rows,
    }
    with open(outdir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nsaved -> {outdir}/")


if __name__ == "__main__":
    mode = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(mode, seed)
