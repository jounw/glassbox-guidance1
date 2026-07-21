# src/glassbox/analysis/compare_policies.py
"""Closed-loop 200-seed comparison: SAC (teacher) vs Q-law (untuned) vs distilled.

Runs the same evaluate.py protocol for all three policies back to back on
the same seed range, so results are directly comparable.
"""

import numpy as np
from stable_baselines3 import SAC

from glassbox.analysis.evaluate import env, evaluate
from glassbox.baselines.qlaw import qlaw_policy
from glassbox.distill.eval_openloop import make_fn, load_expr

UT_DIR, UT_COMPLEXITY = "results/pysr_uT_seed0", 16
UR_DIR, UR_COMPLEXITY = "results/pysr_uR_seed0", 20
SAC_MODEL_PATH = "src/glassbox/models/Orbit_SAC_1"
N_EPISODES = 200
SEED_START = 100


def make_distilled_policy():
    fT, _ = make_fn(load_expr(UT_DIR, UT_COMPLEXITY))
    fR, _ = make_fn(load_expr(UR_DIR, UR_COMPLEXITY))

    def policy(obs):
        p, f, g, L = env.state
        dp = p - 2.0
        uT = fT(dp, f, g, np.cos(L), np.sin(L))
        uR = fR(dp, f, g, np.cos(L), np.sin(L))
        return np.array([np.arctan2(uR, uT) / np.pi])

    return policy


def summarize_row(name, results):
    ok = [r for r in results if r["reached"]]
    n = len(results)
    if not ok:
        return {"name": name, "n_ok": 0, "n": n, "steps_mean": None,
                "steps_std": None, "e_mean": None, "e_std": None}
    steps = np.array([r["steps"] for r in ok])
    e = np.array([r["final_e"] for r in ok])
    return {
        "name": name, "n_ok": len(ok), "n": n,
        "steps_mean": steps.mean(), "steps_std": steps.std(),
        "e_mean": e.mean(), "e_std": e.std(),
    }


def main():
    model = SAC.load(SAC_MODEL_PATH, device="cpu")
    policies = {
        "SAC (teacher)": lambda obs: model.predict(obs, deterministic=True)[0],
        "Q-law (untuned)": lambda obs: qlaw_policy(env.state, 0.001, 2.0, 0.0),
        "Distilled": make_distilled_policy(),
    }

    rows = []
    all_results = {}
    for name, policy_fn in policies.items():
        print(f"running {name}...")
        results = evaluate(policy_fn, n_episodes=N_EPISODES, seed_start=SEED_START)
        all_results[name] = results
        rows.append(summarize_row(name, results))

    print(f"\n{'policy':<18} {'success':>9} {'steps':>16} {'final e':>16}")
    for r in rows:
        if r["n_ok"] == 0:
            print(f"{r['name']:<18} {r['n_ok']:>3}/{r['n']:<5} {'--':>16} {'--':>16}")
            continue
        print(f"{r['name']:<18} {r['n_ok']:>3}/{r['n']:<5} "
              f"{r['steps_mean']:>6.0f} ± {r['steps_std']:<5.0f}  "
              f"{r['e_mean']:>6.4f} ± {r['e_std']:<6.4f}")

    return all_results


if __name__ == "__main__":
    main()
