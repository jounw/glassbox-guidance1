
# src/glassbox/analysis/evaluate.py
import numpy as np
from stable_baselines3 import SAC
from glassbox.env.orbit_env import OrbitEnv
from glassbox.distill.eval_openloop import make_fn, load_expr


from glassbox.baselines.qlaw import qlaw_policy

env = OrbitEnv()

def evaluate(policy_fn, n_episodes=200, seed_start=100):
    results = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        init_p, init_f, init_g, _ = env.state
        init_e = np.sqrt(init_f**2 + init_g**2)

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            steps += 1

        p, f, g, _ = env.state
        results.append({
            "reached": terminated,
            "steps": steps,
            "final_e": np.sqrt(f**2 + g**2),
            "final_p_err": abs(p - env.target_p) / env.target_p,
            "init_p": init_p,
            "init_e": init_e,
        })
        print(i)

    return results


def summarize(results):
    n = len(results)
    ok = [r for r in results if r["reached"]]
    print(f"success rate: {len(ok)}/{n}")
    if ok:
        steps = np.array([r["steps"] for r in ok])
        fe = np.array([r["final_e"] for r in ok])
        print(f"steps to reach: {steps.mean():.0f} ± {steps.std():.0f}")
        print(f"final e: {fe.mean():.4f} ± {fe.std():.4f}")
    for i, r in enumerate(results):
        if not r["reached"]:
            print(f"FAIL ep{i}: init_p={r['init_p']:.3f}, init_e={r['init_e']:.3f}")



eq_T = load_expr("results/pysr_uT_seed0", 16)
eq_R = load_expr("results/pysr_uR_seed0", 20)

fT, _ = make_fn(eq_T)
fR, _ = make_fn(eq_R)

def distilled_policy(state):
    p, f, g, L = state
    p = p-2.0
    uT = fT(p, f, g, np.cos(L), np.sin(L))
    uR = fR(p, f, g, np.cos(L), np.sin(L))
    return np.array([np.arctan2(uR, uT) / np.pi])






if __name__ == "__main__":
#    model = SAC.load("src/glassbox/models/Orbit_SAC_1")
    #policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]

   # policy_fn = lambda obs: qlaw_policy(env.state, 0.001, 2.0, 0.0)
   policy_fn = lambda obs: distilled_policy(env.state)
   summarize(evaluate(policy_fn))
