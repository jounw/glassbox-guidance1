# src/glassbox/training/collect_rollouts.py
"""Generalized rollout collector for the cross-SAC reproducibility study.

Combines collect.py (default starts) and collect_perturbed.py (near-target /
overshoot starts) into one parameterized script so it can be pointed at any
trained SAC model.

Usage:
    python collect_rollouts.py <model_path> <out_path> <mode> [n_episodes] [seed_start]
    mode: "standard" (env default reset) or "perturbed" (p0 in [1.8,2.6], e0 in [0,0.1])
"""

import sys

import numpy as np
from stable_baselines3 import SAC

from glassbox.env.orbit_env import OrbitEnv

P0_RANGE = (1.8, 2.6)
E0_RANGE = (0.0, 0.1)


def collect(model, env, n_episodes, seed_start, mode, rng):
    cols = {k: [] for k in ["p", "f", "g", "L", "cosL", "sinL", "e", "theta",
                             "alpha", "cos_theta", "sin_theta", "episode_id",
                             "step_idx", "actions"]}
    n_fail = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        if mode == "perturbed":
            p0 = rng.uniform(*P0_RANGE)
            e0 = rng.uniform(*E0_RANGE)
            phi = rng.uniform(0, 2 * np.pi)
            L0 = rng.uniform(0, 2 * np.pi)
            env.state = np.array([p0, e0 * np.cos(phi), e0 * np.sin(phi), L0])
            env.step_count = 0
            env.prev_error = env._error()
            obs = env._get_obs()

        ep = {k: [] for k in cols}
        terminated = truncated = False
        step_idx = 0
        while not (terminated or truncated):
            p, f, g, L = np.array(env.state, dtype=np.float64)
            e = np.sqrt(f**2 + g**2)
            alpha = np.arctan2(g, f)
            theta = L - alpha

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep["p"].append(p); ep["f"].append(f); ep["g"].append(g); ep["L"].append(L)
            ep["cosL"].append(np.cos(L)); ep["sinL"].append(np.sin(L))
            ep["e"].append(e); ep["theta"].append(theta); ep["alpha"].append(alpha)
            ep["cos_theta"].append(np.cos(theta)); ep["sin_theta"].append(np.sin(theta))
            ep["episode_id"].append(seed_start + i)
            ep["step_idx"].append(step_idx)
            ep["actions"].append(action)
            step_idx += 1

        if not terminated:
            n_fail += 1
            continue
        for k in cols:
            cols[k].extend(ep[k])

    print(f"{mode}: kept {n_episodes - n_fail}/{n_episodes} episodes, "
          f"{len(cols['p'])} samples")
    return cols


def main():
    model_path = sys.argv[1]
    out_path = sys.argv[2]
    mode = sys.argv[3]
    n_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else (100 if mode == "standard" else 400)
    seed_start = int(sys.argv[5]) if len(sys.argv) > 5 else (0 if mode == "standard" else 1000)

    model = SAC.load(model_path, device="cpu")
    env = OrbitEnv()
    rng = np.random.default_rng(12345)

    cols = collect(model, env, n_episodes, seed_start, mode, rng)

    np.savez(
        out_path,
        **{k: np.array(v, dtype=np.int32 if k in ("episode_id", "step_idx") else np.float64)
           for k, v in cols.items()},
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
