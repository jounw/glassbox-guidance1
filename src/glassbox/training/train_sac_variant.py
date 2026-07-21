# src/glassbox/training/train_sac_variant.py
"""Train one SAC model with a given seed, for the cross-SAC reproducibility study.

Usage: python train_sac_variant.py <seed> <out_path> [total_timesteps]
"""

import sys

from stable_baselines3 import SAC

from glassbox.env.orbit_env import OrbitEnv


def main():
    seed = int(sys.argv[1])
    out_path = sys.argv[2]
    total_timesteps = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000

    env = OrbitEnv()
    model = SAC(
        "MlpPolicy", env, seed=seed, learning_rate=0.0003, verbose=0, device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
