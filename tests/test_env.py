from glassbox.env.orbit_env import OrbitEnv
from gymnasium.utils.env_checker import check_env
import numpy as np

env = OrbitEnv()
check_env(env)
print("check_env passed")

obs, info = env.reset(seed=42)
print("obs:", obs)

obs, _ = env.reset(seed=1)
for i in range(env.max_steps):
    p = env.state[0]
    a = 0.0 if p < env.target_p else 1.
    obs, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
    if term:
        print(f"reached target in {i+1} steps")
        break
    if trunc:
        p, f, g, L = env.state
        e = np.sqrt(f**2 + g**2)
        print(f"p={p:.4f}, e={e:.4f}, steps={env.step_count}, error={env._error():.4f}")
        print(f"truncated at {i+1}")
        break
