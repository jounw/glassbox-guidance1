from stable_baselines3 import SAC
import gymnasium 
import math
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from glassbox.env.orbit_env import OrbitEnv

model = SAC.load("src/glassbox/models/Orbit_SAC_1")

env = OrbitEnv()
obs, info = env.reset()

p_data = []
e_data = []
alpha_data = []
step_data = []

terminated = False
truncated = False
step = 0

while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    
    p, f, g, L = env.state[0], env.state[1], env.state[2], env.state[3]
    
    e = math.sqrt(f**2 + g**2)
    
    alpha = action[0] 
    
    p_data.append(p)
    e_data.append(e)
    alpha_data.append(alpha)
    step_data.append(step)
    
    step += 1

# ==========================================
# ==========================================
plt.style.use(['science', 'no-latex', 'vibrant'])
fig, axs = plt.subplots(3, 1, sharex=True)


axs[0].plot(step_data, p_data)
axs[0].set_ylabel('Semi-latus rectum (p)')

axs[1].plot(step_data, e_data)
axs[1].set_ylabel('Eccentricity (e)')

axs[2].plot(step_data, alpha_data)
axs[2].set_ylabel('Alpha (α) [rad]')
axs[2].set_xlabel('Time Step')

plt.tight_layout()
#fig.savefig("./results/timeseries_seed0.pdf")
plt.show()
