
from stable_baselines3 import SAC
import gymnasium
import math
import matplotlib.pyplot as plt
from glassbox.env.orbit_env import OrbitEnv
model = SAC.load("src/glassbox/models/Orbit_SAC_1")
env = OrbitEnv()
obs, info = env.reset()
x_data = []
y_data = []
terminated = False
truncated = False
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    
    
    p, f, g, L = env.state[0], env.state[1], env.state[2], env.state[3]
    
    
    w = 1 + f * math.cos(L) + g * math.sin(L)
    x = (p / w) * math.cos(L)
    y = (p / w) * math.sin(L)
    
    x_data.append(x)
    y_data.append(y)

plt.figure(figsize=(8, 8))
plt.plot(x_data, y_data, 'b-', linewidth=1.5, label='Orbit Trajectory (from env.state)')
plt.scatter(x_data[0], y_data[0], color='green', marker='o', s=120, label='Start', zorder=5)
plt.scatter(x_data[-1], y_data[-1], color='red', marker='x', s=120, label='End', zorder=5)
plt.scatter(0, 0, color='black', marker='o', s=150, label='Earth (Center)')
plt.title('Orbit Spacecraft Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
