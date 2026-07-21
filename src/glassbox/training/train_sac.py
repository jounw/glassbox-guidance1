from stable_baselines3 import SAC
import gymnasium as gymnasium

from glassbox.env.orbit_env import OrbitEnv

env = OrbitEnv()
# check_env(env, warn=True)


model = SAC(
        "MlpPolicy", env, seed=0, tensorboard_log="./logs", learning_rate=0.0003, verbose=2, device="cpu" )
model.learn(total_timesteps=200_000)
model.save("Orbit_SAC_1")


model = SAC.load("src/glassbox/modles/Orbit_SAC_2")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
