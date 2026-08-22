import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
from .dynamics import dynamics
from scipy.integrate import solve_ivp


class OrbitEnv(gym.Env):
    def __init__(self, thrust_accel=0.001, target_p= 2.0, max_steps=5000):
        self.thrust_accel = thrust_accel
        self.target_p =target_p
        self.max_steps = max_steps

        # self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1, ), dtype=np.float32 )
        self.observation_space = spaces.Box(low=np.array([0, -1, -1, -1, -1]), high=np.array([3*target_p, 1, 1, 1, 1]), shape=(5,), dtype=np.float32)


        self.state: np.ndarray | None = None
        self.step_count = 0

    def _get_obs(self):
        assert self.state is not None
        p, f, g, L = self.state
        Lw = L %(2*np.pi)
        return np.array([p, f, g, np.cos(Lw), np.sin(Lw)], dtype=np.float32)

    def _error(self):
        assert self.state is not None
        p, f, g, _ = self.state
        return abs(p - self.target_p)/self.target_p + np.sqrt(f**2 + g**2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        p0 = self.np_random.uniform(0.5, 0.8)* self.target_p
        e0= self.np_random.uniform(0, 0.1)
        phi = self.np_random.uniform(0, 2*np.pi)
        f0, g0 = e0*np.cos(phi), e0*np.sin(phi)
        L0 = self.np_random.uniform(0, 2*np.pi)

        self.state = np.array([p0, f0, g0, L0])
        self.step_count = 0 
        self.prev_error = self._error()
        return self._get_obs(), {}


    def step(self, action):
        alpha = float(action[0]) * np.pi
        p, f, g, _ = self.state
        a = p / (1 - f**2 - g**2)
        T = 2 * np.pi * a**1.5
        dt = T / 30

        sol = solve_ivp(dynamics, (0, dt), self.state, args=(alpha, self.thrust_accel), rtol=1e-9, atol=1e-12)
        self.state = sol.y[:, -1]
        self.step_count += 1

        error = self._error()
        p, f, g, _ = self.state
        e = np.sqrt(f**2 + g**2)

        reward = 100 * (self.prev_error - error) - 0.01
        self.prev_error = error

        terminated = error < 0.01
        if terminated:
            reward += 100

        truncated = (self.step_count >= self.max_steps or e > 0.9 or p > 3 * self.target_p or not sol.success)

        return self._get_obs(), reward, terminated, truncated, {}


        



