import os
import numpy as np
from stable_baselines3 import SAC
from glassbox.env.orbit_env import OrbitEnv

# p, f, g, L, cosL, sinL, e, theta, alpha, cos_theta, sin_theta, episode_id, step_idx, actions


# p, f, g, L, cosL, sinL, e, theta, alpha, cos_theta, sin_theta, in_dist, actions 


env = OrbitEnv()

def collect(policy_fn, n_episodes=100, seed_start=1000):
    all_p = []
    all_f = []
    all_g = []
    all_L = []
    all_cosL = []
    all_sinL = []
    all_e = []
    all_theta = []
    all_alpha = []
    all_sin_theta = []
    all_cos_theta = []
    all_episode_id = []
    all_step_idx = []
    all_actions = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        
        terminated = False
        truncated = False
        step_idx = 0
        
        while not (terminated or truncated):
            state = np.array(env.state, dtype=np.float64)
            p, f, g, L = state
            
            e = np.sqrt(f**2 + g**2)
            alpha = np.arctan2(g, f)
            theta = L - alpha
            
            cosL = np.cos(L)
            sinL = np.sin(L)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            action = policy_fn(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            all_p.append(p)
            all_f.append(f)
            all_g.append(g)
            all_L.append(L)
            all_cosL.append(cosL)
            all_sinL.append(sinL)
            all_e.append(e)
            all_theta.append(theta)
            all_alpha.append(alpha)
            all_cos_theta.append(cos_theta)
            all_sin_theta.append(sin_theta)
            all_episode_id.append(i)
            all_step_idx.append(step_idx)
            all_actions.append(action)
            
            obs = next_obs
            step_idx +=1
    os.makedirs('data', exist_ok=True)
    
    np.savez(
        'data/sac_seed0_rollout.npz',
        p=np.array(all_p, dtype=np.float64),
        f=np.array(all_f, dtype=np.float64),
        g=np.array(all_g, dtype=np.float64),
        L=np.array(all_L, dtype=np.float64),
        cosL=np.array(all_cosL, dtype=np.float64),
        sinL=np.array(all_sinL, dtype=np.float64),
        e=np.array(all_e, dtype=np.float64),
        theta=np.array(all_theta, dtype=np.float64),
        alpha=np.array(all_alpha, dtype=np.float64),
        cos_theta=np.array(all_cos_theta, dtype=np.float64),
        sin_theta=np.array(all_sin_theta, dtype=np.float64),
        episode_id=np.array(all_episode_id, dtype=np.int32),
        step_idx=np.array(all_step_idx, dtype=np.int32),
        actions=np.array(all_actions, dtype=np.float64)
    )
    print("done")



model = SAC.load("src/glassbox/models/Orbit_SAC_1")
policy_fn= lambda obs: model.predict(obs, deterministic=True)[0]

def set_env_state(environment, state_vector):
    if hasattr(environment, 'set_state'):
        environment.set_state(state_vector)
    elif hasattr(environment, '_state'):
        environment._state = np.array(state_vector, dtype=np.float64)
    else:
        environment.state = np.array(state_vector, dtype=np.float64)

def collect_grid(policy_fn):
    
    p_space = np.linspace(0.8, 2.4, 20)
    e_space = np.linspace(0, 0.15, 10)
    phi_space = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    L_space = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    all_p = []
    all_f = []
    all_g = []
    all_L = []
    all_cosL = []
    all_sinL = []
    all_e = []
    all_theta = []
    all_alpha = []
    all_sin_theta = []
    all_cos_theta = []
    all_in_dist = []
    all_actions = []

    for p in p_space:
        in_dist_flag = 1.0 <= p <= 1.6
        
        for e in e_space:
            for phi in phi_space:
                f = e * np.cos(phi)
                g = e * np.sin(phi)
                
                for L in L_space:
                    state_to_inject = [p, f, g, L]
                    
                    set_env_state(env, state_to_inject)
                    
                    if hasattr(env, 'get_obs'):
                        obs = env.get_obs()
                    elif hasattr(env, '_get_obs'):
                        obs = env._get_obs()
                    else:
                        obs = np.array(state_to_inject, dtype=np.float32)
                    
                    action = policy_fn(obs)
                    
                    alpha = np.arctan2(g, f)
                    theta = L - alpha
                    
                    cosL = np.cos(L)
                    sinL = np.sin(L)
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    
                    all_p.append(p)
                    all_f.append(f)
                    all_g.append(g)
                    all_L.append(L)
                    all_cosL.append(cosL)
                    all_sinL.append(sinL)
                    all_e.append(e)
                    all_theta.append(theta)
                    all_alpha.append(alpha)
                    all_cos_theta.append(cos_theta)
                    all_sin_theta.append(sin_theta)
                    all_in_dist.append(int(in_dist_flag))  
                    all_actions.append(action)

    np.savez_compressed(
        'data/sac_grid_rollout.npz',
        p=np.array(all_p, dtype=np.float64),
        f=np.array(all_f, dtype=np.float64),
        g=np.array(all_g, dtype=np.float64),
        L=np.array(all_L, dtype=np.float64),
        cosL=np.array(all_cosL, dtype=np.float64),
        sinL=np.array(all_sinL, dtype=np.float64),
        e=np.array(all_e, dtype=np.float64),
        theta=np.array(all_theta, dtype=np.float64),
        alpha=np.array(all_alpha, dtype=np.float64),
        cos_theta=np.array(all_cos_theta, dtype=np.float64),
        sin_theta=np.array(all_sin_theta, dtype=np.float64),
        in_dist=np.array(all_in_dist, dtype=np.int32),
        actions=np.array(all_actions, dtype=np.float64)
    )
    print("done")


# =============================================================
if __name__ == '__main__':
    #collect_episodes(policy_fn, n_episodes=100, seed_start=1000)
    
    collect_grid(policy_fn)
