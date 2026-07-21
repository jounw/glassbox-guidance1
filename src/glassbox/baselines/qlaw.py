
import numpy as np


mu = 1.0
h_small = 1e-4

def meeToKepler(state): # state = p, f, g, L
    p, f, g, L = state

    a = p/(1 -f**2 -g**2)
    e = np.sqrt(f**2 + g**2)
    if e < 1e-6:
        e = 1e-6
    theta = L -np.arctan2(g, f)
    w = 1+f*np.cos(L) + g*np.sin(L)
    r = p/w 
    h = np.sqrt(p)

    return a, e, theta, r, h


def Q_func(a, e, A, a_target, e_target=0.0):
    e = np.clip(e, 1e-6, 0.99)
    h = np.sqrt(a * (1 - e**2))   
    adot_xx = 2*A*np.sqrt(a**3*(1+e)/(mu*(1-e)))
    edot_xx = 2*(a*(1-e**2))*A/h
    return ((a - a_target)/adot_xx)**2 + ((e - e_target)/edot_xx)**2


def qlaw_policy(state, thrust_accel, a_target, e_target):

    a, e, theta, r, h = meeToKepler(state)
    p = state[0]
    Q_now = np.inf
    best_alpha = 0.0 

    for alpha in np.linspace(-np.pi, np.pi, 100):
        f_r, f_theta = thrust_accel*np.sin(alpha), thrust_accel*np.cos(alpha)

        st, ct = np.sin(theta), np.cos(theta)

        adot = (2 * a**2 / h) * (e * st * f_r + (p / r) * f_theta)
        edot = (1 / h) * (p * st * f_r + ((p + r) * ct + r * e) * f_theta)
        Q_next = Q_func(a + adot * h_small, e + edot * h_small, thrust_accel, a_target)
        if Q_next < Q_now:
            Q_now = Q_next
            best_alpha = alpha

    return np.array([best_alpha / np.pi], dtype=np.float32)
