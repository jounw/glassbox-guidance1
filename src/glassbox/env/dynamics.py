# env.py v1

import math
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt



mu = 1.0
t = 1

alpha = 0.0
thrust_accel = 0.0001


def dynamics(t, state, alpha, thrust_accel):
    p, f, g, L = state

    w= 1+ f*math.cos(L) + g*math.sin(L)
    a_r = thrust_accel*math.sin(alpha)
    a_t = thrust_accel*math.cos(alpha)

    cL = math.cos(L)
    sL = math.sin(L)

    p_dot = (2*p/w) * math.sqrt(p/mu) * a_t
    f_dot = math.sqrt(p/mu) * (a_r* sL + ((w+1)* cL + f) * a_t/w)
    g_dot = math.sqrt(p/mu)*(- a_r* cL + ((w+1)* sL + g) * a_t/w)
    L_dot = math.sqrt(mu*p)*(w/p)**2


    return p_dot, f_dot, g_dot, L_dot


t_span = (0, 20*np.pi)
initial_state = (1.0, 0.0, 0.0, 0.0)
args = (alpha, thrust_accel)
t_eval=np.linspace(0, 20*np.pi, 2000)

#sol = solve_ivp(dynamics, t_span, initial_state, args=args, t_eval=t_eval, rtol=1e-9, atol=1e-12, method='RK45')

 

#print(np.abs(sol.y[1]).max(), np.abs(sol.y[2]).max())  # f, g 

#print(sol.status)
#print(sol.success)
#print(sol.t[-1])
#print(len(sol.t))
#print(sol.y[3][-1])
# print (dynamics(1, (1, 0, 0, 0),0,0 )

#print(sol.y)
