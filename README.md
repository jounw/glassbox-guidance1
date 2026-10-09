# What Do RL Spacecraft Guidance Policies Learn?

**Symbolic Distillation against a Known Control Law** · NeurIPS 2026 Workshop on Interpretability for Discovery

[[Paper]](https://openreview.net/forum?id=XTuN0nml2m)

We train five SAC policies on planar low-thrust orbit raising, distill each into closed-form steering laws with PySR, and compare the result against Q-law.

- The raw networks steer differently, but 10 of 15 distillations collapse onto one common template: Q-law's small-eccentricity linearization.
- A two-line law built from the median parameters, with no tuning, succeeds on all 200 test episodes and uses less fuel than every teacher.

## Consensus law

```math
u_T = -\frac{0.75 \, e\cos\nu + 0.50 \, \delta p}{\delta p^2 + 0.088}, \qquad
u_R = -4.65 \, e\sin\nu, \qquad
\alpha = \mathrm{atan2}(u_R, u_T)
```

| System (200 episodes) | Success | ΔV |
| --- | --- | --- |
| SAC teachers (5) | 200/200 | 0.1833–0.1904 |
| Q-law (untuned) | 200/200 | 0.1762 |
| **Consensus law** | **200/200** | **0.1734** |

## Quick start

```bash
pip install numpy scipy gymnasium stable-baselines3 sympy matplotlib pysr
export PYTHONPATH=src
```

Reproduce the consensus-law result in about 30 seconds:

```python
import numpy as np
from glassbox.env import OrbitEnv

def consensus_law(state, p_target=2.0):
    p, f, g, L = state
    dp = p - p_target
    uT = -(0.75 * (f*np.cos(L) + g*np.sin(L)) + 0.50 * dp) / (dp**2 + 0.088)
    uR = -4.65 * (f*np.sin(L) - g*np.cos(L))
    return np.array([np.arctan2(uR, uT) / np.pi])

env, ok = OrbitEnv(), 0
for seed in range(100, 300):
    env.reset(seed=seed)
    term = trunc = False
    while not (term or trunc):
        _, _, term, trunc, _ = env.step(consensus_law(env.state))
    ok += term
print(f"{ok}/200 success")
```

## Contents

- `src/glassbox/env/`: orbit environment (modified equinoctial elements)
- `src/glassbox/models/`: five trained SAC teachers
- `src/glassbox/distill/`: dataset building and PySR runs
- `src/glassbox/analysis/`: closed-loop evaluation and gain analysis
- `src/glassbox/baselines/qlaw.py`: Q-law baseline
- `distilled_equations.csv`: all 15 distilled equation pairs

## Citation

```bibtex
@inproceedings{won2026what,
  title     = {What Do {RL} Spacecraft Guidance Policies Learn? Symbolic Distillation against a Known Control Law},
  author    = {Joun Won},
  booktitle = {NeurIPS 2026 Workshop on Interpretability for Discovery},
  year      = {2026},
  url       = {https://openreview.net/forum?id=XTuN0nml2m}
}
```
