"""Stage 4 preprocessing: rollout npz -> stratified, episode-split PySR dataset.

Input : data/sac_seed0_rollout.npz      (Stage 3 output, default starts)
        data/sac_perturbed_rollout.npz  (near-target/overshoot starts)
Output: data/pysr_dataset.npz
        - X_train, X_test : (N, 5) float64  [dp, f, g, cosL, sinL]  (raw-only)
        - uT_train/test, uR_train/test : cos(alpha), sin(alpha) targets
        - alpha_train/test : ground-truth thrust angle (rad), for angle-error eval
        - episode_train/test : episode ids (provenance; perturbed ids >= 1000)
        Prints per-cell counts (distribution log -> paste into commit message).

Design decisions locked in conversation (2026-07-15):
- target from `actions` column (x pi). NEVER from `alpha` column (that is
  longitude of periapsis, a misnamed Stage-3 column).
- stratification is 2-D over (dp, e). The original e-only stratification
  left the dataset 94% cruise (dp < -0.2) with zero dp > +0.05 coverage,
  which is exactly where the distilled policy diverged closed-loop.
- train/test split at EPISODE level to avoid adjacent-step leakage;
  held-out episodes are drawn from BOTH sources so gate 1 actually tests
  the near-target/overshoot region.
"""

import numpy as np

# ---- config (constants in one place) ----------------------------------
INPUT_PATHS = [
    "src/glassbox/data/sac_seed0_rollout.npz",
    "src/glassbox/data/sac_perturbed_rollout.npz",
]
OUTPUT_PATH = "src/glassbox/data/pysr_dataset.npz"
DP_BIN_EDGES = np.array([-1.1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.7])   # 6 bins
E_BIN_EDGES = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])          # 5 bins
PER_CELL = 600            # target samples per (dp, e) cell -> <= 18k total
TEST_EPISODES_PER_SOURCE = 20   # last N episode ids of EACH source held out
SUBSAMPLE_SEED = 42
# -----------------------------------------------------------------------


def main():
    parts = {k: [] for k in ["p", "f", "g", "cosL", "sinL", "e", "episode_id", "actions"]}
    test_id_set = set()
    for path in INPUT_PATHS:
        d = np.load(path)
        for k in parts:
            col = d[k]
            parts[k].append(col[:, 0] if col.ndim == 2 else col)
        ep_ids = np.unique(d["episode_id"])
        test_id_set.update(ep_ids[-TEST_EPISODES_PER_SOURCE:].tolist())
        print(f"{path.split('/')[-1]}: {len(d['p'])} samples, {len(ep_ids)} episodes")

    p, f, g = (np.concatenate(parts[k]) for k in ["p", "f", "g"])
    cosL, sinL = np.concatenate(parts["cosL"]), np.concatenate(parts["sinL"])
    e = np.concatenate(parts["e"])
    ep = np.concatenate(parts["episode_id"])
    actions = np.concatenate(parts["actions"])

    # target: true thrust angle from executed action
    alpha_true = actions * np.pi
    uT = np.cos(alpha_true)
    uR = np.sin(alpha_true)

    dp = p - 2.0
    X = np.column_stack([dp, f, g, cosL, sinL]).astype(np.float64)

    # ---- episode-level split (before subsampling, so test set is untouched
    #      full-distribution data from both sources) ----
    is_test = np.isin(ep, list(test_id_set))
    print(f"held-out episodes: {len(test_id_set)} "
          f"({TEST_EPISODES_PER_SOURCE} per source)")

    # ---- stratified subsampling on TRAIN only, 2-D over (dp, e) ----
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    train_idx_all = np.where(~is_test)[0]
    dp_bins = np.digitize(dp[train_idx_all], DP_BIN_EDGES) - 1   # 0..5
    e_bins = np.digitize(e[train_idx_all], E_BIN_EDGES) - 1     # 0..4
    n_dp, n_e = len(DP_BIN_EDGES) - 1, len(E_BIN_EDGES) - 1
    in_range = (dp_bins >= 0) & (dp_bins < n_dp) & (e_bins >= 0) & (e_bins < n_e)
    n_out = int((~in_range).sum())
    if n_out:
        print(f"  (excluded: {n_out} train samples outside dp/e grid)")

    keep = []
    print(f"\n{'dp bin':>16} {'e bin':>14} {'available':>9} {'kept':>6}")
    for bd in range(n_dp):
        for be in range(n_e):
            pool = train_idx_all[in_range & (dp_bins == bd) & (e_bins == be)]
            if len(pool) == 0:
                continue
            n_keep = min(PER_CELL, len(pool))
            keep.append(rng.choice(pool, size=n_keep, replace=False))
            print(f"[{DP_BIN_EDGES[bd]:+.2f},{DP_BIN_EDGES[bd+1]:+.2f}) "
                  f"[{E_BIN_EDGES[be]:.2f},{E_BIN_EDGES[be+1]:.2f}) "
                  f"{len(pool):>9} {n_keep:>6}")

    train_idx = np.sort(np.concatenate(keep))
    test_idx = np.where(is_test)[0]

    print(f"\ntrain: {len(train_idx)}  test: {len(test_idx)} (full, unsubsampled)")

    np.savez(
        OUTPUT_PATH,
        X_train=X[train_idx], X_test=X[test_idx],
        uT_train=uT[train_idx], uT_test=uT[test_idx],
        uR_train=uR[train_idx], uR_test=uR[test_idx],
        alpha_train=alpha_true[train_idx], alpha_test=alpha_true[test_idx],
        episode_train=ep[train_idx], episode_test=ep[test_idx],
        feature_names=np.array(["dp", "f", "g", "cosL", "sinL"]),
    )
    print(f"saved -> {OUTPUT_PATH}")

    # quick feature stats for the commit message
    for name, col in zip(["dp", "f", "g", "cosL", "sinL"], X[train_idx].T):
        print(f"  {name:>5}: [{col.min():+.4f}, {col.max():+.4f}]")


if __name__ == "__main__":
    main()
