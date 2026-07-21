# src/glassbox/distill/prepare_data_variant.py
"""Generalized version of prepare_data.py: build a 2-D (dp,e)-stratified
PySR dataset from a standard + perturbed rollout pair, for any SAC model.

Usage: python prepare_data_variant.py <standard_npz> <perturbed_npz> <out_npz>
"""

import sys

import numpy as np

DP_BIN_EDGES = np.array([-1.1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.7])
E_BIN_EDGES = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
PER_CELL = 600
TEST_EPISODES_PER_SOURCE = 20
SUBSAMPLE_SEED = 42


def main():
    standard_path, perturbed_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    parts = {k: [] for k in ["p", "f", "g", "cosL", "sinL", "e", "episode_id", "actions"]}
    test_id_set = set()
    for path in [standard_path, perturbed_path]:
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

    alpha_true = actions * np.pi
    uT = np.cos(alpha_true)
    uR = np.sin(alpha_true)

    dp = p - 2.0
    X = np.column_stack([dp, f, g, cosL, sinL]).astype(np.float64)

    is_test = np.isin(ep, list(test_id_set))
    print(f"held-out episodes: {len(test_id_set)} ({TEST_EPISODES_PER_SOURCE} per source)")

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    train_idx_all = np.where(~is_test)[0]
    dp_bins = np.digitize(dp[train_idx_all], DP_BIN_EDGES) - 1
    e_bins = np.digitize(e[train_idx_all], E_BIN_EDGES) - 1
    n_dp, n_e = len(DP_BIN_EDGES) - 1, len(E_BIN_EDGES) - 1
    in_range = (dp_bins >= 0) & (dp_bins < n_dp) & (e_bins >= 0) & (e_bins < n_e)

    keep = []
    for bd in range(n_dp):
        for be in range(n_e):
            pool = train_idx_all[in_range & (dp_bins == bd) & (e_bins == be)]
            if len(pool) == 0:
                continue
            n_keep = min(PER_CELL, len(pool))
            keep.append(rng.choice(pool, size=n_keep, replace=False))

    train_idx = np.sort(np.concatenate(keep))
    test_idx = np.where(is_test)[0]
    print(f"train: {len(train_idx)}  test: {len(test_idx)}")

    np.savez(
        out_path,
        X_train=X[train_idx], X_test=X[test_idx],
        uT_train=uT[train_idx], uT_test=uT[test_idx],
        uR_train=uR[train_idx], uR_test=uR[test_idx],
        alpha_train=alpha_true[train_idx], alpha_test=alpha_true[test_idx],
        episode_train=ep[train_idx], episode_test=ep[test_idx],
        feature_names=np.array(["dp", "f", "g", "cosL", "sinL"]),
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
