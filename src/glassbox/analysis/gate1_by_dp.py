
import numpy as np

from glassbox.distill.eval_openloop import make_fn, load_expr

UT_DIR, UT_COMPLEXITY = "results/pysr_uT_seed0", 16
UR_DIR, UR_COMPLEXITY = "results/pysr_uR_seed0", 20
DATASET_PATH = "src/glassbox/data/pysr_dataset.npz"
DP_BIN_EDGES = [-1.1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.7]


def main():
    eq_T = load_expr(UT_DIR, UT_COMPLEXITY)
    eq_R = load_expr(UR_DIR, UR_COMPLEXITY)
    fT, _ = make_fn(eq_T)
    fR, _ = make_fn(eq_R)

    d = np.load(DATASET_PATH)
    X, alpha_true = d["X_test"], d["alpha_test"]
    cols = [X[:, i] for i in range(5)]

    uT = np.broadcast_to(fT(*cols), alpha_true.shape).astype(float)
    uR = np.broadcast_to(fR(*cols), alpha_true.shape).astype(float)
    alpha_hat = np.arctan2(uR, uT)
    err = np.abs(np.degrees(np.angle(np.exp(1j * (alpha_hat - alpha_true)))))

    dp = X[:, 0]
    print(f"{'dp bin':>16} {'N':>6} {'median':>8} {'p90':>8}")
    for lo, hi in zip(DP_BIN_EDGES[:-1], DP_BIN_EDGES[1:]):
        m = (dp >= lo) & (dp < hi)
        if m.sum() == 0:
            continue
        print(f"[{lo:+.2f},{hi:+.2f}) {m.sum():>6} "
              f"{np.median(err[m]):>7.2f}° {np.percentile(err[m], 90):>7.2f}°")


if __name__ == "__main__":
    main()
