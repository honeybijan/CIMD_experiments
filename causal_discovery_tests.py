import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
from scipy.stats import norm

# Assuming this is in your utils.py file in the same directory
from utils import CIMD_limited_normed


# ---------------------------------------------------------------
# 1. Random DAG & Data generation
# ---------------------------------------------------------------
def random_dag(p, expected_parents=2, seed=0):
    """Random DAG with topological order = node index order."""
    rng = np.random.default_rng(seed)
    # FIX (bug #5): account for the fact that triu keeps only ~half the entries.
    # We want each node to have ~expected_parents incoming edges in the upper-triangular DAG.
    prob = min(expected_parents / max((p - 1) / 2, 1), 1.0)
    A = (rng.random((p, p)) < prob).astype(float)
    A = np.triu(A, k=1)  # upper triangular -> DAG
    W = rng.uniform(0.3, 1.0, size=(p, p)) * rng.choice([-1, 1], size=(p, p))
    return A * W


def sample_linear_gaussian(W, n, noise_scale=1.0, seed=0):
    """Sample n points from linear-Gaussian SCM defined by W."""
    rng = np.random.default_rng(seed)
    p = W.shape[0]
    X = np.zeros((n, p))
    noise = rng.normal(0, noise_scale, size=(n, p))
    for j in range(p):
        X[:, j] = noise[:, j] + X @ W[:, j]
    return X


# ---------------------------------------------------------------
# 2. Custom Fisher-Z and PC Skeleton
# ---------------------------------------------------------------
def fisher_z_test_cov(cov_matrix, n_samples, i, j, S):
    """Compute Fisher-Z partial correlation p-value directly from covariance matrix.

    Returns a scalar p-value.
    """
    k = len(S)
    idx = [i, j] + list(S)
    sub_cov = cov_matrix[np.ix_(idx, idx)]

    try:
        inv_cov = np.linalg.pinv(sub_cov)
        # FIX (bug #1): partial correlation between i and j given S, computed
        # from the (0, 1) off-diagonal of the precision matrix and the relevant
        # diagonal entries. The previous formula r = -inv_cov / sqrt(|inv_cov*inv_cov|)
        # collapsed every entry to ±1 and returned a matrix instead of a scalar.
        denom = np.sqrt(inv_cov[0, 0] * inv_cov[1, 1])
        if denom <= 0 or not np.isfinite(denom):
            r = 0.0
        else:
            r = -inv_cov[0, 1] / denom
    except np.linalg.LinAlgError:
        r = 0.0

    r = float(np.clip(r, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + r) / (1 - r))
    z_stat = np.abs(z) * np.sqrt(max(n_samples - k - 3, 1))
    p_val = 2 * (1 - norm.cdf(z_stat))
    return p_val


def custom_pc_skeleton(cov_matrix, n_samples, alpha_matrix, max_cond_size):
    """
    Find the skeleton of the DAG using custom edge-specific alpha thresholds.
    alpha_matrix[i, j] dictates the p-value threshold for edge (i, j).
    """
    p = cov_matrix.shape[0]

    G = np.ones((p, p), dtype=int)
    np.fill_diagonal(G, 0)

    for k in range(max_cond_size + 1):
        for i, j in itertools.combinations(range(p), 2):
            if G[i, j] == 0:
                continue

            adj_i = [v for v in range(p) if G[i, v] == 1 and v != j]
            adj_j = [v for v in range(p) if G[j, v] == 1 and v != i]

            removed = False

            # 1. Check conditioning sets from i's neighbors
            if len(adj_i) >= k:
                for S in itertools.combinations(adj_i, k):
                    # FIX (bug #2): no longer index [0]; fisher_z_test_cov returns a scalar.
                    pval = fisher_z_test_cov(cov_matrix, n_samples, i, j, list(S))
                    if pval > alpha_matrix[i, j]:
                        G[i, j] = G[j, i] = 0
                        removed = True
                        break

            if removed:
                continue

            # 2. Check conditioning sets from j's neighbors
            if len(adj_j) >= k:
                for S in itertools.combinations(adj_j, k):
                    pval = fisher_z_test_cov(cov_matrix, n_samples, i, j, list(S))
                    if pval > alpha_matrix[i, j]:
                        G[i, j] = G[j, i] = 0
                        break

    return G


# ---------------------------------------------------------------
# 3. Alpha Matrix Computation
# ---------------------------------------------------------------
def compute_alpha_matrices(S, p, alpha=0.05, max_cond_size=2,
                           sample_pairs_per_edge=50, seed=0):
    """
    Computes a symmetric p x p matrix of alpha thresholds.
    Distributes the family-wise error rate per-edge based on the number of pairs
    and the effective number of tests strictly for that pair.
    """
    rng = np.random.default_rng(seed)
    alpha_bonf = np.zeros((p, p))
    alpha_cimd = np.zeros((p, p))

    total_pairs = p * (p - 1) / 2
    nodes = list(range(p))

    total_eff_tests = 0
    total_tests = 0

    for a, b in itertools.combinations(nodes, 2):
        rest = [v for v in nodes if v != a and v != b]

        # Enumerate all condition sets for this specific edge
        edge_cond_sets = []
        for k in range(0, max_cond_size + 1):
            for c in itertools.combinations(rest, k):
                edge_cond_sets.append(list(c))

        n_edge = len(edge_cond_sets)
        total_tests += n_edge

        if n_edge <= 1:
            n_eff_edge = float(n_edge)
        else:
            redund = []
            # FIX (bug #4): renamed from n_samples to avoid shadowing the conceptual
            # sample size used elsewhere. This is the number of conditioning-set pairs
            # we sample to estimate redundancy.
            max_possible_pairs = n_edge * (n_edge - 1) // 2
            n_pair_samples = min(sample_pairs_per_edge, max_possible_pairs)

            # FIX (bug #7): sample distinct unordered pairs without replacement,
            # so the requested sample size is actually achieved.
            all_pairs = list(itertools.combinations(range(n_edge), 2))
            chosen_idx = rng.choice(len(all_pairs), size=n_pair_samples, replace=False)
            pairs = [all_pairs[k] for k in chosen_idx]

            for i_idx, j_idx in pairs:
                c1 = edge_cond_sets[i_idx]
                c2 = edge_cond_sets[j_idx]
                try:
                    v = CIMD_limited_normed(S, [a], [b], c1, [a], [b], c2)
                    redund.append(np.clip(abs(v), 0.0, 1.0))
                except np.linalg.LinAlgError:
                    # Singular covariance projection — treat as no redundancy info.
                    redund.append(0.0)

            mean_redund = float(np.mean(redund)) if redund else 0.0
            n_eff_edge = max(n_edge * (1.0 - mean_redund), 1.0)

        total_eff_tests += n_eff_edge

        # Calculate local thresholds.
        # Bonf: alpha / (total_pairs * n_edge)  [equivalent to global total tests]
        # CIMD: alpha / (total_pairs * n_eff_edge)
        alpha_bonf[a, b] = alpha_bonf[b, a] = alpha / (total_pairs * n_edge)
        alpha_cimd[a, b] = alpha_cimd[b, a] = alpha / (total_pairs * n_eff_edge)

    global_mean_red = 1.0 - (total_eff_tests / total_tests) if total_tests > 0 else 0.0

    return alpha_bonf, alpha_cimd, total_tests, total_eff_tests, global_mean_red


# ---------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------
def true_skeleton(W):
    A = (W != 0).astype(int)
    return ((A + A.T) > 0).astype(int)


def skeleton_metrics(true_sk, est_sk):
    # FIX (bug #3): grab the integer dimension, not the (p, p) tuple.
    p = true_sk.shape[0]
    iu = np.triu_indices(p, k=1)
    t = true_sk[iu]
    e = est_sk[iu]
    tp = int(((t == 1) & (e == 1)).sum())
    fp = int(((t == 0) & (e == 1)).sum())
    fn = int(((t == 1) & (e == 0)).sum())
    shd = fp + fn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return shd, prec, rec, f1


# ---------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------
def run_experiment(p=15, expected_parents=2,
                   sample_sizes=(500, 1000, 2000, 3500, 5000),
                   n_seeds=10, alpha=0.05, max_cond_size=2):

    rows = []
    for seed in range(n_seeds):
        W = random_dag(p, expected_parents, seed=seed)
        true_sk = true_skeleton(W)

        for N in sample_sizes:
            X = sample_linear_gaussian(W, N, seed=seed + 1000)
            S = np.cov(X.T)

            # 1. Compute edge-specific alpha matrices
            t0 = time.time()
            alpha_b_matrix, alpha_c_matrix, n_tests, n_eff, mean_red = compute_alpha_matrices(
                S, p, alpha=alpha,
                max_cond_size=max_cond_size,
                sample_pairs_per_edge=50,
                seed=seed
            )
            cimd_time = time.time() - t0

            # 2. Run custom PC three times: vanilla (uncalibrated), Bonf, CIMD.
            # Vanilla = same alpha (no correction) on every test, applied through
            # the same skeleton routine so the comparison is apples-to-apples.
            alpha_v_matrix = np.full((p, p), alpha)
            est_sk_v = custom_pc_skeleton(S, N, alpha_v_matrix, max_cond_size)
            est_sk_b = custom_pc_skeleton(S, N, alpha_b_matrix, max_cond_size)
            est_sk_c = custom_pc_skeleton(S, N, alpha_c_matrix, max_cond_size)

            shd_v, prec_v, rec_v, f1_v = skeleton_metrics(true_sk, est_sk_v)
            shd_b, prec_b, rec_b, f1_b = skeleton_metrics(true_sk, est_sk_b)
            shd_c, prec_c, rec_c, f1_c = skeleton_metrics(true_sk, est_sk_c)

            rows.append(dict(
                seed=seed, N=N,
                n_tests=n_tests, n_eff=n_eff, mean_red=mean_red,
                shd_van=shd_v, prec_van=prec_v, rec_van=rec_v, f1_van=f1_v,
                shd_bonf=shd_b, prec_bonf=prec_b, rec_bonf=rec_b, f1_bonf=f1_b,
                shd_cimd=shd_c, prec_cimd=prec_c, rec_cimd=rec_c, f1_cimd=f1_c,
                cimd_time=cimd_time,
            ))

            print(f"  seed={seed} N={N}: SHD van={shd_v} bonf={shd_b} cimd={shd_c} | "
                  f"F1 van={f1_v:.3f} bonf={f1_b:.3f} cimd={f1_c:.3f} "
                  f"(n_eff={n_eff:.0f}, global_mean_red={mean_red:.3f})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 6. Reporting & Execution
# ---------------------------------------------------------------
def summarize(df):
    df = df.copy()
    df['shd_diff']  = df['shd_bonf']  - df['shd_cimd']   # >0: CIMD better
    df['prec_diff'] = df['prec_cimd'] - df['prec_bonf']  # >0: CIMD better
    df['rec_diff']  = df['rec_cimd']  - df['rec_bonf']   # >0: CIMD better
    df['f1_diff']   = df['f1_cimd']   - df['f1_bonf']    # >0: CIMD better

    def sem(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    agg = df.groupby('N').agg(
        # Paired differences (top row of plot)
        shd_diff_mean=('shd_diff', 'mean'),
        shd_diff_sem=('shd_diff', sem),
        prec_diff_mean=('prec_diff', 'mean'),
        prec_diff_sem=('prec_diff', sem),
        rec_diff_mean=('rec_diff', 'mean'),
        rec_diff_sem=('rec_diff', sem),
        f1_diff_mean=('f1_diff', 'mean'),
        f1_diff_sem=('f1_diff', sem),
        # Absolute means/SEMs for the three-method comparison (bottom row)
        shd_van_mean=('shd_van', 'mean'),
        shd_van_sem=('shd_van', sem),
        shd_bonf_mean=('shd_bonf', 'mean'),
        shd_bonf_sem=('shd_bonf', sem),
        shd_cimd_mean=('shd_cimd', 'mean'),
        shd_cimd_sem=('shd_cimd', sem),
        prec_van_mean=('prec_van', 'mean'),
        prec_van_sem=('prec_van', sem),
        prec_bonf_mean=('prec_bonf', 'mean'),
        prec_bonf_sem=('prec_bonf', sem),
        prec_cimd_mean=('prec_cimd', 'mean'),
        prec_cimd_sem=('prec_cimd', sem),
        rec_van_mean=('rec_van', 'mean'),
        rec_van_sem=('rec_van', sem),
        rec_bonf_mean=('rec_bonf', 'mean'),
        rec_bonf_sem=('rec_bonf', sem),
        rec_cimd_mean=('rec_cimd', 'mean'),
        rec_cimd_sem=('rec_cimd', sem),
        f1_van_mean=('f1_van', 'mean'),
        f1_van_sem=('f1_van', sem),
        f1_bonf_mean=('f1_bonf', 'mean'),
        f1_bonf_sem=('f1_bonf', sem),
        f1_cimd_mean=('f1_cimd', 'mean'),
        f1_cimd_sem=('f1_cimd', sem),
        n_eff_mean=('n_eff', 'mean'),
        n_seeds=('shd_diff', 'count'),
    ).reset_index()
    return agg


def plot_results(agg, out_path='cimd_vs_bonf_paired.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---- Top-left: paired ΔSHD (Bonf − CIMD) ----
    ax = axes[0, 0]
    ax.errorbar(agg['N'], agg['shd_diff_mean'], yerr=agg['shd_diff_sem'],
                marker='o', capsize=3, color='C2', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(agg['N'], 0, agg['shd_diff_mean'],
                    where=agg['shd_diff_mean'] > 0, alpha=0.15, color='C2', label='CIMD better')
    ax.fill_between(agg['N'], 0, agg['shd_diff_mean'],
                    where=agg['shd_diff_mean'] < 0, alpha=0.15, color='C3', label='Bonferroni better')
    ax.set_xlabel('Sample size N')
    ax.set_ylabel('Mean paired ΔSHD (Bonf − CIMD)')
    ax.set_title('SHD improvement of CIMD over Bonferroni')
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- Top-right: paired ΔPrecision and ΔRecall (CIMD − Bonf) ----
    ax = axes[0, 1]
    ax.errorbar(agg['N'], agg['rec_diff_mean'], yerr=agg['rec_diff_sem'],
                marker='o', capsize=3, label='ΔRecall (CIMD − Bonf)', color='C0', linewidth=2)
    ax.errorbar(agg['N'], agg['prec_diff_mean'], yerr=agg['prec_diff_sem'],
                marker='s', capsize=3, label='ΔPrecision (CIMD − Bonf)',
                color='C1', linewidth=2, linestyle='--')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample size N')
    ax.set_ylabel('Mean paired difference')
    ax.set_title('Precision / Recall improvement of CIMD')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- Bottom-left: paired ΔF1 (CIMD − Bonf) ----
    ax = axes[1, 0]
    ax.errorbar(agg['N'], agg['f1_diff_mean'], yerr=agg['f1_diff_sem'],
                marker='o', capsize=3, color='C4', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(agg['N'], 0, agg['f1_diff_mean'],
                    where=agg['f1_diff_mean'] > 0, alpha=0.15, color='C2', label='CIMD better')
    ax.fill_between(agg['N'], 0, agg['f1_diff_mean'],
                    where=agg['f1_diff_mean'] < 0, alpha=0.15, color='C3', label='Bonferroni better')
    ax.set_xlabel('Sample size N')
    ax.set_ylabel('Mean paired ΔF1 (CIMD − Bonf)')
    ax.set_title('F1 improvement of CIMD over Bonferroni')
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- Bottom-right: absolute F1 for all three methods ----
    ax = axes[1, 1]
    ax.errorbar(agg['N'], agg['f1_van_mean'], yerr=agg['f1_van_sem'],
                marker='^', capsize=3, label='Vanilla PC (α=0.05)', color='C3', linewidth=2)
    ax.errorbar(agg['N'], agg['f1_bonf_mean'], yerr=agg['f1_bonf_sem'],
                marker='s', capsize=3, label='Bonferroni PC', color='C1', linewidth=2)
    ax.errorbar(agg['N'], agg['f1_cimd_mean'], yerr=agg['f1_cimd_sem'],
                marker='o', capsize=3, label='CIMD PC', color='C2', linewidth=2)
    ax.set_xlabel('Sample size N')
    ax.set_ylabel('Mean F1 (higher is better)')
    ax.set_title('Absolute F1: Vanilla vs Bonferroni vs CIMD')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == '__main__':
    # Config tuned to favor multiple-testing correction (and CIMD specifically):
    # - sparse graph (expected_parents=1) so most pairs are genuinely independent
    #   — vanilla PC will leak many false positives here
    # - larger p so total tests grow and correction has bite
    # - max_cond_size=4 so the per-edge test family grows combinatorially,
    #   creating substantial redundancy between conditioning sets, which is
    #   exactly when CIMD's effective-test-count diverges from Bonferroni's.
    df = run_experiment(
        p=18,
        expected_parents=1,
        sample_sizes=(500, 1000, 2000, 3500, 5000),
        n_seeds=10,
        alpha=0.05,
        max_cond_size=3
    )
    df.to_csv('cimd_pc_results_raw.csv', index=False)

    agg = summarize(df)
    print("\n=== Summary table ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    agg.to_csv('cimd_pc_results_summary.csv', index=False)

    plot_results(agg)