"""
Parallel re-run of the CIMD-vs-Bonferroni PC experiment across many seeds.

Each (seed, N) cell is independent and deterministic in (seed, N):
    graph      = random_dag(seed)
    data       = sample_linear_gaussian(..., seed + 1000)
    redundancy = compute_alpha_matrices(..., seed=seed)   # local default_rng(seed)
so the parallel result is identical to the serial one for ANY --workers, and
seeds 0-4 reproduce the original committed CSV exactly.

Usage (from the repo folder, with your venv active):
    python run_parallel.py --seeds 30
    python run_parallel.py --seeds 30 --workers 8

Writes cimd_pc_results_raw.csv and cimd_pc_results_summary.csv (same schema as
causal_discovery_tests.py), which make_pc_figure.py then plots unchanged.
"""
import os
# Keep BLAS single-threaded so the process pool doesn't oversubscribe cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import causal_discovery_tests as cdt  # reuses the tested functions + summarize()

# --- config (matches causal_discovery_tests.py __main__) ---
P = 18
EXPECTED_PARENTS = 1
SAMPLE_SIZES = (500, 1000, 2000, 3500, 5000)
ALPHA = 0.05
MAX_COND_SIZE = 3
SAMPLE_PAIRS_PER_EDGE = 50


def run_one(task):
    """One (seed, N) cell; returns a result row dict. Deterministic in (seed, N)."""
    seed, N = task
    W = cdt.random_dag(P, EXPECTED_PARENTS, seed=seed)
    true_sk = cdt.true_skeleton(W)
    X = cdt.sample_linear_gaussian(W, N, seed=seed + 1000)
    S = np.cov(X.T)

    t0 = time.time()
    alpha_b, alpha_c, n_tests, n_eff, mean_red = cdt.compute_alpha_matrices(
        S, P, alpha=ALPHA, max_cond_size=MAX_COND_SIZE,
        sample_pairs_per_edge=SAMPLE_PAIRS_PER_EDGE, seed=seed)
    cimd_time = time.time() - t0

    iu = np.triu_indices(P, k=1)
    ce, be = alpha_c[iu], alpha_b[iu]
    ratio = ce / be  # >1 => CIMD less strict than Bonferroni

    alpha_v = np.full((P, P), ALPHA)  # vanilla: no correction
    skv = cdt.custom_pc_skeleton(S, N, alpha_v, MAX_COND_SIZE)
    skb = cdt.custom_pc_skeleton(S, N, alpha_b, MAX_COND_SIZE)
    skc = cdt.custom_pc_skeleton(S, N, alpha_c, MAX_COND_SIZE)

    shd_v, pr_v, rc_v, f1_v = cdt.skeleton_metrics(true_sk, skv)
    shd_b, pr_b, rc_b, f1_b = cdt.skeleton_metrics(true_sk, skb)
    shd_c, pr_c, rc_c, f1_c = cdt.skeleton_metrics(true_sk, skc)

    return dict(
        seed=seed, N=N,
        n_tests=n_tests, n_eff=n_eff, mean_red=mean_red,
        cimd_alpha_mean=float(ce.mean()), cimd_alpha_std=float(ce.std()),
        cimd_alpha_cv=float(ce.std() / ce.mean()) if ce.mean() > 0 else 0.0,
        ratio_min=float(ratio.min()), ratio_max=float(ratio.max()), ratio_mean=float(ratio.mean()),
        shd_van=shd_v, prec_van=pr_v, rec_van=rc_v, f1_van=f1_v,
        shd_bonf=shd_b, prec_bonf=pr_b, rec_bonf=rc_b, f1_bonf=f1_b,
        shd_cimd=shd_c, prec_cimd=pr_c, rec_cimd=rc_c, f1_cimd=f1_c,
        cimd_time=cimd_time,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="number of random graphs")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    args = ap.parse_args()

    tasks = [(s, N) for s in range(args.seeds) for N in SAMPLE_SIZES]
    print(f"Running {len(tasks)} (seed, N) cells on {args.workers} workers...")
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        rows = pool.map(run_one, tasks)
    dt = time.time() - t0

    df = pd.DataFrame(rows).sort_values(["seed", "N"]).reset_index(drop=True)
    df.to_csv("cimd_pc_results_raw.csv", index=False)

    agg = cdt.summarize(df)
    agg.to_csv("cimd_pc_results_summary.csv", index=False)

    print("\n=== Summary table ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[{args.seeds} seeds x {len(SAMPLE_SIZES)} N on {args.workers} workers in {dt:.1f}s]")


if __name__ == "__main__":
    main()