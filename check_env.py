"""
Diagnostic for the CIMD experiment runner. Run this from the SAME folder where
you run run_parallel.py, with your venv active:

    python check_env.py

Paste the entire output back. It checks the environment, imports, and actually
computes one (seed=0, N=500) cell so we can see exactly where things stop.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

print("=" * 60)
print("PYTHON     :", sys.version.split()[0], "|", sys.executable)
print("CWD        :", os.getcwd())
py = sorted(f for f in os.listdir(".") if f.endswith(".py"))
csvs = sorted(f for f in os.listdir(".") if f.endswith(".csv"))
print(".py here   :", py)
print(".csv here  :", csvs)
print("has causal_discovery_tests.py :", "causal_discovery_tests.py" in py)
print("has utils.py                  :", "utils.py" in py)
print("=" * 60)


def check_import(name):
    try:
        mod = __import__(name)
        v = getattr(mod, "__version__", "?")
        print(f"[ ok ] import {name}  (version {v})")
        return mod
    except Exception as e:
        print(f"[FAIL] import {name}  ->  {type(e).__name__}: {e}")
        return None


check_import("numpy")
check_import("pandas")
check_import("conditional_independence")   # dependency used by utils.py

cdt = None
try:
    import causal_discovery_tests as cdt
    print("[ ok ] import causal_discovery_tests")
except Exception:
    print("[FAIL] import causal_discovery_tests -- full traceback below:")
    traceback.print_exc()

if cdt is not None:
    needed = ["random_dag", "sample_linear_gaussian", "compute_alpha_matrices",
              "custom_pc_skeleton", "skeleton_metrics", "true_skeleton", "summarize"]
    missing = [n for n in needed if not hasattr(cdt, n)]
    print("missing functions:", missing if missing else "none")

    if not missing:
        print("-" * 60)
        print("Computing ONE cell (seed=0, N=500) -- should take a few seconds...")
        try:
            import numpy as np
            t0 = time.time()
            W = cdt.random_dag(18, 1, seed=0)
            true_sk = cdt.true_skeleton(W)
            X = cdt.sample_linear_gaussian(W, 500, seed=1000)
            S = np.cov(X.T)
            ab, ac, n_tests, n_eff, mean_red = cdt.compute_alpha_matrices(
                S, 18, alpha=0.05, max_cond_size=3, sample_pairs_per_edge=50, seed=0)
            av = np.full((18, 18), 0.05)
            f1s = {}
            for name, am in [("van", av), ("bonf", ab), ("cimd", ac)]:
                sk = cdt.custom_pc_skeleton(S, 500, am, 3)
                f1s[name] = round(cdt.skeleton_metrics(true_sk, sk)[3], 3)
            dt = time.time() - t0
            print(f"  computed in {dt:.1f}s")
            print(f"  F1 van/bonf/cimd = {f1s['van']}/{f1s['bonf']}/{f1s['cimd']}   mean_red={mean_red:.3f}")
            print( "  EXPECTED (seed0,N500) = 0.976/0.923/0.95   mean_red=0.688")
            print("  -> if these match, the machinery works and the issue was in how")
            print("     run_parallel.py was invoked (args/cwd); if they differ, tell me.")
        except Exception:
            print("[FAIL] single-cell computation errored -- traceback:")
            traceback.print_exc()

print("=" * 60)
print("Done. Please paste ALL of the above.")