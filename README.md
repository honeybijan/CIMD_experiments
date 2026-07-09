# Meta-Dependence in Conditional Independence Testing — Experiment Code

Code to reproduce the experiments and figures in **"Meta-Dependence in Conditional
Independence Testing"** (Mazaheri, Zhang, and Uhler, UAI 2026).

The repository has two independent parts:

1. **Tuning causal discovery** (the quantitative PC-algorithm experiment) — driven by
   `run_parallel.py` and plotted by `make_pc_figure.py`.
2. **Illustrative and real-world figures** (CIMD / CIMD-lim / FS-CID heatmaps and the
   real-data dependence matrices) — driven by `cimd.py`.

Everything is deterministic given a seed, so results reproduce exactly.

---

## Repository layout

| File | Purpose |
|------|---------|
| `cimd.py` | Core CIMD / FS-CID definitions and the illustrative + real-world figure drivers (heatmaps, real-data matrices). |
| `utils.py` | Covariance-projection helpers, `CIMD_limited_normed` (the normalized, gated CIMD), and `plot_heatmap`. |
| `causal_discovery_tests.py` | The PC-algorithm experiment: random-DAG generation, the three edge-thresholding rules (vanilla / Bonferroni / CIMD), skeleton metrics, and `summarize`. |
| `run_parallel.py` | Runs the PC experiment over many random graphs (parallel, with a serial fallback) and writes the result CSVs. |
| `make_pc_figure.py` | Plots `cimd_pc_causal_discovery.pdf` from the summary CSV. |
| `check_env.py` | Environment sanity check (imports + one worked example cell). |
| `cimd_pc_results_raw.csv`, `cimd_pc_results_summary.csv` | Committed 100-seed results used in the paper. |
| `cimd_pc_causal_discovery.pdf` | The causal-discovery figure in the paper. |

---

## Environment

Tested with Python 3.14 (any Python 3.9+ should work). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy pandas matplotlib conditional-independence
```

`conditional-independence` imports as `conditional_independence` and is used by the
FS-CID (bootstrap) tests in `cimd.py`/`utils.py`. The PC experiment
(`causal_discovery_tests.py`, `run_parallel.py`) needs only numpy/scipy/pandas/matplotlib.

**Sanity check (recommended first step):**

```bash
python check_env.py
```

This prints your interpreter, confirms every import, and computes one experiment cell
(seed 0, N=500), which should print `F1 van/bonf/cimd = 0.976/0.923/0.95`. If that
matches, the environment is correct.

---

## Part 1 — Tuning causal discovery (main quantitative experiment)

Reproduces the skeleton-recovery comparison of uncorrected ("vanilla") PC, Bonferroni-PC,
and CIMD-corrected PC, and the paper's causal-discovery figure.

### Reproduce

```bash
python run_parallel.py 100        # writes the two result CSVs (~2 min parallel)
python make_pc_figure.py          # writes cimd_pc_causal_discovery.pdf (+ _preview.png)
```

`run_parallel.py` accepts the seed count positionally or as `--seeds`, and takes an
optional `--workers` (defaults to all cores; `--workers 1` forces a serial run, ~15 min
for 100 seeds). It prints a live progress counter and writes results incrementally. If the
process pool fails on your platform, it prints the error and finishes serially
automatically.

### Experimental setup (fixed in `causal_discovery_tests.py`)

- Sparse linear-Gaussian DAGs, `p = 18` variables, ≈1 expected parent per node.
- Skeleton estimated with a partial-correlation (Fisher-Z) CI test, α = 0.05,
  conditioning sets up to size 3.
- Sample sizes `N ∈ {500, 1000, 2000, 3500, 5000}`, averaged over 100 random graphs.
- Three thresholding rules on the **same** skeleton search:
  - **vanilla**: α on every test (no multiple-testing correction);
  - **Bonferroni**: α split across all pairs × conditioning-set tests;
  - **CIMD**: each edge's test count `n_edge` replaced by an effective count
    `n_edge · (1 − r̄)`, where `r̄` is the mean normalized CIMD over sampled pairs of the
    edge's conditioning sets.

### Outputs

- `cimd_pc_results_raw.csv` — one row per (seed, N) with metrics and diagnostics
  (`mean_red`, per-edge threshold ratios, etc.).
- `cimd_pc_results_summary.csv` — per-N means and SEMs across seeds (input to the figure).
- `cimd_pc_causal_discovery.pdf` — (a) absolute skeleton F1 for the three methods and
  (b) the paired ΔF1 (CIMD − Bonferroni).

The committed CSVs already contain the 100-seed results used in the paper.

### Determinism

Each `(seed, N)` cell is fully determined by its seed: the graph uses `seed`, the data
uses `seed + 1000`, and the redundancy sampling uses `numpy.default_rng(seed)`. Results are
therefore identical for any `--workers`, and seeds 0–4 reproduce the original 5-seed run
exactly; increasing the seed count only appends fresh, independent graphs.

### Font matching (optional)

`make_pc_figure.py` renders math in Computer Modern by default (no LaTeX install needed).
Set `USETEX = True` near the top of that file to render **all** text with a local LaTeX
installation for a pixel-perfect match to the paper.

---

## Part 2 — Illustrative and real-world figures

These are driven by `cimd.py`. Uncomment the relevant call(s) at the bottom of `cimd.py`
(or call the function from a Python session) and run:

```bash
python cimd.py
```

### Illustrative heatmaps (CIMD, CIMD-lim, FS-CID)

```python
# Constant is alpha1 = .5; test A ⊥ C vs. B ⊥ C
plot_heatmap('CIMD',    [.5], np.linspace(-.5, .5, 100), np.linspace(-.5, .5, 100), CIMD,                 [0], [2], [], [1], [2], [])
plot_heatmap('CIMD-lim',[.5], np.linspace(-.5, .5, 100), np.linspace(-.5, .5, 100), CIMD_limited,         [0], [2], [], [1], [2], [])
plot_heatmap('FS-CID',  [.5], np.linspace(-.5, .5, 100), np.linspace(-.5, .5, 100), CI_test_dependence_lim,[0], [2], [], [1], [2], [])
```

`plot_heatmap` takes a constant value for the held-out parameter, two `linspace` ranges for
the swept parameters, the quantity to compute (`CIMD`, `CIMD_limited`, or
`CI_test_dependence_lim`), and six lists specifying the two CI tests being compared:
`[a1], [b1], [c1], [a2], [b2], [c2]` means "test `X[a1] ⊥ X[b1] | X[c1]`" versus
"test `X[a2] ⊥ X[b2] | X[c2]`." Change these to probe other conditional independencies.
The helpers `beta_zero()`, `alpha1_zero()`, and `alpha2_zero()` sweep the three canonical
three-node structures.

### Real-world dependence matrices

```python
real_data_FS_CID_matrix_california_housing()
real_data_FS_CID_matrix_apple_watch_fitbit()
real_data_FS_CID_matrix_auto_mpg()
```

(The `real_data_CIMD_matrix_*` variants produce the CIMD counterparts.) These load their
respective datasets and write the corresponding figures.

---

## Citation

```bibtex
@inproceedings{mazaheri2026metadependence,
  title     = {Meta-Dependence in Conditional Independence Testing},
  author    = {Mazaheri, Bijan and Zhang, Jiaqi and Uhler, Caroline},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026}
}
```