"""Regenerate a compact, paper-styled 1x2 figure for the CIMD-tuned PC experiment.
Left:  absolute F1 (vanilla / Bonferroni / CIMD) vs N, SEM error bars.
Right: paired dF1 (CIMD - Bonferroni) vs N, SEM error bars, zero line.
Reads the verified summary CSV; writes PDF (paper) and PNG (preview).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("cimd_pc_results_summary.csv")))
N        = [float(r["N"]) for r in rows]
f1_van   = [float(r["f1_van_mean"])  for r in rows]
f1_bonf  = [float(r["f1_bonf_mean"]) for r in rows]
f1_cimd  = [float(r["f1_cimd_mean"]) for r in rows]
f1_van_e = [float(r["f1_van_sem"])   for r in rows]
f1_bon_e = [float(r["f1_bonf_sem"])  for r in rows]
f1_cim_e = [float(r["f1_cimd_sem"])  for r in rows]
d_mean   = [float(r["f1_diff_mean"]) for r in rows]
d_sem    = [float(r["f1_diff_sem"])  for r in rows]

# Match the paper's font. USETEX=True renders ALL text with real LaTeX
# (true Computer Modern; needs a working latex + dvipng on your machine).
# USETEX=False (default) is robust: it renders math in Computer Modern via
# matplotlib's mathtext -- no LaTeX install needed -- so F_1, N, alpha, etc.
# match the paper; flip to True for pixel-perfect matching of the words too.
USETEX = False
_rc = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "figure.dpi": 200,
    "font.family": "serif",
}
if USETEX:
    _rc.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})
else:
    _rc.update({"mathtext.fontset": "cm"})  # Computer Modern math, no LaTeX needed
plt.rcParams.update(_rc)
C_VAN, C_BONF, C_CIMD = "#999999", "#E69F00", "#009E73"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 2.75))

axL.errorbar(N, f1_van,  yerr=f1_van_e, marker="^", capsize=2.5, color=C_VAN,  label=r"Vanilla PC ($\alpha{=}0.05$)")
axL.errorbar(N, f1_bonf, yerr=f1_bon_e, marker="s", capsize=2.5, color=C_BONF, label="Bonferroni PC")
axL.errorbar(N, f1_cimd, yerr=f1_cim_e, marker="o", capsize=2.5, color=C_CIMD, label="CIMD PC")
axL.set_xlabel(r"Sample size $N$")
axL.set_ylabel(r"Skeleton $F_1$ (higher is better)")
axL.set_title(r"(a) Absolute $F_1$")
axL.set_xticks(N); axL.set_xticklabels([f"{int(n)}" for n in N], rotation=30)
axL.grid(alpha=0.3, linewidth=0.6)
axL.legend(frameon=False, loc="lower right")

axR.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
axR.errorbar(N, d_mean, yerr=d_sem, marker="o", capsize=2.5, color=C_CIMD)
axR.fill_between(N, 0, d_mean, where=[d > 0 for d in d_mean], alpha=0.15, color=C_CIMD, label="CIMD better")
axR.set_xlabel(r"Sample size $N$")
axR.set_ylabel(r"Paired $\Delta F_1$ (CIMD $-$ Bonf)")
axR.set_title("(b) CIMD vs. Bonferroni")
axR.set_xticks(N); axR.set_xticklabels([f"{int(n)}" for n in N], rotation=30)
axR.grid(alpha=0.3, linewidth=0.6)
axR.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig("cimd_pc_causal_discovery.pdf", bbox_inches="tight")
plt.savefig("cimd_pc_causal_discovery_preview.png", bbox_inches="tight", dpi=200)
print("wrote cimd_pc_causal_discovery.pdf and _preview.png")