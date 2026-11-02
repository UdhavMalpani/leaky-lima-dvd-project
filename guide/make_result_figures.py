"""
Result figures for the learning guide (run after scripts 01-06).
Run from the project root:  python guide/make_result_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)
R = config.RESULTS_DIR
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
FAM = {"mlp": ("Neural net (MLP)", C[0]), "poly_ridge": ("Polynomial + Ridge", C[2]),
       "lightgbm": ("LightGBM", C[3]), "random_forest": ("Random Forest", C[1])}
GATE_COL = {"INV": C[0], "NAND2": C[1], "NOR2": C[2], "AND2": C[3], "OR2": C[4]}
plt.rcParams.update({
    "figure.dpi": 160, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
CUR = ["id", "ig", "is", "ib"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


def model_comparison():
    d = pd.read_csv(R / "model_comparison.csv")
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), sharey=True)
    for a, case in zip(ax, config.CASES):
        g = d[d.case == case].set_index("model")
        x = np.arange(len(FAM))
        med = [g.loc[k, [f"{c}_med%" for c in CUR]].mean() for k in FAM]
        p95 = [g.loc[k, [f"{c}_p95%" for c in CUR]].mean() for k in FAM]
        cols = [FAM[k][1] for k in FAM]
        a.bar(x - 0.2, med, 0.4, color=cols, label="median")
        a.bar(x + 0.2, p95, 0.4, color=cols, alpha=0.35, label="p95")
        a.set_yscale("log")
        a.set_xticks(x, ["MLP", "Poly", "LGBM", "RF"], fontsize=8)
        a.axhline(1, c=INK, lw=0.7, ls="--")
        a.set_title(case)
    ax[0].set_ylabel("validation error, mean over Id/Ig/Is/Ib (%)")
    ax[0].text(0.02, 0.97, "solid = median\nfaded = 95th pct\n--- = 1 %", transform=ax[0].transAxes,
               va="top", fontsize=7, color=MUTED)
    save(fig, "model_comparison.png")


VERSIONS = [("v1_uniform_only", "v1: uniform data", C[1]),
            ("v2_rail", "v2: + rail data", C[3]),
            ("", "v3: + W_eff, Vg features", C[0])]


def before_after():
    tabs = [pd.read_csv(R / sub / "gate_results.csv", dtype={"inputs": str}) for sub, _, _ in VERSIONS]
    lab = tabs[0].gate + " " + tabs[0].inputs
    y = np.arange(len(lab))
    fig, ax = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    for a, col, title in ((ax[0], "med_err_pct", "median error (%)"),
                          (ax[1], "p95_err_pct", "95th-percentile error (%)")):
        for k, (t, (_, name, c)) in enumerate(zip(tabs, VERSIONS)):
            a.barh(y - 0.27 * (1 - k), t[col], 0.27, color=c, label=name)
        a.set_xlabel(title)
    ax[0].invert_yaxis()            # axes are shared: invert once, INV at the top
    ax[0].set_yticks(y, lab, fontsize=7)
    ax[1].legend(fontsize=7, loc="lower right")
    save(fig, "rail_before_after.png")


def stack_effect():
    s = pd.read_csv(R / "gate_samples.csv", dtype={"vector": str})
    g = s.groupby(["gate", "vector"], sort=False).spice_leak.median().reset_index()
    fig, ax = plt.subplots(figsize=(9, 3))
    x = np.arange(len(g))
    ax.bar(x, g.spice_leak, color=[GATE_COL[k] for k in g.gate])
    ax.set_yscale("log")
    ax.set_xticks(x, g.gate + "\n" + g.vector, fontsize=6.5)
    ax.set_ylabel("median SPICE leakage (A)")
    for k in ("NAND2", "NOR2"):
        vec = "00" if k == "NAND2" else "11"
        i = g.index[(g.gate == k) & (g.vector == vec)][0]
        ax.annotate("2 OFF in series:\nstack effect", xy=(i, g.spice_leak[i]), xytext=(i, g.spice_leak[i] * 0.04),
                    ha="center", fontsize=7, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_ylim(g.spice_leak.min() * 0.02, g.spice_leak.max() * 3)
    save(fig, "stack_effect.png")


def gate_error_vs_corner():
    s = pd.read_csv(R / "gate_samples.csv", dtype={"vector": str})
    c = pd.read_csv(R / "corners.csv", index_col="corner")
    s = s.join(c, on="corner")
    s["err"] = 100 * (s.ml_leak / s.spice_leak - 1)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for a, col, lab in ((ax[0], "temp", "temperature (°C)"), (ax[1], "vdd", "VDD (V)"),
                        (ax[2], "nmos_vth0", "NMOS vth0 (V)")):
        a.scatter(s[col], s.err, s=2, alpha=0.25, c=C[0])
        bins = np.linspace(s[col].min(), s[col].max(), 9)
        mid = 0.5 * (bins[1:] + bins[:-1])
        q = [np.percentile(np.abs(s.err[(s[col] >= lo) & (s[col] < hi)]), 95)
             for lo, hi in zip(bins[:-1], bins[1:])]
        a.plot(mid, q, c=C[1], lw=1.8, label="95th pct |error| per bin")
        a.plot(mid, -np.array(q), c=C[1], lw=1.8)
        a.axhline(0, c=INK, lw=0.6)
        a.set(xlabel=lab)
    lim = 15
    ax[0].set_ylim(-lim, lim)       # one isolated +126 % point would otherwise squash the plot
    n_off = int((s.err.abs() > lim).sum())
    if n_off:
        ax[-1].text(0.98, 0.97, f"{n_off} point(s) beyond ±{lim} % not shown", transform=ax[-1].transAxes,
                    ha="right", va="top", fontsize=7, color=MUTED)
    ax[0].set_ylabel("gate ML error (%)")
    ax[0].legend(fontsize=7)
    save(fig, "gate_error_vs_corner.png")


def circuit():
    sm = pd.read_csv(R / "circuit_summary.csv")
    labels = ["Hierarchical ML\nvs SPICE circuit", "Hierarchy only\n(SPICE gates vs circuit)",
              "ML only\n(ML gates vs SPICE gates)", "Flat ML on circuit\nnode voltages"]
    fig, ax = plt.subplots(figsize=(8, 2.9))
    x = np.arange(len(sm))
    ax.bar(x - 0.2, sm.med_err_pct, 0.4, color=C[0], label="median |error|")
    ax.bar(x + 0.2, sm.p95_err_pct, 0.4, color=C[0], alpha=0.35, label="95th percentile")
    ax.bar(x - 0.2, sm.bias_pct.abs(), 0.12, color=C[1], label="|bias|")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("error (%)")
    ax.legend(fontsize=7)
    save(fig, "circuit_decomposition.png")

    s = pd.read_csv(R / "circuit_samples.csv", dtype={"vector": str})
    v = s.groupby("vector").spice_leak.median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(np.arange(len(v)), v.to_numpy() * 1e6, "o", ms=3, c=C[0])
    ax.set(xlabel="the 128 input vectors, sorted", ylabel="median CLA leakage (µA)",
           title=f"Leakage depends on the input: {v.max()/v.min():.2f}× between best ({v.index[0]}) "
                 f"and worst ({v.index[-1]}) vector")
    save(fig, "circuit_vs_vector.png")


if __name__ == "__main__":
    for f in (model_comparison, before_after, stack_effect, gate_error_vs_corner, circuit):
        try:
            f()
            print("made", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
