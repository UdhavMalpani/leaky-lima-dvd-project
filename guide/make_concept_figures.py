"""
Concept figures for the learning guide, all made from real ngspice runs of
the assigned 16nm_HP model (not textbook sketches).

Run from the project root:  python guide/make_concept_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from leakml import bipoles, spice

OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
plt.rcParams.update({
    "figure.dpi": 160, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
OUTS = ["i(Vd)", "i(Vg)", "i(Vs)", "i(Vb)"] + [f"@m1[{c}]" for c in bipoles.COMPONENTS]


def sim(dev, rows, temp=27.0, params=None, w=config.WN_GATE):
    """rows: list of dicts with vd, vg, vs, vb. Returns terminal + bipole currents."""
    body = [f"M1 d g s b {dev} W={spice.fmt(w)} L={spice.fmt(config.LMIN)}",
            "Vd d 0 0", "Vg g 0 0", "Vs s 0 0", "Vb b 0 0"]
    pre = [f"option temp={temp}"] + (spice.altermod_cmds(dev, params) if params else [])
    runs = [pre + [f"alter v{t} dc={r[f'v{t}']}" for t in "dgsb"] for r in rows]
    res = spice.run_batch("guide", body, runs, OUTS, tag="guide", n_workers=6)
    df = pd.DataFrame(rows)
    df[["id", "ig", "is", "ib"]] = -res[:, :4]
    comp = pd.DataFrame(res[:, 4:], columns=bipoles.COMPONENTS)
    return df, bipoles.from_components(dev, df, comp)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_iv():
    V = config.VDD_NOM
    vg = np.linspace(0, V, 61)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for t, col in zip((-40, 27, 125), (C[0], C[2], C[1])):
        df, _ = sim("nmos", [dict(vd=V, vg=g, vs=0, vb=0) for g in vg], temp=t)
        ax[0].semilogy(vg, df.id, c=col, lw=1.8, label=f"T = {t} °C")
    vth = config.NOMINAL["nmos"]["vth0"]
    for f, col, ls in ((0.9, C[1], "--"), (1.0, INK, "-"), (1.1, C[0], "--")):
        df, _ = sim("nmos", [dict(vd=V, vg=g, vs=0, vb=0) for g in vg],
                    params={"vth0": vth * f})
        ax[1].semilogy(vg, df.id, c=col, ls=ls, lw=1.6, label=f"vth0 × {f:.1f}")
    for a in ax:
        a.axvspan(0, 0.3, color=C[3], alpha=0.08)
        a.text(0.02, 0.96, "sub-threshold:\nstraight line on log axis\n= exponential", transform=a.transAxes,
               va="top", fontsize=7, color=C[3])
        a.set(xlabel="Gate voltage Vgs (V)", ylabel="Drain current Id (A)")
        a.legend(fontsize=7, loc="lower right")
    ax[0].annotate("I_off (Vgs = 0)\nthis is the leakage", xy=(0.0, 1.35e-7), xytext=(0.45, 1.2e-7),
                   fontsize=7, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax[0].annotate("zero-temperature-\ncoefficient point", xy=(0.19, 4.5e-6), xytext=(0.42, 1.2e-6),
                   fontsize=7, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax[0].set_title("Temperature: hotter → more leakage", fontsize=9)
    ax[1].set_title("Threshold voltage: ±10 % vth0 → ≈3× leakage", fontsize=9)
    save(fig, "iv_curves.png")


def fig_cancellation():
    """Why predicting terminal currents directly is hard."""
    V = config.VDD_NOM
    vd = np.linspace(0.0, V, 121)
    df, b = sim("nmos", [dict(vd=x, vg=0, vs=0.5, vb=0) for x in vd])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax[0].plot(vd, df.id, c=INK, lw=2)
    ax[0].set_yscale("symlog", linthresh=1e-14)
    ax[0].axhline(0, c=MUTED, lw=0.6)
    zc = vd[np.argmin(np.abs(df.id))]
    ax[0].annotate(f"sign flips at Vd ≈ {zc:.2f} V\n(NOT at Vd = Vs = 0.5 V):\nchannel and tunnelling cancel",
                   xy=(zc, 0), xytext=(0.42, -1e-11), fontsize=7,
                   arrowprops=dict(arrowstyle="->", color=MUTED))
    ax[0].set(xlabel="Drain voltage Vd (V)   [Vs = 0.5 V, Vg = 0]", ylabel="Id (A), signed log scale",
              title="Terminal current Id: a sum of mechanisms")
    for name, col, lab in (("ds", C[0], "channel  (D→S)"), ("gd", C[1], "gate tunnelling (G→D)"),
                           ("db", C[2], "junction (D→B)")):
        ax[1].semilogy(vd, np.abs(b[name]), c=col, lw=1.8, label=lab)
    ax[1].semilogy(vd, np.abs(df.id), c=INK, lw=1, ls=":", label="|Id| (their sum)")
    ax[1].set(xlabel="Drain voltage Vd (V)", ylabel="|current| (A)",
              title="Each mechanism alone: smooth, one sign")
    ax[1].legend(fontsize=7)
    save(fig, "cancellation.png")


def fig_decades():
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    bins = np.linspace(-19, -3, 80)
    for case, col in zip(config.CASES, C):
        d = pd.read_csv(config.DATA_DIR / f"{case}.csv")
        ax.hist(np.log10(d.id.abs() + 1e-30), bins=bins, histtype="step", lw=1.6, color=col, label=case)
    ax.set(xlabel="log10 |Id|  (A)", ylabel="samples", title="Our 40,000 drain currents span 14 decades")
    ax.axvspan(-19, -14, color=MUTED, alpha=0.1)
    ax.text(-18.8, ax.get_ylim()[1] * 0.85, "numerical\nnoise floor", fontsize=7, color=MUTED)
    ax.legend(fontsize=7)
    save(fig, "decades.png")


def fig_toy_models():
    """Fit the four learner families to one real 1-D slice: log10 Id vs Vgs."""
    from leakml.models import make_learner
    V = config.VDD_NOM
    vg = np.linspace(0, V, 400)
    df, _ = sim("nmos", [dict(vd=V, vg=g, vs=0, vb=0) for g in vg])
    y = np.log10(df.id.to_numpy())
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(len(vg), 25, replace=False))
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.7), sharey=True)
    names = {"poly_ridge": "Polynomial + Ridge", "random_forest": "Random Forest",
             "lightgbm": "LightGBM (boosting)", "mlp": "Neural net (MLP)"}
    for a, (kind, title) in zip(ax, names.items()):
        hp = {"degree": 4} if kind == "poly_ridge" else {}
        if kind == "lightgbm":
            hp = {"n_estimators": 300}
        if kind == "mlp":
            hp = {"hidden": (16, 16)}
        m = make_learner(kind, **hp)
        if kind == "lightgbm":
            m.set_params(min_child_samples=2)
        m.fit(vg[tr, None], y[tr])
        a.plot(vg, y, c=MUTED, lw=4, alpha=0.35, label="SPICE truth")
        a.plot(vg, m.predict(vg[:, None]), c=C[0], lw=1.4, label="model")
        a.plot(vg[tr], y[tr], "o", ms=3, c=C[1], label="25 training points")
        a.set(title=title, xlabel="Vgs (V)")
    ax[0].set_ylabel("log10 Id")
    ax[0].legend(fontsize=6.5)
    save(fig, "toy_models.png")


def fig_passive():
    d = pd.read_csv(config.DATA_DIR / "nmos_off.csv")
    b = pd.read_csv(config.DATA_DIR / "nmos_off_bipoles.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
    v = bipoles.voltage(d, "gd")
    ax[0].scatter(v, b.gd * 1e9, s=1, c=C[1], alpha=0.4)
    ax[0].axhline(0, c=MUTED, lw=0.6)
    ax[0].axvline(0, c=MUTED, lw=0.6)
    ax[0].set(xlabel="V_gd (V)", ylabel="I_gd (nA)", title="Gate→drain tunnelling: I has the sign of V")
    g = b.gd / v
    ax[1].scatter(v, np.log(np.abs(g)), s=1, c=C[0], alpha=0.4)
    ax[1].set(xlabel="V_gd (V)", ylabel="log G = log(I/V)", title="…so log G is a smooth, positive-G target")
    save(fig, "passive.png")


if __name__ == "__main__":
    for f in (fig_iv, fig_cancellation, fig_decades, fig_toy_models, fig_passive):
        f()
        print("made", f.__name__)
