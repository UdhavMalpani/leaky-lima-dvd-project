"""
Part A, data check (deliverable 3).

1. Kirchhoff check: id + ig + is + ib should be ~0 for every sample.
2. Three trend plots.  Because every parameter is random at once, a raw
   scatter of current vs one parameter is a cloud.  So each plot shows the
   random data (grey) AND a controlled SPICE sweep of only that parameter
   with everything else at nominal (coloured lines) - the sweep shows the
   cause-and-effect, the cloud shows the spread the model has to learn.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from leakml import spice

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
CUR = ["id", "ig", "is", "ib"]


def kirchhoff_table():
    rows = []
    for case in config.CASES:
        d = pd.read_csv(config.DATA_DIR / f"{case}.csv")
        s = d[CUR].sum(axis=1).abs()
        scale = d[CUR].abs().max(axis=1)
        rel = s / scale
        rows.append(dict(case=case, n=len(d),
                         abs_sum_median=s.median(), abs_sum_max=s.max(),
                         rel_median=rel.median(), rel_p95=rel.quantile(.95),
                         rel_max=rel.max()))
    t = pd.DataFrame(rows)
    t.to_csv(config.RESULTS_DIR / "kirchhoff_check.csv", index=False)
    return t


def sweep(dev, vg, vb, vd, vs, w, temps=None, param=None, values=None):
    """Nominal device, sweep either temperature or one process parameter."""
    body = [f"M1 d g s b {dev} W={spice.fmt(w)} L={spice.fmt(config.LMIN)}",
            f"Vd d 0 {vd}", f"Vg g 0 {vg}", f"Vs s 0 {vs}", f"Vb b 0 {vb}"]
    runs = []
    if temps is not None:
        runs = [[f"option temp={t}"] for t in temps]
    else:
        runs = [["option temp=27"] + spice.altermod_cmds(dev, {param: v}) for v in values]
    res = spice.run_batch("sweep", body, runs, ["i(Vd)", "i(Vg)", "i(Vs)", "i(Vb)"],
                          tag="sweep", n_workers=4)
    return -res  # into-device convention


def trend_plots():
    V = config.VDD_NOM
    w = config.WN_GATE
    nom = config.NOMINAL
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) OFF current vs temperature: classic NMOS leakage condition Vd=VDD, Vs=0
    d = pd.read_csv(config.DATA_DIR / "nmos_off.csv")
    ax[0].scatter(d.temp, d.id.abs(), s=2, c="0.8", label="random dataset (all params vary)")
    T = np.linspace(-40, 125, 34)
    for p, col in (("nmos", "C0"), ("pmos", "C3")):
        if p == "nmos":
            I = sweep("nmos", 0, 0, V, 0, w, temps=T)[:, 0]
        else:
            I = -sweep("pmos", V, V, 0, V, w, temps=T)[:, 0]
        ax[0].plot(T, np.abs(I), col, lw=2, label=f"{p.upper()} OFF sweep (|Vds|=VDD, nominal)")
    ax[0].set(yscale="log", xlabel="Temperature (°C)", ylabel="|I_off| = |Id| (A)",
              title="OFF current vs temperature")
    ax[0].legend(fontsize=7)

    # (b) OFF current vs vth0
    ax[1].scatter(d.vth0, d.id.abs(), s=2, c="0.8", label="random dataset")
    v = np.linspace(0.9, 1.1, 21) * nom["nmos"]["vth0"]
    I = sweep("nmos", 0, 0, V, 0, w, param="vth0", values=v)[:, 0]
    ax[1].plot(v, I, "C0", lw=2, label="NMOS OFF sweep (Vds=VDD, 27°C)")
    ax[1].set(yscale="log", xlabel="vth0 (V)", ylabel="|Id| (A)", title="OFF current vs vth0")
    ax[1].legend(fontsize=7)

    # (c) gate current vs toxe: NMOS ON, channel at 0 V, full VDD across the oxide
    d = pd.read_csv(config.DATA_DIR / "nmos_on.csv")
    ax[2].scatter(d.toxe * 1e9, d.ig.abs(), s=2, c="0.8", label="random dataset")
    t = np.linspace(0.9, 1.1, 21) * nom["nmos"]["toxe"]
    I = sweep("nmos", V, 0, 0, 0, w, param="toxe", values=t)[:, 1]
    ax[2].plot(t * 1e9, I, "C0", lw=2, label="NMOS ON sweep (Vgs=VDD, Vd=Vs=0)")
    ax[2].set(yscale="log", xlabel="toxe (nm)", ylabel="|Ig| (A)", title="Gate current vs toxe")
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "data_check_trends.png")

    # Slopes, used in the report to put numbers on the physics
    T2 = np.array([25.0, 125.0])
    I_T = sweep("nmos", 0, 0, V, 0, w, temps=T2)[:, 0]
    v2 = nom["nmos"]["vth0"] * np.array([1.0, 1.1])
    I_v = sweep("nmos", 0, 0, V, 0, w, param="vth0", values=v2)[:, 0]
    t2 = nom["nmos"]["toxe"] * np.array([1.0, 1.1])
    I_t = sweep("nmos", V, 0, 0, 0, w, param="toxe", values=t2)[:, 1]
    return dict(
        ioff_ratio_125C_vs_25C=I_T[1] / I_T[0],
        ioff_ratio_vth0_plus10pct=I_v[1] / I_v[0],
        ig_ratio_toxe_plus10pct=I_t[1] / I_t[0],
        ioff_nominal_27C=sweep("nmos", 0, 0, V, 0, w, temps=[27.0])[0, 0],
    )


def main():
    t = kirchhoff_table()
    print("Kirchhoff check (|id+ig+is+ib|, absolute and relative to largest current):")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    s = trend_plots()
    pd.Series(s).to_csv(config.RESULTS_DIR / "trend_numbers.csv")
    print("\nTrend numbers:")
    for k, v in s.items():
        print(f"  {k:28s} {v:.3g}")


if __name__ == "__main__":
    main()
