"""
Extra analyses built on the trained models (no new SPICE runs).

  1. Minimum-leakage input vector (MLV) of the CLA: does ML rank the 128
     standby vectors like SPICE does?
  2. Leakage mechanisms: split every gate's ML leakage into channel
     (sub-threshold), gate-tunnelling and junction parts across temperature.
     SPICE only reports the total; the bipole models give the split for free.
  3. Sensitivity of the CLA leakage to each PVT parameter (one at a time).
  4. Monte-Carlo: CLA leakage distribution over 5,000 new random corners
     (levels-aware ML), compared with the 200 SPICE corners.
  5. Uncertainty: the spread of the 5 ensemble members as an error bar.
     Is it large where the real error is large?
  6. Cost: SPICE vs ML time per evaluation.

Outputs: results/extras_*.csv, figures/extras_*.png, results/extras_summary.txt
"""
import copy
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from leakml import bipoles, circuits, evaluate, solve
from leakml.models import Ensemble, fast_copy

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def nominal_corners(**over):
    """One-row-per-entry corner table at nominal process, VDD_NOM, 25 C."""
    n = max([len(np.atleast_1d(v)) for v in over.values()] + [1])
    c = pd.DataFrame({"vdd": np.full(n, config.VDD_NOM), "temp": np.full(n, 25.0)})
    for dev in ("nmos", "pmos"):
        for p in config.PROCESS_PARAMS:
            c[f"{dev}_{p}"] = np.full(n, config.NOMINAL[dev][p])
    for k, v in over.items():
        c[k] = v
    c.index.name = "corner"
    return c


def gate_solve(gate, vec, corners, models):
    """ML operating point of one gate input case at every corner."""
    inputs = circuits.GATE_INPUTS[gate]
    cells = circuits.gate_cells(gate, inputs, "out", "g")
    fets = circuits.flatten(cells)
    vdd, n = corners.vdd.to_numpy(), len(corners)
    nodes = circuits.internal_nodes(fets, inputs)
    guess, logic = solve.logic_guess(cells, inputs, vec, nodes, vdd)
    fixed = {"vdd": vdd, "0": np.zeros(n), **{i: b * vdd for i, b in zip(inputs, vec)}}
    P = solve.Problem(fets, fixed, nodes, corners.reset_index(drop=True),
                      {k: np.full(n, v) for k, v in logic.items()}, models)
    V, info = solve.solve(P, guess)
    assert info["unconverged"] == 0, (gate, vec, info)
    return P, V


GROUPS = {"channel": ["ds"], "tunnelling": ["gs", "gd", "gb"], "junction": ["db", "sb"]}


def mechanism_power(P, V):
    """Static power per mechanism group, summed over the FETs."""
    out = {g: np.zeros(P.n) for g in GROUPS}
    for k in range(len(P.fets)):
        bip, df = P.fet_terminals(k, V)
        for g, names in GROUPS.items():
            out[g] += sum(bip[b].to_numpy() * bipoles.voltage(df, b) for b in names)
    return out


def member_models(models, i):
    """Copy of the models that uses only ensemble member i."""
    res = {}
    for case, m in models.items():
        mm = copy.copy(m)
        mm.sub = {k: (v.members[i] if isinstance(v, Ensemble) else v) for k, v in m.sub.items()}
        res[case] = mm
    return res


GATE_CASES = [(g, v) for g in circuits.GATE_INPUTS
              for v in circuits.input_vectors(len(circuits.GATE_INPUTS[g]))]


# --------------------------------------------------------------------------
def mlv(cs):
    say("\n== 1. Minimum-leakage vector (CLA, mean over 200 corners)")
    per = cs.groupby("vector")[["spice_leak", "ml_gate_sum", "ml_levels", "ml_flat"]].mean()
    rows = []
    for col in ["ml_gate_sum", "ml_levels"]:
        rho = stats.spearmanr(per.spice_leak, per[col]).statistic
        pick = per[col].idxmin()
        best = per.spice_leak.idxmin()
        regret = per.spice_leak[pick] / per.spice_leak[best] - 1
        # per corner: SPICE leakage of the vector ML picks vs the true minimum
        piv_s = cs.pivot(index="corner", columns="vector", values="spice_leak")
        piv_m = cs.pivot(index="corner", columns="vector", values=col)
        chosen = piv_s.to_numpy()[np.arange(len(piv_s)), piv_m.to_numpy().argmin(1)]
        reg_c = chosen / piv_s.min(axis=1).to_numpy() - 1
        rows.append(dict(method=col, spearman=rho, ml_mlv=pick, spice_mlv=best,
                         regret_pct=100 * regret, corner_regret_median_pct=100 * np.median(reg_c),
                         corner_regret_max_pct=100 * reg_c.max(),
                         corners_exact=np.mean(reg_c == 0)))
    r = pd.DataFrame(rows)
    r.to_csv(config.RESULTS_DIR / "extras_mlv.csv", index=False)
    say(r.round(4).to_string(index=False))
    say(f"max/min vector leakage (SPICE): {per.spice_leak.max() / per.spice_leak.min():.2f}x; "
        f"MLV {per.spice_leak.idxmin()} = {per.spice_leak.min():.3e} A, "
        f"worst {per.spice_leak.idxmax()} = {per.spice_leak.max():.3e} A")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    o = per.sort_values("spice_leak")
    x = np.arange(len(o))
    ax.plot(x, o.spice_leak * 1e6, "k.-", ms=3, lw=0.8, label="SPICE circuit")
    ax.plot(x, o.ml_levels * 1e6, "o", ms=3, mfc="none", label="levels-aware ML")
    ax.plot(x, o.ml_gate_sum * 1e6, "x", ms=3, alpha=0.6, label="hierarchical ML (ideal levels)")
    ax.set(xlabel="input vector, sorted by SPICE leakage", ylabel="mean leakage (uA)",
           title="CLA standby leakage of all 128 input vectors (mean over 200 corners)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "extras_mlv.png", dpi=130)
    plt.close(fig)
    return per.spice_leak.idxmin(), per.spice_leak.idxmax()


# --------------------------------------------------------------------------
def mechanisms(models):
    say("\n== 2. Leakage mechanisms (nominal process, VDD_NOM)")
    temps = np.linspace(-40, 125, 12)
    cor = nominal_corners(temp=temps)
    rows = []
    for gate, vec in GATE_CASES:
        P, V = gate_solve(gate, vec, cor, models)
        mp = mechanism_power(P, V)
        for j, t in enumerate(temps):
            rows.append(dict(gate=gate, vector="".join(map(str, vec)), temp=t,
                             **{g: mp[g][j] / config.VDD_NOM for g in GROUPS}))
    df = pd.DataFrame(rows)
    df["total"] = df[list(GROUPS)].sum(1)
    for g in GROUPS:
        df[f"{g}_frac"] = df[g] / df.total
    df.to_csv(config.RESULTS_DIR / "extras_mechanisms.csv", index=False)
    for t in (temps[0], 25.0, temps[-1]):
        d = df[np.isclose(df.temp, t, atol=8)].groupby("temp")[[f"{g}_frac" for g in GROUPS]].mean()
        say(f"T={d.index[0]:6.1f} C  mean share over 18 cases: "
            + ", ".join(f"{g} {100 * d[f'{g}_frac'].iloc[0]:.1f}%" for g in GROUPS))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (gate, vec), c in zip([("INV", "1"), ("NAND2", "00"), ("NOR2", "11")], ["C0", "C1", "C2"]):
        g = df[(df.gate == gate) & (df.vector == vec)]
        for mech, ls in zip(GROUPS, ["-", "--", ":"]):
            ax[0].semilogy(g.temp, g[mech], ls, c=c, label=f"{gate} {vec} {mech}")
    ax[0].set(xlabel="temperature (C)", ylabel="leakage component (A)",
              title="Mechanisms vs temperature (ML)")
    ax[0].legend(fontsize=6, ncol=3)
    d = df[np.isclose(df.temp, 25, atol=8)]
    lab = d.gate + " " + d.vector
    bottom = np.zeros(len(d))
    for mech in GROUPS:
        ax[1].bar(lab, 100 * d[f"{mech}_frac"], bottom=bottom, label=mech)
        bottom += 100 * d[f"{mech}_frac"].to_numpy()
    ax[1].set(ylabel="share of leakage (%)", title=f"Share per gate case at {d.temp.iloc[0]:.0f} C")
    ax[1].tick_params(axis="x", rotation=90, labelsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "extras_mechanisms.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def sensitivity(models):
    say("\n== 3. Sensitivity of CLA leakage (levels-aware ML, mean over 128 vectors)")
    cases = [("nominal", {})]
    cases += [("VDD -10%", {"vdd": 0.9 * config.VDD_NOM}), ("VDD +10%", {"vdd": 1.1 * config.VDD_NOM}),
              ("T = -40 C", {"temp": -40.0}), ("T = 125 C", {"temp": 125.0})]
    for dev in ("nmos", "pmos"):
        for p in config.PROCESS_PARAMS:
            nom = config.NOMINAL[dev][p]
            cases += [(f"{dev} {p} -10%", {f"{dev}_{p}": 0.9 * nom}),
                      (f"{dev} {p} +10%", {f"{dev}_{p}": 1.1 * nom})]
    cor = nominal_corners(temp=np.full(len(cases), 25.0))
    for j, (_, over) in enumerate(cases):
        for k, v in over.items():
            cor.loc[j, k] = v
    vecs = circuits.input_vectors(len(circuits.CLA_INPUTS))
    rows = pd.DataFrame([("".join(map(str, v)), c) for v in vecs for c in cor.index],
                        columns=["vector", "corner"])
    t0 = time.time()
    tot, *_ = solve.levels_aware_hierarchy(circuits.cla3_gates(), circuits.CLA_INPUTS, rows, cor, models)
    rows["leak"] = tot
    m = rows.groupby("corner").leak.mean()
    res = pd.DataFrame({"case": [c[0] for c in cases], "leak": m.to_numpy()})
    res["ratio_to_nominal"] = res.leak / res.leak.iloc[0]
    res.to_csv(config.RESULTS_DIR / "extras_sensitivity.csv", index=False)
    say(f"({time.time() - t0:.0f}s) nominal CLA leakage {res.leak.iloc[0]:.3e} A")
    say(res.iloc[1:].sort_values("ratio_to_nominal").round(3).to_string(index=False))

    # tornado: one bar per parameter (low / high)
    par = res.iloc[1:].copy()
    par["param"] = par.case.str.replace(r" [-+]10%| = .*", "", regex=True).str.replace("T", "temperature")
    par.loc[par.case.str.startswith("T ="), "param"] = "temperature (-40 / 125 C)"
    lo = par.groupby("param").ratio_to_nominal.min()
    hi = par.groupby("param").ratio_to_nominal.max()
    order = (np.log(hi) - np.log(lo)).sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(order))
    ax.barh(y, np.log10(hi[order]), color="C3", label="higher leakage")
    ax.barh(y, np.log10(lo[order]), color="C0", label="lower leakage")
    ax.set_yticks(y, order, fontsize=8)
    ax.axvline(0, c="k", lw=0.6)
    ticks = np.array([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticks(np.log10(ticks), [f"{t:g}x" for t in ticks])
    ax.set(xlabel="CLA leakage / nominal (log scale)",
           title="One-at-a-time sensitivity (±10 % process, VDD; T range)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "extras_sensitivity.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def monte_carlo(models, cs, vectors, n=5000):
    say(f"\n== 4. Monte-Carlo: {n} new corners, levels-aware ML")
    cor = evaluate.draw_corners(n=n, seed=config.SEED_CORNERS + 1000)
    rows = pd.DataFrame([(v, c) for v in vectors for c in cor.index], columns=["vector", "corner"])
    t0 = time.time()
    tot, *_ = solve.levels_aware_hierarchy(circuits.cla3_gates(), circuits.CLA_INPUTS, rows, cor, models)
    say(f"{len(rows)} CLA evaluations in {time.time() - t0:.0f}s")
    rows["leak"] = tot
    rows.to_csv(config.RESULTS_DIR / "extras_montecarlo.csv", index=False)
    res = []
    fig, ax = plt.subplots(1, len(vectors), figsize=(5.5 * len(vectors), 3.6))
    for a, v in zip(np.atleast_1d(ax), vectors):
        x = rows[rows.vector == v].leak.to_numpy()
        sp = cs[cs.vector == v].spice_leak.to_numpy()
        mu, sd = np.log(x).mean(), np.log(x).std()
        ks = stats.ks_2samp(np.log(x), np.log(sp))
        res.append(dict(vector=v, n=len(x), median=np.median(x), p01=np.quantile(x, 0.01),
                        p99=np.quantile(x, 0.99), p99_over_median=np.quantile(x, 0.99) / np.median(x),
                        lognorm_sigma=sd, spice200_median=np.median(sp), spice200_p99=np.quantile(sp, 0.99),
                        ks_pvalue_vs_spice200=ks.pvalue))
        bins = np.logspace(np.log10(min(x.min(), sp.min())), np.log10(max(x.max(), sp.max())), 50)
        a.hist(x, bins=bins, density=True, alpha=0.6, label=f"ML, {len(x)} corners")
        a.hist(sp, bins=bins, density=True, histtype="step", color="k", label="SPICE, 200 corners")
        a.set_xscale("log")
        a.set(xlabel="CLA leakage (A)", ylabel="density", title=f"vector {v}")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "extras_montecarlo.png", dpi=130)
    plt.close(fig)
    r = pd.DataFrame(res)
    r.to_csv(config.RESULTS_DIR / "extras_montecarlo_summary.csv", index=False)
    say(r.to_string(index=False, float_format=lambda f: f"{f:.4g}"))


# --------------------------------------------------------------------------
def uncertainty(models, gs, corners):
    say("\n== 5. Ensemble spread as an uncertainty estimate (gates, ML-solved voltages)")
    n_mem = min(len(v.members) for m in models.values() for v in m.sub.values() if isinstance(v, Ensemble))
    mems = [member_models(models, i) for i in range(n_mem)]
    rows = []
    for gate, vec in GATE_CASES:
        leak = []
        for mm in mems:
            P, V = gate_solve(gate, vec, corners, mm)
            leak.append(solve.device_power(P, V) / P.vdd)
        leak = np.array(leak)
        g = gs[(gs.gate == gate) & (gs.vector == "".join(map(str, vec)))].sort_values("corner")
        rows.append(pd.DataFrame({"gate": gate, "vector": "".join(map(str, vec)),
                                  "corner": g.corner.to_numpy(),
                                  "member_spread_pct": 100 * leak.std(0, ddof=1) / leak.mean(0),
                                  "abs_err_pct": 100 * np.abs(g.ml_solved_leak.to_numpy() / g.spice_leak.to_numpy() - 1)}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(config.RESULTS_DIR / "extras_uncertainty.csv", index=False)
    rho = stats.spearmanr(df.member_spread_pct, df.abs_err_pct).statistic
    q = pd.qcut(df.member_spread_pct, 5, labels=False)
    tab = df.groupby(q).agg(spread_med=("member_spread_pct", "median"),
                            err_med=("abs_err_pct", "median"), err_p95=("abs_err_pct",
                                                                         lambda s: s.quantile(0.95)))
    tab.index.name = "spread quintile"
    say(f"{n_mem} members; Spearman(spread, |error|) = {rho:.3f}")
    say(tab.round(3).to_string())
    # the ensemble mean has a smaller spread than any member: standard error = spread / sqrt(n)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.loglog(df.member_spread_pct, df.abs_err_pct, ".", ms=2, alpha=0.4)
    ax.plot(tab.spread_med, tab.err_med, "ko-", label="quintile medians")
    ax.set(xlabel="spread of 5 ensemble members (%)", ylabel="|ML - SPICE| (%)",
           title=f"Uncertainty vs error, 3600 gate sims (rho = {rho:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "extras_uncertainty.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def cost(models, corners):
    say("\n== 6. Cost per evaluation")
    sims, spice_s = pd.read_pickle(config.CACHE_DIR / "cla3_spice.pkl")
    gates = circuits.cla3_gates()
    cells = circuits.circuit_cells(gates)
    fets = circuits.flatten(cells)
    fast = {k: fast_copy(m) for k, m in models.items()}     # same numbers, no sklearn overhead
    t0 = time.time()
    evaluate.ml_leakage(fets, circuits.CLA_INPUTS, sims, corners, fast, cells)
    flat_s = time.time() - t0
    n = len(sims)
    r = pd.DataFrame([
        dict(method="SPICE, full CLA (ngspice, wall clock)", seconds=spice_s,
             cpu_seconds=spice_s * config.N_WORKERS),
        dict(method="ML, flat (204 devices, given node voltages)", seconds=flat_s, cpu_seconds=flat_s),
    ])
    r["evaluations"] = n
    r["ms_per_cpu_eval"] = 1e3 * r.cpu_seconds / n
    r.to_csv(config.RESULTS_DIR / "extras_cost.csv", index=False)
    say(r.round(3).to_string(index=False))
    say(f"device evaluations per second (ML flat): {n * len(fets) / flat_s:.3g}")


def main():
    t0 = time.time()
    models = {c: joblib.load(config.SAVED_MODELS_DIR / f"{c}.joblib") for c in config.CASES}
    corners = pd.read_csv(config.RESULTS_DIR / "corners.csv", index_col="corner")
    cs = pd.read_csv(config.RESULTS_DIR / "circuit_samples.csv", dtype={"vector": str})
    gs = pd.read_csv(config.RESULTS_DIR / "gate_samples.csv", dtype={"vector": str})
    best, worst = mlv(cs)
    mechanisms(models)
    uncertainty(models, gs, corners)
    sensitivity(models)
    cost(models, corners)
    monte_carlo(models, cs, [best, worst])
    say(f"\ndone in {time.time() - t0:.0f}s")
    (config.RESULTS_DIR / "extras_summary.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
