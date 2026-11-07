"""
Part C: test-circuit validation - 3-bit carry-lookahead adder.

  1. SPICE reference: the whole 204-transistor adder as ONE netlist, all
     2^7 = 128 input vectors (a2 a1 a0 b2 b1 b0 c0) x the same 200 corners.
  2. Hierarchical ML (the method of the brief): propagate logic values,
     look up every gate's ML leakage from Part B for its input combination
     and corner, and add the 42 gate leakages.
  3. Compare.

Two extra reference points split the total error into its causes:
  * spice_gate_sum : the same sum, but of SPICE isolated-gate leakages.
        spice_gate_sum vs spice_circuit  -> error of the HIERARCHY itself
        (non-ideal 0/VDD levels between gates, loading)
        ml_gate_sum vs spice_gate_sum    -> error of the ML MODELS
  * ml_flat : transistor models applied to the full circuit using the
        circuit's own SPICE node voltages (no gate-level assumption).

Extension (SPICE-free, levels-aware hierarchy):
  * ml_levels : gates solved in topological order with the ML KCL solver,
        each gate's inputs set to the ML-predicted output voltages of its
        drivers instead of ideal 0 / VDD (leakml/solve.py).  No SPICE voltage
        is used anywhere.

Outputs
  results/circuit_samples.csv, results/circuit_results.csv (per vector)
  results/circuit_summary.csv, figures/circuit_parity.png, netlists/cla3.cir
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from leakml import circuits, evaluate, solve


def main():
    corners = pd.read_csv(config.RESULTS_DIR / "corners.csv", index_col="corner")
    gate_samples = pd.read_csv(config.RESULTS_DIR / "gate_samples.csv",
                               dtype={"vector": str})
    lut = gate_samples.set_index(["gate", "vector", "corner"])[["ml_leak", "spice_leak"]]
    models = {c: joblib.load(config.SAVED_MODELS_DIR / f"{c}.joblib") for c in config.CASES}

    gates = circuits.cla3_gates()
    cells = circuits.circuit_cells(gates)
    fets = circuits.flatten(cells)
    inputs = circuits.CLA_INPUTS
    vectors = circuits.input_vectors(len(inputs))

    # The SPICE reference does not depend on the ML models, so it is cached
    # (delete cache/cla3_spice.pkl to force a re-run, e.g. after changing corners).
    cache = config.CACHE_DIR / "cla3_spice.pkl"
    t0 = time.time()
    if cache.exists():
        sims, spice_s = pd.read_pickle(cache)
    else:
        sims = evaluate.run_spice("cla3", fets, inputs, vectors, corners, cells)
        spice_s = time.time() - t0
        pd.to_pickle((sims, spice_s), cache)
    print(f"SPICE: {len(sims)} full-adder simulations in {spice_s:.0f}s, "
          f"{sims.spice_leak.isna().sum()} failed")
    evaluate.write_netlist(config.NETLIST_DIR / "cla3.cir",
                           f"3-bit carry-lookahead adder, {len(gates)} gates, {len(fets)} transistors",
                           fets, inputs, vectors[0], corners.iloc[0], cells)

    # ---- hierarchical ML: sum of gate-level predictions ----------------
    t0 = time.time()
    ml_sum = np.zeros(len(sims))
    sp_sum = np.zeros(len(sims))
    corner_idx = sims.corner.to_numpy()
    for vec in sims.vector.unique():
        rows = np.where(sims.vector == vec)[0]
        logic = circuits.evaluate_logic(cells, dict(zip(inputs, map(int, vec))))
        for gate, ins, out in gates:
            gvec = "".join(str(logic[i]) for i in ins)
            key = pd.MultiIndex.from_arrays([[gate] * len(rows), [gvec] * len(rows),
                                             corner_idx[rows]])
            vals = lut.loc[key]
            ml_sum[rows] += vals.ml_leak.to_numpy()
            sp_sum[rows] += vals.spice_leak.to_numpy()
    t_ml = time.time() - t0

    # ---- flat ML on the circuit's own node voltages ---------------------
    flat, _ = evaluate.ml_leakage(fets, inputs, sims, corners, models, cells)

    # ---- levels-aware hierarchy (SPICE-free) -----------------------------
    t0 = time.time()
    ml_lev, V_ml, _, lev_stats = solve.levels_aware_hierarchy(gates, inputs, sims[["vector", "corner"]],
                                                    corners, models)
    t_lev = time.time() - t0
    lev_stats.to_csv(config.RESULTS_DIR / "circuit_levels_solver_stats.csv", index=False)
    assert lev_stats.unconverged.sum() == 0, "levels-aware solver did not converge"
    gate_outs = [g[2] for g in gates]
    dv = np.column_stack([V_ml[o] - sims[o].to_numpy() for o in gate_outs])
    print(f"levels-aware ML: {t_lev:.0f}s; gate-output voltages vs SPICE circuit: "
          f"median |dV| {1e3*np.median(np.abs(dv)):.3f} mV, max {1e3*np.abs(dv).max():.2f} mV")

    out = sims[["vector", "corner", "spice_leak", "spice_ivdd"]].copy()
    out["ml_gate_sum"], out["spice_gate_sum"], out["ml_flat"] = ml_sum, sp_sum, flat.ml_leak.to_numpy()
    out["ml_levels"] = ml_lev
    out.to_csv(config.RESULTS_DIR / "circuit_samples.csv", index=False)

    rows = []
    for vec, g in out.groupby("vector", sort=False):
        row = dict(inputs=vec, **evaluate.summarize(g.spice_leak, g.ml_gate_sum))
        lv = evaluate.summarize(g.spice_leak, g.ml_levels)
        row["levels_bias_pct"], row["levels_med_err_pct"], row["levels_p95_err_pct"] = (
            lv["bias_pct"], lv["med_err_pct"], lv["p95_err_pct"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "circuit_results.csv", index=False)

    summary = pd.DataFrame([
        dict(comparison="hierarchical ML (gate sum)  vs SPICE circuit",
             **evaluate.summarize(out.spice_leak, out.ml_gate_sum)),
        dict(comparison="SPICE gate sum             vs SPICE circuit  (hierarchy error)",
             **evaluate.summarize(out.spice_leak, out.spice_gate_sum)),
        dict(comparison="hierarchical ML (gate sum)  vs SPICE gate sum (ML error)",
             **evaluate.summarize(out.spice_gate_sum, out.ml_gate_sum)),
        dict(comparison="flat ML on circuit voltages vs SPICE circuit",
             **evaluate.summarize(out.spice_leak, out.ml_flat)),
        dict(comparison="levels-aware ML hierarchy   vs SPICE circuit  (SPICE-free)",
             **evaluate.summarize(out.spice_leak, out.ml_levels)),
    ])
    summary.to_csv(config.RESULTS_DIR / "circuit_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"hierarchical ML time for all {len(out)} predictions: {t_ml:.2f}s")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.loglog(out.spice_leak, out.ml_gate_sum, ".", ms=2, alpha=0.4, label="hierarchical ML (brief)")
    ax.loglog(out.spice_leak, out.ml_levels, ".", ms=2, alpha=0.25, label="levels-aware ML (extension)")
    lo, hi = out.spice_leak.min(), out.spice_leak.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    for f in (1.01, 1.05):
        ax.plot([lo, hi], [lo * f, hi * f], ":", c="0.6", lw=0.6)
        ax.plot([lo, hi], [lo / f, hi / f], ":", c="0.6", lw=0.6)
    s = summary.iloc[0]
    ax.set(xlabel="SPICE circuit leakage (A)", ylabel="ML leakage (A)",
           title=f"3-bit CLA, {len(out)} sims: median err {s.med_err_pct:.2f}%, p95 {s.p95_err_pct:.2f}%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "circuit_parity.png", dpi=130)


if __name__ == "__main__":
    main()
