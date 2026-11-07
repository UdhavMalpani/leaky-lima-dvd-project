"""
Extra validation on a second, unseen circuit: the 8-bit parity checker
(the brief's other test circuit; our assignment is the CLA).

Nothing is retrained or re-tuned: the same four transistor models, the same
gate tables (Part B) and the same 200 corners.

  * SPICE: the full 112-transistor netlist, all 2^8 = 256 vectors x 200
    corners (cached in cache/parity8_spice.pkl.xz, gate outputs only).
  * hierarchical ML (the brief's method) and the levels-aware ML hierarchy.

Outputs: results/parity_summary.csv, results/parity_samples.csv,
         figures/parity_parity.png, netlists/parity8.cir
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
    gs = pd.read_csv(config.RESULTS_DIR / "gate_samples.csv", dtype={"vector": str})
    lut = gs.set_index(["gate", "vector", "corner"])[["ml_leak", "spice_leak"]]
    models = {c: joblib.load(config.SAVED_MODELS_DIR / f"{c}.joblib") for c in config.CASES}

    gates = circuits.parity8_gates()
    cells = circuits.circuit_cells(gates)
    fets = circuits.flatten(cells)
    inputs = circuits.PARITY_INPUTS
    vectors = circuits.input_vectors(len(inputs))
    outs = [g[2] for g in gates]
    print(f"parity checker: {len(gates)} gates, {len(fets)} transistors, {len(vectors)} vectors")

    cache = config.CACHE_DIR / "parity8_spice.pkl.xz"
    if cache.exists():
        sims, spice_s = pd.read_pickle(cache)
    else:
        t0 = time.time()
        full = evaluate.run_spice("parity8", fets, inputs, vectors, corners, cells)
        spice_s = time.time() - t0
        # keep only what is used later (the cache is tracked: keep it small)
        sims = full[["vector", "corner", "spice_leak", "spice_ivdd"]].copy()
        for o in outs:
            sims[o] = full[o].astype(np.float32)
        pd.to_pickle((sims, spice_s), cache)
    print(f"SPICE: {len(sims)} simulations in {spice_s:.0f}s, {sims.spice_leak.isna().sum()} failed")
    evaluate.write_netlist(config.NETLIST_DIR / "parity8.cir",
                           f"8-bit parity checker, {len(gates)} gates, {len(fets)} transistors",
                           fets, inputs, vectors[0], corners.iloc[0], cells)

    # hierarchical ML (sum of Part-B gate predictions at ideal input levels)
    t0 = time.time()
    ml_sum, sp_sum = np.zeros(len(sims)), np.zeros(len(sims))
    cidx = sims.corner.to_numpy()
    for vec in sims.vector.unique():
        rows = np.where(sims.vector == vec)[0]
        logic = circuits.evaluate_logic(cells, dict(zip(inputs, map(int, vec))))
        for gate, ins, out in gates:
            gvec = "".join(str(logic[i]) for i in ins)
            key = pd.MultiIndex.from_arrays([[gate] * len(rows), [gvec] * len(rows), cidx[rows]])
            vals = lut.loc[key]
            ml_sum[rows] += vals.ml_leak.to_numpy()
            sp_sum[rows] += vals.spice_leak.to_numpy()
    t_ml = time.time() - t0

    t0 = time.time()
    ml_lev, V_ml, _, st = solve.levels_aware_hierarchy(gates, inputs, sims[["vector", "corner"]], corners, models)
    t_lev = time.time() - t0
    assert st.unconverged.sum() == 0
    dv = np.column_stack([V_ml[o] - sims[o].to_numpy() for o in outs])
    print(f"levels-aware ML: {t_lev:.0f}s, max {st.iterations.max()} iterations; gate outputs vs SPICE: "
          f"median |dV| {1e3*np.median(np.abs(dv)):.3f} mV, max {1e3*np.abs(dv).max():.2f} mV")

    out = sims[["vector", "corner", "spice_leak", "spice_ivdd"]].copy()
    out["ml_gate_sum"], out["spice_gate_sum"], out["ml_levels"] = ml_sum, sp_sum, ml_lev
    out.to_csv(config.RESULTS_DIR / "parity_samples.csv", index=False)

    def acc(t, p):
        return 100 - 100 * np.mean(np.abs(p / t - 1))
    summary = pd.DataFrame([
        dict(comparison="hierarchical ML (gate sum) vs SPICE circuit",
             **evaluate.summarize(out.spice_leak, out.ml_gate_sum), mean_acc=acc(out.spice_leak, out.ml_gate_sum)),
        dict(comparison="SPICE gate sum vs SPICE circuit (hierarchy error)",
             **evaluate.summarize(out.spice_leak, out.spice_gate_sum), mean_acc=acc(out.spice_leak, out.spice_gate_sum)),
        dict(comparison="hierarchical ML vs SPICE gate sum (ML error)",
             **evaluate.summarize(out.spice_gate_sum, out.ml_gate_sum), mean_acc=acc(out.spice_gate_sum, out.ml_gate_sum)),
        dict(comparison="levels-aware ML hierarchy vs SPICE circuit (SPICE-free)",
             **evaluate.summarize(out.spice_leak, out.ml_levels), mean_acc=acc(out.spice_leak, out.ml_levels)),
    ])
    summary.to_csv(config.RESULTS_DIR / "parity_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"hierarchical ML time: {t_ml:.2f}s")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.loglog(out.spice_leak, out.ml_gate_sum, ".", ms=2, alpha=0.4, label="hierarchical ML (brief)")
    ax.loglog(out.spice_leak, out.ml_levels, ".", ms=2, alpha=0.25, label="levels-aware ML")
    lo, hi = out.spice_leak.min(), out.spice_leak.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    for f in (1.01, 1.05):
        ax.plot([lo, hi], [lo * f, hi * f], ":", c="0.6", lw=0.6)
        ax.plot([lo, hi], [lo / f, hi / f], ":", c="0.6", lw=0.6)
    ax.set(xlabel="SPICE circuit leakage (A)", ylabel="ML leakage (A)",
           title=f"8-bit parity checker, {len(out)} sims (unseen circuit)")
    ax.legend(fontsize=8)
    from matplotlib.ticker import NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "parity_parity.png", dpi=130)


if __name__ == "__main__":
    main()
