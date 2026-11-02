"""
Part B: gate-level validation.

For INV, NAND2, NOR2, AND2, OR2, every input combination (18 cases) and the
same 200 random corners:
  1. SPICE reference: simulate the full transistor-level gate (evaluation only).
  2. ML prediction: take the SPICE node voltages (internal nodes + output),
     build each transistor's (Vd, Vg, Vs, Vb), pick the right model from
     its gate logic value (ON/OFF), predict, and sum the device powers.
  3. Compare.

Outputs
  results/corners.csv                 the 200 corners (reused in Part C)
  results/gate_samples.csv            every simulation: node voltages, SPICE & ML leakage
  results/gate_results.csv            the 18-row results table
  figures/gate_parity.png             ML vs SPICE, log axes
  netlists/gate_<GATE>.cir            one example netlist per gate
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from leakml import circuits, evaluate


def load_models():
    return {c: joblib.load(config.SAVED_MODELS_DIR / f"{c}.joblib") for c in config.CASES}


def gate_structure(gate):
    inputs = circuits.GATE_INPUTS[gate]
    cells = circuits.gate_cells(gate, inputs, "out", "g")
    return inputs, cells, circuits.flatten(cells)


def friendly(col):
    """Display names for internal nodes: x (NAND stack), y (NOR stack), NAND/NOR output."""
    if col.endswith("_x"):
        return "x"
    if col.endswith("_y"):
        return "y"
    if col in ("g_nd", "g_nr"):
        return {"g_nd": "nand_out", "g_nr": "nor_out"}[col]
    return col


def parity(df, path, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for gate, g in df.groupby("gate", sort=False):
        ax.loglog(g.spice_leak, g.ml_leak, ".", ms=3, alpha=0.6, label=gate)
    lo, hi = df[["spice_leak", "ml_leak"]].min().min(), df[["spice_leak", "ml_leak"]].max().max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    for f in (1.01, 1.05):
        ax.plot([lo, hi], [lo * f, hi * f], ":", c="0.6", lw=0.6)
        ax.plot([lo, hi], [lo / f, hi / f], ":", c="0.6", lw=0.6)
    ax.set(xlabel="SPICE leakage  P/VDD (A)", ylabel="ML leakage (A)", title=title)
    ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    corners = evaluate.draw_corners()
    corners.to_csv(config.RESULTS_DIR / "corners.csv")
    models = load_models()

    samples, rows = [], []
    for gate in circuits.GATES:
        t0 = time.time()
        inputs, cells, fets = gate_structure(gate)
        vectors = circuits.input_vectors(len(inputs))
        sims = evaluate.run_spice(f"gate_{gate}", fets, inputs, vectors, corners)
        ml, _ = evaluate.ml_leakage(fets, inputs, sims, corners, models, cells)
        evaluate.write_netlist(config.NETLIST_DIR / f"gate_{gate}.cir",
                               f"{gate} static CMOS gate ({config.TECH})",
                               fets, inputs, vectors[0], corners.iloc[0])
        s = sims.rename(columns={c: friendly(c) for c in sims.columns})
        s["ml_leak"], s["ml_ivdd"] = ml.ml_leak.to_numpy(), ml.ml_ivdd.to_numpy()
        s.insert(0, "gate", gate)
        samples.append(s)
        for vec, g in s.groupby("vector", sort=False):
            row = dict(gate=gate, inputs=vec, **evaluate.summarize(g.spice_leak, g.ml_leak))
            iv = evaluate.summarize(g.spice_ivdd, g.ml_ivdd)
            row["ivdd_med_err_pct"], row["ivdd_p95_err_pct"] = iv["med_err_pct"], iv["p95_err_pct"]
            rows.append(row)
        print(f"{gate:6s}: {len(sims)} sims, {time.time()-t0:.1f}s")

    samples = pd.concat(samples, ignore_index=True)
    samples.to_csv(config.RESULTS_DIR / "gate_samples.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "gate_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    overall = evaluate.summarize(samples.spice_leak, samples.ml_leak)
    print(f"\nALL {len(samples)} gate simulations: median |err| {overall['med_err_pct']:.3f}%, "
          f"p95 {overall['p95_err_pct']:.3f}%, mean accuracy "
          f"{100 - np.mean(np.abs(samples.ml_leak/samples.spice_leak - 1))*100:.3f}%")
    parity(samples, config.FIG_DIR / "gate_parity.png",
           f"Gate leakage, {len(samples)} simulations (18 cases x 200 corners)")


if __name__ == "__main__":
    main()
