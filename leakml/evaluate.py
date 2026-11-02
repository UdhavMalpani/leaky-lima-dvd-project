"""
SPICE-vs-ML leakage evaluation for any flat list of transistors.

Leakage definition used throughout the project
-----------------------------------------------
    I_leak = P_static / VDD,   P_static = total DC power delivered by ALL
                               sources (VDD rail and every input driver).
This is the 'equivalent leakage current' (P_leak = VDD * I_leak).  It counts
gate-tunnelling current that flows into an input pin, which in a real
circuit is supplied from VDD through the driving gate - so gate leakages
defined this way add up to the circuit leakage.  We also record the plain
supply current I(VDD) as a secondary metric.

ML side: by Tellegen's theorem the power delivered by the sources equals the
power absorbed by the devices, and for each transistor
    P = sum over its six bipoles of  V_xy * I_xy   ( = G * V^2 >= 0 ).
"""
import numpy as np
import pandas as pd

import config
from leakml import bipoles, circuits, spice


# --------------------------------------------------------------------------
def draw_corners(n=config.N_CORNERS, seed=config.SEED_CORNERS):
    """One corner = VDD, T, one process draw for all NMOS, one for all PMOS."""
    rng = np.random.default_rng(seed)
    s = config.PROCESS_SPREAD
    c = pd.DataFrame({"vdd": rng.uniform(*config.VDD_RANGE, n),
                      "temp": rng.uniform(*config.TEMP_RANGE, n)})
    for dev in ("nmos", "pmos"):
        for p in config.PROCESS_PARAMS:
            c[f"{dev}_{p}"] = config.NOMINAL[dev][p] * rng.uniform(1 - s, 1 + s, n)
    c.index.name = "corner"
    return c


# --------------------------------------------------------------------------
def netlist_body(fets, inputs):
    lines = ["Vdd vdd 0 0"] + [f"V{i} {i} 0 0" for i in inputs]
    lines += [f"M{f.name} {f.d} {f.g} {f.s} {f.b} {f.dev} "
              f"W={spice.fmt(f.w)} L={spice.fmt(config.LMIN)}" for f in fets]
    return lines


def corner_cmds(corner, vec, inputs):
    cmds = []
    for dev in ("nmos", "pmos"):
        cmds += spice.altermod_cmds(dev, {p: corner[f"{dev}_{p}"] for p in config.PROCESS_PARAMS})
    vdd = corner["vdd"]
    cmds.append(f"alter vdd dc={spice.fmt(vdd)}")
    cmds += [f"alter v{i} dc={spice.fmt(bit * vdd)}" for i, bit in zip(inputs, vec)]
    cmds.append(f"option temp={corner['temp']:.6f}")
    return cmds


def run_spice(name, fets, inputs, vectors, corners):
    """Simulate every (input vector, corner).  Returns a DataFrame with one
    row per simulation: vector, corner, node voltages, source currents and
    the SPICE leakage."""
    nodes = circuits.internal_nodes(fets, inputs)
    outputs = ["i(Vdd)"] + [f"i(V{i})" for i in inputs] + [f"v({n})" for n in nodes]
    runs, meta = [], []
    for vec in vectors:
        for k, c in corners.iterrows():
            runs.append(corner_cmds(c, vec, inputs))
            meta.append((vec, k))
    res = spice.run_batch(name, netlist_body(fets, inputs), runs, outputs, tag=name)
    df = pd.DataFrame(res, columns=["I_vdd"] + [f"I_{i}" for i in inputs] + nodes)
    df.insert(0, "vector", ["".join(map(str, v)) for v, _ in meta])
    df.insert(1, "corner", [k for _, k in meta])
    vdd = corners.loc[df.corner, "vdd"].to_numpy()
    # power delivered by each source = -V * I(V)  (ngspice: + -> - through source)
    p = -vdd * df["I_vdd"].to_numpy()
    for i in inputs:
        bit = np.array([int(v[inputs.index(i)]) for v in df.vector])
        p += -(bit * vdd) * df[f"I_{i}"].to_numpy()
    df["spice_leak"] = p / vdd
    df["spice_ivdd"] = -df["I_vdd"]
    return df


def write_netlist(path, title, fets, inputs, vec, corner):
    """Standalone netlist for one vector at one corner (deliverable 2)."""
    vdd = corner["vdd"]
    models = []
    from importlib import import_module
    alt = import_module("01_generate_transistor_data").altered_model_text
    for dev in ("nmos", "pmos"):
        models.append(alt(dev, {p: corner[f"{dev}_{p}"] for p in config.PROCESS_PARAMS}))
    body = netlist_body(fets, inputs)
    body[0] = f"Vdd vdd 0 {spice.fmt(vdd)}"
    for j, (i, bit) in enumerate(zip(inputs, vec)):
        body[1 + j] = f"V{i} {i} 0 {spice.fmt(bit * vdd)}"
    nodes = circuits.internal_nodes(fets, inputs)
    path.write_text("\n".join([
        f"* {title}", f"* inputs {''.join(inputs)} = {''.join(map(str, vec))}, "
        f"VDD = {vdd:.4f} V, T = {corner['temp']:.2f} C",
        config.SPICE_OPTIONS, f".temp {corner['temp']:.6f}", *models, *body,
        ".control", "op", "print i(Vdd) " + " ".join(f"i(V{i})" for i in inputs),
        "print " + " ".join(f"v({n})" for n in nodes), ".endc", ".end", ""]))


# --------------------------------------------------------------------------
def ml_leakage(fets, inputs, sims, corners, models, cells):
    """ML leakage for every row of `sims` (output of run_spice), using the
    SPICE node voltages and the four transistor models."""
    n = len(sims)
    vdd = corners.loc[sims.corner, "vdd"].to_numpy()
    temp = corners.loc[sims.corner, "temp"].to_numpy()

    def node_v(node):
        if node == "vdd":
            return vdd
        if node == "0":
            return np.zeros(n)
        if node in inputs:
            j = inputs.index(node)
            return np.array([int(v[j]) for v in sims.vector]) * vdd
        return sims[node].to_numpy()

    # logic value of every node for each distinct input vector
    logic = {vec: circuits.evaluate_logic(cells, dict(zip(inputs, map(int, vec))))
             for vec in sims.vector.unique()}

    power = np.zeros(n)
    i_vdd = np.zeros(n)
    per_fet = {}
    for f in fets:
        gate_bit = np.array([logic[v][f.g] for v in sims.vector])
        on = gate_bit == (1 if f.dev == "nmos" else 0)
        df = pd.DataFrame({"vdd": vdd, "temp": temp, "w": f.w,
                           "vd": node_v(f.d), "vg": node_v(f.g),
                           "vs": node_v(f.s), "vb": node_v(f.b)})
        for p in config.PROCESS_PARAMS:
            df[p] = corners.loc[sims.corner, f"{f.dev}_{p}"].to_numpy()
        bip = pd.DataFrame(0.0, index=df.index, columns=bipoles.BIPOLES)
        for state, mask in (("on", on), ("off", ~on)):
            if mask.any():
                bip.loc[mask] = models[f"{f.dev}_{state}"].predict_bipoles(df[mask]).to_numpy()
        p_fet = sum(bip[b].to_numpy() * bipoles.voltage(df, b) for b in bipoles.BIPOLES)
        power += p_fet
        per_fet[f.name] = p_fet / vdd
        term = bipoles.terminals(bip)
        for t, node in (("id", f.d), ("ig", f.g), ("is", f.s), ("ib", f.b)):
            if node == "vdd":
                i_vdd += term[t].to_numpy()
    out = sims[["vector", "corner"]].copy()
    out["ml_leak"] = power / vdd
    out["ml_ivdd"] = i_vdd
    return out, pd.DataFrame(per_fet, index=sims.index)


# --------------------------------------------------------------------------
def summarize(true, pred):
    """Row of the results table: medians, bias, median / p95 |error|."""
    rel = (np.asarray(pred) - np.asarray(true)) / np.asarray(true)
    return dict(spice_median=np.median(true), ml_median=np.median(pred),
                bias_pct=100 * np.median(rel), med_err_pct=100 * np.median(np.abs(rel)),
                p95_err_pct=100 * np.quantile(np.abs(rel), 0.95),
                max_err_pct=100 * np.max(np.abs(rel)), n=len(rel))
