"""Fast invariants of the pipeline (no SPICE needed).   Run:  .venv/bin/python -m pytest -q"""
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from leakml import bipoles, circuits, evaluate, models  # noqa: E402


def test_bsim4_weff_clamp():
    # W = 1 Lmin -> Weff = 6 nm -> BSIM4 clamp gives 14.17 nm; continuous at 20 nm; identity above
    df = pd.DataFrame({"w": [16e-9, 30e-9, 30e-9 + 1e-15, 64e-9]})
    w = models.w_eff_clamped(df)
    assert w[0] == pytest.approx(2e-8 * (4e-8 - 6e-9) / (6e-8 - 12e-9))
    assert w[1] == pytest.approx(20e-9) and w[2] == pytest.approx(20e-9, rel=1e-6)
    assert w[3] == pytest.approx(64e-9 - 2 * config.WINT)


@pytest.mark.parametrize("gate,truth", [
    ("INV", {(0,): 1, (1,): 0}),
    ("NAND2", {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}),
    ("NOR2", {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}),
    ("AND2", {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}),
    ("OR2", {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
])
def test_gate_truth_tables(gate, truth):
    inputs = circuits.GATE_INPUTS[gate]
    cells = circuits.gate_cells(gate, inputs, "out", "g")
    for vec, want in truth.items():
        assert circuits.evaluate_logic(cells, dict(zip(inputs, vec)))["out"] == want


def test_cla3_adds():
    cells = circuits.circuit_cells(circuits.cla3_gates())
    for a, b, c0 in product(range(8), range(8), range(2)):
        bits = dict(a2=a >> 2 & 1, a1=a >> 1 & 1, a0=a & 1, b2=b >> 2 & 1, b1=b >> 1 & 1, b0=b & 1, c0=c0)
        v = circuits.evaluate_logic(cells, bits)
        assert v["s0"] + 2 * v["s1"] + 4 * v["s2"] + 8 * v["c3"] == a + b + c0


def test_cla3_size():
    fets = circuits.flatten(circuits.circuit_cells(circuits.cla3_gates()))
    assert len(circuits.cla3_gates()) == 42 and len(fets) == 204


def test_netlist_and_ml_share_topology():
    # every FET in the SPICE netlist body is exactly one Fet of the ML description
    inputs = circuits.GATE_INPUTS["AND2"]
    fets = circuits.flatten(circuits.gate_cells("AND2", inputs, "out", "g"))
    body = evaluate.netlist_body(fets, inputs)
    mlines = [ln for ln in body if ln.startswith("M")]
    assert len(mlines) == len(fets)
    for f, ln in zip(fets, mlines):
        assert ln.split()[1:6] == [f.d, f.g, f.s, f.b, f.dev]


def test_nodeset_uses_logic_levels():
    inputs = circuits.GATE_INPUTS["AND2"]
    cells = circuits.gate_cells("AND2", inputs, "out", "g")
    fets = circuits.flatten(cells)
    line = evaluate.nodeset_line(cells, inputs, (0, 0), circuits.internal_nodes(fets, inputs))[0]
    assert "v(g_nd)=1" in line and "v(out)=0" in line


@pytest.mark.parametrize("case", list(config.CASES))
def test_saved_models_obey_kcl(case):
    m = joblib.load(config.SAVED_MODELS_DIR / f"{case}.joblib")
    df = pd.read_csv(config.DATA_DIR / f"{case}.csv").head(200)
    pred = m.predict(df)
    s = pred.sum(axis=1).abs() / pred.abs().max(axis=1)
    assert s.max() < 1e-9


@pytest.mark.parametrize("case", list(config.CASES))
def test_bipoles_are_passive(case):
    bip = pd.read_csv(config.DATA_DIR / f"{case}_bipoles.csv")
    df = pd.read_csv(config.DATA_DIR / f"{case}.csv")
    for b in bipoles.BIPOLES:
        v, i = bipoles.voltage(df, b), bip[b].to_numpy()
        big = (np.abs(i) > 1e-15) & (np.abs(v) > 1e-3)
        if big.sum() < 50:          # mechanism absent in this case (e.g. Vgb = 0 when OFF)
            continue
        assert np.mean(np.sign(i[big]) == np.sign(v[big])) > 0.999, b


# ---------------------------------------------------------------------------
# SPICE-free solver (leakml/solve.py)
# ---------------------------------------------------------------------------
from leakml import solve  # noqa: E402
from leakml.models import fast_copy  # noqa: E402


def _models():
    return {c: joblib.load(config.SAVED_MODELS_DIR / f"{c}.joblib") for c in config.CASES}


def test_fast_inference_identical():
    m = joblib.load(config.SAVED_MODELS_DIR / "nmos_off.joblib")
    df = pd.read_csv(config.DATA_DIR / "nmos_off.csv").head(100)
    a, b = m.predict_bipoles(df).to_numpy(), fast_copy(m).predict_bipoles(df).to_numpy()
    assert np.allclose(a, b, rtol=1e-12, atol=0)


def test_stack_guess_nand():
    inputs = circuits.GATE_INPUTS["NAND2"]
    cells = circuits.gate_cells("NAND2", inputs, "out", "g")
    nodes = circuits.internal_nodes(circuits.flatten(cells), inputs)
    vdd = np.array([1.0])
    x = next(n for n in nodes if n.endswith("_x"))
    assert solve.logic_guess(cells, inputs, (0, 1), nodes, vdd)[0][x][0] == 0.0   # bottom NMOS on -> x at ground
    assert solve.logic_guess(cells, inputs, (1, 0), nodes, vdd)[0][x][0] == 1.0   # top NMOS on -> x follows out (=1)


@pytest.mark.parametrize("gate,vec", [("INV", (1,)), ("NAND2", (1, 0)), ("OR2", (0, 1))])
def test_solver_converges_and_closes_kcl(gate, vec):
    corners = pd.read_csv(config.RESULTS_DIR / "corners.csv", index_col="corner").head(20)
    inputs = circuits.GATE_INPUTS[gate]
    cells = circuits.gate_cells(gate, inputs, "out", "g")
    fets = circuits.flatten(cells)
    nodes = circuits.internal_nodes(fets, inputs)
    vdd, n = corners.vdd.to_numpy(), len(corners)
    guess, logic = solve.logic_guess(cells, inputs, vec, nodes, vdd)
    fixed = {"vdd": vdd, "0": np.zeros(n), **{i: b * vdd for i, b in zip(inputs, vec)}}
    P = solve.Problem(fets, fixed, nodes, corners, {k: np.full(n, v) for k, v in logic.items()}, _models())
    V, info = solve.solve(P, guess)
    assert info["unconverged"] == 0
    F, _ = P.residual(V)
    # residual tiny compared with the current through the network
    assert np.max(np.abs(F)) < 1e-12
    out = logic["out"]
    assert np.all(np.abs(V["out"] - out * vdd) < 0.1 * vdd)   # output at the right logic level


def test_levels_aware_single_gate_equals_gate_solve():
    # one-gate "circuit": the levels-aware hierarchy must reduce to the plain gate solve
    corners = pd.read_csv(config.RESULTS_DIR / "corners.csv", index_col="corner").head(10)
    models = _models()
    rows = pd.DataFrame([(v, c) for v in ("00", "01", "10", "11") for c in corners.index],
                        columns=["vector", "corner"])
    tot, V, _, st = solve.levels_aware_hierarchy([("NAND2", ("a", "b"), "out")], ["a", "b"],
                                                 rows, corners, models)
    assert st.unconverged.sum() == 0
    inputs = circuits.GATE_INPUTS["NAND2"]
    cells = circuits.gate_cells("NAND2", inputs, "out", "g")
    fets = circuits.flatten(cells)
    nodes = circuits.internal_nodes(fets, inputs)
    vdd, n = corners.vdd.to_numpy(), len(corners)
    for j, vec in enumerate(circuits.input_vectors(2)):
        guess, logic = solve.logic_guess(cells, inputs, vec, nodes, vdd)
        fixed = {"vdd": vdd, "0": np.zeros(n), **{i: b * vdd for i, b in zip(inputs, vec)}}
        P = solve.Problem(fets, fixed, nodes, corners, {k: np.full(n, v) for k, v in logic.items()}, models)
        Vg, _ = solve.solve(P, guess)
        ref = solve.device_power(P, Vg) / vdd
        assert np.allclose(tot[j * n:(j + 1) * n], ref, rtol=1e-6)


def test_cla_levels_results_consistent():
    # saved CLA results: levels-aware column present, finite, positive, and better than the ideal-level hierarchy
    cs = pd.read_csv(config.RESULTS_DIR / "circuit_samples.csv")
    assert len(cs) == 128 * 200 and np.isfinite(cs.ml_levels).all() and (cs.ml_levels > 0).all()
    e_lev = np.median(np.abs(cs.ml_levels / cs.spice_leak - 1))
    e_hier = np.median(np.abs(cs.ml_gate_sum / cs.spice_leak - 1))
    assert e_lev < e_hier


def test_parity8_computes_parity():
    gates = circuits.parity8_gates()
    cells = circuits.circuit_cells(gates)
    assert len(gates) == 28 and len(circuits.flatten(cells)) == 112
    for vec in circuits.input_vectors(8):
        v = circuits.evaluate_logic(cells, dict(zip(circuits.PARITY_INPUTS, vec)))
        assert v["par"] == sum(vec) % 2
