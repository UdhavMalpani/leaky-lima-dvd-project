"""
Gate and circuit descriptions shared by the SPICE netlists and the ML side.

A circuit is a list of transistors (name, type, drain, gate, source, body, W).
The same list is used to (1) write the SPICE netlist and (2) build the
per-transistor terminal voltages fed to the ML models, so the two can never
disagree about topology.

Conventions
  * supply node "vdd", ground "0"; NMOS body -> 0, PMOS body -> vdd
  * NAND2: MN_A is the TOP nmos (drain on the output), MN_B the bottom one;
    x is the node between them.
  * NOR2:  MP_A is the TOP pmos (source on vdd), MP_B the bottom one;
    y is the node between them.
"""
from dataclasses import dataclass
from itertools import product

import config


@dataclass
class Fet:
    name: str
    dev: str        # "nmos" / "pmos"
    d: str
    g: str
    s: str
    b: str
    w: float


# --------------------------------------------------------------------------
# Primitive static-CMOS cells.  Nodes "a", "b" are inputs, "out" the output,
# anything else is internal and gets prefixed when instantiated.
# --------------------------------------------------------------------------
WN, WP = config.WN_GATE, config.WP_GATE
PRIMITIVES = {
    "INV": dict(inputs=["a"], logic=lambda a: 1 - a, fets=[
        Fet("MP", "pmos", "out", "a", "vdd", "vdd", WP),
        Fet("MN", "nmos", "out", "a", "0", "0", WN)]),
    "NAND2": dict(inputs=["a", "b"], logic=lambda a, b: 1 - (a & b), fets=[
        Fet("MP_A", "pmos", "out", "a", "vdd", "vdd", WP),
        Fet("MP_B", "pmos", "out", "b", "vdd", "vdd", WP),
        Fet("MN_A", "nmos", "out", "a", "x", "0", WN),
        Fet("MN_B", "nmos", "x", "b", "0", "0", WN)]),
    "NOR2": dict(inputs=["a", "b"], logic=lambda a, b: 1 - (a | b), fets=[
        Fet("MP_A", "pmos", "y", "a", "vdd", "vdd", WP),
        Fet("MP_B", "pmos", "out", "b", "y", "vdd", WP),
        Fet("MN_A", "nmos", "out", "a", "0", "0", WN),
        Fet("MN_B", "nmos", "out", "b", "0", "0", WN)]),
}

# The five library gates of the brief, built from primitives.
# Each entry: list of (primitive, {pin: node}, output node)
GATES = {
    "INV":   [("INV",   {"a": "a"},           "out")],
    "NAND2": [("NAND2", {"a": "a", "b": "b"}, "out")],
    "NOR2":  [("NOR2",  {"a": "a", "b": "b"}, "out")],
    "AND2":  [("NAND2", {"a": "a", "b": "b"}, "nd"), ("INV", {"a": "nd"}, "out")],
    "OR2":   [("NOR2",  {"a": "a", "b": "b"}, "nr"), ("INV", {"a": "nr"}, "out")],
}
GATE_INPUTS = {g: (["a"] if g == "INV" else ["a", "b"]) for g in GATES}


def flatten(cells, prefix=""):
    """cells: list of (primitive, pin map, output node) -> flat list of Fets.
    Internal nodes of each primitive instance get a unique prefix."""
    fets = []
    for k, (prim, pins, out) in enumerate(cells):
        tag = f"{prefix}{prim.lower()}{k}"
        def node(n):
            if n in pins:
                return pins[n]
            if n == "out":
                return out
            if n in ("vdd", "0"):
                return n
            return f"{tag}_{n}"
        for f in PRIMITIVES[prim]["fets"]:
            fets.append(Fet(f"{tag}_{f.name}", f.dev, node(f.d), node(f.g),
                            node(f.s), node(f.b), f.w))
    return fets


def evaluate_logic(cells, values):
    """Propagate 0/1 values through the cells (in order); returns node->bit."""
    v = dict(values)
    v["vdd"], v["0"] = 1, 0
    for prim, pins, out in cells:
        args = [v[pins[p]] for p in PRIMITIVES[prim]["inputs"]]
        v[out] = PRIMITIVES[prim]["logic"](*args)
    return v


def internal_nodes(fets, inputs):
    nodes = {n for f in fets for n in (f.d, f.g, f.s)}
    return sorted(nodes - set(inputs) - {"vdd", "0"})


def input_vectors(n):
    """All 2**n input combinations, MSB first (e.g. AB = 00, 01, 10, 11)."""
    return [tuple(bits) for bits in product((0, 1), repeat=n)]


# --------------------------------------------------------------------------
# 3-bit carry-lookahead adder, built only from library gates.
#   p_i = a_i XOR b_i          g_i = a_i AND b_i
#   c1 = g0 + p0 c0
#   c2 = g1 + p1 g0 + p1 p0 c0
#   c3 = g2 + p2 g1 + p2 p1 g0 + p2 p1 p0 c0       (= carry out)
#   s_i = p_i XOR c_i
# XOR is the classic 4-NAND2 structure.  Each carry is computed directly
# from g/p and c0 (that is what makes it look-ahead, not ripple).
# --------------------------------------------------------------------------
CLA_INPUTS = ["a2", "a1", "a0", "b2", "b1", "b0", "c0"]


def _xor(out, a, b, tag):
    n1, n2, n3 = f"{tag}_n1", f"{tag}_n2", f"{tag}_n3"
    return [("NAND2", (a, b), n1), ("NAND2", (a, n1), n2),
            ("NAND2", (b, n1), n3), ("NAND2", (n2, n3), out)]


def cla3_gates():
    """Gate-level netlist: list of (GATE, (inputs...), output)."""
    G = []
    for i in range(3):
        G += _xor(f"p{i}", f"a{i}", f"b{i}", f"xp{i}")
        G.append(("AND2", (f"a{i}", f"b{i}"), f"g{i}"))
    # c1
    G += [("AND2", ("p0", "c0"), "t_p0c0"),
          ("OR2", ("g0", "t_p0c0"), "c1")]
    # c2
    G += [("AND2", ("p1", "g0"), "t_p1g0"),
          ("AND2", ("p1", "p0"), "t_p1p0"),
          ("AND2", ("t_p1p0", "c0"), "t_p1p0c0"),
          ("OR2", ("t_p1g0", "t_p1p0c0"), "t_c2a"),
          ("OR2", ("g1", "t_c2a"), "c2")]
    # c3
    G += [("AND2", ("p2", "g1"), "t_p2g1"),
          ("AND2", ("p2", "p1"), "t_p2p1"),
          ("AND2", ("t_p2p1", "g0"), "t_p2p1g0"),
          ("AND2", ("t_p2p1", "p0"), "t_p2p1p0"),
          ("AND2", ("t_p2p1p0", "c0"), "t_p2p1p0c0"),
          ("OR2", ("g2", "t_p2g1"), "t_c3a"),
          ("OR2", ("t_p2p1g0", "t_p2p1p0c0"), "t_c3b"),
          ("OR2", ("t_c3a", "t_c3b"), "c3")]
    for i in range(3):
        G += _xor(f"s{i}", f"p{i}", f"c{i}", f"xs{i}")
    return G


# --------------------------------------------------------------------------
# 8-bit parity checker (the brief's other test circuit), used as a second,
# unseen validation circuit: a balanced tree of seven 4-NAND2 XORs.
#   par = a7 ^ a6 ^ ... ^ a0   (1 when an odd number of inputs is 1)
# --------------------------------------------------------------------------
PARITY_INPUTS = [f"a{i}" for i in range(7, -1, -1)]


def parity8_gates():
    G, level, k = [], [f"a{i}" for i in range(8)], 0
    while len(level) > 1:
        nxt = []
        for a, b in zip(level[::2], level[1::2]):
            out = "par" if len(level) == 2 else f"x{k}"
            G += _xor(out, a, b, f"xr{k}")
            nxt.append(out)
            k += 1
        level = nxt
    return G


def gate_cells(gate, ins, out, tag):
    """Expand one library gate instance into primitive cells."""
    pin = dict(zip(GATE_INPUTS[gate], ins))
    cells = []
    for prim, pins, o in GATES[gate]:
        cells.append((prim, {p: pin.get(n, f"{tag}_{n}") for p, n in pins.items()},
                      out if o == "out" else f"{tag}_{o}"))
    return cells


def circuit_cells(gates):
    cells = []
    for k, (gate, ins, out) in enumerate(gates):
        cells += gate_cells(gate, ins, out, f"G{k}")
    return cells
