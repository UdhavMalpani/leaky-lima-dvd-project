"""
SPICE-free operating point: solve a circuit's internal node voltages with
the ML transistor models.

At DC every internal node n obeys KCL: the currents flowing INTO the devices
from n sum to zero,
        F_n(V) = sum over FET terminals t attached to n of  I_t(V) = 0,
where I_t is the ML-predicted terminal current (positive into the device).
We solve F(V) = 0 for all unknown nodes at once with a damped Newton method
and a finite-difference Jacobian, vectorised over rows (corners x vectors).
Like SPICE, each Newton step is limited (here to 50 mV) because the device
currents are exponential in voltage.

Which model (ON or OFF) a FET uses is fixed by the logic value at its gate,
exactly as in the hierarchical leakage estimate.
"""
import numpy as np
import pandas as pd

import config
from leakml import bipoles, circuits
from leakml.models import fast_copy

STEP_LIMIT = 0.05      # V, max Newton update per iteration
DV_JAC = 1e-4          # V, finite-difference step
TOL_V = 1e-7           # V, convergence on the update
MAX_IT = 60


class Problem:
    """fets + fixed node voltages + which nodes are unknown, for N rows."""

    def __init__(self, fets, fixed, unknown, corners_rows, logic_rows, models):
        """
        fets         : list of circuits.Fet
        fixed        : dict node -> (N,) voltage array (vdd, 0, inputs, ...)
        unknown      : list of node names to solve for
        corners_rows : DataFrame, one corner (vdd, temp, nmos_*, pmos_*) per row
        logic_rows   : dict node -> (N,) logic bits for every node that drives a gate
        models       : {"nmos_on": TransistorModel, ...}
        """
        self.fets, self.fixed, self.unknown = fets, fixed, list(unknown)
        self.models = {k: fast_copy(m) for k, m in models.items()}   # same numbers, faster
        self.n = len(corners_rows)
        self.vdd = corners_rows["vdd"].to_numpy()
        self.temp = corners_rows["temp"].to_numpy()
        self.proc = {dev: {p: corners_rows[f"{dev}_{p}"].to_numpy() for p in config.PROCESS_PARAMS}
                     for dev in ("nmos", "pmos")}
        self.on = [logic_rows[f.g] == (1 if f.dev == "nmos" else 0) for f in fets]
        # FETs touching each unknown node (only these need re-evaluating in the Jacobian)
        self.touch = {u: [k for k, f in enumerate(fets) if u in (f.d, f.g, f.s, f.b)]
                      for u in self.unknown}

    def take(self, idx):
        """The same problem restricted to rows `idx` (used to iterate only
        on rows that have not converged yet)."""
        q = object.__new__(Problem)
        q.fets, q.unknown, q.models, q.touch = self.fets, self.unknown, self.models, self.touch
        q.n = len(idx)
        q.vdd, q.temp = self.vdd[idx], self.temp[idx]
        q.proc = {d: {p: a[idx] for p, a in pp.items()} for d, pp in self.proc.items()}
        q.on = [o[idx] for o in self.on]
        q.fixed = {k: v[idx] for k, v in self.fixed.items()}
        return q

    def _v(self, V, node):
        return V[node] if node in V else self.fixed[node]

    def fet_terminals(self, k, V, rows=slice(None)):
        """ML terminal currents (id, ig, is, ib) of FET k for the given rows."""
        f = self.fets[k]
        df = pd.DataFrame({"vdd": self.vdd[rows], "temp": self.temp[rows], "w": f.w,
                           "vd": self._v(V, f.d)[rows], "vg": self._v(V, f.g)[rows],
                           "vs": self._v(V, f.s)[rows], "vb": self._v(V, f.b)[rows]})
        for p in config.PROCESS_PARAMS:
            df[p] = self.proc[f.dev][p][rows]
        on = self.on[k][rows]
        bip = pd.DataFrame(0.0, index=df.index, columns=bipoles.BIPOLES)
        for state, mask in (("on", on), ("off", ~on)):
            if mask.any():
                bip.loc[mask] = self.models[f"{f.dev}_{state}"].predict_bipoles(df[mask]).to_numpy()
        return bip, df

    def residual(self, V, fet_cache=None, only=None):
        """KCL residual per unknown node.  `fet_cache` holds per-FET currents so
        a Jacobian column only re-evaluates the FETs that touch the moved node."""
        cache = {} if fet_cache is None else dict(fet_cache)
        for k in (range(len(self.fets)) if only is None else only):
            bip, df = self.fet_terminals(k, V)
            cache[k] = bipoles.terminals(bip).to_numpy()        # columns id ig is ib
        F = {u: np.zeros(self.n) for u in self.unknown}
        for k, f in enumerate(self.fets):
            for j, node in enumerate((f.d, f.g, f.s, f.b)):
                if node in F:
                    F[node] += cache[k][:, j]
        return np.column_stack([F[u] for u in self.unknown]), cache


def solve(problem, guess):
    """Damped Newton, iterating only on rows that have not converged.
    guess: dict node -> (N,) initial voltages.  Returns (dict node -> voltages, info)."""
    P = problem
    V = {u: np.array(guess[u], dtype=float) for u in P.unknown}
    m = len(P.unknown)
    active = np.arange(P.n)
    iters = np.zeros(P.n, int)
    for it in range(MAX_IT):
        Q = P.take(active)
        Va = {u: V[u][active] for u in P.unknown}
        F, cache = Q.residual(Va)
        J = np.empty((Q.n, m, m))
        for c, u in enumerate(P.unknown):
            Vp = dict(Va)
            Vp[u] = Va[u] + DV_JAC
            Fp, _ = Q.residual(Vp, cache, only=Q.touch[u])
            J[:, :, c] = (Fp - F) / DV_JAC
        # tiny diagonal conductance keeps J invertible on rows where a node is floating-ish
        J += np.eye(m)[None] * 1e-18
        dV = -np.linalg.solve(J, F[..., None])[..., 0]
        dV = np.clip(dV, -STEP_LIMIT, STEP_LIMIT)
        for c, u in enumerate(P.unknown):
            V[u][active] = np.clip(Va[u] + dV[:, c], -0.1 * Q.vdd, 1.1 * Q.vdd)
        iters[active] += 1
        active = active[np.max(np.abs(dV), axis=1) >= TOL_V]
        if not len(active):
            break
    F, _ = P.residual(V)
    return V, dict(iterations=int(iters.max()), mean_iterations=float(iters.mean()),
                   unconverged=int(len(active)), max_residual=float(np.max(np.abs(F))))


def stack_guess(node, fets, logic_rows, vdd):
    """Initial guess for a stack node from which of its two series devices is ON."""
    nmos = node.endswith("_x")
    on_rail = [f for f in fets if f.s == node or f.d == node]
    # the device whose other end is a rail, and the one whose other end is the output
    to_rail = next(f for f in on_rail if ({f.d, f.s} - {node}) & {"0", "vdd"})
    to_out = next(f for f in on_rail if f is not to_rail)
    other = (({to_out.d, to_out.s} - {node}).pop())
    rail_on = logic_rows[to_rail.g] == (1 if nmos else 0)
    out_on = logic_rows[to_out.g] == (1 if nmos else 0)
    rail_v = np.zeros_like(vdd) if nmos else vdd
    return np.where(rail_on, rail_v, np.where(out_on, logic_rows[other] * vdd,
                                              (0.2 if nmos else 0.8) * vdd))


def logic_guess(cells, inputs, vec, nodes, vdd):
    """Initial guess: logic nodes at 0 / VDD, stack nodes from stack_guess."""
    logic = circuits.evaluate_logic(cells, dict(zip(inputs, vec)))
    fets = circuits.flatten(cells)
    lr = {k: np.full(len(vdd), v) for k, v in logic.items()}
    g = {n: (logic[n] * vdd if n in logic else stack_guess(n, fets, lr, vdd)) for n in nodes}
    return g, logic


# --------------------------------------------------------------------------
PRIM_FETS = {p: len(d["fets"]) for p, d in circuits.PRIMITIVES.items()}


def device_power(problem, V):
    """Static power absorbed by every FET of `problem` (sum of V*I over its bipoles)."""
    p = np.zeros(problem.n)
    for k in range(len(problem.fets)):
        bip, df = problem.fet_terminals(k, V)
        p += sum(bip[b].to_numpy() * bipoles.voltage(df, b) for b in bipoles.BIPOLES)
    return p


def levels_aware_hierarchy(gates, inputs, rows, corners, models):
    """Hierarchical ML leakage with REAL logic levels between gates.

    The brief's hierarchy feeds every gate ideal 0 / VDD inputs.  Here the
    gates are visited in topological order; each one is solved with the ML
    KCL solver using, as its inputs, the ML-predicted output voltages of the
    gates that drive it.  A slightly-off '0' turns the next gate's OFF device
    exponentially more on, which is exactly the effect the ideal-level
    hierarchy misses.  Loading (gate-tunnelling current drawn by fan-out pins)
    is still neglected: each gate is solved in isolation.

    rows : DataFrame with columns vector (str of bits, in `inputs` order) and corner
    Returns (total leakage per row, dict node -> ML voltage per row,
             per-gate leakage, per-gate solver statistics).
    """
    cr = corners.loc[rows.corner].reset_index(drop=True)
    vdd, n = cr.vdd.to_numpy(), len(rows)
    cells = circuits.circuit_cells(gates)
    fets = circuits.flatten(cells)
    vecs = rows.vector.to_numpy()
    uniq = {v: circuits.evaluate_logic(cells, dict(zip(inputs, map(int, v)))) for v in np.unique(vecs)}
    all_nodes = set(uniq[vecs[0]])
    logic_rows = {nd: np.array([uniq[v][nd] for v in vecs]) for nd in all_nodes}
    V = {"vdd": vdd, "0": np.zeros(n), **{i: logic_rows[i] * vdd for i in inputs}}
    total = np.zeros(n)
    per_gate, stats = {}, {}
    c0 = 0
    for k, (gate, ins, out) in enumerate(gates):
        gcells = circuits.gate_cells(gate, ins, out, f"G{k}")
        nf = sum(PRIM_FETS[c[0]] for c in gcells)
        gfets = fets[c0:c0 + nf]
        c0 += nf
        unknown = sorted({nd for f in gfets for nd in (f.d, f.s)} - {"vdd", "0"})
        fixed = {nd: V[nd] for f in gfets for nd in (f.d, f.g, f.s, f.b) if nd not in unknown}
        # Rows with the same corner and the same input voltages have the same
        # solution: solve each distinct one once and scatter the result back.
        key = np.column_stack([rows.corner.to_numpy()] + [V[i] for i in ins])
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        lr = {nd: a[first] for nd, a in logic_rows.items()}
        vu = vdd[first]
        guess = {nd: (lr[nd] * vu if nd in lr else stack_guess(nd, gfets, lr, vu)) for nd in unknown}
        P = Problem(gfets, {nd: a[first] for nd, a in fixed.items()}, unknown,
                    cr.iloc[first].reset_index(drop=True), lr, models)
        sol, info = solve(P, guess)
        stats[k] = dict(gate=gate, out=out, unique_rows=len(first), **info)
        V.update({nd: a[inv] for nd, a in sol.items()})
        per_gate[k] = (device_power(P, sol) / vu)[inv]
        total += per_gate[k]
    return total, V, per_gate, pd.DataFrame(stats.values())
