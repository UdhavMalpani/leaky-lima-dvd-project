"""
Part A, step 1: generate the four single-transistor datasets.

For each case (nmos_on, nmos_off, pmos_on, pmos_off) we draw N_SAMPLES random
(VDD, T, W, Vd, Vs, process) points, simulate one transistor with a separate
voltage source on every terminal, and record the four terminal currents.

Sign convention (used everywhere in the project): a terminal current is
POSITIVE when it flows INTO the transistor through that terminal.
ngspice reports I(Vx) as the current flowing + -> - through the source, i.e.
OUT of the transistor, so we negate it.  With this convention Kirchhoff's
current law says id + ig + is + ib = 0 for every sample.

Output: data/<case>.csv with columns
    vdd temp w vd vs vg vb vth0 toxe ndep u0 xj id ig is ib
and data/<case>_bipoles.csv with the six internal bipole currents
(ds gs gd gb db sb), rebuilt from BSIM4's internal current components.
These let the ML learn each physical leakage mechanism separately
(see leakml/bipoles.py).
"""
import re
import sys
import time

import numpy as np
import pandas as pd

import config
from leakml import bipoles, spice

COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
           *config.PROCESS_PARAMS, "id", "ig", "is", "ib"]

BODY = [
    "M1 d g s b {dev} W={w} L={l}",
    "Vd d 0 0",
    "Vg g 0 0",
    "Vs s 0 0",
    "Vb b 0 0",
]
OUTPUTS = ["i(Vd)", "i(Vg)", "i(Vs)", "i(Vb)"] + [f"@m1[{c}]" for c in bipoles.COMPONENTS]


def draw_samples(case, n, rng):
    """Draw every input parameter independently (VDD first, as Vd/Vs depend on it)."""
    spec = config.CASES[case]
    nominal = config.NOMINAL[spec["dev"]]
    vdd = rng.uniform(*config.VDD_RANGE, n)
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": rng.uniform(*config.TEMP_RANGE, n),
        "w": rng.uniform(config.W_MIN, config.W_MAX, n),
        "vd": rng.uniform(0, 1, n) * vdd,
        "vs": rng.uniform(0, 1, n) * vdd,
        "vg": vdd if spec["vg"] == "vdd" else np.zeros(n),
        "vb": vdd if spec["vb"] == "vdd" else np.zeros(n),
    })
    s = config.PROCESS_SPREAD
    for p in config.PROCESS_PARAMS:
        df[p] = nominal[p] * rng.uniform(1 - s, 1 + s, n)
    return df


def draw_rail_samples(case, n, rng):
    """Supplementary 'rail-focused' samples (NOT part of the 10k deliverable).

    In a real gate the leaking transistor almost always has a terminal sitting
    exactly on a rail (source at 0 V / VDD, drain at the output ~ VDD or ~0).
    Uniform sampling almost never lands there (3 in 10,000 samples are within
    2 % of both rails), so the model would be extrapolating exactly where it
    is used.  Here each of Vd, Vs is exactly 0 or VDD one third of the time and
    otherwise follows an arcsine (Beta(0.5, 0.5)) law, dense near both rails.

    The gate voltage is also moved slightly off its ideal level: inside a
    circuit a gate is driven by another gate's output, which sits tens of mV
    away from the rail (e.g. a logic 0 at 16-60 mV).  Half the samples keep
    the ideal level, the rest are up to 10 % of VDD inside the rail."""
    df = draw_samples(case, n, rng)
    for t in ("vd", "vs"):
        u = rng.uniform(size=n)
        frac = rng.beta(0.5, 0.5, n)
        frac = np.where(u < 1 / 6, 0.0, np.where(u < 1 / 3, 1.0, frac))
        df[t] = frac * df["vdd"]
    off = np.where(rng.uniform(size=n) < 0.5, 0.0, rng.uniform(0, 0.1, n)) * df["vdd"]
    df["vg"] = off if config.CASES[case]["vg"] == "0" else df["vdd"] - off
    return df


def control_cmds(dev, row):
    return [
        *spice.altermod_cmds(dev, {p: row[p] for p in config.PROCESS_PARAMS}),
        f"alter m1 w={spice.fmt(row['w'])}",
        f"alter vd dc={spice.fmt(row['vd'])}",
        f"alter vg dc={spice.fmt(row['vg'])}",
        f"alter vs dc={spice.fmt(row['vs'])}",
        f"alter vb dc={spice.fmt(row['vb'])}",
        f"option temp={row['temp']:.6f}",
    ]


def body_for(dev, w=config.W_MIN):
    return [ln.format(dev=dev, w=spice.fmt(w), l=spice.fmt(config.LMIN)) for ln in BODY]


def simulate(case, df):
    dev = config.CASES[case]["dev"]
    runs = [control_cmds(dev, r) for r in df.to_dict("records")]
    res = spice.run_batch(f"single {case}", body_for(dev), runs, OUTPUTS, tag=case)
    out = df.copy()
    out[["id", "ig", "is", "ib"]] = -res[:, :4]      # into-the-device convention
    comp = pd.DataFrame(res[:, 4:], columns=bipoles.COMPONENTS, index=df.index)
    bip = bipoles.from_components(dev, out, comp)
    return out, bip


def write_example_netlist(case, row):
    """Deliverable 2: a standalone, human-readable netlist for one sample."""
    dev = config.CASES[case]["dev"]
    model = altered_model_text(dev, {p: row[p] for p in config.PROCESS_PARAMS})
    path = config.NETLIST_DIR / f"single_{case}.cir"
    path.write_text("\n".join([
        f"* Single-transistor netlist: {case.upper()}  ({config.TECH}, one random sample)",
        f"* VDD = {row['vdd']:.4f} V, T = {row['temp']:.2f} C, W = {row['w']*1e9:.2f} nm",
        "* Every terminal has its own voltage source; currents are read through them.",
        config.SPICE_OPTIONS,
        f".temp {row['temp']:.6f}",
        model,
        f"M1 d g s b {dev} W={spice.fmt(row['w'])} L={spice.fmt(config.LMIN)}",
        f"Vd d 0 {spice.fmt(row['vd'])}",
        f"Vg g 0 {spice.fmt(row['vg'])}",
        f"Vs s 0 {spice.fmt(row['vs'])}",
        f"Vb b 0 {spice.fmt(row['vb'])}",
        ".control", "op", "print i(Vd) i(Vg) i(Vs) i(Vb)", ".endc", ".end", "",
    ]))
    return path


def altered_model_text(dev, params):
    """Return the .model card for `dev` with the five process params replaced."""
    text = config.MODEL_FILE.read_text()
    blocks = re.split(r"(?im)^(?=\.model\s+)", text)
    blk = next(b for b in blocks if re.match(rf"(?i)\.model\s+{dev}\b", b))
    for p, v in params.items():
        blk = re.sub(rf"(?i)\b({p}\s*=\s*)[-+0-9.eE]+", rf"\g<1>{spice.fmt(v)}", blk, count=1)
    return blk


def verify_altermod(case, df, n=5):
    """Self-test: the fast altermod loop must equal fresh standalone netlists."""
    import subprocess
    worst = 0.0
    for k in range(n):
        row = df.iloc[k]
        path = write_example_netlist(case, row)
        txt = subprocess.run([config.NGSPICE, "-b", str(path)],
                             capture_output=True, text=True).stdout
        vals = [float(re.search(rf"i\({v}\)\s*=\s*([-+0-9.eE]+)", txt, re.I).group(1))
                for v in ("vd", "vg", "vs", "vb")]
        fresh = -np.array(vals)
        fast = row[["id", "ig", "is", "ib"]].to_numpy(float)
        # Normalise by the sample's largest current: some components (e.g. the
        # PMOS-OFF gate current, ~1e-16 A) sit at double-precision round-off
        # where a relative difference is meaningless.
        worst = max(worst, np.max(np.abs(fresh - fast)) / np.max(np.abs(fresh)))
    return worst


def main():
    rng = np.random.default_rng(config.SEED_DATA)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else config.N_SAMPLES
    for case in config.CASES:
        t0 = time.time()
        df, bip = simulate(case, draw_samples(case, n, rng))
        ok = df[["id", "ig", "is", "ib"]].notna().all(axis=1)
        n_fail = (~ok).sum()
        df = df[ok].reset_index(drop=True)
        bip = bip[ok].reset_index(drop=True)
        df[COLUMNS].to_csv(config.DATA_DIR / f"{case}.csv", index=False)
        bip.to_csv(config.DATA_DIR / f"{case}_bipoles.csv", index=False)

        # Check 1: fast altermod loop == fresh standalone netlists
        err = verify_altermod(case, df)
        # Check 2: the six bipoles rebuild the four terminal currents (KCL)
        rebuilt = bipoles.terminals(bip).to_numpy()
        term = df[["id", "ig", "is", "ib"]].to_numpy()
        kcl = np.max(np.abs(rebuilt - term).max(axis=1) / np.abs(term).max(axis=1))
        write_example_netlist(case, df.iloc[0])
        print(f"{case:9s}: {len(df)} samples in {time.time()-t0:.1f}s, {n_fail} failed, "
              f"altermod-vs-fresh {err:.1e}, bipole rebuild {kcl:.1e}")

    # Supplementary rail-focused data (separate files, separate seed stream)
    rng = np.random.default_rng(config.SEED_DATA + 1)
    for case in config.CASES:
        df, bip = simulate(case, draw_rail_samples(case, config.N_RAIL_SAMPLES, rng))
        ok = df[["id", "ig", "is", "ib"]].notna().all(axis=1)
        df[ok][COLUMNS].to_csv(config.DATA_DIR / f"{case}_rail.csv", index=False)
        bip[ok].to_csv(config.DATA_DIR / f"{case}_rail_bipoles.csv", index=False)
        print(f"{case:9s}: {ok.sum()} rail-focused supplementary samples")


if __name__ == "__main__":
    main()
