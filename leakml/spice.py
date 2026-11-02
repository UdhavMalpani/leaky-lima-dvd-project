"""
Thin wrapper around ngspice for running many DC operating points fast.

Key idea: instead of launching ngspice once per sample (≈0.1 s of start-up
each), we write ONE netlist per worker whose .control block loops over its
share of samples.  For each sample it:
    altermod  -> change process parameters of the nmos/pmos model card
    alter     -> change transistor width and source voltages
    option temp=...  -> change temperature
    op        -> solve the DC operating point
    wrdata    -> append the requested currents/voltages to a text file
This is the same simulation as a fresh netlist per sample, just without the
start-up cost.  `verify_altermod()` in 01_generate_transistor_data.py proves it.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config


def fmt(x):
    return f"{x:.9e}"


def build_netlist(title, body, runs, outputs, outfile):
    """
    body    : netlist lines (devices + sources), without .include/.options
    runs    : list of lists of ngspice control commands, one list per sample
    outputs : list of ngspice vector expressions to record, e.g. "i(Vd)"
    """
    lines = [
        f"* {title}",
        f".include {config.MODEL_FILE.as_posix()}",
        config.SPICE_OPTIONS,
        *body,
        ".control",
        "set appendwrite",
        "set wr_vecnames=0",
    ]
    for k, cmds in enumerate(runs):
        lines += cmds
        lines += [
            "op",
            f"let idx = {k}",
            f"wrdata {outfile.as_posix()} idx {' '.join(outputs)}",
            "destroy all",
        ]
    lines += [".endc", ".end", ""]
    return "\n".join(lines)


def _run_chunk(args):
    title, body, runs, outputs, tag = args
    outfile = config.WORK_DIR / f"{tag}.out"
    cir = config.WORK_DIR / f"{tag}.cir"
    outfile.unlink(missing_ok=True)
    cir.write_text(build_netlist(title, body, runs, outputs, outfile))
    # one thread per ngspice process: parallelism comes from running many
    # processes, and nested OpenMP threads would only oversubscribe the CPU
    proc = subprocess.run([config.NGSPICE, "-b", str(cir)], capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "1"})
    (config.WORK_DIR / f"{tag}.log").write_text(proc.stdout + proc.stderr)

    result = np.full((len(runs), len(outputs)), np.nan)
    if outfile.exists():
        raw = np.atleast_2d(np.loadtxt(outfile))
        # wrdata writes (scale, value) pairs -> keep every second column
        vals = raw[:, 1::2]
        idx = vals[:, 0].astype(int)
        result[idx] = vals[:, 1:]
    return result


def run_batch(title, body, runs, outputs, tag, n_workers=config.N_WORKERS):
    """Split `runs` across workers, return array (len(runs), len(outputs)).
    Rows that failed to converge are NaN."""
    chunks = np.array_split(np.arange(len(runs)), n_workers)
    jobs = [(title, body, [runs[i] for i in c], outputs, f"{tag}_{j}")
            for j, c in enumerate(chunks) if len(c)]
    with ThreadPoolExecutor(n_workers) as ex:
        parts = list(ex.map(_run_chunk, jobs))
    return np.vstack(parts)


def altermod_cmds(dev, params):
    """ngspice commands that set the process parameters of a model.

    One parameter per command on purpose: ngspice 44.2 silently applies only
    the FIRST parameter of a multi-parameter `altermod` line (found by the
    altermod-vs-fresh-netlist self-test in 01_generate_transistor_data.py).
    """
    return [f"altermod {dev} {k}={fmt(v)}" for k, v in params.items()]
