"""
Single source of truth for the whole project.

Every script imports from here, so changing the technology node, VDD or
seed in this one file regenerates everything consistently.
"""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Team assignment (Team 1 "Leaky Lima" in the TA spreadsheet)
# ---------------------------------------------------------------------------
TECH = "16nm_HP"
VDD_NOM = 1.0             # volts; note PTM's own nominal for this card is 0.7 V
TEST_CIRCUIT = "CLA3"     # 3-bit carry-lookahead adder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / f"{TECH}.pm"
DATA_DIR = ROOT / "data"
NETLIST_DIR = ROOT / "netlists"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
SAVED_MODELS_DIR = ROOT / "saved_models"
WORK_DIR = ROOT / "work"          # scratch netlists / raw ngspice output
for _d in (DATA_DIR, NETLIST_DIR, FIG_DIR, RESULTS_DIR, SAVED_MODELS_DIR, WORK_DIR):
    _d.mkdir(exist_ok=True)

NGSPICE = r"C:\NG Spice\Spice64\bin\ngspice_con.exe"
N_WORKERS = 12            # parallel ngspice processes

# ---------------------------------------------------------------------------
# Device geometry
# ---------------------------------------------------------------------------
LMIN = 16e-9                       # channel length = feature size
W_MIN, W_MAX = 1 * LMIN, 8 * LMIN  # width range for transistor training data

# Sizing used inside the logic gates (must lie inside the training range).
# We pick interior values (not the 1x edge) so the ML models never have to
# extrapolate, and PMOS is 2x NMOS as is conventional for static CMOS.
WN_GATE = 2 * LMIN
WP_GATE = 4 * LMIN

# ---------------------------------------------------------------------------
# Sampling ranges (Section 4 of the brief)
# ---------------------------------------------------------------------------
VDD_RANGE = (0.9 * VDD_NOM, 1.1 * VDD_NOM)
TEMP_RANGE = (-40.0, 125.0)
PROCESS_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
PROCESS_SPREAD = 0.10              # +/- 10 % around nominal

N_SAMPLES = 10_000                 # per transistor case (uniform, the deliverable)
N_RAIL_SAMPLES = 5_000             # per case, rail-focused supplement (see 01_...)
N_CORNERS = 200                    # gate / circuit corners

SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

# Gate and body voltages fixed by each case (as functions of VDD)
CASES = {
    "nmos_on":  dict(dev="nmos", vg="vdd", vb="0"),
    "nmos_off": dict(dev="nmos", vg="0",   vb="0"),
    "pmos_on":  dict(dev="pmos", vg="0",   vb="vdd"),
    "pmos_off": dict(dev="pmos", vg="vdd", vb="vdd"),
}

# ---------------------------------------------------------------------------
# Reproducibility: every random draw in the project comes from these seeds
# ---------------------------------------------------------------------------
SEED_DATA = 2026
SEED_SPLIT = 42
SEED_CORNERS = 7


def read_nominal_params(model_file=MODEL_FILE):
    """Parse nominal vth0/toxe/ndep/u0/xj for nmos and pmos from the PTM card."""
    text = model_file.read_text()
    blocks = re.split(r"(?im)^\.model\s+", text)[1:]
    out = {}
    for blk in blocks:
        name = blk.split()[0].lower()
        vals = {}
        for p in PROCESS_PARAMS:
            m = re.search(rf"(?i)\b{p}\s*=\s*([-+0-9.eE]+)", blk)
            vals[p] = float(m.group(1))
        out[name] = vals
    return out


NOMINAL = read_nominal_params()

# Effective width W_eff = W - 2*WINT: BSIM4 removes `wint` from each edge of
# the drawn width.  At W = 16 nm only 6 nm conducts, so currents are far from
# proportional to the drawn W for narrow devices.
WINT = 5e-9
