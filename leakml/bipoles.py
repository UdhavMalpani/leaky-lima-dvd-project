"""
The "bipole" view of a MOSFET (slide 1 of the course material).

A transistor in DC is described as six two-terminal current paths between
its four terminals:

    ds : channel (sub-threshold / strong inversion)      drain  -> source
    gs : gate-oxide tunnelling, gate <-> source side     gate   -> source
    gd : gate-oxide tunnelling, gate <-> drain side      gate   -> drain
    gb : gate-oxide tunnelling, gate <-> body            gate   -> body
    db : drain junction + GIDL + impact ionisation       drain  -> body
    sb : source junction + GISL (+ impact ionisation)     source -> body

Each bipole current is defined positive in the arrow direction above.
Every bipole is PASSIVE: current flows from the higher to the lower voltage,
so I_xy = G_xy * V_xy with G_xy > 0.  We verified this on all samples.

Terminal currents (positive INTO the device) follow from KCL:
    Id =  I_ds - I_gd + I_db
    Ig =  I_gs + I_gd + I_gb
    Is = -I_ds - I_gs + I_sb
    Ib = -I_gb - I_db - I_sb
so Id + Ig + Is + Ib = 0 holds by construction.
"""
import numpy as np
import pandas as pd

BIPOLES = ["ds", "gs", "gd", "gb", "db", "sb"]

# BSIM4 instance quantities recorded from ngspice (@m1[...])
COMPONENTS = ["id", "igs", "igd", "igcs", "igcd", "igb",
              "igidl", "igisl", "isub", "ibd", "ibs"]

# terminal voltage pair across each bipole: I_xy flows x -> y
BIPOLE_NODES = {"ds": ("vd", "vs"), "gs": ("vg", "vs"), "gd": ("vg", "vd"),
                "gb": ("vg", "vb"), "db": ("vd", "vb"), "sb": ("vs", "vb")}


def forward_mode(dev, vd, vs):
    """BSIM4 relabels drain/source internally so its 'drain' is the terminal
    that receives channel current from the supply side.  For NMOS that is
    the higher terminal, for PMOS the lower one."""
    s = 1 if dev == "nmos" else -1
    return (np.asarray(vd) - np.asarray(vs)) * s >= 0


def from_components(dev, df, comp):
    """BSIM4 components (ngspice sign convention) -> six physical bipoles.

    BSIM4 swaps igcs/igcd and moves isub to the other side when the device
    runs in reverse mode; PMOS components come out in n-type sign, so they
    are negated first (verified by the KCL rebuild check in
    01_generate_transistor_data.py)."""
    k = comp if dev == "nmos" else -comp
    fwd = forward_mode(dev, df["vd"], df["vs"])
    return pd.DataFrame({
        "ds": np.where(fwd, k.id, -k.id),
        "gs": k.igs + np.where(fwd, k.igcs, k.igcd),
        "gd": k.igd + np.where(fwd, k.igcd, k.igcs),
        "gb": k.igb,
        "db": -k.ibd + k.igidl + np.where(fwd, k.isub, 0.0),
        "sb": -k.ibs + k.igisl + np.where(fwd, 0.0, k.isub),
    }, index=df.index)


def terminals(b):
    """Six bipole currents -> four terminal currents (into the device)."""
    return pd.DataFrame({
        "id": b["ds"] - b["gd"] + b["db"],
        "ig": b["gs"] + b["gd"] + b["gb"],
        "is": -b["ds"] - b["gs"] + b["sb"],
        "ib": -b["gb"] - b["db"] - b["sb"],
    }, index=b.index)


def voltage(df, name):
    a, c = BIPOLE_NODES[name]
    return np.asarray(df[a]) - np.asarray(df[c])
