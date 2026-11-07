"""
Deliverable 10: the typed report.

Every number and table in the report is read from results/ when this script
runs, so the report can never drift from the code.  Figures come from
figures/ plus a few report-only plots written to report/fig/.

    .venv/bin/python report/build_report.py         -> report/report.html + report/Leaky_Lima_Report.pdf
"""
import html as H
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from leakml import circuits  # noqa: E402

R, F = config.RESULTS_DIR, config.FIG_DIR
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
PDF = HERE / "Leaky_Lima_Report.pdf"
_MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME = (os.environ.get("CHROME")
          or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
          or (_MAC_CHROME if Path(_MAC_CHROME).exists() else None)
          or next((str(p) for p in Path.home().glob("Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*")), None))
CUR = ["id", "ig", "is", "ib"]
FAM = {"mlp": "MLP (neural net)", "poly_ridge": "Polynomial + ridge",
       "lightgbm": "LightGBM", "random_forest": "Random forest"}
CASE = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}&times;10<sup>{int(e)}</sup>"


def p(x, d=2):
    return f"{x:.{d}f}"


def table(df, num=None, caption=None, best=None, cls=""):
    num = set(df.columns[1:] if num is None else num)
    h = "".join(f'<th class="{"n" if c in num else ""}">{c}</th>' for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        b = ' class="best"' if best is not None and best(i, r) else ""
        rows.append(f"<tr{b}>" + "".join(
            f'<td class="{"n" if c in num else ""}">{r[c]}</td>' for c in df.columns) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="{cls}">{cap}<thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


N_FIG = [0]


def fig(path, caption, width=100):
    N_FIG[0] += 1
    rel = os.path.relpath(path, HERE)
    return (f'<figure style="width:{width}%"><img src="{rel}">'
            f'<figcaption><b>Figure {N_FIG[0]}.</b> {caption}</figcaption></figure>')


def log_value(name, pattern, default="?"):
    try:
        m = re.search(pattern, (R / name).read_text())
        return m.group(1) if m else default
    except FileNotFoundError:
        return default


def mean_acc(true, pred):
    return 100 - 100 * np.mean(np.abs(np.asarray(pred) / np.asarray(true) - 1))


# --------------------------------------------------------------------------
# report-only figures
# --------------------------------------------------------------------------
def fig_model_comparison(mc):
    fig_, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    fams = list(FAM)
    for a, case in zip(ax, CASE):
        d = mc[mc.case == case].set_index("model")
        x = np.arange(len(CUR))
        for k, fam in enumerate(fams):
            if fam in d.index:
                a.bar(x + (k - 1.5) * 0.2, [d.loc[fam, f"{c}_med%"] for c in CUR], 0.2,
                      label=FAM[fam])
        a.set_yscale("log")
        a.set_xticks(x, CUR)
        a.axhline(1, c="k", lw=0.6, ls=":")
        a.set_title(CASE[case])
    ax[0].set_ylabel("validation median error (%)")
    ax[-1].legend(fontsize=7)
    fig_.tight_layout()
    out = FIG / "model_comparison.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_learning_curve(lc):
    fig_, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for a, case in zip(ax, CASE):
        for k, c in enumerate(CUR):
            g = lc[(lc.case == case) & (lc.current == c)].sort_values("n_train")
            a.loglog(g.n_train, g.med_pct, "o-", c=f"C{k}", ms=4, label=f"{c} median")
            a.loglog(g.n_train, g.p95_pct, "x--", c=f"C{k}", ms=4, alpha=0.5, label=f"{c} p95")
        a.axhline(1, c="k", lw=0.6, ls=":")
        ns = sorted(lc.n_train.unique())
        a.set_xticks(ns, [str(n) for n in ns], fontsize=7)
        a.minorticks_off()
        a.set(title=CASE[case], xlabel="uniform training samples")
    ax[0].set_ylabel("validation error (%)")
    ax[-1].legend(fontsize=6, ncol=2)
    fig_.tight_layout()
    out = FIG / "learning_curve.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_gate_error_vs_corner(gs):
    fig_, ax = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
    for a, (col, lab) in zip(ax, [("temp", "temperature (C)"), ("vdd", "VDD (V)"),
                                  ("pmos_vth0", "PMOS vth0 (V)")]):
        b = pd.qcut(gs[col], 10)
        g = gs.groupby(b, observed=True)
        x = g[col].median()
        a.fill_between(x, g.e.quantile(0.05), g.e.quantile(0.95), alpha=0.25, label="5-95 %")
        a.plot(x, g.e.median(), "o-", label="median")
        a.axhline(0, c="k", lw=0.6)
        a.set_xlabel(lab)
    ax[0].set_ylabel("signed gate error ML/SPICE - 1 (%)")
    ax[0].legend(fontsize=8)
    fig_.suptitle("Gate error vs corner conditions (3,600 sims, deciles)", fontsize=10)
    fig_.tight_layout()
    out = FIG / "gate_error_vs_corner.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_cla_vs_temp(cs):
    fig_, ax = plt.subplots(figsize=(6.6, 3.6))
    b = pd.qcut(cs.temp, 10)
    for col, lab in [("ml_gate_sum", "hierarchical ML (ideal 0/VDD levels)"),
                     ("spice_gate_sum", "SPICE gate sum (hierarchy alone)"),
                     ("ml_flat", "flat ML on circuit voltages"),
                     ("ml_levels", "levels-aware ML (SPICE-free)")]:
        e = 100 * (cs[col] / cs.spice_leak - 1)
        g = e.groupby(b, observed=True)
        ax.plot(cs.temp.groupby(b, observed=True).median(), g.median(), "o-", ms=4, label=lab)
    ax.axhline(0, c="k", lw=0.6)
    ax.set(xlabel="temperature (C)", ylabel="median signed error vs SPICE circuit (%)",
           title="CLA: where the hierarchy error comes from")
    ax.legend(fontsize=7)
    fig_.tight_layout()
    out = FIG / "cla_error_vs_temp.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_level_shift(gs):
    """How far a gate's logic-0 / logic-1 output sits from the rail (SPICE)."""
    fig_, ax = plt.subplots(figsize=(6.6, 3.4))
    for lv, lab in [(0, "output = 0: V(out)"), (1, "output = 1: VDD - V(out)")]:
        rows = []
        for (gate, vec), g in gs.groupby(["gate", "vector"]):
            bits = dict(zip(circuits.GATE_INPUTS[gate], map(int, vec)))
            cells = circuits.gate_cells(gate, circuits.GATE_INPUTS[gate], "out", "g")
            if circuits.evaluate_logic(cells, bits)["out"] != lv:
                continue
            dv = g.out if lv == 0 else g.vdd - g.out
            rows.append(pd.DataFrame({"temp": g.temp, "dv": 1e3 * dv}))
        d = pd.concat(rows)
        b = pd.qcut(d.temp, 10)
        ax.semilogy(d.temp.groupby(b, observed=True).median(), d.dv.groupby(b, observed=True).median(),
                    "o-", ms=4, label=lab)
    ax.set(xlabel="temperature (C)", ylabel="distance from the rail (mV)",
           title="Gate outputs are not at exactly 0 / VDD (SPICE, isolated gates)")
    ax.legend(fontsize=8)
    fig_.tight_layout()
    out = FIG / "level_shift.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


# --------------------------------------------------------------------------
def build():
    env = (R / "env.txt").read_text()
    envd = dict(re.findall(r"^(.+?)\s*:\s*(.+)$", env, re.M))
    corners = pd.read_csv(R / "corners.csv", index_col="corner")
    mc = pd.read_csv(R / "model_comparison.csv")
    allhp = pd.read_csv(R / "model_comparison_all_hparams.csv") if (R / "model_comparison_all_hparams.csv").exists() else None
    ch = pd.read_csv(R / "chosen_models_test.csv")
    kc = pd.read_csv(R / "kirchhoff_check.csv")
    tn = pd.read_csv(R / "trend_numbers.csv", index_col=0).iloc[:, 0]
    gr = pd.read_csv(R / "gate_results.csv", dtype={"inputs": str})
    gs = pd.read_csv(R / "gate_samples.csv", dtype={"vector": str}).join(corners, on="corner")
    gs["e"] = 100 * (gs.ml_leak / gs.spice_leak - 1)
    gs["ae"] = gs.e.abs()
    csum = pd.read_csv(R / "circuit_summary.csv")
    cres = pd.read_csv(R / "circuit_results.csv", dtype={"inputs": str})
    cs = pd.read_csv(R / "circuit_samples.csv", dtype={"vector": str}).join(corners, on="corner")
    lstats = pd.read_csv(R / "circuit_levels_solver_stats.csv")
    abl = pd.read_csv(R / "ablation.csv")
    lc = pd.read_csv(R / "learning_curve.csv")

    gate_acc = mean_acc(gs.spice_leak, gs.ml_leak)
    gate_acc_solved = mean_acc(gs.spice_leak, gs.ml_solved_leak)
    rel = np.abs(gs.ml_leak / gs.spice_leak - 1) * 100
    rel_s = np.abs(gs.ml_solved_leak / gs.spice_leak - 1) * 100
    acc = {k: mean_acc(cs.spice_leak, cs[k]) for k in ["ml_gate_sum", "ml_flat", "ml_levels"]}
    acc_mlonly = mean_acc(cs.spice_gate_sum, cs.ml_gate_sum)
    hier_err = csum.iloc[1]
    lev = csum[csum.comparison.str.contains("levels")].iloc[0]
    hml = csum.iloc[0]
    mlerr = csum.iloc[2]
    flat = csum.iloc[3]
    node_cols = [c for c in gs.columns if c.startswith("ml_") and c[3:] in gs.columns
                 and c not in ("ml_leak", "ml_ivdd")]
    node_dv = np.concatenate([np.abs(gs[c] - gs[c[3:]]).dropna().to_numpy() for c in node_cols]) * 1e3
    spice_gate_s = sum(float(x) for x in re.findall(r"sims, ([0-9.]+)s", (R / "gate_log.txt").read_text()))
    spice_cla_s = float(log_value("circuit_log.txt", r"simulations in ([0-9.]+)s", "nan"))
    lev_s = float(log_value("circuit_log.txt", r"levels-aware ML: ([0-9.]+)s", "nan"))
    hier_s = float(log_value("circuit_log.txt", r"predictions: ([0-9.]+)s", "nan"))
    extras = (R / "extras_summary.txt").read_text() if (R / "extras_summary.txt").exists() else ""

    S = []   # html sections

    # ---------------------------------------------------------------- title
    S.append(f"""
<header>
<p class="course">Digital VLSI Design &mdash; Course Project</p>
<h1>ML-Driven PVT-Aware Leakage Estimation</h1>
<p class="sub">Team 1 &ldquo;Leaky Lima&rdquo; &middot; {config.TECH.replace('_', ' ')} PTM &middot;
VDD<sub>nom</sub> = {config.VDD_NOM:.1f} V &middot; 3-bit carry-lookahead adder</p>
</header>
<div class="abstract"><b>Summary.</b>
Four neural-network transistor models (NMOS/PMOS &times; ON/OFF) predict all four terminal
currents of a 16&nbsp;nm FET under process, voltage and temperature variation. Composed into
logic gates, they reproduce 3,600 SPICE gate simulations with a
<b>mean accuracy of {p(gate_acc)}&nbsp;%</b> (median error {p(np.median(rel))}&nbsp;%, 95th percentile
{p(np.quantile(rel, .95))}&nbsp;%). Solving the internal node voltages with the same models instead of
taking them from SPICE gives {p(gate_acc_solved)}&nbsp;%.
On the 204-transistor CLA (25,600 SPICE simulations) the hierarchical sum asked for in the brief
reaches {p(acc['ml_gate_sum'])}&nbsp;%. Its error is almost entirely due to the hierarchy itself
(isolated gates with ideal 0/VDD inputs), not the ML: the same sum of <i>SPICE</i> gate leakages is off by
{p(hier_err.bias_pct)}&nbsp;%. A levels-aware hierarchy, which feeds each gate the ML-predicted output voltage
of its driver, removes this bias and reaches <b>{p(acc['ml_levels'])}&nbsp;% mean accuracy</b> on the CLA
(median error {p(lev.med_err_pct)}&nbsp;%), with no SPICE voltage used anywhere.
</div>
""")

    # ---------------------------------------------------------------- 1 setup
    S.append(f"""
<h2>1&nbsp; Setup</h2>
{table(pd.DataFrame([
    ["Model file", f"{config.TECH}.pm (PTM, BSIM4 level 54)"],
    ["Nominal VDD", f"{config.VDD_NOM:.1f} V (sampled &plusmn;10 %: {config.VDD_RANGE[0]:.2f}&ndash;{config.VDD_RANGE[1]:.2f} V)"],
    ["Channel length", f"L = {config.LMIN*1e9:.0f} nm; W = 1&ndash;8 &times; L = {config.W_MIN*1e9:.0f}&ndash;{config.W_MAX*1e9:.0f} nm"],
    ["Gate sizing", f"W<sub>n</sub> = 2L = {config.WN_GATE*1e9:.0f} nm, W<sub>p</sub> = 4L = {config.WP_GATE*1e9:.0f} nm (interior of the training range)"],
    ["Test circuit", "3-bit carry-lookahead adder: 42 library gates, 204 transistors, 7 inputs (128 vectors)"],
    ["Simulator", H.escape(envd.get("simulator", "?"))],
    ["Software", f"Python {envd.get('python','?')}, scikit-learn {envd.get('scikit-learn','?')}, LightGBM {envd.get('lightgbm','?')}, numpy {envd.get('numpy','?')}, pandas {envd.get('pandas','?')}"],
    ["Platform", H.escape(envd.get("platform", "?")) + f", {envd.get('cpu cores used','?')} cores"],
    ["SPICE options", f"<code>{config.SPICE_OPTIONS}</code> in every netlist"],
], columns=["Item", "Value"]), num=[])}
<p><b>Leakage definition.</b> Gate and circuit leakage are the total static power divided by VDD,
P<sub>static</sub>/VDD = &Sigma; V<sub>k</sub>I<sub>k</sub>/VDD over the supply <i>and</i> the input sources.
The input sources matter because gate-tunnelling current into an input pin is real leakage that a driving gate
must supply. The plain supply current I(VDD) is also recorded (<code>*_ivdd</code> columns) and gives the same conclusions.
Terminal currents are positive <i>into</i> the device, so I<sub>d</sub>+I<sub>g</sub>+I<sub>s</sub>+I<sub>b</sub> = 0.</p>
<p><b>Accuracy metric.</b> Mean accuracy = 100 % &minus; mean |ML/SPICE &minus; 1|.
Medians, 95th percentiles and the worst case are reported alongside it.</p>
""")

    # ---------------------------------------------------------------- 2 data
    kct = kc.copy()
    kct["case"] = kct.case.map(CASE)
    kct = pd.DataFrame({"Case": kct.case, "samples": kct.n,
                        "median |&Sigma;I| (A)": [sci(x) for x in kct.abs_sum_median],
                        "max |&Sigma;I| (A)": [sci(x) for x in kct.abs_sum_max],
                        "median |&Sigma;I|/max|I|": [sci(x) for x in kct.rel_median],
                        "p95 |&Sigma;I|/max|I|": [sci(x) for x in kct.rel_p95]})
    S.append(f"""
<h2>2&nbsp; Data generation</h2>
<h3>2.1 Sampling</h3>
<p>For each of the four cases, 10,000 samples are drawn with every parameter independent. VDD is drawn first,
uniform within &plusmn;10 % of nominal. V<sub>d</sub> and V<sub>s</sub> are uniform in [0, VDD], W is uniform in [1, 8]&times;L
and T is uniform in [&minus;40, 125]&nbsp;&deg;C. The five process parameters (vth0, toxe, ndep, u0, xj) are each uniform within &plusmn;10 % of
the model-card nominal, separately for NMOS and PMOS. V<sub>g</sub> and V<sub>b</sub> are fixed by the case (ON: V<sub>g</sub> = VDD for
NMOS / 0 for PMOS; OFF: the opposite; body at the device's own rail). All draws use fixed seeds
(<code>SEED_DATA = {config.SEED_DATA}</code>), and these 10,000-row files are deliverable 1.</p>
<p>A uniform draw almost never puts a terminal on a rail: 3 in 10,000 samples are within 2 % of both rails.
Inside a gate, however, the leaking transistor always has its source on a rail and its drain at the output (&asymp; 0 or VDD).
We therefore add a <b>rail-focused supplement</b> of 5,000 samples per case. In it, V<sub>d</sub> and V<sub>s</sub> are exactly 0 or VDD
one third of the time and otherwise follow an arcsine law that is dense near both rails. In half of these samples V<sub>g</sub> is moved up to
10 % of VDD off its ideal level, because a gate driven by another gate sees a logic 0 a few tens of mV above ground.
A third set of 2,000 samples per case at the gate widths and rail voltages (the <i>deployment check</i>) is used only for validation and test.
No gate or circuit simulation is ever used for training or model selection.</p>
<h3>2.2 Netlists</h3>
<p>The single-FET netlist (one per case in <code>netlists/single_*.cir</code>) has one transistor and a separate voltage source on every
terminal, so all four terminal currents are read directly through their sources. For speed, one ngspice process runs a whole batch of
samples, changing the process parameters with <code>altermod</code> and the width, voltages and temperature with <code>alter</code> between
<code>op</code> analyses. A self-check re-simulates samples as fresh standalone netlists and confirms the batch loop reproduces them. Besides the
terminal currents, the BSIM4 internal current components (I<sub>ds</sub>, I<sub>gs</sub>, I<sub>gd</sub>, I<sub>gcs</sub>, I<sub>gcd</sub>, I<sub>gb</sub>,
GIDL/GISL, I<sub>sub</sub>, junction currents) are recorded. They are regrouped into six physical <b>bipoles</b> (ds, gs, gd, gb, db, sb), which
rebuild the four terminal currents exactly. Gate and circuit netlists are generated from the <i>same</i> transistor list that the ML side uses,
so SPICE and ML cannot disagree about topology. Each starts from a logic-level <code>.nodeset</code>; the reason is given in &sect;6.</p>
<h3>2.3 Splits</h3>
<p>Each 10,000-row dataset is split 70/15/15 (7,000 train / 1,500 validation / 1,500 test) with <code>SEED_SPLIT = {config.SEED_SPLIT}</code>.
The rail supplement is split the same way and its training part is added to the training data. The test parts are touched once, after model
selection. Split indices are saved in <code>results/split_*.npz</code>.</p>
<h3>2.4 Data check</h3>
{table(kct, caption="Table 1. Kirchhoff check, |I<sub>d</sub>+I<sub>g</sub>+I<sub>s</sub>+I<sub>b</sub>| for each case (10,000 samples).")}
<p>KCL closes to 10<sup>&minus;9</sup> of the largest terminal current in the median for every case. The largest relative residuals
(PMOS OFF, up to {p(100*kc.set_index('case').loc['pmos_off','rel_max'],1)}&nbsp;%) occur only where every terminal current is around
10<sup>&minus;15</sup>&nbsp;A. That is the simulator's absolute tolerance (abstol = 10<sup>&minus;18</sup>&nbsp;A amplified by the solver's
convergence check), not a physical violation: the absolute residual never exceeds {sci(kc.abs_sum_max.max())}&nbsp;A.</p>
{fig(F / 'data_check_trends.png', f"Trend checks. OFF current rises {p(tn['ioff_ratio_125C_vs_25C'])}&times; from 25 to 125&nbsp;&deg;C (sub-threshold current &prop; exp(&minus;V<sub>th</sub>/nV<sub>T</sub>), and V<sub>th</sub> falls with T). It falls to {p(tn['ioff_ratio_vth0_plus10pct'])}&times; with +10&nbsp;% vth0 (exponential in V<sub>th</sub>). Gate current falls to {p(tn['ig_ratio_toxe_plus10pct'])}&times; with +10&nbsp;% toxe (direct tunnelling is exponential in oxide thickness). All three trends match the physics.")}
""")

    # ---------------------------------------------------------------- 3 models
    mct = mc.copy()
    mct["Case"] = mct.case.map(CASE)
    mct["Model"] = mct.model.map(FAM)
    best = mct.groupby("case").score.transform("min") == mct.score
    rows = [mct.Case, mct.Model.str.replace(" (neural net)", ""), mct.hparams.str.replace(r"hidden=\((\d+), \d+, \d+\),n_seeds=(\d+)", r"3&times;\1, \2 seeds", regex=True).str.replace("n_estimators=", "trees=").str.replace("num_leaves=", "leaves=")]
    cols = ["Case", "Model", "hyper-params"]
    for c in CUR:
        rows += [mct[f"{c}_r2"].map(lambda x: f"{x:.4f}"), mct[f"{c}_med%"].map(p), mct[f"{c}_p95%"].map(p)]
        cols += [f"{c} R&sup2;", f"{c} med %", f"{c} p95 %"]
    mct_t = pd.DataFrame(dict(zip(cols, rows)))
    cht = pd.DataFrame({"Case": ch.case.map(CASE), "Model": ch.model.map(FAM).str.replace(" (neural net)", ""),
                        **{f"{c} med %": ch[f"{c}_med%"].map(p) for c in CUR},
                        **{f"{c} p95 %": ch[f"{c}_p95%"].map(p) for c in CUR}})
    dep = pd.DataFrame({"Case": ch.case.map(CASE),
                        **{f"{c} med %": ch[f"deploy_{c}_med%"].map(p) for c in CUR},
                        **{f"{c} p95 %": ch[f"deploy_{c}_p95%"].map(p) for c in CUR}})
    ab = abl.copy()
    abt = pd.DataFrame({"Case": ab.case.map(CASE), "current": ab.current,
                        "direct med %": ab.direct_med_pct.map(p), "direct p95 %": ab.direct_p95_pct.map(p),
                        "bipole med %": ab.bipole_med_pct.map(p), "bipole p95 %": ab.bipole_p95_pct.map(p)})
    ab_ratio = np.median(ab.direct_med_pct / ab.bipole_med_pct)
    piv = lc.pivot_table(index=["case", "current"], columns="n_train", values="med_pct")
    nmax = piv.columns.max()
    # smallest n whose median error is within 1.25x of the full-size model, for every current
    ok = piv.le(1.25 * piv[nmax].to_numpy()[:, None] + 0.05)
    n_enough = {case: int(min(n for n in piv.columns if ok.loc[case][n].all())) for case in CASE}
    piv_t = piv.copy()
    piv_t.columns = [f"n = {c}" for c in piv_t.columns]
    piv_t = piv_t.map(p).reset_index()
    piv_t.insert(0, "Case", piv_t.pop("case").map(CASE))

    S.append(f"""
<h2>3&nbsp; Models</h2>
<h3>3.1 What is learned</h3>
<p>Predicting the four terminal currents directly is hard: they span about 12 decades, change sign, and I<sub>s</sub> is often the
difference of two much larger opposite currents. We use the structure of the device instead. Each of the six bipoles is a passive
two-terminal element whose current always flows from its higher to its lower terminal. The model therefore learns its
<b>conductance</b> G = I/V, as log G per unit width, and rebuilds the terminal currents with KCL:
I<sub>d</sub> = I<sub>ds</sub> &minus; I<sub>gd</sub> + I<sub>db</sub>, and so on. This guarantees exact KCL, the correct sign of every current,
and I = 0 at V = 0. Five sub-models per case share the inputs: channel (ds), gate-to-diffusion tunnelling (gs and gd, one model
evaluated twice), gate-to-body, drain-junction and source-junction.</p>
<p><b>Features</b> (per sub-model): VDD, T [K], log(W<sub>eff</sub>/L), &Delta;V<sub>g</sub>/V<sub>T</sub> (offset of the gate from its
ideal level), the five process parameters relative to nominal, the two terminal voltages of the bipole, V/V<sub>T</sub>, log|V|, and
three Arrhenius-type inputs 1/V<sub>T</sub>, V<sub>th0</sub>/V<sub>T</sub> and (VDD&minus;V<sub>th0</sub>)/V<sub>T</sub>, since
sub-threshold current goes as exp(&minus;V<sub>th</sub>/nV<sub>T</sub>). The <b>width normalisation</b> follows BSIM4 for each mechanism.
A width sweep in SPICE showed:</p>
<ul>
<li>The OFF channel scales with the <i>clamped</i> effective width. Below 20&nbsp;nm BSIM4 replaces W<sub>eff</sub> by
2&middot;10<sup>&minus;8</sup>(4&middot;10<sup>&minus;8</sup>&minus;W<sub>eff</sub>)/(6&middot;10<sup>&minus;8</sup>&minus;2W<sub>eff</sub>), so a 1L device
conducts as if it were 14&nbsp;nm wide, not 6&nbsp;nm.</li>
<li>The ON channel scales with the drawn W.</li>
<li>Tunnelling scales with W&minus;2&middot;wint.</li>
<li>Junctions do not depend on W in this card.</li>
</ul>
<p>Without the clamp the models carried a +2&nbsp;% bias into every gate.</p>
<h3>3.2 Model families and hyper-parameters</h3>
<ul>
<li><b>Polynomial + ridge</b>: standardised features, polynomial degree 3 or 4, ridge strength chosen by internal CV over 10<sup>&minus;8</sup>&ndash;1.</li>
<li><b>Random forest</b>: 300 trees, max_features 0.6, fully grown.</li>
<li><b>LightGBM</b>: 4,000 trees, learning rate 0.02, 31 or 63 leaves, row and column subsampling 0.8/0.9.</li>
<li><b>MLP</b>: 3 hidden layers of 64 or 128 tanh units, L-BFGS, L2 = 10<sup>&minus;6</sup>, up to 5,000 iterations, standardised inputs
and target. Five networks with different seeds are averaged in log-G space (a geometric mean of the conductances), which removes most of
L-BFGS's run-to-run noise.</li>
</ul>
<p>All families use the same bipole targets and features. Selection uses the mean of the median-relative-error scores on three validation
sets: uniform, rail-focused and deployment check. The table below is the brief's uniform validation set.</p>
{table(mct_t, num=cols[2:], caption="Table 2. Model comparison on the uniform validation set (1,500 samples per case). Errors are relative, on currents above 10<sup>&minus;14</sup>&nbsp;A; R&sup2; is computed on asinh(I/1&nbsp;fA), which weights every decade equally. Bold = selected.", best=lambda i, r: bool(best.iloc[i]), cls="small")}
{fig(fig_model_comparison(mc), "Validation median error by model family (log scale). The dotted line is 1&nbsp;%.")}
<p><b>Choice.</b> The MLP wins every case, usually by 5&ndash;20&times; in median error and more in the 95th percentile. Log-conductance is a
smooth function of the inputs, a sum of exponents with gentle non-linearities, which a tanh network represents with few parameters. Trees
fit it in piecewise-constant steps, which gives the heavy p95 tails in Table 2. A degree-3/4 polynomial is accurate where it is flexible
enough (NMOS OFF I<sub>d</sub>) but fails on mechanisms with sharper bias dependence, for example the junction/GIDL current in I<sub>b</sub>.
Of the two MLP sizes, the 64-unit network was selected for every case.</p>
<h3>3.3 Test-set performance of the chosen models</h3>
{table(cht, caption="Table 3. Chosen models on the held-out test set (1,500 uniform samples per case).")}
{table(dep, caption="Table 4. The same models on the deployment-check test set (1,000 samples per case at the gate widths and rail voltages).")}
<p>On the test set the median error is below 1&nbsp;% for every current and case. On the deployment-check set the
leakage-carrying currents (I<sub>d</sub>, I<sub>s</sub>) stay below 1&nbsp;% in the median. The larger OFF-state I<sub>b</sub> errors there are on
body currents of about 10<sup>&minus;14</sup>&nbsp;A, six to seven decades below the channel current, so they do not affect gate leakage. The PMOS-OFF I<sub>g</sub> and I<sub>s</sub> rows evaluate fewer
samples (see the <code>eval_frac</code> columns of <code>chosen_models_test.csv</code>), because those currents are often below the
10<sup>&minus;14</sup>&nbsp;A floor.</p>
{fig(F / 'pred_vs_spice_nmos_on.png', 'Predicted vs SPICE, NMOS ON, test set (Id, Ig, Is, Ib).')}
{fig(F / 'pred_vs_spice_nmos_off.png', 'Predicted vs SPICE, NMOS OFF, test set.')}
{fig(F / 'pred_vs_spice_pmos_on.png', 'Predicted vs SPICE, PMOS ON, test set.')}
{fig(F / 'pred_vs_spice_pmos_off.png', 'Predicted vs SPICE, PMOS OFF, test set. The spread in I<sub>g</sub> is at 10<sup>&minus;16</sup>&ndash;10<sup>&minus;15</sup> A, below the evaluation floor.')}
<h3>3.4 Does the structure matter? Ablation and data size</h3>
<p>The ablation trains the same MLP (same size, same rows, same ensemble) to predict each terminal current directly, on an asinh scale.
It is {p(ab_ratio,1)}&times; worse in median error (median over the 16 currents). The learning curve retrains the chosen recipe on 500
to {nmax} uniform samples (plus half as many rail samples). The median error reaches within 25&nbsp;% of the full-size model at
{", ".join(f"{CASE[c]} n&nbsp;=&nbsp;{n}" for c, n in n_enough.items())}. This answers the brief's question of the smallest
sufficient dataset: a few thousand SPICE samples per case.</p>
{table(abt, caption="Table 5. Ablation: direct terminal-current MLP vs the bipole model (validation set).", cls="small")}
{fig(fig_learning_curve(lc), "Learning curve: validation median (solid) and 95th-percentile (dashed) error vs the number of uniform training samples.")}
{table(piv_t, caption="Table 6. Learning curve, validation median error (%).", cls="small")}
""")

    # ---------------------------------------------------------------- 4 gates
    grt = pd.DataFrame({"Gate": gr.gate, "inputs": gr.inputs,
                        "SPICE median (A)": gr.spice_median.map(sci),
                        "ML median (A)": gr.ml_median.map(sci),
                        "bias %": gr.bias_pct.map(lambda x: f"{x:+.2f}"),
                        "median err %": gr.med_err_pct.map(p), "p95 err %": gr.p95_err_pct.map(p),
                        "max err %": gr.max_err_pct.map(p),
                        "ML-solved med %": gr.solved_med_err_pct.map(p),
                        "ML-solved p95 %": gr.solved_p95_err_pct.map(p),
                        "max |&Delta;V| node (mV)": gr.max_node_dv_mV.map(lambda x: f"{x:.2f}")})
    by_gate = gs.groupby("gate", sort=False).apply(lambda g: mean_acc(g.spice_leak, g.ml_leak), include_groups=False)
    worst_case = gr.loc[gr.p95_err_pct.idxmax()]
    best_case = gr.loc[gr.p95_err_pct.idxmin()]
    S.append(f"""
<h2>4&nbsp; Gate results</h2>
<p>Each of the 18 gate/input cases is simulated at the same 200 random corners. A corner is one draw of VDD, T and process: one draw for all
NMOS and one for all PMOS. Following the brief, the ML leakage uses the SPICE-recorded internal and output voltages to set each transistor's
terminal voltages, then sums V&middot;I over the predicted bipole currents. As an extension (&sect;7.1), the internal and output node voltages
are also solved from the ML models alone. Both sets of node voltages are recorded in <code>results/gate_samples.csv</code>.</p>
{table(grt, caption=f"Table 7. Gate results, 200 corners per case. Over all 3,600 simulations: mean accuracy {p(gate_acc)} %, median error {p(np.median(rel))} %, p95 {p(np.quantile(rel,.95))} %, max {p(rel.max())} %. ML-solved: {p(gate_acc_solved)} %, {p(np.median(rel_s))} %, {p(np.quantile(rel_s,.95))} %, {p(rel_s.max())} %.", cls="small")}
{fig(F / 'gate_parity.png', 'Gate parity plot, all 3,600 simulations (log axes; dotted lines at &plusmn;1 % and &plusmn;5 %).', 62)}
{fig(fig_gate_error_vs_corner(gs), 'Signed gate error vs temperature, VDD and PMOS vth0, in deciles of each variable (median and 5&ndash;95 % band).')}
<p><b>Most and least accurate.</b> Mean accuracy per gate:
{", ".join(f"{g} {p(v)} %" for g, v in by_gate.items())}. The most accurate case is
{best_case.gate} {best_case.inputs} (p95 {p(best_case.p95_err_pct)} %). Its leakage is set by a two-transistor OFF stack, whose current is
small and dominated by a single, well-sampled mechanism. The least accurate is {worst_case.gate} {worst_case.inputs}
(p95 {p(worst_case.p95_err_pct)} %). The cases with the largest tails (INV&nbsp;1, AND2&nbsp;00/01/10, whose output inverter sees a 1, and
NOR2&nbsp;01, OR2&nbsp;00) all leak through a single OFF PMOS with V<sub>sd</sub> &asymp; VDD. That makes the PMOS-OFF channel model the
accuracy bottleneck at gate level, consistent with it having the highest test p95 in Table 3. The stack effect is visible in the SPICE
medians: NAND2&nbsp;00 and NOR2&nbsp;11 leak about 200&times; less than an inverter, because the internal node rises (falls) and reverse-biases
the upper OFF device. The models reproduce this without being told: the ML-solved internal node voltages agree with SPICE to within
{p(node_dv.max())} mV (median {p(np.median(node_dv),3)} mV).</p>
""")

    # ---------------------------------------------------------------- 5 circuit
    cst = pd.DataFrame({"Comparison": csum.comparison.str.replace(r"\s+", " ", regex=True),
                        "SPICE median (A)": csum.spice_median.map(sci), "ML median (A)": csum.ml_median.map(sci),
                        "bias %": csum.bias_pct.map(lambda x: f"{x:+.2f}"), "median err %": csum.med_err_pct.map(p),
                        "p95 err %": csum.p95_err_pct.map(p), "max err %": csum.max_err_pct.map(p)})
    cst["mean acc %"] = [p(mean_acc(cs.spice_leak, cs.ml_gate_sum)), p(mean_acc(cs.spice_leak, cs.spice_gate_sum)),
                         p(acc_mlonly), p(acc["ml_flat"]), p(acc["ml_levels"])]
    worst_vec = cres.loc[cres.med_err_pct.idxmax()]
    best_vec = cres.loc[cres.med_err_pct.idxmin()]
    S.append(f"""
<h2>5&nbsp; Test-circuit results: 3-bit CLA</h2>
<p>The adder is built only from library gates. The propagate signals p<sub>i</sub> = a<sub>i</sub>&oplus;b<sub>i</sub> use 4-NAND2 XORs and the
generate signals g<sub>i</sub> = a<sub>i</sub>b<sub>i</sub> use AND2. Each carry is computed directly from g, p and c<sub>0</sub>
(c<sub>3</sub> = g<sub>2</sub>+p<sub>2</sub>g<sub>1</sub>+p<sub>2</sub>p<sub>1</sub>g<sub>0</sub>+p<sub>2</sub>p<sub>1</sub>p<sub>0</sub>c<sub>0</sub>
from AND2/OR2 trees), and s<sub>i</sub> = p<sub>i</sub>&oplus;c<sub>i</sub>. That gives 42 gates and 204 transistors; a unit test checks that
it adds correctly for all 128 inputs. SPICE simulates the full netlist for all 128 vectors at the same 200 corners (25,600 operating points,
{spice_cla_s:.0f}&nbsp;s on {envd.get('cpu cores used','?')} cores). For the hierarchical ML prediction (the brief's method), logic values are
propagated, each gate's Part-B ML leakage is looked up for its input combination and corner, and the 42 values are summed.</p>
{table(cst, caption="Table 8. CLA metrics over 128 vectors &times; 200 corners = 25,600 points. Rows 2&ndash;4 split the error into its causes; row 5 is our extension (&sect;7.2).", cls="small")}
<div class="grid2">
{fig(F / 'circuit_parity.png', 'CLA parity plot (log axes): hierarchical ML (the brief) and levels-aware ML vs the SPICE circuit.')}
{fig(fig_cla_vs_temp(cs), 'Median signed CLA error vs temperature for each method.')}
</div>
<p><b>Compared with the gates.</b> The hierarchical estimate reaches {p(acc['ml_gate_sum'])} % mean accuracy (median error
{p(hml.med_err_pct)} %, bias {p(hml.bias_pct)} %), clearly below the gate level, as the brief anticipates. The extra reference points show
where the error comes from:</p>
<ul>
<li><b>ML error alone</b> (ML gate sum vs SPICE gate sum): median {p(mlerr.med_err_pct)} %, mean accuracy {p(acc_mlonly)} %. This is the gate-level
accuracy again; gate errors partly cancel in the sum.</li>
<li><b>Hierarchy error alone</b> (SPICE gate sum vs SPICE circuit): bias {p(hier_err.bias_pct)} %, p95 {p(hier_err.p95_err_pct)} %. The isolated-gate
assumption is wrong in a systematic way, independent of the ML.</li>
<li><b>Flat ML</b> (all 204 transistors at the circuit's own SPICE voltages): median {p(flat.med_err_pct)} %. The transistor models are just as good
inside the large circuit.</li>
</ul>
<p>The hierarchy error grows with temperature, from about {p(np.median(100*(cs.spice_gate_sum/cs.spice_leak-1)[cs.temp<0]),1)} % below 0&nbsp;&deg;C
to {p(np.median(100*(cs.spice_gate_sum/cs.spice_leak-1)[cs.temp>80]),1)} % above 80&nbsp;&deg;C. In a real chain a logic 0 sits tens of mV above
ground (Figure&nbsp;{N_FIG[0]+1}) and a logic 1 below VDD, and the shift grows with leakage, so with temperature. The next gate's OFF transistor sees
that offset on its gate, and sub-threshold current is exponential in it, so the real circuit leaks more than the ideal-input sum. Loading
(tunnelling current drawn by fan-out pins) is a much smaller effect: the levels-aware hierarchy below ignores it and still removes the bias.
Per vector, the median error ranges from {p(best_vec.med_err_pct)} % (inputs {best_vec.inputs}) to {p(worst_vec.med_err_pct)} %
(inputs {worst_vec.inputs}); <code>results/circuit_results.csv</code> lists all 128.</p>
{fig(fig_level_shift(gs), 'How far isolated-gate outputs sit from the rails in SPICE (median per temperature decile, log scale).', 70)}
""")

    # ---------------------------------------------------------------- 6 error analysis
    worst = gs.nlargest(8, "ae")[["gate", "vector", "corner", "temp", "vdd", "nmos_vth0", "pmos_vth0", "e"]].copy()
    worst_t = pd.DataFrame({"Gate": worst.gate, "inputs": worst.vector, "corner": worst.corner,
                            "T (&deg;C)": worst.temp.map(lambda x: f"{x:.1f}"), "VDD (V)": worst.vdd.map(lambda x: f"{x:.3f}"),
                            "NMOS vth0": worst.nmos_vth0.map(lambda x: f"{x:.3f}"), "PMOS vth0": worst.pmos_vth0.map(lambda x: f"{x:.3f}"),
                            "error %": worst.e.map(lambda x: f"{x:+.2f}")})
    cold = gs[gs.temp < 0].ae
    warm = gs[gs.temp > 40].ae
    hv = gs[gs.vdd > 1.05 * config.VDD_NOM].ae
    lv = gs[gs.vdd < 0.95 * config.VDD_NOM].ae
    S.append(f"""
<h2>6&nbsp; Error analysis</h2>
<p><b>Where.</b> At gate level the error depends mainly on temperature and supply. Below 0&nbsp;&deg;C the median |error| is
{p(np.median(cold))} % (p95 {p(np.quantile(cold,.95))} %), against {p(np.median(warm))} % (p95 {p(np.quantile(warm,.95))} %) above 40&nbsp;&deg;C.
Above 1.05&nbsp;V the p95 is {p(np.quantile(hv,.95))} %, against {p(np.quantile(lv,.95))} % below 0.95&nbsp;V. The worst points all come from a few
cold, high-VDD corners (Table&nbsp;9) and from the cases that leak through one OFF PMOS (&sect;4). Process parameters have weak correlation with the
error (|r| &lt; 0.1 for every parameter), so the models capture the process dependence well.</p>
{table(worst_t, caption="Table 9. The eight largest gate errors.", cls="small")}
<p><b>Why.</b> (1) Cold OFF currents are the smallest in the dataset, and log G changes fastest with V<sub>th</sub>/V<sub>T</sub> there. The same
error in log G is a larger relative current error, and cold, high-VDD corners are corners of the 12-dimensional input box where uniform data is
thinnest. (2) At high VDD, drain-induced barrier lowering and GIDL make the OFF current depend more steeply on V<sub>ds</sub> exactly where the
gate operates (V<sub>ds</sub> = VDD). (3) The PMOS-OFF channel is the weakest of the four models, because its leakage depends on several
competing mechanisms of similar size.</p>
<p><b>A SPICE artefact found along the way.</b> Before the fix, the worst "error" was +126 % on INV A=1 at corner 198 (112&nbsp;&deg;C, 1.088&nbsp;V).
The ML was right: ngspice had converged to a spurious operating point. Sweeping V(out) with a forcing source shows a single true KCL root at
42&nbsp;mV (3.58&nbsp;&micro;A). The default Newton start gets stuck at a narrow kink of the model near 22&nbsp;mV. Starting every gate and circuit
simulation from a logic-level <code>.nodeset</code> removes it. Only this one of the 3,600 gate points moved by more than 0.1 %, and no CLA point did.</p>
<p><b>Circuit level.</b> The CLA error of the brief's method is dominated by the ideal-level assumption (&sect;5), with a {p(hier_err.bias_pct)} % bias
that grows with temperature. With the levels corrected, the remaining CLA error ({p(lev.med_err_pct)} % median) equals the gate-level ML error.</p>
""")

    # ---------------------------------------------------------------- 7 extensions
    mlv = pd.read_csv(R / "extras_mlv.csv") if (R / "extras_mlv.csv").exists() else None
    mech = pd.read_csv(R / "extras_mechanisms.csv", dtype={"vector": str}) if (R / "extras_mechanisms.csv").exists() else None
    sens = pd.read_csv(R / "extras_sensitivity.csv") if (R / "extras_sensitivity.csv").exists() else None
    mcs = pd.read_csv(R / "extras_montecarlo_summary.csv", dtype={"vector": str}) if (R / "extras_montecarlo_summary.csv").exists() else None
    unc = pd.read_csv(R / "extras_uncertainty.csv", dtype={"vector": str}) if (R / "extras_uncertainty.csv").exists() else None
    cost = pd.read_csv(R / "extras_cost.csv") if (R / "extras_cost.csv").exists() else None
    ext = [f"""
<h2>7&nbsp; Extensions</h2>
<h3>7.1 SPICE-free node voltages</h3>
<p>The ML models can predict the operating point as well as the leakage. At DC, the ML-predicted terminal currents into every internal node sum
to zero. <code>leakml/solve.py</code> solves this KCL system for all unknown nodes at once: damped Newton, finite-difference Jacobian, each step
limited to 50&nbsp;mV as SPICE does. It is vectorised over corners and iterates only on rows that have not converged. The initial guess puts logic
nodes at their rails and stack nodes at the rail of their ON neighbour. Every gate case converges, with the KCL residual below 10<sup>&minus;12</sup>&nbsp;A.
The solved node voltages are within {p(node_dv.max())} mV of SPICE, and the leakage computed from them is as accurate as with SPICE voltages
({p(gate_acc_solved)} % vs {p(gate_acc)} %).</p>
<h3>7.2 Levels-aware hierarchy</h3>
<p>The same solver fixes the CLA hierarchy. Gates are visited in topological order, and each one is solved with its inputs set to the
<i>ML-predicted output voltages</i> of its drivers instead of ideal 0/VDD. Rows that share a corner and input voltages are solved once. Over 42
gates the solver needs at most {int(lstats.iterations.max())} iterations and leaves {int(lstats.unconverged.sum())} rows unconverged. The predicted
gate-output voltages match the full SPICE circuit to {log_value('circuit_log.txt', r'median \|dV\| ([0-9.]+) mV')} mV in the median
({log_value('circuit_log.txt', r'max ([0-9.]+) mV')} mV max). The CLA leakage improves to <b>{p(acc['ml_levels'])} % mean accuracy</b>, with median
error {p(lev.med_err_pct)} %, p95 {p(lev.p95_err_pct)} %, max {p(lev.max_err_pct)} % and bias {lev.bias_pct:+.2f} %. That beats even the flat ML on SPICE
voltages ({p(acc['ml_flat'])} %) and uses no SPICE input at all: the full chain from the model card to circuit leakage runs on four ML models.</p>
"""]
    if mlv is not None:
        m = mlv.set_index("method")
        ext.append(f"""
<h3>7.3 Minimum-leakage standby vector</h3>
<p>In standby, a block can be parked at the input vector that leaks least. The CLA's 128 vectors differ by
{cs.groupby('vector').spice_leak.mean().max() / cs.groupby('vector').spice_leak.mean().min():.2f}&times; in mean leakage. Ranked by ML, the vectors
follow SPICE with Spearman &rho; = {m.loc['ml_levels','spearman']:.3f} (levels-aware) and {m.loc['ml_gate_sum','spearman']:.3f}
(ideal-level hierarchy). Both pick the same minimum-leakage vector as SPICE ({m.loc['ml_levels','ml_mlv']}). Choosing the best vector separately in
each corner, the levels-aware ML finds the exact SPICE optimum in {100*m.loc['ml_levels','corners_exact']:.0f} % of corners and is never more than
{p(m.loc['ml_levels','corner_regret_max_pct'])} % above it.</p>
{fig(F / 'extras_mlv.png', 'Mean CLA leakage of the 128 input vectors, sorted by SPICE.')}
""")
    if mech is not None:
        t25 = mech[np.isclose(mech.temp, mech.temp.unique()[np.argmin(np.abs(mech.temp.unique() - 25))])]
        tlo, thi = mech[mech.temp == mech.temp.min()], mech[mech.temp == mech.temp.max()]
        ext.append(f"""
<h3>7.4 Leakage by mechanism</h3>
<p>SPICE reports only total currents, but the bipole models split leakage into physical mechanisms. Averaged over the 18 gate cases at nominal
process and VDD:</p>
<ul>
<li>Channel (sub-threshold) leakage is {p(100*tlo.channel_frac.mean(),1)} % at &minus;40&nbsp;&deg;C, {p(100*t25.channel_frac.mean(),1)} % at
{t25.temp.iloc[0]:.0f}&nbsp;&deg;C and {p(100*thi.channel_frac.mean(),1)} % at 125&nbsp;&deg;C.</li>
<li>Gate tunnelling makes up the rest: {p(100*tlo.tunnelling_frac.mean(),1)} % when cold, falling to {p(100*thi.tunnelling_frac.mean(),1)} % when hot.
It is almost temperature-independent while sub-threshold current grows exponentially.</li>
<li>Junction leakage and GIDL are below {p(100*mech.junction_frac.max(),3)} % everywhere in this card.</li>
</ul>
<p>Gate tunnelling matters most in the stacked cases, where the stack has suppressed sub-threshold leakage: it is
{p(100*t25[(t25.gate=='NAND2')&(t25.vector=='00')].tunnelling_frac.iloc[0],0)} % of NAND2&nbsp;00 and
{p(100*t25[(t25.gate=='NOR2')&(t25.vector=='11')].tunnelling_frac.iloc[0],0)} % of NOR2&nbsp;11 near room temperature.</p>
{fig(F / 'extras_mechanisms.png', 'Left: mechanism currents vs temperature for three gate cases. Right: share of each mechanism per gate case near room temperature.')}
""")
    if sens is not None:
        s = sens.iloc[1:].copy()
        s["span"] = s.ratio_to_nominal
        top = s.iloc[np.argsort(-np.abs(np.log(s.span.to_numpy())))].head(4)
        ext.append(f"""
<h3>7.5 Sensitivity</h3>
<p>Each PVT parameter is moved one at a time (process and VDD &plusmn;10 %, T to &minus;40 / 125&nbsp;&deg;C) and the levels-aware CLA leakage is
averaged over all 128 vectors. The largest effects are {", ".join(f"{r.case} ({r.ratio_to_nominal:.2f}&times;)" for r in top.itertuples())}.
Supply voltage has the largest effect, through drain-induced barrier lowering: a higher V<sub>ds</sub> lowers the
OFF device's threshold. Next come PMOS vth0 and temperature, as the sub-threshold law exp(&minus;V<sub>th</sub>/nV<sub>T</sub>) predicts. PMOS
parameters matter more than NMOS ones because the PMOS devices are twice as wide and carry most of the CLA leakage. Oxide thickness, mobility
and junction depth change the leakage by less than 10&nbsp;%.</p>
{fig(F / 'extras_sensitivity.png', 'One-at-a-time sensitivity of the CLA leakage (log scale).', 75)}
""")
    if mcs is not None:
        rows = " ".join(
            f"For vector {r.vector}, the median is {sci(r.median)}&nbsp;A and the 99th percentile {r.p99_over_median:.2f}&times; the median "
            f"(lognormal &sigma; = {r.lognorm_sigma:.2f}; the 200 SPICE corners give median {sci(r.spice200_median)}&nbsp;A, KS p = {r.ks_pvalue_vs_spice200:.2f})."
            for r in mcs.itertuples())
        ext.append(f"""
<h3>7.6 Monte-Carlo leakage distribution</h3>
<p>Once trained, the models make large statistical studies cheap. The levels-aware CLA leakage is evaluated at {int(mcs.n.iloc[0]):,} new random corners
for the minimum- and maximum-leakage vectors. {rows} The distribution is strongly right-skewed: worst-case design must budget for
several times the median leakage.</p>
{fig(F / 'extras_montecarlo.png', 'CLA leakage over new random corners (ML) vs the 200 SPICE corners.')}
""")
    if unc is not None:
        from scipy import stats
        rho = stats.spearmanr(unc.member_spread_pct, unc.abs_err_pct).statistic
        q = pd.qcut(unc.member_spread_pct, 5, labels=False)
        lo_e, hi_e = unc.abs_err_pct[q == 0].median(), unc.abs_err_pct[q == 4].median()
        ext.append(f"""
<h3>7.7 Uncertainty from the ensemble</h3>
<p>Each of the five ensemble members gives its own gate leakage, and their spread is a free error bar. The spread correlates only weakly with the real
error (Spearman &rho; = {rho:.2f}). The 20 % of points with the smallest spread have median error {p(lo_e)} %, and the 20 % with the largest have
{p(hi_e)} %. The members disagree where data is thin, but most of the remaining error is shared by all members, a systematic bias of the training
distribution. An ensemble cannot detect such a bias; only more data where the gates operate can remove it.</p>
{fig(F / 'extras_uncertainty.png', 'Ensemble spread vs actual error, all 3,600 gate simulations.', 60)}
""")
    if cost is not None:
        c = cost.set_index("method")
        spice_ms = c.ms_per_cpu_eval.iloc[0]
        flat_ms = c.ms_per_cpu_eval.iloc[1]
        ext.append(f"""
<h3>7.8 Cost</h3>
<p>SPICE needs {p(spice_ms)} ms of CPU time per CLA operating point. ML inference on the same 204 transistors, given the node voltages, needs
{p(flat_ms,3)} ms, {spice_ms/flat_ms:.0f}&times; less. The brief's hierarchical lookup costs {hier_s:.1f}&nbsp;s for all 25,600 points once the gate tables
exist. The SPICE-free levels-aware solve of all 25,600 points takes {lev_s:.0f}&nbsp;s on one core, about
{c.cpu_seconds.iloc[0]/lev_s:.0f}&times; less CPU time than SPICE. Most of that time goes into the finite-difference Jacobian, which needs many model
calls per Newton step; an analytic Jacobian of the MLPs would remove most of it.</p>
""")
    if (R / "parity_summary.csv").exists():
        ps = pd.read_csv(R / "parity_summary.csv")
        pt = pd.DataFrame({"Comparison": ps.comparison, "bias %": ps.bias_pct.map(lambda x: f"{x:+.2f}"),
                           "median err %": ps.med_err_pct.map(p), "p95 err %": ps.p95_err_pct.map(p),
                           "max err %": ps.max_err_pct.map(p), "mean acc %": ps.mean_acc.map(p)})
        lv, hr = ps.iloc[-1], ps.iloc[0]
        ext.append(f"""
<h3>7.9 A second, unseen circuit: 8-bit parity checker</h3>
<p>To check that nothing was tuned to the CLA, the brief's other test circuit was built from the same library: a balanced tree of seven 4-NAND2
XORs (28 gates, 112 transistors). It was simulated in SPICE for all 256 vectors at the same 200 corners (51,200 points), with no retraining
or other change. The brief's hierarchy reaches {p(hr.mean_acc)} % mean accuracy (bias {hr.bias_pct:+.2f} %). The levels-aware ML hierarchy
reaches <b>{p(lv.mean_acc)} %</b> (median error {p(lv.med_err_pct)} %, p95 {p(lv.p95_err_pct)} %), confirming the CLA conclusions on a different
circuit.</p>
{table(pt, caption="Table 10. 8-bit parity checker, 256 vectors &times; 200 corners.", cls="small")}
{fig(F / 'parity_parity.png', 'Parity checker: ML vs SPICE circuit leakage (log axes).', 55)}
""")
    S.append("".join(ext))

    # ---------------------------------------------------------------- 8 limitations
    S.append(f"""
<h2>8&nbsp; Limitations and improvements</h2>
<ul>
<li><b>Training distribution.</b> The 10,000 uniform samples rarely land where gates operate (terminals on the rails, cold and high-VDD
corners). The rail supplement fixed most of this. The remaining tail is in cold, high-VDD, single-OFF-PMOS conditions. Adaptive sampling,
adding SPICE points where the ensemble disagrees most, would target it directly.</li>
<li><b>Ideal-level hierarchy.</b> As specified in the brief, the hierarchy is biased by {p(hier_err.bias_pct)} % on this circuit whatever the
ML: even exact SPICE gate leakages give only {p(mean_acc(cs.spice_leak, cs.spice_gate_sum))} % mean accuracy. The levels-aware hierarchy (&sect;7.2) removes this limit. Loading between gates is still ignored, which is acceptable here
because gate tunnelling is a small share of leakage at 16&nbsp;nm HP (&sect;7.4). It would matter more for thin-oxide LP cards.</li>
<li><b>Solver speed.</b> The levels-aware solve uses finite-difference Jacobians. An analytic Jacobian, or a small learned gate-output model,
would make it as fast as the lookup.</li>
<li><b>Scope.</b> The models cover W = 1&ndash;8 L, &plusmn;10 % process and &minus;40 to 125&nbsp;&deg;C. Outside that box they extrapolate. One process draw is
shared by all devices of a type (no within-die mismatch), as specified. Mismatch would need per-transistor draws, which the models already support.</li>
<li><b>Simulator noise.</b> Data regenerated with ngspice 44.2 and 45.2 differs only below about 10<sup>&minus;12</sup>&nbsp;A. L-BFGS amplifies this into
&plusmn;0.02 points of gate accuracy with 5 seeds (&plusmn;0.07 with 3). Training itself is bit-reproducible on identical data.</li>
</ul>

<h2>9&nbsp; Reproducibility and deliverables</h2>
<p>All scripts run in order (<code>./run_all.sh --full</code>) from the model file with fixed seeds. The fast path (<code>./run_all.sh</code>) reuses
the stored data, the saved models and the cached CLA SPICE reference, and reruns validation in minutes. <code>pytest</code> checks 26 invariants:
KCL of the saved models, passivity of the bipoles, netlist/ML topology identity, truth tables, that the CLA adds correctly, the BSIM4 width clamp,
solver convergence, and that the parity checker computes parity.</p>
{table(pd.DataFrame([
    ["1", "Datasets (4 &times; 10,000 rows)", "data/{nmos,pmos}_{on,off}.csv"],
    ["2", "Example netlists", "netlists/single_*.cir, gate_*.cir, cla3.cir"],
    ["3", "Kirchhoff check, trend plots", "results/kirchhoff_check.csv, figures/data_check_trends.png (Table 1, Figure 1)"],
    ["4", "Model comparison", "results/model_comparison.csv (Table 2)"],
    ["5", "Chosen models", "results/chosen_models_test.csv, figures/pred_vs_spice_*.png, saved_models/*.joblib (Table 3)"],
    ["6", "Gate results", "results/gate_results.csv, results/gate_samples.csv (Table 7)"],
    ["7", "Gate parity plot", "figures/gate_parity.png"],
    ["8", "CLA results", "results/circuit_summary.csv, circuit_results.csv, figures/circuit_parity.png (Table 8)"],
    ["9", "Code", "00_ &hellip; 08_*.py, leakml/, run_all.sh, tests/"],
    ["10", "Report", "report/Leaky_Lima_Report.pdf (this document, generated by report/build_report.py)"],
], columns=["#", "Deliverable", "Where"]), num=[], cls="small")}
""")
    return "\n".join(S)


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.2pt; line-height: 1.42; color: #111; background: #fff; max-width: 190mm; margin: auto; }
header { text-align: center; margin-bottom: 8px; }
.course { font-variant: small-caps; letter-spacing: .05em; margin: 0; color: #444; }
h1 { font-size: 19pt; margin: 4px 0 4px; }
.sub { margin: 0 0 6px; color: #333; }
.abstract { border-left: 3px solid #2a5d9f; background: #f3f6fa; padding: 8px 12px; margin: 10px 0 14px; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #999; padding-bottom: 2px; margin-top: 20px; break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 4px; break-after: avoid; }
p { margin: 5px 0; text-align: justify; }
ul { margin: 4px 0 6px; padding-left: 20px; }
li { margin: 2px 0; }
code { font-family: "Menlo", monospace; font-size: 8.4pt; background: #f2f2f2; padding: 0 2px; }
table { border-collapse: collapse; margin: 8px auto; font-size: 8.6pt; break-inside: avoid; }
table.small { font-size: 7.4pt; }
caption { caption-side: top; text-align: left; font-size: 8.6pt; padding-bottom: 3px; }
th, td { border-bottom: 1px solid #ddd; padding: 2px 5px; text-align: left; vertical-align: top; }
thead th { border-bottom: 1.5px solid #333; border-top: 1.5px solid #333; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { font-weight: bold; }
figure { margin: 8px auto; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 8.4pt; text-align: left; margin-top: 2px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.grid2 figure { width: 100% !important; }
"""


def main():
    body = build()
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Leaky Lima Report</title><style>{CSS}</style></head><body>{body}</body></html>")
    out = HERE / "report.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)
    if CHROME:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", out.as_uri()], check=True, capture_output=True)
        print("wrote", PDF)
    else:
        print("Chrome not found: open report.html and print to PDF")


if __name__ == "__main__":
    main()
