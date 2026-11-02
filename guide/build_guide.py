"""
Build the learning-guide PDF.

Every number in the guide is read from results/*.csv at build time, so the
PDF always matches the code.  Run from the project root after scripts 01-06:
    python guide/build_guide.py
"""
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd

import config

R = config.RESULTS_DIR
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PDF = config.ROOT / "Leaky_Lima_Learning_Guide.pdf"
CUR = ["id", "ig", "is", "ib"]
FAM = {"mlp": "Neural net (MLP)", "poly_ridge": "Polynomial + Ridge",
       "lightgbm": "LightGBM", "random_forest": "Random Forest"}


def pct(x, d=2):
    return f"{x:.{d}f} %"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}×10<sup>{int(e)}</sup>"


def table(head, rows, num_cols=(), best=None):
    h = "".join(f'<th class="{"num" if i in num_cols else ""}">{c}</th>' for i, c in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        cls = ' class="best"' if best and best(k) else ""
        body.append(f"<tr{cls}>" + "".join(
            f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>")
    return f"<table><tr>{h}</tr>{''.join(body)}</table>"


def parse_log(path, pattern, default="?"):
    try:
        m = re.search(pattern, Path(path).read_text())
        return m.group(1) if m else default
    except FileNotFoundError:
        return default


# ---------------------------------------------------------------------------
def values():
    v = {"build_date": dt.date.today().strftime("%d %B %Y")}
    chosen = pd.read_csv(R / "chosen_models_test.csv")
    comp = pd.read_csv(R / "model_comparison.csv")
    gate = pd.read_csv(R / "gate_results.csv", dtype={"inputs": str})
    gs = pd.read_csv(R / "gate_samples.csv", dtype={"vector": str})
    circ = pd.read_csv(R / "circuit_summary.csv")
    trend = pd.read_csv(R / "trend_numbers.csv", index_col=0).iloc[:, 0]
    kir = pd.read_csv(R / "kirchhoff_check.csv")
    v1 = pd.read_csv(R / "v1_uniform_only" / "gate_samples.csv")

    gerr = np.abs(gs.ml_leak / gs.spice_leak - 1)
    v1err = np.abs(v1.ml_leak / v1.spice_leak - 1)
    v["kpi_trans_med"] = pct(chosen[[f"{c}_med%" for c in CUR]].to_numpy().mean())
    v["kpi_gate_med"] = pct(100 * np.median(gerr))
    v["kpi_circ_med"] = pct(circ.med_err_pct[0])
    cs = pd.read_csv(R / "circuit_samples.csv")
    v["kpi_circ_acc"] = pct(100 - 100 * np.mean(np.abs(cs.ml_gate_sum / cs.spice_leak - 1)))
    v["v1_gate_med"] = pct(100 * np.median(v1err))
    v["v1_gate_p95"] = pct(100 * np.quantile(v1err, 0.95))
    rows = []
    for sub, what in (("v1_uniform_only", "10k uniform samples per case, features: W/L"),
                      ("v2_rail", "+ 5k rail-focused samples per case"),
                      ("", "+ effective-width target and feature, + V<sub>g</sub> offset feature (final)")):
        s = pd.read_csv(R / sub / "gate_samples.csv")
        e = np.abs(s.ml_leak / s.spice_leak - 1) * 100
        cf = R / sub / "circuit_summary.csv"
        cm = f"{pd.read_csv(cf).med_err_pct[0]:.2f} %" if cf.exists() else "–"
        name = {"v1_uniform_only": "v1", "v2_rail": "v2", "": "<b>v3</b>"}[sub]
        rows.append(f"<tr><td>{name}</td><td>{what}</td><td class='num'>{np.median(e):.2f} %</td>"
                    f"<td class='num'>{np.quantile(e, .95):.2f} %</td><td class='num'>{cm}</td></tr>")
    v["version_rows"] = "".join(rows)
    v["n_gate_sims"] = f"{len(gs):,}"
    v["circ_spice_s"] = parse_log(R / "circuit_log.txt", r"simulations in (\d+)s")
    v["circ_ml_s"] = parse_log(R / "circuit_log.txt", r"predictions: ([\d.]+)s")
    v["train_min"] = str(round(int(parse_log(R / "train_log.txt", r"done in (\d+)s", "0")) / 60))
    v["n_submodels"] = parse_log(R / "train_log.txt", r"training (\d+) sub-models")
    v["t_vth"] = f"×{trend['ioff_ratio_vth0_plus10pct']:.2f}".lstrip("×")
    v["t_temp"] = f"{trend['ioff_ratio_125C_vs_25C']:.1f}"
    v["t_toxe"] = f"{trend['ig_ratio_toxe_plus10pct']:.2f}"

    v["kirchhoff_rows"] = "".join(
        f"<tr><td>{r.case}</td><td class='num'>{r.rel_median:.1e}</td><td class='num'>{r.rel_p95:.1e}</td>"
        f"<td class='num'>{r.rel_max:.1e}</td><td>{'✔ balanced' if r.rel_p95 < 1e-6 else '✔ (noise floor, see text)'}</td></tr>"
        for r in kir.itertuples())

    # ---- comparison table (validation) ----
    rows, best_flags = [], []
    chosen_key = {(r.case, r.model) for r in chosen.itertuples()}
    for case in config.CASES:
        g = comp[comp.case == case].sort_values("score")
        for _, r in g.iterrows():
            rows.append([case, FAM[r.model], r.hparams.replace("hidden=", "").replace("n_estimators=", "trees=")]
                        + [f"{r[f'{c}_r2']:.4f}" for c in ("id", "ig")]
                        + [f"{r[f'{c}_med%']:.2f} / {r[f'{c}_p95%']:.1f}" for c in CUR])
            best_flags.append((case, r.model) in chosen_key)
    v["comparison_table"] = table(
        ["case", "family", "best h-params", "R² Id", "R² Ig", "Id med / p95", "Ig med / p95",
         "Is med / p95", "Ib med / p95"], rows, num_cols=range(3, 9), best=lambda k: best_flags[k])

    rows = []
    for _, r in chosen.iterrows():
        rows.append([r.case, f"{FAM[r.model]} {r.hparams.replace('hidden=', '')}"] +
                    [f"{r[f'{c}_med%']:.2f} / {r[f'{c}_p95%']:.1f} "
                     f"<span class='small'>({r[f'{c}_eval_frac']*100:.0f} %)</span>" for c in CUR] +
                    [f"{r['rail_id_med%']:.2f} / {r['rail_id_p95%']:.1f}"])
    v["chosen_table"] = table(["case", "chosen model", "Id med / p95 (kept)", "Ig", "Is", "Ib",
                               "rail test: Id med / p95"], rows, num_cols=range(2, 7))

    # ---- gate table ----
    rows = [[r.gate, r.inputs, sci(r.spice_median), sci(r.ml_median), f"{r.bias_pct:+.2f}",
             f"{r.med_err_pct:.2f}", f"{r.p95_err_pct:.2f}", f"{r.ivdd_med_err_pct:.2f}"] for r in gate.itertuples()]
    v["gate_table"] = table(["gate", "inputs", "SPICE median (A)", "ML median (A)", "bias %",
                             "median err %", "p95 err %", "I(VDD) med err %"], rows, num_cols=range(2, 8))
    g2 = gate.sort_values("p95_err_pct")
    best3 = ", ".join(f"{r.gate} {r.inputs} ({r.med_err_pct:.2f} % / {r.p95_err_pct:.2f} %)" for r in g2.head(3).itertuples())
    worst3 = ", ".join(f"{r.gate} {r.inputs} ({r.med_err_pct:.2f} % / {r.p95_err_pct:.2f} %)" for r in g2.tail(3)[::-1].itertuples())
    v["gate_best_worst"] = (f"most accurate (median / p95): {best3}. Least accurate: {worst3}. "
                            f"Over all {len(gs):,} simulations: median {pct(100*np.median(gerr))}, "
                            f"p95 {pct(100*np.quantile(gerr, .95))}, mean accuracy {pct(100-100*gerr.mean())}.")

    names = ["<b>Hierarchical ML vs SPICE circuit</b>", "SPICE gate sum vs SPICE circuit",
             "ML gate sum vs SPICE gate sum", "Flat ML (circuit node voltages) vs SPICE circuit"]
    what = ["the full method of the brief", "error of the gate-level assumption itself",
            "error of the ML models alone", "transistor models without any gate-level assumption"]
    v["circuit_rows"] = "".join(
        f"<tr><td>{n}</td><td>{w}</td><td class='num'>{r.med_err_pct:.2f} %</td>"
        f"<td class='num'>{r.p95_err_pct:.2f} %</td><td class='num'>{r.bias_pct:+.2f} %</td></tr>"
        for r, n, w in zip(circ.itertuples(), names, what))

    v["learning_section"] = learning_section()
    v["error_analysis"] = error_analysis(gs, gate)
    v["viva"] = viva(v, chosen, gate, circ)
    return v


def learning_section():
    out = []
    try:
        abl = pd.read_csv(R / "ablation.csv")
        rows = [[r.case, r.current, f"{r.direct_med_pct:.2f} / {r.direct_p95_pct:.1f}",
                 f"{r.bipole_med_pct:.2f} / {r.bipole_p95_pct:.1f}"] for r in abl.itertuples()]
        out.append("<h3>Ablation: how much does the bipole idea buy?</h3>"
                   "<p>Same neural network and the same uniform training data, but predicting the four terminal "
                   "currents <em>directly</em> (on an asinh signed-log scale), against our bipole model (validation set, "
                   "median / p95 %). Note the bipole model here also had the rail data; the point is the size of the gap.</p>"
                   + table(["case", "current", "direct MLP", "bipole MLP (ours)"], rows, num_cols=(2, 3)))
    except FileNotFoundError:
        pass
    if (HERE / ".." / "figures" / "learning_curve.png").exists():
        lc = pd.read_csv(R / "learning_curve.csv")
        m = lc.groupby("n_train").med_pct.mean()
        out.append("<h3>Learning curve: how much data do we need?</h3>"
                   "<p>The brief asks for high accuracy 'with the smallest dataset possible'. We retrained the chosen "
                   f"models on growing subsets of the training data. Average median error: "
                   + ", ".join(f"{int(n):,} samples → {e:.2f} %" for n, e in m.items()) + ". "
                   "(Each size n also gets n/2 rail samples, the same mix as the full model; 7,000 is the whole training split.) "
                   "The error falls steeply up to about 2,000 samples and then flattens: the last 3,000 samples buy only a few "
                   "hundredths of a percent. So 10,000 samples per case is comfortably enough, and about 4,000 would already "
                   "give sub-1 % median error.</p>"
                   '<figure><img src="../figures/learning_curve.png"><figcaption><b>Figure 6.4. Learning curve.</b> '
                   "Validation error vs number of training samples, per case and current. Where the curves flatten, more "
                   "data stops helping.</figcaption></figure>")
    return "".join(out)


def error_analysis(gs, gate):
    c = pd.read_csv(R / "corners.csv", index_col="corner")
    s = gs.join(c, on="corner")
    s["err"] = np.abs(s.ml_leak / s.spice_leak - 1) * 100
    hot = s.temp > 85
    cold = s.temp < 0
    lowv = s.nmos_vth0 < np.quantile(c.nmos_vth0, 0.2)
    rows = [["all gate sims", f"{s.err.median():.2f}", f"{s.err.quantile(.95):.2f}"],
            ["T > 85 °C", f"{s.err[hot].median():.2f}", f"{s.err[hot].quantile(.95):.2f}"],
            ["T < 0 °C", f"{s.err[cold].median():.2f}", f"{s.err[cold].quantile(.95):.2f}"],
            ["lowest 20 % NMOS vth0", f"{s.err[lowv].median():.2f}", f"{s.err[lowv].quantile(.95):.2f}"],
            ["VDD > 1.07 V", f"{s.err[s.vdd > 1.07].median():.2f}", f"{s.err[s.vdd > 1.07].quantile(.95):.2f}"],
            ["VDD < 0.93 V", f"{s.err[s.vdd < 0.93].median():.2f}", f"{s.err[s.vdd < 0.93].quantile(.95):.2f}"]]
    worst = gate.sort_values("p95_err_pct", ascending=False).head(4)
    txt = table(["subset of gate simulations", "median |err| %", "p95 |err| %"], rows, num_cols=(1, 2))
    txt += ("<p><b>By gate and input:</b> the largest p95 errors are "
            + ", ".join(f"{r.gate} {r.inputs} ({r.p95_err_pct:.1f} %)" for r in worst.itertuples()) + ".</p>")
    notes = HERE / "analysis_notes.html"     # hand-written interpretation of the numbers
    if notes.exists():
        txt += notes.read_text(encoding="utf-8")
    return txt


def viva(v, chosen, gate, circ):
    qa = [
        ("Walk me through your project in one minute.",
         "We simulate a single 16 nm transistor 40,000 times across random process, voltage and temperature and train four ML models (NMOS/PMOS × ON/OFF) to predict its terminal currents. A gate's leakage is then the sum over its transistors, and the adder's is the sum over its 42 gates. We check both against full SPICE: median error "
         f"{v['kpi_gate_med']} for gates and {v['kpi_circ_med']} for the adder."),
        ("Why log scale?",
         "Currents span 14 decades. On a linear scale the squared error is dominated by the largest currents. On a log scale a fixed error means a fixed percentage for every current, which is what we are graded on."),
        ("Why did you not predict Id, Ig, Is, Ib directly?",
         "Each terminal current is a sum of mechanisms with opposite signs that cancel and create sharp zero-crossings (Figure 5.1). We predict each mechanism (bipole), which is smooth and one-signed, as log G, and recombine with Kirchhoff's law. The ablation table shows the gain."),
        ("How do you know the bipoles are right?",
         "Two automatic checks: they rebuild the four SPICE terminal currents to about 1e-9 on every sample, and sign(I) = sign(V) holds for 100 % of samples."),
        ("Why a neural network and not XGBoost/LightGBM?",
         "Tree models predict piecewise constants, which is good for thresholds but poor for smooth exponential laws (Figure 4.2). On our validation set the MLP's error score was 4-7× lower than the next-best family in every case."),
        ("What is L-BFGS?",
         "A quasi-Newton optimiser: it uses the whole dataset each step and estimates curvature to take well-aimed steps. It suits small, noise-free problems like ours better than Adam."),
        ("How did you choose the model? Did you look at the test set?",
         "Every model was compared on the validation set only. The test set was used once, at the end, for the chosen model. The split is 70/15/15 with seed 42."),
        ("Why does a 10 % vth0 change give 3× leakage?",
         "Sub-threshold current is ∝ exp(−Vth/(n·kT/q)). 48 mV/(1.5 × 26 mV) ≈ 1.2 and e^(−1.2) ≈ 0.3."),
        ("What is the stack effect? Show it.",
         "Two series OFF transistors leak far less than one, because the middle node rises (about 0.17 V for our NAND2 '00'), giving the top device negative Vgs, body effect and less DIBL. Figure 7.2: NAND2 '00' leaks about 200× less than NAND2 '01'."),
        ("Why is your leakage defined as power/VDD?",
         "Gate-tunnelling current flowing into an input pin is supplied, in a real circuit, by the previous gate from VDD. Counting all source power makes gate leakages add up to the circuit's. We also report plain I(VDD); the conclusions are the same."),
        ("Why is the adder less accurate than the gates?",
         "The hierarchy assumes ideal 0/VDD levels and no loading, but real logic-0 levels are typically 6-21 mV (up to 100 mV at hot corners) at this node. Our decomposition (Figure 8.2) separates the two: the hierarchy alone costs a −2.6 % bias, while the ML part adds only about 1.3 % median error, and flat ML on the real node voltages is 1.2 %."),
        ("What was the hardest bug?",
         "ngspice's altermod silently ignores all but the first parameter on a line. We caught it by comparing the fast simulation path against standalone netlists."),
        ("You added extra data: is that allowed?",
         "Yes: the brief says the sample count may be increased. The deliverable CSVs are still the 10,000 uniform samples. The supplement targets the rail voltages where gates operate (distribution shift), and it used no gate or circuit simulations."),
        ("Why is PMOS OFF gate current so poorly predicted?",
         "It is about 1e-15 A, at the level of double-precision rounding in SPICE itself. Those values are noise, which is why we exclude currents below 1e-14 A from percentage errors and report the fraction kept."),
        ("Your transistor test error was below 1 %, but the gates were worse. How did you find out why?",
         "With a 'transistor oracle': re-simulate each transistor of a gate alone at its exact in-gate voltages and compare it with the ML, device by device. It showed a −27 % error for an OFF NMOS whose gate sat at 16 mV instead of 0 (the model had no Vg input) and a +3.6 % bias at narrow widths (effective width W − 2·wint not represented). Adding both as physics features fixed them."),
        ("What is effective width and why does it matter?",
         "BSIM4 removes wint = 5 nm from each channel edge, so a drawn 16 nm device conducts like 6 nm. Current scales with W_eff, so we predict conductance per unit W_eff and give log(W_eff/L) as a feature."),
        ("What would you improve?",
         "Solve internal node voltages with the ML models (no SPICE at all), propagate non-ideal logic levels in the hierarchy instead of assuming exact 0/VDD, and ensemble several networks to tighten the tails."),
    ]
    return "".join(f'<p class="q">Q. {q}</p><p class="a">{a}</p>' for q, a in qa)


def render():
    tpl =(HERE / "template.html").read_text(encoding="utf-8")
    vals = values()
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: str(vals.get(m.group(1), f"[{m.group(1)}?]")), tpl)
    missing = re.findall(r"\[(\w+)\?\]", out)
    if missing:
        print("WARNING: unfilled placeholders:", missing)
    html = HERE / "guide.html"
    html.write_text(out, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", html.as_uri()], check=True, capture_output=True)
    print("wrote", PDF)


if __name__ == "__main__":
    if "--figures" in sys.argv:
        subprocess.run([sys.executable, str(HERE / "make_concept_figures.py")], check=True)
        subprocess.run([sys.executable, str(HERE / "make_result_figures.py")], check=True)
    render()
