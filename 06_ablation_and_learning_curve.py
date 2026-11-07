"""
Extra analysis (not required, but answers two obvious questions).

1. Ablation: does the bipole decomposition matter?
   Same MLP size, same training rows (uniform + rail), same ensemble size,
   same validation set, but predicting the four terminal currents DIRECTLY
   from the raw inputs on a signed-log (asinh) scale.

2. Learning curve: how many SPICE samples are actually needed?
   Retrain the chosen model family on 500 ... 7000 training samples and
   measure validation error.  The brief asks for 'the smallest dataset
   possible without compromising accuracy'.

Outputs: results/ablation.csv, results/learning_curve.csv,
         figures/learning_curve.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import importlib
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from leakml.models import TransistorModel, make_learner, metrics

warnings.filterwarnings("ignore")
train_mod = importlib.import_module("03_train_models")
CURRENTS = train_mod.CURRENTS
I_REF = 1e-15
SIZES = [500, 1000, 2000, 4000, 7000]


def raw_features(df):
    return df[["vdd", "temp", "w", "vd", "vs", *config.PROCESS_PARAMS]].to_numpy()


def direct_job(case, cur, df_tr, df_va, n_seeds=1):
    """Baseline: MLP(s) per terminal current, target asinh(I / I_REF).
    Same training rows (uniform + rail), same ensemble size and same
    validation set as the bipole model it is compared with."""
    y = np.arcsinh(df_tr[cur].to_numpy() / I_REF)
    mu, sd = y.mean(), y.std()
    preds = []
    for seed in range(n_seeds):
        est = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64, 64), activation="tanh", solver="lbfgs",
            max_iter=5000, max_fun=10**7, tol=1e-10, random_state=seed))
        est.fit(raw_features(df_tr), (y - mu) / sd)
        preds.append(est.predict(raw_features(df_va)))
    pred = np.sinh(np.mean(preds, axis=0) * sd + mu) * I_REF
    return case, cur, metrics(df_va[cur], pred)


def curve_job(case, kind, hp, n_seeds, n, df_tr, bip_tr, df_va, rdf_tr, rbip_tr):
    """n uniform samples + n/2 rail-focused samples (the same 2:1 mix as the full model)."""
    k = n // 2
    d = pd.concat([df_tr.iloc[:n], rdf_tr.iloc[:k]], ignore_index=True)
    b = pd.concat([bip_tr.iloc[:n], rbip_tr.iloc[:k]], ignore_index=True)
    m = TransistorModel(case, kind, n_seeds=n_seeds, **hp).fit(d, b)
    pred = m.predict(df_va)
    return case, n, {c: metrics(df_va[c], pred[c]) for c in CURRENTS}


def main():
    chosen = pd.read_csv(config.RESULTS_DIR / "chosen_models_test.csv")
    data = {}
    for case in config.CASES:
        df, bip = train_mod.load(case)
        sp = np.load(config.RESULTS_DIR / f"split_{case}.npz")
        rdf, rbip = train_mod.load(case, "_rail")
        rsp = np.load(config.RESULTS_DIR / f"split_{case}_rail.npz")
        data[case] = (df.iloc[sp["train"]], bip.iloc[sp["train"]], df.iloc[sp["val"]],
                      rdf.iloc[rsp["train"]], rbip.iloc[rsp["train"]])

    seeds = {}
    jobs = []
    for _, r in chosen.iterrows():
        m = joblib.load(config.SAVED_MODELS_DIR / f"{r.case}.joblib")
        seeds[r.case] = getattr(m, "n_seeds", 1)
        for n in SIZES:
            jobs.append(delayed(curve_job)(r.case, m.kind, m.hp, seeds[r.case], n, *data[r.case]))
    for c in config.CASES:
        full_tr = pd.concat([data[c][0], data[c][3]], ignore_index=True)   # uniform + rail train
        jobs += [delayed(direct_job)(c, cur, full_tr, data[c][2], seeds[c]) for cur in CURRENTS]
    out = Parallel(n_jobs=config.N_WORKERS + 2)(jobs)

    direct = [o for o in out if len(o) == 3 and isinstance(o[1], str)]
    curve = [o for o in out if not (len(o) == 3 and isinstance(o[1], str))]

    # ---- ablation table ------------------------------------------------
    # comparator: the bipole model trained on the full training set (n = 7000
    # uniform + 3500 rail, i.e. the chosen recipe) from the learning curve
    full = {case: m for case, n, m in curve if n == max(SIZES)}
    rows = []
    for case, cur, m in direct:
        b = full[case][cur]
        rows.append(dict(case=case, current=cur,
                         direct_med_pct=100 * m["med"], direct_p95_pct=100 * m["p95"],
                         bipole_med_pct=100 * b["med"], bipole_p95_pct=100 * b["p95"]))
    abl = pd.DataFrame(rows)
    abl.to_csv(config.RESULTS_DIR / "ablation.csv", index=False)
    print(abl.round(3).to_string(index=False))

    # ---- learning curve ------------------------------------------------
    rows = []
    for case, n, m in curve:
        for c in CURRENTS:
            rows.append(dict(case=case, n_train=n, current=c,
                             med_pct=100 * m[c]["med"], p95_pct=100 * m[c]["p95"]))
    lc = pd.DataFrame(rows).sort_values(["case", "current", "n_train"])
    lc.to_csv(config.RESULTS_DIR / "learning_curve.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
    for a, case in zip(ax, config.CASES):
        for c in CURRENTS:
            g = lc[(lc.case == case) & (lc.current == c)]
            a.loglog(g.n_train, g.med_pct, "o-", label=f"{c} median")
            a.loglog(g.n_train, g.p95_pct, "x--", alpha=0.5, label=f"{c} p95")
        a.axhline(1, c="k", lw=0.6)
        a.set_xticks(SIZES, [str(n) for n in SIZES], fontsize=7)
        a.minorticks_off()
        a.set(title=case, xlabel="training samples")
    ax[0].set_ylabel("validation error (%)")
    ax[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "learning_curve.png", dpi=130)
    print(lc.pivot_table(index=["case", "current"], columns="n_train", values="med_pct").round(2))


if __name__ == "__main__":
    main()
