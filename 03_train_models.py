"""
Part A, step 2: train, compare and select the transistor models.

For each case: split 70 / 15 / 15 (train / validation / test) with a fixed
seed, train every learner family on TRAIN, compare them on VALIDATION, pick
the best, and only then touch TEST once to report the final numbers.

Both datasets are split this way: the 10k uniform deliverable set and the 5k
rail-focused supplement.  Training uses uniform-train + rail-train.  The
comparison table reports the uniform validation set (the brief's
distribution); model selection averages the uniform and rail validation
scores, because the rails are where the gates actually operate.

Outputs
  results/model_comparison.csv      every learner x case, validation metrics
  results/chosen_models_test.csv    chosen model per case, test metrics
  saved_models/<case>.joblib        the chosen TransistorModel objects
  figures/pred_vs_spice_<case>.png  predicted-vs-SPICE parity plots (test)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one core per training job
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from leakml.models import SUBMODELS, TransistorModel, metrics

warnings.filterwarnings("ignore")
CURRENTS = ["id", "ig", "is", "ib"]

# Learner families and the (small) hyper-parameter grid tried for each
GRID = {
    "poly_ridge":    [dict(degree=3), dict(degree=4)],
    "random_forest": [dict(n_estimators=300)],
    "lightgbm":      [dict(num_leaves=31), dict(num_leaves=63)],
    "mlp":           [dict(hidden=(64, 64, 64)), dict(hidden=(128, 128, 128))],
}


def load(case, suffix=""):
    df = pd.read_csv(config.DATA_DIR / f"{case}{suffix}.csv")
    bip = pd.read_csv(config.DATA_DIR / f"{case}{suffix}_bipoles.csv")
    return df, bip


def split_indices(n, seed=config.SEED_SPLIT, frac=(0.70, 0.15, 0.15)):
    idx = np.random.default_rng(seed).permutation(n)
    a = int(frac[0] * n)
    b = a + int(frac[1] * n)
    return idx[:a], idx[a:b], idx[b:]


def hp_key(hp):
    return ",".join(f"{k}={v}" for k, v in hp.items())


def _fit_job(case, kind, hp, name, df, bip):
    t0 = time.time()
    model = TransistorModel(case, kind, **hp)
    _, est = model.fit_submodel(name, df, bip)
    return case, kind, hp_key(hp), name, est, time.time() - t0


def evaluate(model, df):
    pred = model.predict(df)
    return {c: metrics(df[c], pred[c]) for c in CURRENTS}, pred


def score(m):
    """Selection score: mean over the four currents of (median + p95)/2."""
    return np.mean([(m[c]["med"] + m[c]["p95"]) / 2 for c in CURRENTS])


def parity_plot(case, df, pred, path):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, c in zip(ax, CURRENTS):
        t, p = df[c].abs().to_numpy(), pred[c].abs().to_numpy()
        lo = max(min(t.min(), p.min()), 1e-19)
        hi = max(t.max(), p.max())
        a.loglog(t, p, ".", ms=2, alpha=0.5)
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        m = metrics(df[c], pred[c])
        a.set(title=f"{c}: median {m['med']*100:.2f}%  p95 {m['p95']*100:.1f}%",
              xlabel="|SPICE| (A)", ylabel="|ML| (A)")
    fig.suptitle(f"{case}: ML vs SPICE on the test set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    data, splits, rail, rsplits = {}, {}, {}, {}
    for case in config.CASES:
        df, bip = load(case)
        tr, va, te = split_indices(len(df))
        data[case] = (df, bip)
        splits[case] = (tr, va, te)
        np.savez(config.RESULTS_DIR / f"split_{case}.npz", train=tr, val=va, test=te)
        rdf, rbip = load(case, "_rail")
        rail[case] = (rdf, rbip)
        rsplits[case] = split_indices(len(rdf))
        np.savez(config.RESULTS_DIR / f"split_{case}_rail.npz", train=rsplits[case][0],
                 val=rsplits[case][1], test=rsplits[case][2])

    jobs = []
    for case, (df, bip) in data.items():
        tr, rtr = splits[case][0], rsplits[case][0]
        rdf, rbip = rail[case]
        df_tr = pd.concat([df.iloc[tr], rdf.iloc[rtr]], ignore_index=True)
        bip_tr = pd.concat([bip.iloc[tr], rbip.iloc[rtr]], ignore_index=True)
        for kind, hps in GRID.items():
            for hp in hps:
                for name in SUBMODELS:
                    jobs.append((case, kind, hp, name, df_tr, bip_tr))
    print(f"training {len(jobs)} sub-models in parallel ...", flush=True)
    t0 = time.time()
    fitted = Parallel(n_jobs=config.N_WORKERS + 2, verbose=0)(
        delayed(_fit_job)(*j) for j in jobs)
    print(f"done in {time.time()-t0:.0f}s")

    # assemble TransistorModels
    models = {}
    for case, kind, key, name, est, dt in fitted:
        hp = next(h for h in GRID[kind] if hp_key(h) == key)
        m = models.setdefault((case, kind, key), TransistorModel(case, kind, **hp))
        m.sub[name] = est
        m.train_seconds = getattr(m, "train_seconds", 0) + dt

    # validation comparison
    rows = []
    for (case, kind, key), m in models.items():
        df = data[case][0].iloc[splits[case][1]]
        met, _ = evaluate(m, df)
        rmet, _ = evaluate(m, rail[case][0].iloc[rsplits[case][1]])
        row = dict(case=case, model=kind, hparams=key,
                   score=(score(met) + score(rmet)) / 2,
                   uniform_score=score(met), rail_score=score(rmet),
                   train_s=round(m.train_seconds, 1))
        for c in CURRENTS:
            row[f"{c}_r2"] = met[c]["r2"]
            row[f"{c}_med%"] = 100 * met[c]["med"]
            row[f"{c}_p95%"] = 100 * met[c]["p95"]
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values(["case", "score"])
    comp.to_csv(config.RESULTS_DIR / "model_comparison_all_hparams.csv", index=False)
    # best hyper-parameters per (case, family) -> the 4 x 4 comparison table
    best_per_family = comp.groupby(["case", "model"], sort=False).head(1)
    best_per_family.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(best_per_family.round(4).to_string(index=False))

    # choose the best per case on validation, report on test
    test_rows = []
    for case in config.CASES:
        best = comp[comp.case == case].iloc[0]
        m = models[(case, best.model, best.hparams)]
        df = data[case][0].iloc[splits[case][2]]
        met, pred = evaluate(m, df)
        joblib.dump(m, config.SAVED_MODELS_DIR / f"{case}.joblib")
        parity_plot(case, df, pred, config.FIG_DIR / f"pred_vs_spice_{case}.png")
        rmet, _ = evaluate(m, rail[case][0].iloc[rsplits[case][2]])
        row = dict(case=case, model=best.model, hparams=best.hparams)
        for c in CURRENTS:
            row[f"{c}_med%"] = 100 * met[c]["med"]
            row[f"{c}_p95%"] = 100 * met[c]["p95"]
            row[f"{c}_eval_frac"] = met[c]["frac_eval"]
        for c in CURRENTS:
            row[f"rail_{c}_med%"] = 100 * rmet[c]["med"]
            row[f"rail_{c}_p95%"] = 100 * rmet[c]["p95"]
        test_rows.append(row)
    test = pd.DataFrame(test_rows)
    test.to_csv(config.RESULTS_DIR / "chosen_models_test.csv", index=False)
    print("\nChosen models, TEST set:")
    print(test.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
