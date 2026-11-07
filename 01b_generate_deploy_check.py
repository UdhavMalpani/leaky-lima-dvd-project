"""
Part A, step 1b: a transistor-level "deployment check" set (validation only).

The uniform 10k set says how well a model fits the whole input box, but the
gates only ever use one small corner of it: W fixed at the gate sizing
(Wn = 2 Lmin, Wp = 4 Lmin), source on its rail, drain on the opposite rail
(OFF) or within tens of mV of the source (ON), and stack nodes a little off
the rails.  A model can be excellent on the uniform set and still biased
there; that is exactly how the +2 % gate bias of the v3 models showed up.

So we SPICE-simulate 2,000 single transistors per case in those conditions
(still with random VDD, T and process, and still with one FET per netlist -
no gate simulation is involved).  03_train_models.py splits it 50/50 into a
validation half (used, with the uniform and rail sets, to select models)
and a test half (reported).  It is never used for training.

Output: data/<case>_deploy.csv and data/<case>_deploy_bipoles.csv
"""
import importlib

import numpy as np

import config

gen = importlib.import_module("01_generate_transistor_data")
N_DEPLOY = 2000


def draw_deploy_samples(case, n, rng):
    df = gen.draw_samples(case, n, rng)
    dev = config.CASES[case]["dev"]
    df["w"] = config.WN_GATE if dev == "nmos" else config.WP_GATE
    rail = 0.0 if dev == "nmos" else 1.0          # the rail this device's source sits on

    def near(level):
        # a node within 0-20 % of VDD of `level`, strongly concentrated near it
        return np.abs(level * df.vdd - df.vdd * rng.uniform(0, 0.2, n) * rng.uniform(size=n) ** 3)

    on_rail = rng.uniform(size=n) < 0.7
    df["vs"] = np.where(on_rail, rail * df.vdd, near(rail))
    if case.endswith("off"):
        df["vd"] = np.where(rng.uniform(size=n) < 0.7, (1 - rail) * df.vdd, near(1 - rail))
    else:
        df["vd"] = near(rail)
    return df


def main():
    rng = np.random.default_rng(config.SEED_DATA + 2)
    for case in config.CASES:
        df, bip = gen.simulate(case, draw_deploy_samples(case, N_DEPLOY, rng))
        ok = df[["id", "ig", "is", "ib"]].notna().all(axis=1)
        df[ok][gen.COLUMNS].to_csv(config.DATA_DIR / f"{case}_deploy.csv", index=False)
        bip[ok].to_csv(config.DATA_DIR / f"{case}_deploy_bipoles.csv", index=False)
        print(f"{case:9s}: {ok.sum()} deployment-check samples")


if __name__ == "__main__":
    main()
