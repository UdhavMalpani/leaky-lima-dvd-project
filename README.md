# Leaky Lima — ML-driven PVT-aware leakage estimation

DVD course project (Team 1). Assignment: **16nm HP PTM card, VDD = 1.0 V, 3-bit carry-lookahead adder**.

Four ML transistor models (NMOS/PMOS × ON/OFF) predict Id, Ig, Is, Ib under
process / voltage / temperature variation. They are composed into gate leakage
(INV, NAND2, NOR2, AND2, OR2; 18 input cases × 200 corners) and then into the
CLA (128 input vectors × 200 corners). Every prediction is checked against ngspice.

## Setup

```bash
brew install ngspice            # or apt install ngspice / Windows installer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python 00_environment.py   # checks ngspice, writes results/env.txt
```

ngspice is found automatically on `PATH`. Override it with `NGSPICE=/path/to/ngspice`.
Parallel SPICE processes default to all cores; override with `N_WORKERS=4`.

## Run

```bash
./run_all.sh          # fast: reuse data, saved models and the CLA SPICE cache (~2 min)
./run_all.sh --full   # everything from the model file (SPICE + training: hours)
```

| Step | Script | What it does | Time (11-core M-series) |
|---|---|---|---|
| 0 | `00_environment.py` | simulator / package versions → `results/env.txt` | s |
| 1 | `01_generate_transistor_data.py` | 4 × 10,000 uniform samples + 4 × 5,000 rail-focused samples, SPICE | minutes |
| 1b | `01b_generate_deploy_check.py` | 4 × 2,000 single-FET samples at gate conditions (validation/test only) | s |
| 2 | `02_data_check.py` | Kirchhoff check, 3 trend plots | s |
| 3 | `03_train_models.py` | 4 learner families × 4 cases, select on validation, report test | ~40 min |
| 4 | `04_gate_validation.py` | 18 gate cases × 200 corners, SPICE vs ML (SPICE and ML-solved node voltages) | ~2 min |
| 5 | `05_circuit_validation.py` | CLA, 25,600 SPICE sims (cached in `cache/`, ~2 min to redo) vs hierarchical, flat and levels-aware ML | ~7 min |
| 6 | `06_ablation_and_learning_curve.py` | bipole-vs-direct ablation, learning curve | several h |
| 7 | `07_extras.py` | MLV search, mechanisms, sensitivity, Monte-Carlo, uncertainty, cost (ML only) | ~15 min |
| 8 | `08_parity_validation.py` | second, unseen circuit: 8-bit parity checker, 256 vectors × 200 corners (SPICE cached in `cache/`) | ~5 min |

All randomness comes from the seeds in `config.py` (`SEED_DATA`, `SEED_SPLIT`, `SEED_CORNERS`).
`config.py` is the single source of truth: change `TECH` / `VDD_NOM` there and rerun.

## Layout

```
config.py                  assignment, ranges, seeds, paths
leakml/                    library: spice runner, bipole model, circuits, evaluation,
                           solve.py (ML KCL solver, levels-aware hierarchy)
0X_*.py                    pipeline steps (run in order)
data/                      deliverable 1: <case>.csv (10k rows) + rail supplement + bipole currents
netlists/                  deliverable 2: single-FET, gate and CLA example netlists
results/                   tables (CSV) and logs for deliverables 3–8
figures/                   plots for deliverables 3, 5, 7, 8
saved_models/              deliverable 5: the four chosen models (.joblib, scikit-learn 1.9.0)
cache/                     CLA SPICE reference (expensive; delete to recompute)
guide/, docs/              learning guide and the course hand-outs
work/                      scratch netlists / raw ngspice output (not tracked)
```

## Deliverables map (brief §6)

| # | Deliverable | Where |
|---|---|---|
| 1 | 4 datasets, 10,000 rows, required columns | `data/{nmos,pmos}_{on,off}.csv` |
| 2 | example netlists | `netlists/single_*.cir`, `netlists/gate_*.cir`, `netlists/cla3.cir` |
| 3 | Kirchhoff check + 3 trend plots | `results/kirchhoff_check.csv`, `figures/data_check_trends.png` |
| 4 | model comparison (4 × 4, R², median, p95 on validation) | `results/model_comparison.csv` |
| 5 | chosen models: test metrics, parity plots, saved models | `results/chosen_models_test.csv`, `figures/pred_vs_spice_*.png`, `saved_models/` |
| 6 | gate results table (18 cases) | `results/gate_results.csv` |
| 7 | gate parity plot (log axes) | `figures/gate_parity.png` |
| 8 | CLA metrics + parity plot | `results/circuit_summary.csv`, `results/circuit_results.csv`, `figures/circuit_parity.png` |
| + | extras | `results/extras_*.csv`, `results/extras_summary.txt`, `figures/extras_*.png`, `results/ablation.csv`, `figures/learning_curve.png` |
| 9 | code, ordered, seeded | this repo, `run_all.sh` |
| 10 | report | (in progress) |

## Conventions

- Terminal currents are positive **into** the device, so id + ig + is + ib = 0.
- Gate / circuit leakage = static power ÷ VDD, counting the VDD rail **and** the input drivers
  (gate-tunnelling into an input pin is supplied by the driving gate in a real circuit).
  Plain I(VDD) is recorded too (`*_ivdd` columns).
- Gate sizing: Wn = 2·Lmin, Wp = 4·Lmin (inside the 1–8 × Lmin training range).
- "Mean accuracy" = 100 % − mean |ML/SPICE − 1|.

## Headline numbers (v4, see `CHANGES.md`)

- Gates, 3,600 sims: mean accuracy **99.10 %** (median 0.69 %, p95 2.60 %); SPICE-free 99.12 %.
- CLA, 25,600 sims, levels-aware ML (no SPICE anywhere): mean accuracy **99.21 %**, median 0.66 %, p95 2.18 %.
- Unseen 8-bit parity checker (51,200 sims, no retraining): levels-aware ML **99.24 %**, the brief's hierarchy 94.80 %.
- `.venv/bin/python -m pytest -q tests` runs 26 invariant checks in under a minute.
