# Changes on branch `phase0-setup` (v4 models)

## Results: before → after

| Metric | v3 (main) | v4 (this branch) |
|---|---|---|
| Gate mean accuracy (3,600 sims, SPICE node voltages) | 98.02 % | **99.10 %** |
| Gate median / p95 / max error | 1.40 / 5.55 / 125.5 % | **0.69 / 2.60 / 5.7 %** |
| Gate, SPICE-free (ML solves the internal node voltages) | — | **99.12 %**, median 0.68 %, p95 2.42 %, max 4.9 % |
| ML vs SPICE internal node voltages (gates, worst case) | — | 2.5 mV |
| Worst gate-case bias (18 cases) | +2.32 % | +0.79 % |
| CLA, brief's hierarchy (ideal 0/VDD levels) vs SPICE circuit, median | 2.85 % | 2.59 % (bias −2.56 %, all from the hierarchy) |
| CLA, ML error alone (ML gate sum vs SPICE gate sum), median / p95 | 1.26 / 3.70 % | **0.68 / 2.51 %** |
| CLA, flat ML on circuit voltages, median | 1.21 % | **0.68 %** (mean accuracy 99.14 %) |
| CLA, **levels-aware ML hierarchy, SPICE-free**, median / p95 / max | — | **0.66 / 2.18 / 3.5 %** (bias +0.28 %, mean accuracy **99.21 %**) |
| **8-bit parity checker** (unseen circuit, 51,200 sims), levels-aware ML | — | **99.24 %** mean accuracy, median 0.65 % (brief's hierarchy: 94.80 %) |
| Transistor test, NMOS-OFF id median / p95 | 0.29 / 3.82 % | 0.17 / 2.23 % |

## Phase 0: portability and reproducibility
- `config.NGSPICE` resolves `$NGSPICE`, then `ngspice` on PATH, then the old Windows path. `N_WORKERS` defaults to all cores.
- `00_environment.py` writes `results/env.txt` with the simulator and package versions.
- Added `requirements.txt` (scikit-learn pinned to 1.9.0, which the joblib files need), `README.md`, `run_all.sh` (fast mode and `--full`) and `tests/` (26 pytest invariants).
- Untracked `work/` and `__pycache__/`. Moved the course hand-outs to `docs/course/` and the guide PDF to `docs/`. Removed `model.tar` (it duplicated `models/`).
- The CLA SPICE reference cache now lives in the tracked `cache/` folder.
- Verified on macOS with ngspice 45.2:
  - the v3 gate results reproduce exactly;
  - a fresh SPICE run of all 25,600 CLA sims matches the cache to 1e-8.

## Phase 1: deliverable audit
All of D1–D9 pass:
- **D1:** 10,000 rows per dataset, exact columns, every sampled range inside the brief's bounds.
- **D2:** every netlist has the `.options` line; single-FET netlists give each terminal its own source; the netlists run standalone and reproduce the recorded values.
- **D3–D8:** the expected tables and figures are present.
- **Splits:** 7,000/1,500/1,500, disjoint.

The brief asks for "internal node voltages from both" SPICE and ML. v3 reused the SPICE voltages for ML; this is now closed (Extra 1).

## Phase 2: accuracy
1. **SPICE reference fix.** The +126 % "outlier" (INV A=1, corner 198) was a spurious ngspice operating point.
   - Forcing V(out) shows a single true KCL root at 42 mV (3.58 µA).
   - The default Newton start gets trapped at a narrow model kink near 22 mV (1.85 µA).
   - Gate and circuit runs now start from a logic-level `.nodeset` (`evaluate.nodeset_line`).
   - Of the 3,600 gate points only that one moves by more than 0.1 %; in the CLA none do.
2. **Per-mechanism width scaling** (`models.WNORM`).
   - Root cause of the +2 % gate bias: BSIM4 clamps the effective channel width below 20 nm (`Weff' = 2e-8 (4e-8 − Weff)/(6e-8 − 2 Weff)`), so I/Weff jumps 2.4× between W = 1L and 2L.
   - A nominal SPICE width sweep picked the width law for each mechanism:
     - OFF channel: clamped Weff.
     - ON channel: drawn W.
     - Gate tunnelling: W − 2·wint.
     - Junctions: no width scaling.
3. **Arrhenius features** (`features(..., version=2)`): 1/V_T, Vth0/V_T and (VDD − Vth0)/V_T. These remove a residual temperature trend; signed gate error was correlated −0.38 with T before.
4. **5-seed MLP ensembles** (`models.Ensemble`).
   - Averaging in log-G space removes most of the run-to-run noise of L-BFGS.
   - With 3 seeds, retraining on data from ngspice 44.2 vs 45.2 moved gate accuracy by 0.07 points; with 5 seeds, by 0.02 points.
   - The two simulator versions differ only below the 1e-12 A measurement noise floor.
   - Training itself is bit-identical on identical data.
   - The spread between members serves as an uncertainty estimate (Extra 5 below).
5. **Deployment-check set** (`01b_generate_deploy_check.py`): 2,000 single-FET SPICE samples per case at gate conditions (gate widths, rail voltages).
   - Split 50/50 into validation and test.
   - Used in model selection alongside the uniform and rail sets, never trained on.
   - No gate simulation is used for training or selection.

Old pickles still load: new attributes default to v3 behaviour through `getattr`.

## Phase 4: extras
1. **SPICE-free node voltages** (`leakml/solve.py`).
   - A damped Newton solve of KCL at every internal node using the ML terminal currents:
     - finite-difference Jacobian;
     - 50 mV step limit (like SPICE);
     - stack-node initial guess from which series device is ON.
   - Columns `ml_<node>` and `ml_solved_leak` in `results/gate_samples.csv`, `solved_*` and `max_node_dv_mV` in `gate_results.csv`.
2. **Levels-aware hierarchy for the CLA** (`solve.levels_aware_hierarchy`).
   - Gates are solved in topological order; each gate's inputs are the ML-predicted output voltages of its drivers, not ideal 0/VDD.
   - Identical (corner, input voltage) rows are solved once.
   - This removes the −2.6 % hierarchy bias: median 2.59 % → 0.66 %.
   - Gate-output voltages agree with the SPICE circuit to 0.044 mV (median), 2.9 mV (max).
   - Every gate converged within 12 iterations (`results/circuit_levels_solver_stats.csv`).
3. **Fair ablation** (`06`): the direct baseline now uses the same training rows (uniform + rail) and the same ensemble size; the comparator is the bipole model at the full training size.
4. **`07_extras.py`** (ML only, no new SPICE):
   - minimum-leakage-vector search;
   - leakage split by mechanism vs temperature;
   - one-at-a-time PVT sensitivity;
   - 5,000-corner Monte-Carlo;
   - ensemble uncertainty vs error;
   - cost per evaluation.

   Summary in `results/extras_summary.txt`.
5. **`08_parity_validation.py`**: the brief's other test circuit (8-bit parity checker, 28 NAND2, 112 FETs) as an unseen validation, with no retraining. The brief's hierarchy gives 94.80 % (bias −4.0 %, all of it hierarchy error); the levels-aware ML gives 99.24 %.
   - SPICE cache: `cache/parity8_spice.pkl.xz`, 0.6 MB (gate outputs only).
6. **Ablation result** (fair version): predicting terminal currents directly is about 25× worse in median error than the bipole model. The learning curve flattens at 2,000–4,000 uniform samples per case.

## Report and guide
- `report/build_report.py` builds the typed report (deliverable 10) from `results/` into `report/Leaky_Lima_Report.pdf`. It uses Chrome or Playwright's Chromium for the PDF.
- `guide/` has been rebuilt for v4: a version 3 → 4 chapter, the levels-aware hierarchy, and updated viva answers.
