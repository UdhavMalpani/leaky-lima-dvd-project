#!/usr/bin/env bash
# Regenerate every deliverable, in order, with fixed seeds.
#
#   ./run_all.sh            fast path: reuse data/, saved_models/ and cache/
#                           (gate + circuit validation only, a few minutes)
#   ./run_all.sh --full     everything from the model file: SPICE data
#                           generation, training (hours), validation, extras
#
# Environment overrides: PYTHON, NGSPICE, N_WORKERS
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

run() { echo; echo "=== $*"; "$PY" "$@"; }

run 00_environment.py
if [[ "${1:-}" == "--full" ]]; then
  rm -f cache/cla3_spice.pkl
  run 01_generate_transistor_data.py
  run 01b_generate_deploy_check.py
  run 02_data_check.py
  run 03_train_models.py
fi
run 04_gate_validation.py
run 05_circuit_validation.py
run 08_parity_validation.py
if [[ "${1:-}" == "--full" ]]; then
  run 06_ablation_and_learning_curve.py
  run 07_extras.py
fi
echo; echo "done - see results/ and figures/"
