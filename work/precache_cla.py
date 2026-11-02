# One-off: compute the CLA SPICE reference into the cache used by 05 while models train.
import sys, time; sys.path.insert(0, ".")
import pandas as pd, config
config.N_WORKERS = 4
from leakml import circuits, evaluate
corners = pd.read_csv(config.RESULTS_DIR / "corners.csv", index_col="corner")
cells = circuits.circuit_cells(circuits.cla3_gates()); fets = circuits.flatten(cells)
t0 = time.time()
sims = evaluate.run_spice("cla3", fets, circuits.CLA_INPUTS, circuits.input_vectors(7), corners)
pd.to_pickle((sims, time.time() - t0), config.WORK_DIR / "cla3_spice.pkl")
print("cached", len(sims), round(time.time() - t0), "s")
