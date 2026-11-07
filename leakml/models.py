"""
Physics-structured transistor model.

One TransistorModel per case (nmos_on, nmos_off, pmos_on, pmos_off).
Inside, it is a set of small regressors, one per leakage MECHANISM
(bipole), each learning

        log( G / W_eff ),   G = I_bipole / V_bipole   (G > 0 always: passive)

i.e. the log-conductance per unit effective width, and predicting
I = exp(prediction) * W_eff * V.  The four terminal currents are then
assembled with KCL (leakml/bipoles.py), so Id+Ig+Is+Ib = 0 exactly.

Why this instead of predicting Id, Ig, Is, Ib directly?
  * terminal currents are SUMS of mechanisms with opposite signs, which
    cancel and cross zero -> very hard to learn in log space;
  * every single mechanism is one-signed and smooth, so log G is easy;
  * the transistor is symmetric, so gate<->source and gate<->drain
    tunnelling are the SAME function with d and s swapped: one regressor
    learns both from twice the data.  (The junctions are NOT symmetric
    in this PTM card, which we verified in SPICE, so they get separate
    regressors.)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import config
from leakml import bipoles

K_B = 8.617333e-5      # Boltzmann constant, eV/K  ->  V_T = K_B * T
V_MIN = 1e-3           # below this |V| we use G(V_MIN): any smooth I-V is linear near 0
I_FLOOR = 1e-17        # below this the SPICE current is numerical noise

# sub-model name -> list of (bipole, own-terminal, other-terminal)
# "own" is the diffusion terminal the bipole touches; for symmetric bipoles
# we use (lower, higher) of the two diffusion voltages instead.
SUBMODELS = {
    "ch": [("ds", "lo", "hi")],
    "gx": [("gs", "vs", "vd"), ("gd", "vd", "vs")],
    "gb": [("gb", "lo", "hi")],
    "db": [("db", "vd", "vs")],
    "sb": [("sb", "vs", "vd")],
}


def _terminal(df, which):
    vd, vs = df["vd"].to_numpy(), df["vs"].to_numpy()
    if which == "lo":
        return np.minimum(vd, vs)
    if which == "hi":
        return np.maximum(vd, vs)
    return df[which].to_numpy()


def w_eff(df):
    return df["w"].to_numpy() - 2 * config.WINT


def w_eff_clamped(df):
    """BSIM4's effective width as used for the CHANNEL current.

    BSIM4 replaces Weff below 20 nm by a smooth clamp,
        Weff' = 2e-8 * (4e-8 - Weff) / (6e-8 - 2 Weff),
    so at W = 1 x Lmin (Weff = 6 nm) the channel behaves like 14.2 nm.
    Normalising the channel current by this makes I/W exactly flat in W."""
    we = w_eff(df)
    return np.where(we < 2e-8, 2e-8 * (4e-8 - we) / (6e-8 - 2 * we), we)


# Width each mechanism scales with, verified by a nominal SPICE width sweep
# (log(I / norm) varies by < 0.01 over W = 1..8 Lmin for these choices):
#   channel, OFF      -> clamped Weff (sub-threshold: log swing 0.003)
#   channel, ON       -> drawn W (strong inversion adds narrow-width Vth and
#                        series-resistance effects; drawn W is the smoothest
#                        law, swing 0.10, and avoids the clamp's kink at
#                        W = 30 nm right next to the 32 nm gate NMOS)
#   gate tunnelling   -> unclamped Weff (W - 2 wint)
#   junctions (db/sb) -> no width scaling (no AS/AD given, so diode area is fixed)
WNORM = {"ch": "clamp", "gx": "weff", "gb": "weff", "db": "none", "sb": "none"}
WNORM_ON = {**WNORM, "ch": "w"}


def width_norm(df, how):
    if how == "clamp":
        return w_eff_clamped(df)
    if how == "none":
        return np.full(len(df), config.LMIN)
    if how == "w":
        return df["w"].to_numpy()
    return w_eff(df)


def vg_ideal(case, df):
    return df["vdd"].to_numpy() if config.CASES[case]["vg"] == "vdd" else np.zeros(len(df))


def features(df, case, own, other, v, version=1):
    """Inputs to every sub-model.

    Process parameters are normalised by their nominal value (so they are all
    ~1 +/- 0.1), temperature is in kelvin, and physics features are added:
      log(W_eff/L) - width enters currents through the EFFECTIVE width
      dVg / V_T    - how far the gate is from its ideal logic level (a gate
                     driven by another gate sits a few tens of mV off the
                     rail, and sub-threshold current is exponential in it)
      V / V_T      - the exponent of every diode / sub-threshold law
      log|V|       - lets the model represent saturating currents (I ~ const
                     means log G = const - log V)."""
    dev = config.CASES[case]["dev"]
    nom = config.NOMINAL[dev]
    T = df["temp"].to_numpy() + 273.15
    vt = K_B * T
    dvg = df["vg"].to_numpy() - vg_ideal(case, df)
    cols = [df["vdd"].to_numpy(), T, np.log(w_eff(df) / config.LMIN), dvg / vt]
    cols += [df[p].to_numpy() / nom[p] for p in config.PROCESS_PARAMS]
    vabs = np.maximum(np.abs(v), V_MIN)
    cols += [_terminal(df, own), _terminal(df, other), np.sign(v) * vabs / vt, np.log(vabs)]
    if version >= 2:
        # Arrhenius-type inputs: sub-threshold current ~ exp(-Vth / (n kT/q)),
        # junction/GIDL ~ exp(-Eg / kT).  Handing the MLP 1/VT and Vth0/VT
        # directly removes a residual temperature trend in the gate errors.
        vth = np.abs(df["vth0"].to_numpy())
        cols += [1.0 / vt, vth / vt, (df["vdd"].to_numpy() - vth) / vt]
    return np.column_stack(cols)


FEATURE_NAMES = (["vdd", "T_K", "log_Weff_over_L", "dVg_over_VT"]
                 + [f"{p}_rel" for p in config.PROCESS_PARAMS]
                 + ["v_own", "v_other", "V_over_VT", "log_absV"])
FEATURE_NAMES_V2 = FEATURE_NAMES + ["inv_VT", "vth0_over_VT", "overdrive_over_VT"]


# ---------------------------------------------------------------------------
# The four learner families compared in the project
# ---------------------------------------------------------------------------
class _ScaledTarget:
    """Standardise y before fitting (helps the MLP), undo it on predict."""
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.mu, self.sd = y.mean(), y.std() + 1e-12
        self.est.fit(X, (y - self.mu) / self.sd)
        return self

    def predict(self, X):
        return self.est.predict(X) * self.sd + self.mu


def make_learner(kind, seed=0, **hp):
    if kind == "poly_ridge":
        deg = hp.get("degree", 3)
        return make_pipeline(StandardScaler(), PolynomialFeatures(deg),
                             RidgeCV(alphas=np.logspace(-8, 0, 17)))
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=hp.get("n_estimators", 300),
                                     min_samples_leaf=1, max_features=0.6,
                                     n_jobs=1, random_state=seed)
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=hp.get("n_estimators", 4000),
                                 learning_rate=0.02, num_leaves=hp.get("num_leaves", 31),
                                 min_child_samples=5, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.9, random_state=seed, verbose=-1,
                                 n_jobs=1)
    if kind == "mlp":
        net = MLPRegressor(hidden_layer_sizes=hp.get("hidden", (64, 64, 64)),
                           activation="tanh", solver="lbfgs",
                           alpha=hp.get("alpha", 1e-6),
                           max_iter=hp.get("max_iter", 5000), max_fun=10**7,
                           tol=1e-10, random_state=seed)
        return _ScaledTarget(make_pipeline(StandardScaler(), net))
    raise ValueError(kind)


class Ensemble:
    """Average of several learners trained with different seeds (in log-G space,
    i.e. a geometric mean of the predicted conductances).  The spread between
    members is an uncertainty estimate."""
    def __init__(self, members):
        self.members = members

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)

    def predict_std(self, X):
        return np.std([m.predict(X) for m in self.members], axis=0)


# ---------------------------------------------------------------------------
class TransistorModel:
    """Predicts (id, ig, is, ib) for one case from the 12 dataset inputs."""

    def __init__(self, case, kind, physics_width=True, n_seeds=1, feat_version=2, **hp):
        self.case, self.kind, self.hp = case, kind, hp
        self.dev = config.CASES[case]["dev"]
        self.sub = {}
        # per-mechanism width scaling (see WNORM); False = original v3 behaviour
        base = WNORM_ON if case.endswith("_on") else WNORM
        self.wnorm = dict(base) if physics_width else {k: "weff" for k in SUBMODELS}
        self.n_seeds = n_seeds
        self.feat_version = feat_version

    def _X(self, df, own, other, v):
        return features(df, self.case, own, other, v, getattr(self, "feat_version", 1))

    def _wn(self, name, df):
        # models pickled before `wnorm` existed normalise everything by Weff
        how = getattr(self, "wnorm", {}).get(name, "weff")
        return width_norm(df, how)

    def _training_rows(self, name, df, bip):
        Xs, ys = [], []
        for b, own, other in SUBMODELS[name]:
            v = bipoles.voltage(df, b)
            i = bip[b].to_numpy()
            keep = (np.abs(v) > V_MIN) & (np.abs(i) > I_FLOOR) & (np.sign(i) == np.sign(v))
            Xs.append(self._X(df[keep], own, other, v[keep]))
            # target: log of conductance per unit of the width this mechanism scales with
            ys.append(np.log(i[keep] / v[keep] / self._wn(name, df[keep])))
        return np.vstack(Xs), np.concatenate(ys)

    def fit_submodel(self, name, df, bip, seed=0):
        X, y = self._training_rows(name, df, bip)
        if len(y) < 50:        # mechanism absent in this case (e.g. Vgb = 0)
            return name, None
        n = getattr(self, "n_seeds", 1)
        ests = [make_learner(self.kind, seed=seed + k, **self.hp).fit(X, y) for k in range(n)]
        return name, ests[0] if n == 1 else Ensemble(ests)

    def fit(self, df, bip):
        for name in SUBMODELS:
            self.sub[name] = self.fit_submodel(name, df, bip)[1]
        return self

    def predict_bipoles(self, df):
        out = {}
        for name, parts in SUBMODELS.items():
            est = self.sub.get(name)
            for b, own, other in parts:
                v = bipoles.voltage(df, b)
                if est is None:
                    out[b] = np.zeros(len(df))
                    continue
                logg = est.predict(self._X(df, own, other, v))
                out[b] = np.exp(np.clip(logg, -120, 60)) * self._wn(name, df) * v
        return pd.DataFrame(out, index=df.index)

    def predict(self, df):
        return bipoles.terminals(self.predict_bipoles(df))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
EVAL_FLOOR = 1e-14   # A; terminal currents below this are excluded from %-errors


def metrics(true, pred, floor=EVAL_FLOOR):
    """Relative-error statistics on currents above `floor`, plus R^2 on a
    signed-log scale (asinh), which weights every decade equally."""
    true, pred = np.asarray(true), np.asarray(pred)
    m = np.abs(true) > floor
    rel = np.abs(pred[m] - true[m]) / np.abs(true[m])
    a, b = np.arcsinh(true / 1e-15), np.arcsinh(pred / 1e-15)
    r2 = 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    return dict(r2=r2, med=np.median(rel), p95=np.quantile(rel, 0.95),
                mean=np.mean(rel), frac_eval=m.mean())


# ---------------------------------------------------------------------------
# Fast inference (same numbers, no scikit-learn call overhead)
# ---------------------------------------------------------------------------
class FastMLP:
    """Plain-numpy forward pass of a fitted _ScaledTarget(StandardScaler -> MLP).

    scikit-learn spends ~1 ms per predict() call on input validation, which
    dominates when the KCL solver calls the models tens of thousands of times
    on small batches.  This reproduces the prediction to round-off."""
    def __init__(self, scaled_target):
        scaler, net = scaled_target.est.steps[0][1], scaled_target.est.steps[-1][1]
        assert net.activation == "tanh" and net.out_activation_ == "identity"
        self.mean, self.scale = scaler.mean_, scaler.scale_
        self.W, self.b = net.coefs_, net.intercepts_
        self.mu, self.sd = scaled_target.mu, scaled_target.sd

    def predict(self, X):
        h = (X - self.mean) / self.scale
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.tanh(h @ W + b)
        return (h @ self.W[-1] + self.b[-1]).ravel() * self.sd + self.mu


def _fast(est):
    if isinstance(est, Ensemble):
        return Ensemble([_fast(m) for m in est.members])
    if isinstance(est, _ScaledTarget):
        return FastMLP(est)
    return est


def fast_copy(model):
    """Copy of a TransistorModel whose MLP sub-models use FastMLP."""
    import copy
    m = copy.copy(model)
    m.sub = {k: (None if v is None else _fast(v)) for k, v in model.sub.items()}
    return m
