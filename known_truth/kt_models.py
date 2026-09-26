"""
Demand models of the known-truth benchmark and their planning interfaces.

    B0f  classical structural model: the SIDN functional form with pair-level
         intercepts and *global* elasticities, fitted by nonlinear least squares
    B1   MLP                     (src.models.baselines.MLPDemand)
    B2   XGBoost                 (src.models.baselines.XGBoostDemand)
    B2m  XGBoost with monotone constraints in price (-), advertising (+), and
         recycled content (+)
    B4   BiLSTM + attention      (src.models.recurrent.BiLSTMAttnDemand)
    A1   SIDN                    (src.models.sidn.SIDNDemand)
Conformal robust variants (no retraining): A4 = A1 + CRO, B2c = B2 + CRO.

Every model is exposed as rows -> demand, and as a batched week demand
function X (B, n) -> D (B, I, J) for the planner.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sidn import SIDNDemand                                 # noqa: E402
from src.models.baselines import MLPDemand, XGBoostDemand              # noqa: E402
from src.models.recurrent import BiLSTMAttnDemand                      # noqa: E402

DEC_KEYS = ("p", "b_s", "b_m", "b_r", "u_s", "u_m", "u_r")


# ----------------------------------------------------------------------------- helpers
def _pairs(kt, split):
    cats = kt.bundle.schema.cat_feats
    pc = [k for k, c in enumerate(cats) if c.startswith("product_")]
    rc = [k for k, c in enumerate(cats) if c.startswith("retailer_")]
    return split.X_cat[:, pc].argmax(1), split.X_cat[:, rc].argmax(1)


def _p0_rows(kt, split):
    i, j = _pairs(kt, split)
    return kt.p0[i, j]


# ----------------------------------------------------------------------------- B0f
class FixedFormStructural:
    """Khorshidvand-type law with pair intercepts and global elasticities."""
    name = "B0f"

    def __init__(self, kt):
        self.kt = kt
        self.dec = kt.dec
        self.IJ = kt.I * kt.J

    def _design(self, split_like_X, Xcat):
        cats = self.kt.bundle.schema.cat_feats
        pc = [k for k, c in enumerate(cats) if c.startswith("product_")]
        rc = [k for k, c in enumerate(cats) if c.startswith("retailer_")]
        i = Xcat[:, pc].argmax(1)
        j = Xcat[:, rc].argmax(1)
        X = split_like_X
        rel = X[:, self.dec["p"]] / self.kt.p0[i, j]
        lb = np.stack([np.log1p(np.maximum(X[:, self.dec[k]], 0)) for k in ("b_s", "b_m", "b_r")], 1)
        ubar = (X[:, self.dec["u_s"]] + X[:, self.dec["u_m"]] + X[:, self.dec["u_r"]]) / 3.0
        return i * self.kt.J + j, rel, lb, ubar

    def _pred(self, th, pair, rel, lb, ubar):
        a = th[:self.IJ]
        beta, k, lam = th[self.IJ], th[self.IJ + 1:self.IJ + 4], th[self.IJ + 4]
        return np.exp(a[pair]) * rel ** (-beta) * (1 + lb @ k) * (1 + lam * ubar)

    def fit(self, seed=0):
        tr = self.kt.bundle.train
        pair, rel, lb, ubar = self._design(tr.X_cont_u.astype(float), tr.X_cat)
        y = tr.y_u.astype(float)
        a0 = np.log(np.maximum([y[pair == q].mean() if np.any(pair == q) else 1.0
                                for q in range(self.IJ)], 1e-3))
        th0 = np.r_[a0, 1.0, 0.03, 0.03, 0.03, 0.5]
        lo = np.r_[np.full(self.IJ, -20.0), 0.05, 0.0, 0.0, 0.0, 0.0]
        hi = np.r_[np.full(self.IJ, 20.0), 10.0, 2.0, 2.0, 2.0, 5.0]
        res = least_squares(lambda th: np.log1p(self._pred(th, pair, rel, lb, ubar)) - np.log1p(y),
                            th0, bounds=(lo, hi), max_nfev=2000)
        self.th = res.x
        return self

    def predict_rows(self, Xc_u, Xcat, week=None):
        pair, rel, lb, ubar = self._design(Xc_u, Xcat)
        return self._pred(self.th, pair, rel, lb, ubar)


# ----------------------------------------------------------------------------- src-model wrappers
class SrcModel:
    """Wraps a src.models DemandModel (rows in original units -> demand)."""

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def predict_rows(self, Xc_u, Xcat, week=None):
        return np.asarray(self.model.raw_predict(Xc_u.astype(np.float32), Xcat), dtype=float)


class SeqModel:
    """BiLSTM with week-aware windows: the previous (window-1) weeks of each
    pair (historical rows from train/val/test in time order) followed by the
    live row with the planned decisions."""

    def __init__(self, name, model, kt):
        self.name = name
        self.model = model
        self.kt = kt
        b = kt.bundle
        W = model.window
        rows = []
        for split in (b.train, b.val, b.test):
            i, j = _pairs(kt, split)
            feats = np.concatenate([split.X_cont, split.X_cat], axis=1).astype(np.float32)
            ws = split.panel_df["week_start"].values
            for r in range(len(feats)):
                rows.append((ws[r], int(i[r]), int(j[r]), feats[r]))
        rows.sort(key=lambda t: t[0])
        self.hist = {}
        self.pos = {}
        for ws, i, j, f in rows:
            self.hist.setdefault((i, j), []).append(f)
            self.pos[(ws, i, j)] = len(self.hist[(i, j)]) - 1
        self.W = W
        self.d = rows[0][3].shape[0]

    def predict_rows(self, Xc_u, Xcat, week=None):
        """week: array of week_start values aligned with rows (historical rows
        are looked up; the live row replaces the last window position)."""
        b = self.kt.bundle
        scaled = b.x_scaler.transform(Xc_u).astype(np.float32)
        live = np.concatenate([scaled, Xcat.astype(np.float32)], axis=1)
        cats = b.schema.cat_feats
        pc = [k for k, c in enumerate(cats) if c.startswith("product_")]
        rc = [k for k, c in enumerate(cats) if c.startswith("retailer_")]
        ii = Xcat[:, pc].argmax(1)
        jj = Xcat[:, rc].argmax(1)
        # the (W-1)-week history is shared by every row of the same (week, pair)
        keys = list(zip(week, ii.tolist(), jj.tolist()))
        uniq = {}
        for k in keys:
            if k not in uniq:
                uniq[k] = len(uniq)
        prefix = np.zeros((len(uniq), self.W - 1, self.d), dtype=np.float32)
        for (wk, i, j), u in uniq.items():
            h = self.hist[(i, j)]
            pos = self.pos[(wk, i, j)]
            win = h[max(0, pos - self.W + 1):pos]
            if win:
                prefix[u, self.W - 1 - len(win):] = np.stack(win)
        idx = np.fromiter((uniq[k] for k in keys), dtype=np.int64, count=len(keys))
        seqs = np.empty((len(live), self.W, self.d), dtype=np.float32)
        seqs[:, :-1] = prefix[idx]
        seqs[:, -1] = live
        self.model.model.eval()
        with torch.no_grad():
            ys = self.model.model(torch.tensor(seqs)).numpy().reshape(-1, 1)
        return b.y_scaler.inverse_transform(ys).ravel()


def monotone_xgb(kt, seed):
    from xgboost import XGBRegressor
    b = kt.bundle
    cont = b.schema.cont_feats
    sign = {"p": -1, "b_s": 1, "b_m": 1, "b_r": 1, "u_s": 1, "u_m": 1, "u_r": 1}
    cons = tuple(sign.get(c, 0) for c in cont) + (0,) * b.schema.n_cat
    m = XGBoostDemand(b.schema, b.x_scaler, b.y_scaler)
    m.model = XGBRegressor(max_depth=6, n_estimators=400, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
                           n_jobs=1, verbosity=0, monotone_constraints=cons)
    m.fit(b.train.X_cont, b.train.X_cat, b.train.y, seed=seed)
    return SrcModel("B2m", m)


def fit_all(kt, seed):
    """Train every base model on the training split; returns name -> model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = kt.bundle
    tr = b.train
    out = {"B0f": FixedFormStructural(kt).fit(seed)}
    mlp = MLPDemand(b.schema, b.x_scaler, b.y_scaler, hidden=64, n_layers=2, dropout=0.10, device="cpu")
    mlp.fit(tr.X_cont, tr.X_cat, tr.y, epochs=200, batch_size=64, lr=1e-3, weight_decay=1e-5, seed=seed)
    out["B1"] = SrcModel("B1", mlp)
    xgb = XGBoostDemand(b.schema, b.x_scaler, b.y_scaler)
    xgb.model.set_params(n_jobs=1)
    xgb.fit(tr.X_cont, tr.X_cat, tr.y, seed=seed)
    out["B2"] = SrcModel("B2", xgb)
    out["B2m"] = monotone_xgb(kt, seed)
    bil = BiLSTMAttnDemand(b.schema, b.x_scaler, b.y_scaler, window=8, hidden=32, dropout=0.10, device="cpu")
    bil.fit(tr.X_cont, tr.X_cat, tr.y, week_idx=tr.week_idx, epochs=100, batch_size=32, lr=1e-3,
            weight_decay=1e-5, seed=seed)
    out["B4"] = SeqModel("B4", bil, kt)
    sidn = SIDNDemand(b.schema, b.x_scaler, b.y_scaler, hidden=64, dropout=0.10, device="cpu")
    sidn.fit(tr.X_cont, tr.X_cat, tr.y, epochs=300, batch_size=64, lr=5e-4, weight_decay=1e-5,
             patience=25, seed=seed)
    out["A1"] = SrcModel("A1", sidn)
    return out


# ----------------------------------------------------------------------------- week interface
def week_rows(kt, split, wd):
    """Context rows of a week (IJ, n_cont) in original units, categorical rows,
    and the week_start value of each row."""
    Xc = split.X_cont_u[wd.rows].astype(float)
    Xk = split.X_cat[wd.rows]
    ws = split.panel_df["week_start"].values[wd.rows]
    return Xc, Xk, ws


def week_demand_fn(model, kt, econ, Xc, Xk, ws):
    """Batched demand function X (B, n) -> (B, I, J) for one planning week."""
    I, J = kt.I, kt.J
    dec = kt.dec

    def fn(X):
        p, b, u = econ.unpack(X)
        B = len(X)
        C = np.repeat(Xc[None], B, axis=0)
        C[:, :, dec["p"]] = p.reshape(B, -1)
        for k, key in enumerate(("b_s", "b_m", "b_r")):
            C[:, :, dec[key]] = b[:, k].reshape(B, -1)
        for k, key in enumerate(("u_s", "u_m", "u_r")):
            C[:, :, dec[key]] = np.repeat(u[:, k], J, axis=1)
        D = model.predict_rows(C.reshape(B * I * J, -1), np.tile(Xk, (B, 1)), np.tile(ws, B))
        return np.maximum(np.asarray(D, float).reshape(B, I, J), 0.0)

    return fn


# ----------------------------------------------------------------------------- conformal layer
class Conformal:
    """Joint (weekly), two-sided, pair-normalised conformal band on log(1 + D):

        |log1p(D_ij) - log1p(Dhat_ij)| <= q * sigma_ij   for all pairs,

    with sigma_ij the training residual scale of the pair and q the
    ceil((n+1)(1-alpha))-th smallest weekly max-score on the validation weeks.
    """

    def __init__(self, model, kt, alpha=0.10):
        self.model = model
        self.alpha = alpha
        b = kt.bundle
        I, J = kt.I, kt.J
        res = []
        for wd in kt.train_weeks:
            Xc, Xk, ws = week_rows(kt, b.train, wd)
            yhat = model.predict_rows(Xc, Xk, ws)
            res.append((np.log1p(b.train.y_u[wd.rows]) - np.log1p(np.maximum(yhat, 0))).reshape(I, J))
        self.sigma = np.maximum(np.std(np.array(res), axis=0), 0.02)
        scores = []
        for wd in kt.val_weeks:
            Xc, Xk, ws = week_rows(kt, b.val, wd)
            yhat = model.predict_rows(Xc, Xk, ws)
            r = np.abs(np.log1p(b.val.y_u[wd.rows]) - np.log1p(np.maximum(yhat, 0))).reshape(I, J)
            scores.append(np.max(r / self.sigma))
        self.scores_chrono = list(scores)          # validation weeks in time order
        self.q = self.quantile(scores, alpha)
        self.n_cal = len(scores)

    @staticmethod
    def quantile(scores, alpha):
        s = np.sort(np.asarray(scores, float))
        n = len(s)
        k = math.ceil((n + 1) * (1 - alpha))
        return float(s[k - 1]) if k <= n else float("inf")

    def band(self):
        return self.q * self.sigma

    def lower(self, D):
        return np.maximum((D + 1.0) * np.exp(-self.band()) - 1.0, 0.0)

    def upper(self, D):
        return (D + 1.0) * np.exp(self.band()) - 1.0


class RollingConformal(Conformal):
    """Weekly re-calibration for deployment: the calibration set is a sliding
    window of the most recent n weekly scores. It starts with the validation
    weeks; after each planning week, the score observed at the implemented plan
    (realised demand versus the model's forecast at that plan) is appended."""

    def __init__(self, base: Conformal):
        self.model = base.model
        self.alpha = base.alpha
        self.sigma = base.sigma
        self.n_cal = base.n_cal
        self.window = list(base.scores_chrono)
        self.q = base.q

    def update(self, realised, forecast):
        s = float(np.max(np.abs(np.log1p(realised) - np.log1p(np.maximum(forecast, 0))) / self.sigma))
        self.window = (self.window + [s])[-self.n_cal:]
        self.q = self.quantile(self.window, self.alpha)


class ScaledConformal:
    """The band of a calibrated Conformal object scaled by a conservatism level
    rho in [0, 1] (rho = 1 carries the guarantee; rho < 1 trades coverage for profit)."""

    def __init__(self, base: Conformal, rho: float):
        self.base = base
        self.rho = rho
        self.sigma = base.sigma
        self.q = base.q * rho

    def band(self):
        return self.q * self.sigma

    def lower(self, D):
        return np.maximum((D + 1.0) * np.exp(-self.band()) - 1.0, 0.0)

    def upper(self, D):
        return (D + 1.0) * np.exp(self.band()) - 1.0
