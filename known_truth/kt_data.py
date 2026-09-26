"""
Semi-synthetic data with a known ground truth.

Real components (from the preprocessed panels): base weekly demand L_ijt
(qty_baseline), calendar and market-context features, product and segment
identifiers, and the historical advertising and recycled-content overlays.
Synthetic components: promotional price variation p = p0 exp(eta),
eta ~ N(0, sigma_p^2) truncated at +-2 sigma_p, the demand response S_f of the
chosen family, and log-normal noise.  Observed demand is
    y_ijt = L_ijt * S_f(z_ijt) * exp(eps_ijt).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.dataset import load_dataset, SplitArrays, DatasetBundle   # noqa: E402
from src.models.base import decision_columns                            # noqa: E402
from kt_truth import draw_truth                                          # noqa: E402

DATASETS = {"dataco": "Preprocessed", "olist": "Preprocessed_Olist", "synth": "Preprocessed_Synth2026"}


@dataclass
class WeekData:
    rows: np.ndarray        # (IJ,) row indices of this week in the split, ordered (i, j)
    L: np.ndarray           # (I, J) true base demand
    woy: float
    eps_real: np.ndarray    # (I, J) noise of the realised demand at the planned decision


@dataclass
class KTData:
    name: str
    family: str
    seed: int
    bundle: DatasetBundle   # scaled/unscaled arrays with synthetic prices and targets
    truth: object
    I: int
    J: int
    p0: np.ndarray
    L_mean: np.ndarray
    b_hist: np.ndarray      # (3, I, J)
    val_weeks: list
    test_weeks: list
    train_weeks: list
    dec: dict


def _pair_index(bundle):
    cats = bundle.schema.cat_feats
    pc = [k for k, c in enumerate(cats) if c.startswith("product_")]
    rc = [k for k, c in enumerate(cats) if c.startswith("retailer_")]
    return pc, rc


def _weeks(split, pc, rc, I, J):
    """Group rows by week, ordered by (product, segment)."""
    out = []
    for w in sorted(np.unique(split.week_idx)):
        idx = np.where(split.week_idx == w)[0]
        if len(idx) != I * J:
            continue
        i = split.X_cat[idx][:, pc].argmax(1)
        j = split.X_cat[idx][:, rc].argmax(1)
        order = np.lexsort((j, i))
        out.append((int(w), idx[order]))
    return out


def build(dataset: str, family: str, seed: int, sigma_p: float = 0.15, sigma: float = 0.10,
          truth_seed: int = 2026) -> KTData:
    base = load_dataset(ROOT / "data" / DATASETS[dataset], name=dataset)
    sch = base.schema
    dec = decision_columns(sch)
    pc, rc = _pair_index(base)
    I, J = len(pc), len(rc)
    splits = {"train": base.train, "val": base.val, "test": base.test}
    weeks = {k: _weeks(s, pc, rc, I, J) for k, s in splits.items()}

    # reference price, historical advertising, and base demand level (training weeks)
    tr = base.train
    P = np.stack([tr.X_cont_u[r, dec["p"]].reshape(I, J) for _, r in weeks["train"]])
    p0 = np.median(P, axis=0)
    b_hist = np.stack([np.mean([tr.X_cont_u[r, dec[k]].reshape(I, J) for _, r in weeks["train"]], axis=0)
                       for k in ("b_s", "b_m", "b_r")])
    L_mean = np.mean([tr.panel_df["qty_baseline"].values[r].reshape(I, J) for _, r in weeks["train"]], axis=0)
    # the same base parameters for every family, so that families differ only in form
    truth = draw_truth(family, p0, b_hist, seed=truth_seed, sigma=sigma)

    rng = np.random.default_rng(1000 * seed + 17)
    new = {}
    week_data = {}
    for k, s in splits.items():
        Xu = s.X_cont_u.astype(np.float64).copy()
        y = np.zeros(len(Xu))
        wd = []
        for w, r in weeks[k]:
            eta = np.clip(rng.normal(0.0, sigma_p, size=(I, J)), -2 * sigma_p, 2 * sigma_p)
            p = p0 * np.exp(eta)
            Xu[r, dec["p"]] = p.ravel()
            b = np.stack([Xu[r, dec[c]].reshape(I, J) for c in ("b_s", "b_m", "b_r")])
            u = np.stack([Xu[r, dec[c]].reshape(I, J)[:, 0] for c in ("u_s", "u_m", "u_r")])
            L = s.panel_df["qty_baseline"].values[r].reshape(I, J).astype(float)
            woy = float(s.panel_df["weekofyear"].values[r][0])
            S = truth.response(p[None], b[None], u[None], woy)[0]
            eps = rng.normal(0.0, sigma, size=(I, J))
            y[r] = (L * S * np.exp(eps)).ravel()
            wd.append(WeekData(rows=r, L=L, woy=woy, eps_real=rng.normal(0.0, sigma, size=(I, J))))
        new[k] = (Xu, y)
        week_data[k] = wd

    x_scaler = StandardScaler().fit(new["train"][0])
    y_scaler = StandardScaler().fit(new["train"][1].reshape(-1, 1))

    def mk(k):
        s = splits[k]
        Xu, y = new[k]
        return SplitArrays(X_cont=x_scaler.transform(Xu).astype(np.float32), X_cat=s.X_cat,
                           y=y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32),
                           X_cont_u=Xu.astype(np.float32), y_u=y.astype(np.float32),
                           week_idx=s.week_idx, panel_df=s.panel_df)

    bundle = DatasetBundle(name=dataset, preprocessed_dir=base.preprocessed_dir, schema=sch,
                           x_scaler=x_scaler, y_scaler=y_scaler,
                           train=mk("train"), val=mk("val"), test=mk("test"))
    return KTData(name=dataset, family=family, seed=seed, bundle=bundle, truth=truth, I=I, J=J,
                  p0=p0, L_mean=L_mean, b_hist=b_hist, val_weeks=week_data["val"],
                  test_weeks=week_data["test"], train_weeks=week_data["train"], dec=dec)
