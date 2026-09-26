"""
Sensitivity runs of the known-truth benchmark.

Two parameters of the benchmark are varied one at a time around the base
setting of the main runs (historical price variation sigma_p = 0.15, demand
noise sigma = 0.10):

    sp0.05, sp0.30   historical price variation sigma_p (families F1 and F2)
    sd0.05, sd0.20   demand noise sigma                 (family F1)

Everything else (truth parameters, operational limits, models, conformal layer,
planner, metrics) is identical to kt_run.py, and all models are seeded, so the
base setting is the main run itself. To save time only the models needed for
the sensitivity analysis are fitted and planned: B0f, B2m, A1, and the conformal
robust plans A4 (SIDN) and B2c (XGBoost).

Usage:  python kt_run_sens.py --workers 15 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "results" / "known_truth_sens"

MODELS = ["B0f", "B2m", "A1", "A4", "B2c"]
SETTINGS = {                       # tag -> (sigma_p, sigma, families)
    "sp0.05": (0.05, 0.10, ("F1", "F2")),
    "sp0.30": (0.30, 0.10, ("F1", "F2")),
    "sd0.05": (0.15, 0.05, ("F1",)),
    "sd0.20": (0.15, 0.20, ("F1",)),
}


def fit_subset(kt, seed):
    """B0f, B2, B2m, A1 exactly as in kt_models.fit_all (each fit is seeded)."""
    import torch
    from kt_models import FixedFormStructural, SrcModel, monotone_xgb
    from src.models.baselines import XGBoostDemand
    from src.models.sidn import SIDNDemand
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = kt.bundle
    tr = b.train
    out = {"B0f": FixedFormStructural(kt).fit(seed)}
    xgb = XGBoostDemand(b.schema, b.x_scaler, b.y_scaler)
    xgb.model.set_params(n_jobs=1)
    xgb.fit(tr.X_cont, tr.X_cat, tr.y, seed=seed)
    out["B2"] = SrcModel("B2", xgb)
    out["B2m"] = monotone_xgb(kt, seed)
    sidn = SIDNDemand(b.schema, b.x_scaler, b.y_scaler, hidden=64, dropout=0.10, device="cpu")
    sidn.fit(tr.X_cont, tr.X_cat, tr.y, epochs=300, batch_size=64, lr=5e-4, weight_decay=1e-5,
             patience=25, seed=seed)
    out["A1"] = SrcModel("A1", sidn)
    return out


def run_task(ds, fam, seed, sigma_p, sigma, alpha=0.10, max_weeks=None):
    import torch
    torch.set_num_threads(1)
    from kt_data import build
    from kt_models import Conformal, week_rows, week_demand_fn
    from kt_run import get_econ
    from kt_solver import Planner, default_starts

    t0 = time.time()
    kt = build(ds, fam, seed, sigma_p=sigma_p, sigma=sigma)
    econ = get_econ(kt)                      # the limits of the main runs (cached per panel)
    models = fit_subset(kt, seed)
    t_fit = time.time() - t0
    conf = {"A4": Conformal(models["A1"], kt, alpha), "B2c": Conformal(models["B2"], kt, alpha)}
    plan_models = {"B0f": (models["B0f"], None), "B2m": (models["B2m"], None), "A1": (models["A1"], None),
                   "A4": (models["A1"], conf["A4"]), "B2c": (models["B2"], conf["B2c"])}
    tru = kt.truth
    I, J = kt.I, kt.J
    IJ = I * J
    test = kt.bundle.test
    rec = {m: {k: [] for k in ("NR", "viol", "cover", "cover_hist", "dist_p", "elast_err", "elast_bias",
                               "mape", "price_loss", "feas_planned", "profit_true")} for m in MODELS}
    oracle = {"pi_star": [], "viol": [], "woy": []}
    for wd in kt.test_weeks[:max_weeks]:
        Xc, Xk, ws = week_rows(kt, test, wd)
        true_fn = (lambda X, wd=wd: wd.L * tru.response(*econ.unpack(X), wd.woy)
                   * np.exp(tru.sigma ** 2 / 2))
        orc = Planner(econ, true_fn, fd_step=1e-5).solve(default_starts(econ, kt.b_hist, extra=3, seed=seed))
        x_or = orc["x"]
        pi_star = float(econ.profit(x_or[None], true_fn(x_or[None]))[0])
        D_or_real = wd.L * tru.response(*econ.unpack(x_or[None]), wd.woy)[0] * np.exp(wd.eps_real)
        oracle["pi_star"].append(pi_star)
        oracle["viol"].append(bool(np.any(econ.constraints(x_or[None], D_or_real[None])[0][I:] < -1e-9)))
        oracle["woy"].append(wd.woy)
        y_hist = test.y_u[wd.rows].astype(float)
        live = wd.L > 0.5
        for name in MODELS:
            m, cf = plan_models[name]
            fn = week_demand_fn(m, kt, econ, Xc, Xk, ws)
            kw = {} if cf is None else {"obj_transform": cf.lower, "con_transform": cf.upper}
            res = Planner(econ, fn, fd_step=1e-3, **kw).solve(default_starts(econ, kt.b_hist), max_iter=150)
            x = res["x"][None]
            pi = float(econ.profit(x, true_fn(x))[0])
            p, b, u = econ.unpack(x)
            D_real = wd.L * tru.response(p, b, u, wd.woy)[0] * np.exp(wd.eps_real)
            g = econ.constraints(x, D_real[None])[0]
            Dhat = fn(x)[0]
            x_d = x.copy()
            x_d[0, :IJ] *= 1.01
            Dd = fn(x_d)[0]
            el_hat = np.clip(-(np.log(Dd + 1e-9) - np.log(Dhat + 1e-9)) / np.log(1.01), -10.0, 10.0)
            el_true = tru.log_elasticity(p, b, u, wd.woy)[0]
            yhat_hist = np.asarray(m.predict_rows(Xc, Xk, ws), float)
            p_hist = Xc[:, kt.dec["p"]].reshape(I, J)
            x_mix = x_or.copy()
            x_mix[:IJ] = x[0, :IJ]
            pi_mix = float(econ.profit(x_mix[None], true_fn(x_mix[None]))[0])
            r = rec[name]
            r["NR"].append(100.0 * (pi_star - pi) / abs(pi_star))
            r["viol"].append(bool(np.any(g[I:] < -1e-9)))
            r["cover"].append(None if cf is None else
                              bool(np.all(np.abs(np.log1p(D_real) - np.log1p(Dhat)) <= cf.band() + 1e-12)))
            r["cover_hist"].append(None if cf is None else bool(np.all(
                np.abs(np.log1p(y_hist) - np.log1p(np.maximum(yhat_hist, 0))).reshape(I, J)
                <= cf.band() + 1e-12)))
            r["dist_p"].append(float(np.mean(np.abs(np.log(p[0] / p_hist)))))
            r["elast_err"].append(float(np.mean(np.abs(el_hat - el_true)[live])))
            r["elast_bias"].append(float(np.mean((el_hat - el_true)[live])))
            r["mape"].append(float(np.mean(np.abs(yhat_hist - y_hist) / (np.abs(y_hist) + 1.0)) * 100))
            r["price_loss"].append(100.0 * (pi_star - pi_mix) / abs(pi_star))
            r["feas_planned"].append(bool(res["feasible_planned"]))
            r["profit_true"].append(pi)
    return {"dataset": ds, "family": fam, "seed": seed, "alpha": alpha, "sigma_p": sigma_p, "sigma": sigma,
            "q": {k: conf[k].q for k in ("A4", "B2c")}, "n_cal": conf["A4"].n_cal,
            "econ": {"K": econ.K, "A": econ.A.tolist()}, "oracle": oracle, "models": rec,
            "time_fit_s": t_fit, "time_total_s": time.time() - t0}


def _worker(args):
    tag, ds, fam, seed = args
    sp, sd, _ = SETTINGS[tag]
    out = OUT / tag / f"{ds}_{fam}_seed{seed}.json"
    if out.exists():
        return str(out), "cached", 0.0
    t0 = time.time()
    try:
        res = run_task(ds, fam, seed, sp, sd)
        out.write_text(json.dumps(res))
        return str(out), "ok", time.time() - t0
    except Exception:
        (out.parent / f"{ds}_{fam}_seed{seed}.error.txt").write_text(traceback.format_exc())
        return str(out), "error", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dataco", "olist", "synth"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    for tag in a.settings:
        (OUT / tag).mkdir(parents=True, exist_ok=True)
    # seed-major order, so that every completed round adds whole seeds to all settings
    tasks = [(tag, ds, f, s) for s in a.seeds for tag in a.settings for ds in a.datasets
             for f in SETTINGS[tag][2]]
    print(f"{len(tasks)} tasks on {a.workers} workers", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from kt_awake import stay_awake
    print("keep-awake:", stay_awake(), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for n, fu in enumerate(as_completed(futs), 1):
            path, status, dt = fu.result()
            print(f"[{n}/{len(tasks)}] {Path(path).parent.name}/{Path(path).name} {status} {dt:.0f}s "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
