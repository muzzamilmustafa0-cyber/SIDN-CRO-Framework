"""
Known-truth benchmark runner.

For every (dataset, truth family, seed): build the semi-synthetic data, train
all models, calibrate the conformal layer on the validation weeks, and for each
test week solve (i) the oracle plan under the true expected demand and (ii) each
model's plan; then score each plan under the truth.

Per model and test week it records
    NR           100 (pi* - pi(x_hat)) / |pi*|, both under the true expected demand
    viol         the plan violates a demand-dependent limit (recycled input or
                 capacity) under the realised demand
    cover        the realised demand lies in the conformal band at the plan (CRO)
    elast_err    mean |model price elasticity - true elasticity| at the plan
    mape         forecast error at the historical decisions of the week
    price_loss   100 (pi* - pi(oracle plan with the model's prices)) / |pi*|

Usage:  python kt_run.py --workers 14 --seeds 42 43 44 45 46
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
OUT = HERE.parent / "results" / "known_truth"

MODELS = ["B0f", "B1", "B2", "B2m", "B4", "A1", "A4", "A4r", "A4_75", "A4_50", "A4_25", "B2c"]


def _econ_cache(ds):
    return OUT / f"econ_{ds}.json"


def get_econ(kt):
    from kt_calib import calibrate_limits
    from kt_econ import build_econ
    f = _econ_cache(kt.name)
    econ = build_econ(kt.p0, kt.L_mean, kt.b_hist)
    if f.exists():
        d = json.loads(f.read_text())
        econ.K = d["K"]
        econ.A = np.array(d["A"])
        return econ
    econ = calibrate_limits(kt)
    f.write_text(json.dumps({"K": econ.K, "A": econ.A.tolist()}))
    return econ


def run_task(ds, fam, seed, alpha=0.10):
    import torch
    torch.set_num_threads(1)
    from kt_data import build
    from kt_models import (fit_all, Conformal, RollingConformal, ScaledConformal, week_rows,
                           week_demand_fn)
    from kt_solver import Planner, default_starts

    t0 = time.time()
    kt = build(ds, fam, seed)
    econ = get_econ(kt)
    models = fit_all(kt, seed)
    t_fit = time.time() - t0
    conf = {"A4": Conformal(models["A1"], kt, alpha), "B2c": Conformal(models["B2"], kt, alpha)}
    conf["A4r"] = RollingConformal(conf["A4"])
    for rho in (75, 50, 25):
        conf[f"A4_{rho}"] = ScaledConformal(conf["A4"], rho / 100)
    plan_models = {"B0f": (models["B0f"], None), "B1": (models["B1"], None), "B2": (models["B2"], None),
                   "B2m": (models["B2m"], None), "B4": (models["B4"], None), "A1": (models["A1"], None),
                   "A4": (models["A1"], conf["A4"]), "A4r": (models["A1"], conf["A4r"]),
                   "A4_75": (models["A1"], conf["A4_75"]), "A4_50": (models["A1"], conf["A4_50"]),
                   "A4_25": (models["A1"], conf["A4_25"]), "B2c": (models["B2"], conf["B2c"])}
    tru = kt.truth
    I, J = kt.I, kt.J
    IJ = I * J
    test = kt.bundle.test
    rec = {m: {k: [] for k in ("NR", "viol", "cover", "cover_hist", "dist_p", "elast_err", "elast_bias",
                               "mape", "price_loss", "feas_planned", "profit_true")} for m in MODELS}
    oracle = {"pi_star": [], "viol": [], "woy": []}
    for wd in kt.test_weeks:
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
            # coverage at the week's historical decisions isolates temporal shift from decision shift
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
            if name == "A4r":
                r.setdefault("q_path", []).append(cf.q)
                cf.update(D_real, Dhat)
    return {"dataset": ds, "family": fam, "seed": seed, "alpha": alpha,
            "q": {k: conf[k].q for k in ("A4", "B2c")}, "n_cal": conf["A4"].n_cal,
            "econ": {"K": econ.K, "A": econ.A.tolist()}, "oracle": oracle, "models": rec,
            "time_fit_s": t_fit, "time_total_s": time.time() - t0}


def _worker(args):
    ds, fam, seed = args
    out = OUT / f"{ds}_{fam}_seed{seed}.json"
    if out.exists():
        return str(out), "cached", 0.0
    t0 = time.time()
    try:
        res = run_task(ds, fam, seed)
        out.write_text(json.dumps(res))
        return str(out), "ok", time.time() - t0
    except Exception:
        (OUT / f"{ds}_{fam}_seed{seed}.error.txt").write_text(traceback.format_exc())
        return str(out), "error", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dataco", "olist", "synth"])
    ap.add_argument("--families", nargs="+", default=["F1", "F2", "F3", "F4", "F5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reverse", action="store_true",
                    help="process the task list from the end (to run beside another runner)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # calibrate the operational limits once per dataset before the parallel run
    from kt_data import build
    for ds in a.datasets:
        get_econ(build(ds, "F1", a.seeds[0]))
    # seed-major order: every completed round of workers adds whole seeds to all cells
    tasks = [(ds, f, s) for s in a.seeds for ds in a.datasets for f in a.families]
    if a.reverse:
        tasks = tasks[::-1]
    print(f"{len(tasks)} tasks on {a.workers} workers", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from kt_awake import stay_awake
    print('keep-awake:', stay_awake(), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for n, fu in enumerate(as_completed(futs), 1):
            path, status, dt = fu.result()
            print(f"[{n}/{len(tasks)}] {Path(path).name} {status} {dt:.0f}s "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
