"""
Runner for SIDN-P (partial pooling) on the known-truth benchmark.

Identical data, economics, oracle, planner, and metrics to kt_run.py; the models
are A1p (SIDN-P), A4p (A1p + CRO, rho = 1), A4rp (rolling recalibration), and
A4p_50 (rho = 0.5). Results: results/known_truth_pool/{ds}_{fam}_seed{seed}.json
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
OUT = HERE.parent / "results" / "known_truth_pool"
MODELS = ["A1p", "A4p", "A4rp", "A4p_50"]


def run_task(ds, fam, seed, alpha=0.10):
    import torch
    torch.set_num_threads(1)
    from kt_data import build
    from kt_run import get_econ
    from kt_models import (FixedFormStructural, SrcModel, Conformal, RollingConformal, ScaledConformal,
                           week_rows, week_demand_fn)
    from kt_pool import fit_sidn_pooled
    from kt_solver import Planner, default_starts

    t0 = time.time()
    kt = build(ds, fam, seed)
    econ = get_econ(kt)
    torch.manual_seed(seed)
    np.random.seed(seed)
    b0f = FixedFormStructural(kt).fit(seed)
    sidn_p, pooled = fit_sidn_pooled(kt, b0f, seed)
    a1p = SrcModel("A1p", sidn_p)
    base = Conformal(a1p, kt, alpha)
    conf = {"A4p": base, "A4rp": RollingConformal(base), "A4p_50": ScaledConformal(base, 0.5)}
    plan_models = {"A1p": (a1p, None), "A4p": (a1p, conf["A4p"]), "A4rp": (a1p, conf["A4rp"]),
                   "A4p_50": (a1p, conf["A4p_50"])}
    t_fit = time.time() - t0
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
            if name == "A4rp":
                r.setdefault("q_path", []).append(cf.q)
                cf.update(D_real, Dhat)
    return {"dataset": ds, "family": fam, "seed": seed, "alpha": alpha, "lam": sidn_p.lam,
            "pooled": pooled, "q": {"A4p": base.q}, "n_cal": base.n_cal, "oracle": oracle,
            "models": rec, "time_fit_s": t_fit, "time_total_s": time.time() - t0}


def _worker(args):
    ds, fam, seed = args
    out = OUT / f"{ds}_{fam}_seed{seed}.json"
    if out.exists():
        return str(out), "cached", 0.0
    t0 = time.time()
    try:
        out.write_text(json.dumps(run_task(ds, fam, seed)))
        return str(out), "ok", time.time() - t0
    except Exception:
        (OUT / f"{ds}_{fam}_seed{seed}.error.txt").write_text(traceback.format_exc())
        return str(out), "error", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dataco", "olist", "synth"])
    ap.add_argument("--families", nargs="+", default=["F1", "F2", "F3", "F4", "F5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--reverse", action="store_true",
                    help="process the task list from the end (to run beside another runner)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
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
