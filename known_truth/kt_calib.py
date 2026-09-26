"""
Operational limits of the benchmark, calibrated once per dataset on the
validation weeks (the most recent history before the test period) under the
matched (F1) truth: capacity K is set to the 30th percentile and recycled-input
availability A_j to the median of the weekly usage at the *unconstrained*
optimum, so that both limits bind in a substantial share of weeks.  The same limits are
then used for every truth family of that dataset.
"""
from __future__ import annotations

import numpy as np

from kt_econ import build_econ
from kt_solver import Planner, default_starts
from kt_truth import draw_truth


def calibrate_limits(kt, every=1, q_cap=0.3, q_rec=0.5):
    econ = build_econ(kt.p0, kt.L_mean, kt.b_hist)
    econ.K = 1e12
    econ.A = np.full(kt.J, 1e12)
    f1 = draw_truth("F1", kt.p0, kt.b_hist, seed=2026, sigma=kt.truth.sigma)
    use_cap, use_rec = [], []
    for wd in kt.val_weeks[::every]:
        fn = lambda X, wd=wd: wd.L * f1.response(*econ.unpack(X), wd.woy) * np.exp(f1.sigma ** 2 / 2)
        res = Planner(econ, fn, fd_step=1e-5).solve(default_starts(econ, kt.b_hist, extra=1, seed=3))
        x = res["x"][None]
        D = fn(x)[0]
        _, _, u = econ.unpack(x)
        use_cap.append(D.sum())
        use_rec.append((u[0].sum(axis=0)[:, None] * D).sum(axis=0))
    econ = build_econ(kt.p0, kt.L_mean, kt.b_hist)
    econ.K = float(np.quantile(use_cap, q_cap))
    econ.A = np.quantile(np.array(use_rec), q_rec, axis=0)
    return econ
