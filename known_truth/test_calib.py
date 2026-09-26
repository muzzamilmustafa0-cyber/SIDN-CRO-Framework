"""Check the calibrated limits: how often do capacity and recycled input bind on
test weeks, and do F1 oracle prices match the closed-form markup where the
demand-dependent constraints are slack?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_data import build                        # noqa: E402
from kt_calib import calibrate_limits            # noqa: E402
from kt_solver import Planner, default_starts    # noqa: E402

for ds in sys.argv[1:] or ["dataco"]:
    kt = build(ds, "F1", seed=42)
    econ = calibrate_limits(kt)
    tru = kt.truth
    print("=" * 78)
    print(f"{ds}: K={econ.K:,.0f} (mean base demand {kt.L_mean.sum():,.0f}), A={np.round(econ.A, 1)}")
    bind_cap = bind_rec = 0
    dev = []
    for wd in kt.test_weeks:
        fn = lambda X, wd=wd: wd.L * tru.response(*econ.unpack(X), wd.woy) * np.exp(tru.sigma ** 2 / 2)
        res = Planner(econ, fn, fd_step=1e-5).solve(default_starts(econ, kt.b_hist, extra=2, seed=1))
        x = res["x"][None]
        g = econ.constraints(x, fn(x))[0]
        cap_b = g[-1] / econ.K < 1e-4
        rec_b = np.any(g[kt.I:-1] / econ.A < 1e-4)
        bind_cap += cap_b
        bind_rec += rec_b
        if not cap_b and not rec_b:
            p, b, u = econ.unpack(x)
            c = econ.c0 * (1 - 0.25 * u.mean(axis=1)[0][:, None])
            p_star = np.clip(tru.beta * c / (tru.beta - 1), 0.6 * econ.p0, 1.6 * econ.p0)
            dev.append(np.max(np.abs(p[0] / p_star - 1)))
    n = len(kt.test_weeks)
    print(f"  capacity binds in {bind_cap}/{n} test weeks, recycled input in {bind_rec}/{n}")
    if dev:
        print(f"  weeks with slack constraints: max |p/p* - 1| = {max(dev):.2e} over {len(dev)} weeks")
