"""Sanity checks of the oracle planner on the known-truth benchmark.

1. Where the demand-dependent constraints are slack, the F1 oracle prices must
   equal the closed-form markup beta c / (beta - 1) (clipped to the bounds).
2. Different starting points must reach the same optimum.
3. Report how often capacity and recycled-input constraints bind.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_data import build                       # noqa: E402
from kt_econ import build_econ                  # noqa: E402
from kt_solver import Planner, default_starts   # noqa: E402

for ds in sys.argv[1:] or ["dataco"]:
    for fam in ("F1", "F2", "F3"):
        kt = build(ds, fam, seed=42)
        econ = build_econ(kt.p0, kt.L_mean, kt.b_hist)
        tru = kt.truth
        print("=" * 78)
        print(f"{ds} {fam}: I={kt.I} J={kt.J}, capacity K={econ.K:.0f}, mean weekly base demand "
              f"{kt.L_mean.sum():.0f}; test weeks {len(kt.test_weeks)}")
        for wd in kt.test_weeks[:4]:
            fn = lambda X, wd=wd: wd.L * tru.response(*econ.unpack(X), wd.woy) * np.exp(tru.sigma ** 2 / 2)
            pl = Planner(econ, fn, fd_step=1e-5)
            t0 = time.time()
            res = pl.solve(default_starts(econ, kt.b_hist, extra=3, seed=1))
            dt = time.time() - t0
            # single-start solutions from each start, to check agreement
            profs = []
            for s in default_starts(econ, kt.b_hist, extra=3, seed=1):
                r1 = Planner(econ, fn, fd_step=1e-5).solve([s], polish=False)
                profs.append(r1["profit_planned"])
            x = res["x"][None]
            p, b, u = econ.unpack(x)
            D = fn(x)
            g = econ.constraints(x, D)[0]
            ubar = u.mean(axis=1)[0][:, None]
            c = econ.c0 * (1 - 0.25 * ubar)
            msg = ""
            if fam == "F1":
                p_star = np.clip(tru.beta * c / (tru.beta - 1), 0.6 * econ.p0, 1.6 * econ.p0)
                msg = f"| max |p/p*-1| = {np.max(np.abs(p[0] / p_star - 1)):.3%}"
            slack = g / np.concatenate([econ.AB, econ.A, [econ.K]])
            print(f"  week woy={wd.woy:>4.0f}: profit={res['profit_planned']:>12,.1f} status={res['status']} "
                  f"{dt:4.1f}s | starts spread={(max(profs) - min(profs)) / abs(max(profs)):.2e} "
                  f"| p/p0 in [{(p[0] / econ.p0).min():.2f},{(p[0] / econ.p0).max():.2f}] "
                  f"| u={u.mean():.2f} | cap slack={slack[-1]:+.2f} recyc slack min={slack[kt.I:-1].min():+.2f} "
                  f"budget slack min={slack[:kt.I].min():+.2f} {msg}")
