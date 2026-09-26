"""
Planner for the known-truth benchmark.

SLSQP on variables normalised to [0, 1] (the original solver worked on raw
variables of very different scales and could stop far from the optimum), with
one batched demand evaluation per gradient (forward differences), multiple
starting points, and a polishing re-solve from the best point.

A demand function maps a batch of decision vectors X (B, n) to demand (B, I, J).
For robust (conformal) planning, the objective is evaluated at the lower demand
bound and the demand-dependent constraints at the upper bound.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class Planner:
    def __init__(self, econ, demand_fn, fd_step=1e-3, obj_transform=None, con_transform=None):
        self.econ = econ
        self.demand_fn = demand_fn
        self.h = fd_step
        self.lo, self.hi = econ.bounds()
        self.rng = self.hi - self.lo
        self.obj_t = obj_transform or (lambda D: D)
        self.con_t = con_transform or (lambda D: D)
        # constraint scales: budget (I), recycled input (J), capacity (1)
        self.g_scale = np.concatenate([np.maximum(econ.AB, 1.0), np.maximum(econ.A, 1.0),
                                       [max(econ.K, 1.0)]])
        self._cache_key = None
        self.n_eval = 0

    # ---------------------------------------------------------------- core
    def x_of(self, y):
        return self.lo + np.clip(y, 0.0, 1.0) * self.rng

    def _batch(self, Y):
        X = self.lo + np.clip(Y, 0.0, 1.0) * self.rng
        D = self.demand_fn(X)
        self.n_eval += len(X)
        f = self.econ.profit(X, self.obj_t(D))
        g = self.econ.constraints(X, self.con_t(D)) / self.g_scale
        return f, g

    def _evaluate(self, y):
        key = y.tobytes()
        if key == self._cache_key:
            return self._cache
        n = len(y)
        step = np.where(y + self.h <= 1.0, self.h, -self.h)
        Y = np.vstack([y, y + np.diag(step)])
        f, g = self._batch(Y)
        grad_f = (f[1:] - f[0]) / step
        jac_g = ((g[1:] - g[0]) / step[:, None]).T
        self._cache_key = key
        self._cache = (f[0], grad_f, g[0], jac_g)
        return self._cache

    def _solve_from(self, y0, scale, max_iter):
        fun = lambda y: (-self._evaluate(y)[0] / scale, -self._evaluate(y)[1] / scale)
        cons = [{"type": "ineq", "fun": lambda y: self._evaluate(y)[2],
                 "jac": lambda y: self._evaluate(y)[3]}]
        res = minimize(fun, y0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * len(y0),
                       constraints=cons, options={"maxiter": max_iter, "ftol": 1e-10})
        y = np.clip(res.x, 0.0, 1.0)
        f, _, g, _ = self._evaluate(y)
        return y, f, g, res

    def solve(self, starts, max_iter=300, polish=True):
        """starts: list of decision vectors x0 (raw units). Returns dict."""
        Y0 = [np.clip((x0 - self.lo) / self.rng, 0.0, 1.0) for x0 in starts]
        f0, _ = self._batch(np.vstack(Y0))
        scale = max(1.0, float(np.max(np.abs(f0))))
        best = None
        for y0 in Y0:
            y, f, g, res = self._solve_from(y0, scale, max_iter)
            feas = bool(np.all(g >= -1e-6))
            cand = (feas, f, y, g, res)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        if polish:
            y, f, g, res = self._solve_from(best[2], scale, max_iter)
            feas = bool(np.all(g >= -1e-6))
            if (feas, f) >= (best[0], best[1]):
                best = (feas, f, y, g, res)
        feas, f, y, g, res = best
        return {"x": self.x_of(y), "profit_planned": float(f), "feasible_planned": feas,
                "status": int(res.status), "message": str(res.message)}


def default_starts(econ, b_hist, extra=0, seed=0):
    """Common starting points (raw units): reference prices with historical
    advertising, a higher-price start, and optional random interior starts."""
    I, J = econ.I, econ.J
    lo, hi = econ.bounds()
    b0 = np.minimum(b_hist, econ.b_max)
    s1 = econ.pack(econ.p0, b0, np.full((3, I), 0.05))
    s2 = econ.pack(1.25 * econ.p0, np.minimum(2.0 * b0, econ.b_max), np.full((3, I), 0.20))
    starts = [s1, s2]
    rng = np.random.default_rng(seed)
    for _ in range(extra):
        starts.append(lo + rng.uniform(0.1, 0.9, size=len(lo)) * (hi - lo))
    return starts
