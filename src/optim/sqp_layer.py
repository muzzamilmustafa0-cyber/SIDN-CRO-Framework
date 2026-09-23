"""
SIDN-CRO -- Optimisation core (Step 2a).

Re-implementation in pure NumPy / SciPy of the centralised three-echelon
profit-maximisation problem from the companion paper (Salman et al., under
review at Scientific Reports), restructured so that the demand model is a
pluggable callable.  This allows the same solver to be invoked with:

    *   the closed-form analytical demand of the original paper
    *   any of the ML baselines (XGBoost, LSTM, Bi-LSTM+Attention, MLP, ...)
    *   the Structure-Informed Demand Network (SIDN) developed in
        src/models/sidn.py

Decision variables (per planning period, for J retailers serving I products)
---------------------------------------------------------------------------
For each (product i, retailer j) pair we expose the following decisions:

    p_ij    >= p_min        retail selling price                       [USD]
    b_s_ij  >= 0            supplier ad spend earmarked for retailer  [USD]
    b_m_ij  >= 0            manufacturer ad spend                      [USD]
    b_r_ij  >= 0            retailer's own ad spend                    [USD]

Plus three echelon-level circularity decisions per product:

    u_s_i, u_m_i, u_r_i  in [0, 1]    circular-input fractions

Cycle time T, integer multiples z1/z2 and shipment counts y1/y2 are taken
as fixed parameters at this stage (they are exogenous in the SCM model and
will be re-introduced in the supplementary mixed-integer extension).

Objective
---------
        maximise  TP =  sum_ij [ p_ij - C_var_ij(u_s,u_m,u_r) ] * D_ij
                       + sum_i  [ Inc_s + Inc_m + Inc_r ]
                       - sum_ij [ b_s_ij + b_m_ij + b_r_ij ]
                       - sum_i  [ Penalty_i (u_s, u_m, u_r) ]
                       - FixedCosts

where the demand D_ij is supplied by the pluggable demand model.

Constraints
-----------
    *   Advertising-budget cap          sum_j (b_s_ij + b_m_ij + b_r_ij)
                                          <=  AdBudget_i        for each i
    *   Recycled-input availability     sum_i (u_s_i + u_m_i + u_r_i) D_ij
                                          <=  A_R               for each j
    *   Capacity                        sum_ij D_ij  <=  Capacity
    *   Box bounds on prices, ads, circular fractions

Solver
------
SciPy SLSQP with analytical Jacobians where possible, finite-difference
fall-back where not.  We deliberately do NOT use SLSQP for the *integer*
parts (z1, z2, y1, y2) -- those are fixed at the original-paper values for
the ML benchmark.

References
----------
Salman et al. (under review, Scientific Reports).  Khorshidvand, Soleimani,
Sibdari & Esfahani (2021).  Aust & Buscher (2014, IJPE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.optimize import (minimize, OptimizeResult, NonlinearConstraint,
                            LinearConstraint)


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
@dataclass
class SCMParameters:
    """Problem parameters for one planning period.

    Indices
    -------
    I = number of products,  J = number of retailers.
    """
    I: int                                                # products
    J: int                                                # retailers
    # Cost coefficients ----------------------------------------------------
    C_s:        np.ndarray                                # (I,)  supplier base unit cost (USD/unit)
    C_m:        np.ndarray                                # (I,)  manufacturer unit cost
    C_r:        np.ndarray                                # (I,J) retailer unit handling cost
    CrCAd:      np.ndarray                                # (I,)  circular cost adjustment ($/unit per unit u)
    h_s:        np.ndarray                                # (I,)  supplier holding cost
    h_m:        np.ndarray                                # (I,)  manufacturer holding cost
    h_r:        np.ndarray                                # (I,J) retailer holding cost
    A_s:        np.ndarray                                # (I,)  supplier setup cost
    A_m:        np.ndarray                                # (I,)  manufacturer setup cost
    A_r:        np.ndarray                                # (I,J) retailer ordering cost
    # Penalty coefficients -------------------------------------------------
    c2_s:       np.ndarray                                # (I,)
    c2_m:       np.ndarray                                # (I,)
    c2_r:       np.ndarray                                # (I,J)
    # Incentives -----------------------------------------------------------
    inc_rate:   float                                     # USD/unit
    alpha_inc:  float                                     # circular-effect gain
    # Limits ---------------------------------------------------------------
    A_R:        np.ndarray                                # (J,)  recycled-input availability per retailer
    AdBudget:   np.ndarray                                # (I,)  per-product cooperative ad budget cap
    Capacity:   float                                     # total throughput cap
    # Box bounds -----------------------------------------------------------
    p_lo:       np.ndarray                                # (I,J) lower price
    p_hi:       np.ndarray                                # (I,J) upper price
    b_hi:       np.ndarray                                # (I,J) per-(i,j) cooperative ad upper bound
    u_hi:       float = 0.50                              # max circular fraction per echelon


@dataclass
class SCMSolution:
    """Solution returned by `solve_sqp`."""
    p:        np.ndarray         # (I,J)
    b_s:      np.ndarray         # (I,J)
    b_m:      np.ndarray         # (I,J)
    b_r:      np.ndarray         # (I,J)
    u_s:      np.ndarray         # (I,)
    u_m:      np.ndarray         # (I,)
    u_r:      np.ndarray         # (I,)
    profit:   float
    demand:   np.ndarray         # (I,J)
    feasible: bool
    raw:      OptimizeResult


# --------------------------------------------------------------------------- #
# Variable packing / unpacking
# --------------------------------------------------------------------------- #
def _layout(params: SCMParameters) -> Dict[str, slice]:
    I, J = params.I, params.J
    n_pij = I * J          # p, b_s, b_m, b_r each
    o = 0
    sl = {"p":   slice(o,           o + n_pij)};    o += n_pij
    sl["b_s"] = slice(o,           o + n_pij);     o += n_pij
    sl["b_m"] = slice(o,           o + n_pij);     o += n_pij
    sl["b_r"] = slice(o,           o + n_pij);     o += n_pij
    sl["u_s"] = slice(o,           o + I);          o += I
    sl["u_m"] = slice(o,           o + I);          o += I
    sl["u_r"] = slice(o,           o + I);          o += I
    sl["__total__"] = slice(0, o)
    return sl


def _pack(p, b_s, b_m, b_r, u_s, u_m, u_r) -> np.ndarray:
    return np.concatenate([p.ravel(), b_s.ravel(), b_m.ravel(), b_r.ravel(),
                           u_s, u_m, u_r])


def _unpack(x: np.ndarray, params: SCMParameters):
    I, J = params.I, params.J
    sl = _layout(params)
    p   = x[sl["p"]].reshape(I, J)
    b_s = x[sl["b_s"]].reshape(I, J)
    b_m = x[sl["b_m"]].reshape(I, J)
    b_r = x[sl["b_r"]].reshape(I, J)
    u_s = x[sl["u_s"]]
    u_m = x[sl["u_m"]]
    u_r = x[sl["u_r"]]
    return p, b_s, b_m, b_r, u_s, u_m, u_r


# --------------------------------------------------------------------------- #
# Closed-form analytical demand (Khorshidvand-style, original paper eq. 9-10)
# --------------------------------------------------------------------------- #
@dataclass
class AnalyticalDemandModel:
    """Reference demand model implementing Khorshidvand et al. eqs. (9), (10)
    and the multiplicative circular uplift of the companion paper eq. (7)."""
    D0:        np.ndarray             # (I,J) baseline demand
    p0:        np.ndarray             # (I,J) reference price
    beta:      float = 0.40           # price elasticity
    k1:        float = 0.030
    k2:        float = 0.030
    k3:        float = 0.040
    alpha_inc: float = 0.80

    def __call__(self, p, b_s, b_m, b_r, u_s, u_m, u_r,
                 context: Optional[np.ndarray] = None) -> np.ndarray:
        I, J = self.D0.shape
        ad_eff = (1.0
                  + self.k1 * np.log1p(b_s)
                  + self.k2 * np.log1p(b_m)
                  + self.k3 * np.log1p(b_r))
        # Cir_eff is per-product (depends only on echelon-level u's)
        avg_circ = (u_s + u_m + u_r) / 3.0      # (I,)
        cir_eff  = (1.0 + self.alpha_inc * avg_circ).reshape(I, 1)   # broadcast over J
        elast    = (p / self.p0) ** (-self.beta)
        return self.D0 * elast * ad_eff * cir_eff


# --------------------------------------------------------------------------- #
# Objective and constraints
# --------------------------------------------------------------------------- #
def _objective(x: np.ndarray, params: SCMParameters,
               demand_fn: Callable, context, sign: float = -1.0) -> float:
    """Returns *negative* total profit (because SciPy minimises)."""
    p, b_s, b_m, b_r, u_s, u_m, u_r = _unpack(x, params)

    D = demand_fn(p, b_s, b_m, b_r, u_s, u_m, u_r, context=context)

    # Variable cost per product (supplier+manuf+retailer) and circular adj.
    # C_var_ij = C_s_i + C_m_i + C_r_ij + CrCAd_i * (u_s+u_m+u_r)/3
    avg_u = (u_s + u_m + u_r) / 3.0
    base_cost = (params.C_s + params.C_m).reshape(-1, 1) + params.C_r        # (I,J)
    circ_cost = params.CrCAd.reshape(-1, 1) * avg_u.reshape(-1, 1)           # (I,1)
    C_var = base_cost + circ_cost                                            # (I,J)

    revenue   = (p * D).sum()
    var_cost  = (C_var * D).sum()
    ad_cost   = b_s.sum() + b_m.sum() + b_r.sum()

    # Cooperative incentives (paid per circular unit served, eq. (3)-(6))
    Inc = (params.inc_rate
           * (u_s + u_m + u_r).reshape(-1, 1)
           * D).sum()

    # Convex penalty on high circular fractions (eq. 8)
    pen_s = (params.c2_s * u_s ** 2).sum()
    pen_m = (params.c2_m * u_m ** 2).sum()
    pen_r = (params.c2_r * u_r.reshape(-1, 1) ** 2).sum()
    Penalty = pen_s + pen_m + pen_r

    # Inventory / setup fixed costs (constant w.r.t. continuous decisions
    # in this stage of the model -- they are scaled by D as proxy for
    # throughput-related holding/setup).
    holding  = ((params.h_s + params.h_m).reshape(-1, 1) * D).sum() * 0.05
    holding += (params.h_r * D).sum() * 0.05

    profit = revenue - var_cost - ad_cost - Penalty + Inc - holding
    return sign * profit


def _make_constraints(params: SCMParameters, demand_fn: Callable, context):
    cons = []

    # Advertising budget per product:  sum_j (b_s_ij + b_m_ij + b_r_ij) <= AdBudget_i
    for i in range(params.I):
        def fun_ad(x, i=i):
            _, b_s, b_m, b_r, *_ = _unpack(x, params)
            return params.AdBudget[i] - (b_s[i].sum() + b_m[i].sum() + b_r[i].sum())
        cons.append({"type": "ineq", "fun": fun_ad})

    # Recycled-input availability per retailer:
    # sum_i (u_s+u_m+u_r) * D_ij <= A_R[j]
    for j in range(params.J):
        def fun_avail(x, j=j):
            p, b_s, b_m, b_r, u_s, u_m, u_r = _unpack(x, params)
            D = demand_fn(p, b_s, b_m, b_r, u_s, u_m, u_r, context=context)
            return params.A_R[j] - ((u_s + u_m + u_r) * D[:, j]).sum()
        cons.append({"type": "ineq", "fun": fun_avail})

    # Total capacity
    def fun_cap(x):
        p, b_s, b_m, b_r, u_s, u_m, u_r = _unpack(x, params)
        D = demand_fn(p, b_s, b_m, b_r, u_s, u_m, u_r, context=context)
        return params.Capacity - D.sum()
    cons.append({"type": "ineq", "fun": fun_cap})

    return cons


def _make_bounds(params: SCMParameters):
    I, J = params.I, params.J
    bs = []
    # p
    for i in range(I):
        for j in range(J):
            bs.append((float(params.p_lo[i, j]), float(params.p_hi[i, j])))
    # b_s, b_m, b_r each
    for _ in range(3):
        for i in range(I):
            for j in range(J):
                bs.append((0.0, float(params.b_hi[i, j])))
    # u_s, u_m, u_r each
    for _ in range(3):
        for _ in range(I):
            bs.append((0.0, float(params.u_hi)))
    return bs


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def solve_sqp(params: SCMParameters,
              demand_fn: Callable,
              context=None,
              x0: Optional[np.ndarray] = None,
              max_iter: int = 400,
              tol: float = 1e-6,
              verbose: bool = False,
              method: str = "trust-constr",
              objective_scale: Optional[float] = None) -> SCMSolution:
    """Solve the centralised SCM profit-maximisation.

    Two solvers are exposed:

    * ``method="trust-constr"`` (default) -- SciPy's interior-point trust-
      region method.  More robust on nonlinear constraints than SLSQP,
      especially when constraint Jacobians are estimated by finite
      differences.

    * ``method="SLSQP"``                   -- classic SQP, kept for
      benchmarking (this is the solver used by the original paper).

    The objective is automatically scaled to keep magnitudes in [-1, 1]
    so the line search and trust-region radii behave well.
    """
    I, J = params.I, params.J

    if x0 is None:
        p0   = 0.5 * (params.p_lo + params.p_hi)
        b0   = 0.10 * params.b_hi
        u0   = np.full(I, 0.05)
        x0   = _pack(p0, b0, b0, b0, u0, u0, u0)

    bounds = _make_bounds(params)

    if objective_scale is None:
        # estimate scale from a quick evaluation at x0
        prof0 = -_objective(x0, params, demand_fn, context, sign=-1.0)
        objective_scale = max(1.0, abs(prof0))

    sign = -1.0 / objective_scale

    if method == "SLSQP":
        cons = _make_constraints(params, demand_fn, context)
        res = minimize(_objective, x0,
                       args=(params, demand_fn, context, sign),
                       method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": max_iter, "ftol": tol,
                                "disp": verbose})
    elif method == "trust-constr":
        # convert constraint dicts -> NonlinearConstraint objects
        # (trust-constr supports vectorised constraints with bounds)
        nl_cons = _make_constraints_trust(params, demand_fn, context)
        from scipy.optimize import Bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        res = minimize(_objective, x0,
                       args=(params, demand_fn, context, sign),
                       method="trust-constr",
                       bounds=Bounds(lb, ub),
                       constraints=nl_cons,
                       options={"maxiter": max_iter, "xtol": tol,
                                "gtol": tol * 10, "verbose": int(verbose) * 2})
    else:
        raise ValueError(f"unknown method {method!r}")

    p, b_s, b_m, b_r, u_s, u_m, u_r = _unpack(res.x, params)
    D = demand_fn(p, b_s, b_m, b_r, u_s, u_m, u_r, context=context)

    profit = -float(res.fun) * objective_scale

    # feasibility: re-evaluate using dict-style constraints (clean check)
    cons_check = _make_constraints(params, demand_fn, context)
    feasible = all(c["fun"](res.x) >= -tol * 100 for c in cons_check)

    return SCMSolution(p=p, b_s=b_s, b_m=b_m, b_r=b_r,
                       u_s=u_s, u_m=u_m, u_r=u_r,
                       profit=profit, demand=D,
                       feasible=feasible, raw=res)


def _make_constraints_trust(params: SCMParameters, demand_fn, context):
    """Vectorised NonlinearConstraint objects for trust-constr."""
    I, J = params.I, params.J

    def all_g(x):
        p, b_s, b_m, b_r, u_s, u_m, u_r = _unpack(x, params)
        D = demand_fn(p, b_s, b_m, b_r, u_s, u_m, u_r, context=context)
        # I ad-budget + J availability + 1 capacity = I + J + 1
        out = np.empty(I + J + 1)
        for i in range(I):
            out[i] = (params.AdBudget[i]
                      - (b_s[i].sum() + b_m[i].sum() + b_r[i].sum()))
        for j in range(J):
            out[I + j] = (params.A_R[j]
                          - ((u_s + u_m + u_r) * D[:, j]).sum())
        out[I + J] = params.Capacity - D.sum()
        return out

    n = I + J + 1
    return NonlinearConstraint(all_g, lb=np.zeros(n), ub=np.full(n, np.inf))


# --------------------------------------------------------------------------- #
# Convenience -- build SCMParameters from preprocessed-panel mean values
# --------------------------------------------------------------------------- #
def params_from_panel_mean(panel_df, n_products: int, n_retailers: int,
                           rng_seed: int = 0) -> SCMParameters:
    """Build a default SCMParameters object using the mean values observed in
    a preprocessed panel (DataCo / Olist / H&M / Synth).  This gives every
    experiment a self-consistent starting set of cost coefficients without
    requiring an external configuration file."""
    rng = np.random.default_rng(rng_seed)
    I, J = n_products, n_retailers

    p_mean = panel_df["p"].mean()

    return SCMParameters(
        I=I, J=J,
        C_s   = np.full(I, 0.30 * p_mean),
        C_m   = np.full(I, 0.20 * p_mean),
        C_r   = np.full((I, J), 0.10 * p_mean),
        CrCAd = np.full(I, -5.0),       # circular costs cheaper than virgin (-5)
        h_s   = np.full(I, 0.05 * p_mean),
        h_m   = np.full(I, 0.05 * p_mean),
        h_r   = np.full((I, J), 0.04 * p_mean),
        A_s   = np.full(I, 200.0),
        A_m   = np.full(I, 150.0),
        A_r   = np.full((I, J), 80.0),
        c2_s  = np.full(I, 40.0),
        c2_m  = np.full(I, 30.0),
        c2_r  = np.full((I, J), 10.0),
        inc_rate  = 1.5,
        alpha_inc = 0.80,
        A_R       = np.full(J, 1.5 * panel_df["qty_aug"].mean() * I),
        AdBudget  = np.full(I, 4.0 * panel_df.groupby("product")[["b_s", "b_m", "b_r"]]
                                            .sum().sum(axis=1).max()
                            if "b_s" in panel_df.columns else 50_000.0),
        Capacity  = float(panel_df["qty_aug"].sum() * 1.50),
        p_lo      = np.full((I, J), 0.50 * p_mean),
        p_hi      = np.full((I, J), 1.80 * p_mean),
        b_hi      = np.full((I, J), 5_000.0),
        u_hi      = 0.50,
    )
