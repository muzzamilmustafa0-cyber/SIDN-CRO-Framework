"""
SIDN-CRO -- Differentiable decision-focused learning bridge (Step 2d).

This module connects the SIDN demand predictions to the SQP optimisation layer
in a way that allows gradient-based training with a decision-focused (SPO+)
loss.  The fundamental challenge is that ``solve_sqp`` calls SciPy under the
hood, which breaks the PyTorch autograd graph.

Strategy: Straight-Through Envelope Theorem (STET)
---------------------------------------------------
1. **Forward** (no gradient):
       Call ``solve_sqp(params, sidn, context)`` to obtain z* = (p*, b*, u*)
       ? the optimal decisions under the current SIDN predictions.

2. **Re-evaluate demand at z*** (with gradient):
       Build the feature matrix corresponding to z* and call
       ``sidn.forward_tensor()`` to obtain D?(z*; theta) ? a PyTorch tensor whose
       gradient w.r.t. theta is tracked by autograd.

3. **Differentiable profit**:
       Compute pi(z*; D?(z*; theta)) in pure PyTorch.  The decisions z* are treated
       as *constants* here (straight-through estimator); gradient flows only
       through D?.  By the envelope theorem, this is the correct first-order
       approximation when z* is near-optimal.

4. **SPO+ surrogate loss**:
       L = pi*(D_true) ? pi(z*(D?); D?(z*; theta))

   Minimising L maximises the realised profit at the predicted-optimal
   decisions ? equivalent to the SPO+ principle (Elmachtoub & Grigas, 2022)
   adapted to nonlinear supply-chain objectives.

The gradient is:
       dL/dtheta = ?dpi/dD? ? dD?/dtheta
             = ???? (p*_ij ? C_var*_ij + gamma*_ij) ? dD?_ij/dtheta

where C_var* and gamma* are the variable cost and circularity-incentive rate at z*.
This tells the SIDN: "predict higher demand at (product, retailer, week) triples
where doing so would increase profit at the current recommended price."

Computational cost per training step
-------------------------------------
* One full SQP solve          : ~0.5?2 s (warm-started; cheap for I<=10, J<=5)
* One SIDN re-evaluation      : ~1 ms  (single forward pass)
* One backward pass           : ~5 ms

For large-scale training, the SQP step dominates.  We amortise it by:
    (a) Running the solver per *week* (the unit of the panel), not per sample.
    (b) Using 'trust-constr' with warm-starting from the previous week's z*.

References
----------
Elmachtoub & Grigas (2022)  -- SPO+, Management Science
Donti, Amos & Kolter (2017) -- Task-based end-to-end model learning, NeurIPS
Amos & Kolter (2017)        -- OptNet, differentiable QP layers, ICML
Berthet et al. (2020)       -- Differentiable perturbed optimisers, NeurIPS
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.optim.sqp_layer import (
    SCMParameters, SCMSolution, solve_sqp, _unpack, _layout,
)


# --------------------------------------------------------------------------- #
#  Profit formula ? differentiable PyTorch version
# --------------------------------------------------------------------------- #
def profit_torch(
    p:    torch.Tensor,   # (I,J) ? from z* (detached)
    b_s:  torch.Tensor,   # (I,J)
    b_m:  torch.Tensor,   # (I,J)
    b_r:  torch.Tensor,   # (I,J)
    u_s:  torch.Tensor,   # (I,)
    u_m:  torch.Tensor,   # (I,)
    u_r:  torch.Tensor,   # (I,)
    D:    torch.Tensor,   # (I,J) ? D? with gradient
    params: SCMParameters,
    device: str,
) -> torch.Tensor:
    """Differentiable profit evaluation mirroring the NumPy formula in sqp_layer.py.

    Decisions z* = (p, b_s, b_m, b_r, u_s, u_m, u_r) are **detached** tensors
    (they come from the SQP solver and carry no gradient).  Only ``D`` carries
    a gradient, which is what flows back to the SIDN parameters.

    Returns
    -------
    profit : scalar tensor (with gradient through D)
    """
    def t(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=device)

    I, J = params.I, params.J

    # -- Variable cost per (product, retailer) -----------------------------
    avg_u    = (u_s + u_m + u_r) / 3.0                           # (I,)
    base_c   = t((params.C_s + params.C_m).reshape(I, 1)) + t(params.C_r)  # (I,J)
    circ_c   = t(params.CrCAd.reshape(I, 1)) * avg_u.reshape(I, 1)         # (I,J)
    C_var    = base_c + circ_c                                              # (I,J)

    # -- Revenue and cost terms ---------------------------------------------
    revenue  = (p * D).sum()
    var_cost = (C_var * D).sum()
    ad_cost  = b_s.sum() + b_m.sum() + b_r.sum()

    # -- Cooperative incentive (income effect of circular fractions) --------
    Inc = (t(params.inc_rate)
           * (u_s + u_m + u_r).reshape(I, 1) * D).sum()

    # -- Convex penalty on high circular-input fractions --------------------
    pen_s = (t(params.c2_s) * u_s ** 2).sum()
    pen_m = (t(params.c2_m) * u_m ** 2).sum()
    pen_r  = (t(params.c2_r) * u_r.reshape(I, 1) ** 2).sum()
    Penalty = pen_s + pen_m + pen_r

    # -- Holding / setup cost (proxy: 5% of throughput-proportional cost) --
    hold_rate = t((params.h_s + params.h_m).reshape(I, 1)) + t(params.h_r)
    holding   = (hold_rate * D).sum() * 0.05

    profit = revenue - var_cost - ad_cost - Penalty + Inc - holding
    return profit


# --------------------------------------------------------------------------- #
#  Helper: insert z* decisions into a feature matrix
# --------------------------------------------------------------------------- #
def _apply_zstar(base_cont: np.ndarray,
                 sol: SCMSolution,
                 dec_idx: Dict[str, int],
                 I: int, J: int) -> np.ndarray:
    """Return a copy of ``base_cont`` with decision columns set to z*.

    ``base_cont`` is (I*J, n_cont); rows are ordered as product-major
    (row k = product i=k//J, retailer j=k%J).
    """
    cont = base_cont.copy()
    cont[:, dec_idx["p"]]   = sol.p.ravel()
    cont[:, dec_idx["b_s"]] = sol.b_s.ravel()
    cont[:, dec_idx["b_m"]] = sol.b_m.ravel()
    cont[:, dec_idx["b_r"]] = sol.b_r.ravel()
    # u_s/u_m/u_r are per-product (I,); broadcast over J retailers
    cont[:, dec_idx["u_s"]] = np.repeat(sol.u_s, J)
    cont[:, dec_idx["u_m"]] = np.repeat(sol.u_m, J)
    cont[:, dec_idx["u_r"]] = np.repeat(sol.u_r, J)
    return cont


def _sol_to_tensors(sol: SCMSolution, params: SCMParameters,
                    device: str) -> Dict[str, torch.Tensor]:
    """Convert SCMSolution decision arrays to detached PyTorch tensors."""
    def t(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=device)
    I, J = params.I, params.J
    return dict(
        p   = t(sol.p),           # (I,J)
        b_s = t(sol.b_s),         # (I,J)
        b_m = t(sol.b_m),         # (I,J)
        b_r = t(sol.b_r),         # (I,J)
        u_s = t(sol.u_s),         # (I,)
        u_m = t(sol.u_m),         # (I,)
        u_r = t(sol.u_r),         # (I,)
    )


# --------------------------------------------------------------------------- #
#  Core DFL loss for one optimisation instance (one week)
# --------------------------------------------------------------------------- #
def dfl_loss(
    sidn,                          # SIDNDemand instance
    sqp_params: SCMParameters,
    context: Dict,                 # {"base_cont": np.ndarray, "X_cat": np.ndarray, "shape": tuple}
    *,
    true_profit_star: Optional[float] = None,
    solver_x0: Optional[np.ndarray]   = None,
    sqp_method: str = "trust-constr",
    sqp_tol: float  = 1e-5,
    sqp_max_iter: int = 300,
    verbose: bool   = False,
) -> Tuple[torch.Tensor, SCMSolution]:
    """Compute the SPO+ surrogate loss for one planning-period instance.

    Parameters
    ----------
    sidn            : fitted SIDNDemand model (parameters theta to be trained)
    sqp_params      : SCMParameters for this week / instance
    context         : dict with keys ``base_cont`` (I*J, n_cont unscaled),
                      ``X_cat`` (I*J, n_cat), ``shape`` (I, J)
    true_profit_star: pi*(D_true) ? upper bound on achievable profit under the
                      true demand.  If provided, returns *regret*; otherwise
                      returns *negative profit* (equivalent up to a constant).
    solver_x0       : warm-start point for the SQP solver
    sqp_method      : 'trust-constr' (default) or 'SLSQP'
    sqp_tol, sqp_max_iter : solver tolerances
    verbose         : print solver diagnostics

    Returns
    -------
    loss : scalar tensor (with gradient through SIDN parameters theta)
    sol  : SCMSolution at z*(D?)  ? returned for warm-starting the next call
    """
    I, J = context["shape"]
    base_cont = context["base_cont"]   # (I*J, n_cont)  unscaled
    X_cat     = context["X_cat"]       # (I*J, n_cat)
    dec_idx   = sidn._dec_idx

    # -- Step 1: Solve SQP with current SIDN (no gradient) -----------------
    with torch.no_grad():
        sidn.model.eval()
        sol = solve_sqp(
            sqp_params, sidn, context=context,
            x0=solver_x0, method=sqp_method,
            tol=sqp_tol, max_iter=sqp_max_iter, verbose=verbose,
        )

    # -- Step 2: Build feature matrix at z* and re-evaluate D? with gradient
    sidn.model.train()
    cont_zstar = _apply_zstar(base_cont, sol, dec_idx, I, J)

    tensors = sidn.make_tensors(cont_zstar, X_cat)    # dict of torch tensors
    D_hat   = sidn.forward_tensor(tensors)             # (I*J,) with gradient
    D_hat_2d = D_hat.reshape(I, J)                    # (I,J)

    # -- Step 3: Differentiable profit at z* under D?(z*; theta) ---------------
    zt = _sol_to_tensors(sol, sqp_params, sidn.device)
    profit = profit_torch(
        zt["p"], zt["b_s"], zt["b_m"], zt["b_r"],
        zt["u_s"], zt["u_m"], zt["u_r"],
        D_hat_2d, sqp_params, sidn.device,
    )

    # -- Step 4: SPO+ surrogate loss (minimise regret) ---------------------
    if true_profit_star is not None:
        loss = (torch.tensor(true_profit_star, dtype=torch.float32,
                             device=sidn.device) - profit)
    else:
        loss = -profit   # minimise negative profit ? maximise profit

    return loss, sol


# --------------------------------------------------------------------------- #
#  True-optimal profit: pi*(D_true) for a demand oracle
# --------------------------------------------------------------------------- #
def solve_true_optimum(
    sqp_params: SCMParameters,
    true_demand_fn,            # callable with the DemandModel interface
    context: Dict,
    *,
    sqp_method: str   = "trust-constr",
    sqp_tol: float    = 1e-5,
    sqp_max_iter: int = 400,
    verbose: bool     = False,
) -> SCMSolution:
    """Solve SQP under the *true* demand model to obtain the regret baseline.

    In evaluation this is called once per test week to obtain pi*(D_true), which
    defines the achievable upper bound on profit.  The regret of any ML model
    is then:   regret_k = pi*(D_true) ? pi(z*(D?_k); D_true).

    Parameters
    ----------
    true_demand_fn : a DemandModel that approximates the true demand.  In the
                     evaluation pipeline this is an AnalyticalDemand fitted on
                     the full train+val set, or an interpolated look-up table.

    Returns
    -------
    SCMSolution at z*(D_true)
    """
    return solve_sqp(
        sqp_params, true_demand_fn, context=context,
        method=sqp_method, tol=sqp_tol, max_iter=sqp_max_iter,
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
#  Regret evaluation (no gradient, for val / test)
# --------------------------------------------------------------------------- #
def evaluate_regret(
    demand_model,              # any DemandModel instance
    sqp_params: SCMParameters,
    context: Dict,
    true_profit_star: float,
    true_demand_fn=None,       # if provided, re-evaluates profit under true demand
    sqp_method: str   = "trust-constr",
    sqp_tol: float    = 1e-5,
    sqp_max_iter: int = 400,
) -> Dict[str, float]:
    """Evaluate regret and supplementary metrics for one week.

    Parameters
    ----------
    demand_model     : model whose z*(D?) we evaluate
    true_profit_star : pi*(D_true), pre-computed via solve_true_optimum()
    true_demand_fn   : optional ? if given, realised profit is computed with
                       the true demand at z*(D?) instead of the model's D?.
                       This gives the *true regret*; otherwise it is the
                       *in-model regret* (lower bound on true regret).

    Returns
    -------
    dict with keys:
        profit_hat       : profit at z*(D?) under D?
        profit_realised  : profit at z*(D?) under D_true  (if true_demand_fn given)
        regret_model     : true_profit_star ? profit_hat
        regret_realised  : true_profit_star ? profit_realised
        feasible         : whether z*(D?) satisfies all constraints
        n_sqp_iter       : solver iteration count
    """
    sol = solve_sqp(
        sqp_params, demand_model, context=context,
        method=sqp_method, tol=sqp_tol, max_iter=sqp_max_iter,
    )

    profit_hat      = sol.profit
    regret_model    = true_profit_star - profit_hat

    result = dict(
        profit_hat    = profit_hat,
        regret_model  = regret_model,
        feasible      = sol.feasible,
        n_sqp_iter    = getattr(sol.raw, "nit", -1),
    )

    if true_demand_fn is not None:
        # Re-evaluate profit at z*(D?) using the true demand model
        I, J   = sqp_params.I, sqp_params.J
        D_true = true_demand_fn(
            sol.p, sol.b_s, sol.b_m, sol.b_r,
            sol.u_s, sol.u_m, sol.u_r,
            context=context,
        )
        # Manually replicate the profit formula (pure NumPy for speed)
        avg_u    = (sol.u_s + sol.u_m + sol.u_r) / 3.0
        base_c   = (sqp_params.C_s + sqp_params.C_m).reshape(I, 1) + sqp_params.C_r
        circ_c   = sqp_params.CrCAd.reshape(I, 1) * avg_u.reshape(I, 1)
        C_var    = base_c + circ_c
        revenue  = (sol.p * D_true).sum()
        var_cost = (C_var * D_true).sum()
        ad_cost  = sol.b_s.sum() + sol.b_m.sum() + sol.b_r.sum()
        Inc      = (sqp_params.inc_rate
                   * (sol.u_s + sol.u_m + sol.u_r).reshape(I, 1) * D_true).sum()
        pen_s    = (sqp_params.c2_s * sol.u_s ** 2).sum()
        pen_m    = (sqp_params.c2_m * sol.u_m ** 2).sum()
        pen_r    = (sqp_params.c2_r * sol.u_r.reshape(I, 1) ** 2).sum()
        hold     = (((sqp_params.h_s + sqp_params.h_m).reshape(I, 1)
                    + sqp_params.h_r) * D_true).sum() * 0.05
        profit_realised = float(
            revenue - var_cost - ad_cost - (pen_s + pen_m + pen_r) + Inc - hold
        )
        result["profit_realised"] = profit_realised
        result["regret_realised"] = true_profit_star - profit_realised

    return result


# --------------------------------------------------------------------------- #
#  Batch regret over a panel of test weeks
# --------------------------------------------------------------------------- #
def batch_regret(
    demand_model,
    sqp_params_list,               # list of SCMParameters, one per week
    context_list,                  # list of context dicts, one per week
    true_profit_star_list,         # list of floats
    true_demand_fn=None,
    sqp_method: str = "trust-constr",
    sqp_tol: float  = 1e-5,
    sqp_max_iter: int = 400,
    verbose: bool   = False,
) -> Dict[str, np.ndarray]:
    """Run evaluate_regret for each week and aggregate results.

    Parameters
    ----------
    sqp_params_list, context_list, true_profit_star_list :
        Parallel lists of length n_weeks.

    Returns
    -------
    dict of numpy arrays, each of length n_weeks:
        profit_hat, regret_model, feasible, n_sqp_iter,
        (profit_realised, regret_realised  if true_demand_fn given)
    """
    records = []
    x0 = None   # warm-start chain

    for k, (params, ctx, pi_star) in enumerate(
            zip(sqp_params_list, context_list, true_profit_star_list)):
        res = evaluate_regret(
            demand_model, params, ctx, pi_star,
            true_demand_fn=true_demand_fn,
            sqp_method=sqp_method,
            sqp_tol=sqp_tol,
            sqp_max_iter=sqp_max_iter,
        )
        records.append(res)
        if verbose and (k + 1) % 10 == 0:
            print(f"  [batch_regret] week {k+1}  "
                  f"regret={res.get('regret_realised', res['regret_model']):.1f}")

    # Aggregate
    keys = list(records[0].keys())
    return {k: np.array([r[k] for r in records]) for k in keys}


# --------------------------------------------------------------------------- #
#  SPO+ gradient oracle (analytical ? for linear-objective verification)
# --------------------------------------------------------------------------- #
def spo_plus_gradient_oracle(
    D_hat: np.ndarray,
    D_true: np.ndarray,
    sqp_params: SCMParameters,
    context: Dict,
    sqp_method: str = "trust-constr",
) -> np.ndarray:
    """Compute the exact SPO+ gradient oracle for verification purposes.

    For a *linear* objective f(z) = c ? z, the SPO+ gradient is:
        dL_SPO+ / d? = ?(z*(2? ? c_true) ? z*(c_true))

    For our nonlinear problem we use the linearised version:
        g = ?(z*(2D? ? D_true) ? z*(D_true))

    This requires two additional SQP solves and is used only for gradient
    verification in the unit tests (not in the main training loop).

    Parameters
    ----------
    D_hat, D_true : (I*J,) demand predictions and true demands (flat, row-major)

    Returns
    -------
    g : (I*J,) gradient of SPO+ loss w.r.t. D?  (matches shape of D_hat)
    """
    from src.models.base import DemandModel

    I, J = sqp_params.I, sqp_params.J

    # Construct a demand function that returns a fixed array (D = const)
    class _FixedDemand:
        def __init__(self, d_flat: np.ndarray):
            self._D = d_flat.reshape(I, J)
        def __call__(self, p, b_s, b_m, b_r, u_s, u_m, u_r, context=None):
            return self._D

    sol_perturbed = solve_sqp(
        sqp_params, _FixedDemand(2 * D_hat - D_true), context=context,
        method=sqp_method,
    )
    sol_true = solve_sqp(
        sqp_params, _FixedDemand(D_true), context=context,
        method=sqp_method,
    )
    # z* is expressed as demand (since demand = D in FixedDemand, gradient is
    # in demand space)
    g = -(sol_perturbed.demand.ravel() - sol_true.demand.ravel())
    return g


# --------------------------------------------------------------------------- #
#  Normalised regret (for tables / figures)
# --------------------------------------------------------------------------- #
def normalised_regret(regret: np.ndarray,
                      profit_star: np.ndarray) -> np.ndarray:
    """Regret / pi*(D_true), expressed as a fraction of maximum achievable profit.

    Ranges in [0, 1]; lower is better.  A value of 0.05 means the model
    achieves 95% of the optimal profit.
    """
    denom = np.where(profit_star > 0, profit_star, 1.0)
    return np.clip(regret / denom, 0.0, None)
