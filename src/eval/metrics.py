"""
SIDN-CRO -- Evaluation metrics (Step 2g).

Primary metric : Normalised Regret (NR)
    NR = (pi*(D_true) ? pi(z*(D?); D_true)) / pi*(D_true) x 100 %

Secondary metrics :
    MAE, RMSE, MAPE, R2  -- demand forecasting accuracy
    Feasibility rate      -- fraction of weeks where z*(D?) is feasible
    Profit realised       -- pi(z*(D?); D_true) in absolute units
    CI coverage           -- conformal interval coverage (A4 only)

All metrics are computed per week and then summarised as mean +/- std over
the test set and over multiple random seeds.

The main entry point is ``evaluate_model_on_dataset()``, which:
    1. For each test week, solves SQP with the candidate model -> z*(D?).
    2. Evaluates true profit at z*(D?) using the true demand proxy.
    3. Computes true optimal profit pi*(D_true) from a pre-cached solution.
    4. Returns a ModelResult with all per-week metrics.
"""

from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.base import DemandModel, decision_columns
from src.optim.sqp_layer import SCMParameters
from src.optim.differentiable import (
    batch_regret, solve_true_optimum, normalised_regret, evaluate_regret,
)
from src.train.dataset import (
    DatasetBundle, SplitArrays,
    compute_demand_metrics, evaluate_split, build_week_context,
)

# Default number of parallel SQP workers (use physical CPU cores, cap at 8)
_DEFAULT_N_WORKERS = min(8, max(1, (os.cpu_count() or 4)))


# --------------------------------------------------------------------------- #
# Configuration and result containers
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    """Hyperparameters for the evaluation procedure."""
    sqp_method:   str   = "SLSQP"
    sqp_tol:      float = 1e-5
    sqp_max_iter: int   = 300
    alpha:        float = 0.10   # conformal coverage level (A4 only)
    rho:          float = 0.50   # CRO robustness level (A4 only)


@dataclass
class WeekResult:
    """Per-week evaluation result."""
    week:             int
    profit_hat:       float   # pi(z*(D?); D?)  ? in-model profit
    profit_realised:  float   # pi(z*(D?); D_true) ? true realised profit
    profit_star:      float   # pi*(D_true)
    regret_abs:       float   # profit_star ? profit_realised
    regret_norm:      float   # regret_abs / profit_star  (in [0, 1])
    feasible:         bool
    n_sqp_iter:       int


@dataclass
class ModelResult:
    """Evaluation results for one model x one dataset x one seed."""
    model_name:    str
    dataset_name:  str
    seed:          int

    # Demand-accuracy metrics (test set)
    demand_metrics: Dict[str, float] = field(default_factory=dict)
    # MAE, RMSE, MAPE, R2

    # Optimisation metrics (per week, test set)
    week_results:   List[WeekResult] = field(default_factory=list)

    # Conformal coverage metrics (A4 only)
    conformal_metrics: Dict[str, float] = field(default_factory=dict)

    # Runtime
    eval_time_s: float = 0.0

    # -- Derived summary statistics ------------------------------------------
    @property
    def regret_norm_mean(self) -> float:
        return float(np.mean([w.regret_norm for w in self.week_results]))

    @property
    def regret_norm_std(self) -> float:
        return float(np.std([w.regret_norm for w in self.week_results]))

    @property
    def profit_realised_mean(self) -> float:
        return float(np.mean([w.profit_realised for w in self.week_results]))

    @property
    def feasibility_rate(self) -> float:
        if not self.week_results:
            return 0.0
        return float(np.mean([w.feasible for w in self.week_results]))

    def summary(self) -> Dict[str, float]:
        wr = self.week_results
        if not wr:
            return {}
        return {
            "NR_mean":     self.regret_norm_mean * 100,   # percentage
            "NR_std":      self.regret_norm_std * 100,
            "profit_mean": self.profit_realised_mean,
            "feasible":    self.feasibility_rate * 100,
            **{f"demand_{k}": v for k, v in self.demand_metrics.items()},
            **{f"conf_{k}":   v for k, v in self.conformal_metrics.items()},
        }


# --------------------------------------------------------------------------- #
# True-optimal profit cache (pi* per test week)  — parallel version
# --------------------------------------------------------------------------- #
def build_pi_star_cache(
    true_demand_fn: DemandModel,
    sqp_params: SCMParameters,
    test_split: SplitArrays,
    n_products: int,
    n_retailers: int,
    dec_idx: Dict,
    cfg: EvalConfig,
    verbose: bool = True,
    n_workers: int = _DEFAULT_N_WORKERS,
) -> Dict[int, float]:
    """Pre-compute pi*(D_true) for each test week in parallel.

    Parameters
    ----------
    true_demand_fn : demand model treated as the ground truth (typically B0
                     fitted on train+val, or AnalyticalDemand with known params)
    n_workers      : number of parallel threads for SQP solves (default: CPU count)
    """
    weeks = sorted(np.unique(test_split.week_idx).tolist())

    def _solve_one(week):
        ctx = build_week_context(
            test_split, week, n_products, n_retailers, dec_idx)
        if ctx is None:
            return week, None
        sol = solve_true_optimum(
            sqp_params, true_demand_fn, ctx,
            sqp_method=cfg.sqp_method,
            sqp_tol=cfg.sqp_tol,
            sqp_max_iter=cfg.sqp_max_iter,
        )
        return week, sol.profit

    cache: Dict[int, float] = {}
    _print_lock = threading.Lock()
    n_done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_solve_one, w): w for w in weeks}
        for fut in as_completed(futures):
            week, profit = fut.result()
            if profit is not None:
                cache[week] = profit
            n_done += 1
            if verbose and n_done % 10 == 0:
                with _print_lock:
                    print(f"    pi* cache: {n_done}/{len(weeks)} weeks done")

    return cache


# --------------------------------------------------------------------------- #
# Main evaluation function
# --------------------------------------------------------------------------- #
def evaluate_model_on_dataset(
    model:           DemandModel,
    bundle:          DatasetBundle,
    sqp_params:      SCMParameters,
    true_demand_fn:  DemandModel,
    pi_star_cache:   Optional[Dict[int, float]] = None,
    seed:            int   = 42,
    cfg:             EvalConfig = None,
    verbose:         bool  = False,
    n_workers:       int   = _DEFAULT_N_WORKERS,
) -> ModelResult:
    """Full evaluation of one demand model on the test split.

    Parameters
    ----------
    model          : fitted demand model to evaluate (B0-B4 or A1-A4)
    bundle         : DatasetBundle (uses test split)
    sqp_params     : SCMParameters for this dataset
    true_demand_fn : demand model used as D_true proxy for realised profit
    pi_star_cache  : pre-computed pi*(D_true) per test week (if None, computed here)
    seed           : used in ModelResult for labelling
    cfg            : EvalConfig; uses defaults if None
    verbose        : print per-week progress

    Returns
    -------
    ModelResult
    """
    if cfg is None:
        cfg = EvalConfig()

    test  = bundle.test
    I, J  = bundle.n_products, bundle.n_retailers
    dec   = model._dec_idx if hasattr(model, "_dec_idx") else \
            (model.base_model._dec_idx if hasattr(model, "base_model") else {})
    # fallback dec_idx from schema
    if not dec:
        dec = decision_columns(bundle.schema)

    t0 = time.time()

    # -- 1. Demand-accuracy metrics (no SQP needed) -------------------------
    demand_m = compute_demand_metrics(
        model.raw_predict(test.X_cont_u, test.X_cat), test.y_u)

    # -- 2. Build pi* cache if not provided ---------------------------------
    if pi_star_cache is None:
        print(f"  [{model.name}] Building pi* cache ...")
        pi_star_cache = build_pi_star_cache(
            true_demand_fn, sqp_params, test,
            I, J, dec, cfg, verbose=verbose,
        )

    # -- 3. Per-week regret evaluation (parallel SQP via ThreadPoolExecutor) -
    weeks       = sorted(np.unique(test.week_idx).tolist())
    week_results: List[WeekResult] = []

    # Put PyTorch models in eval mode ONCE before spawning threads to avoid
    # concurrent state modification.  raw_predict() also calls eval() internally
    # but doing it here ensures consistency when threads start.
    import torch.nn as _nn
    def _set_eval(m):
        if hasattr(m, "model") and isinstance(m.model, _nn.Module):
            m.model.eval()
        if hasattr(m, "base_model"):
            _set_eval(m.base_model)
    _set_eval(model)
    _set_eval(true_demand_fn)

    _print_lock = threading.Lock()
    n_done_ref  = [0]   # mutable counter shared across threads

    def _eval_one_week(week):
        ctx = build_week_context(test, week, I, J, dec)
        if ctx is None:
            return None
        pi_star = pi_star_cache.get(week, 0.0)

        res = evaluate_regret(
            model, sqp_params, ctx, pi_star,
            true_demand_fn=true_demand_fn,
            sqp_method=cfg.sqp_method,
            sqp_tol=cfg.sqp_tol,
            sqp_max_iter=cfg.sqp_max_iter,
        )

        nr = float(normalised_regret(
            np.array([res.get("regret_realised", res["regret_model"])]),
            np.array([pi_star]),
        )[0])

        wr = WeekResult(
            week=week,
            profit_hat=res["profit_hat"],
            profit_realised=res.get("profit_realised", res["profit_hat"]),
            profit_star=pi_star,
            regret_abs=res.get("regret_realised", res["regret_model"]),
            regret_norm=nr,
            feasible=res["feasible"],
            n_sqp_iter=res["n_sqp_iter"],
        )

        if verbose:
            with _print_lock:
                n_done_ref[0] += 1
                k = n_done_ref[0]
                if k % 10 == 0:
                    print(f"  [{model.name}] week {k}/{len(weeks)}  "
                          f"NR={nr*100:.1f}%  feasible={res['feasible']}")
        return wr

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_one_week, w): w for w in weeks}
        raw_results = {}
        for fut in as_completed(futures):
            week = futures[fut]
            wr = fut.result()
            if wr is not None:
                raw_results[week] = wr

    # Restore original week order
    week_results = [raw_results[w] for w in weeks if w in raw_results]

    # -- 4. Conformal coverage metrics (A4 only) ----------------------------
    conf_m: Dict[str, float] = {}
    if hasattr(model, "evaluate_coverage") and hasattr(model, "_calibrated"):
        if model._calibrated:
            conf_m = model.evaluate_coverage(
                test.X_cont_u, test.X_cat, test.y_u)

    elapsed = time.time() - t0
    result  = ModelResult(
        model_name=getattr(model, "name", type(model).__name__),
        dataset_name=bundle.name,
        seed=seed,
        demand_metrics=demand_m,
        week_results=week_results,
        conformal_metrics=conf_m,
        eval_time_s=elapsed,
    )

    s = result.summary()
    print(f"\n  [{result.model_name}] on {bundle.name}  seed={seed}")
    print(f"    NR = {s.get('NR_mean', 0):.2f} +/- {s.get('NR_std', 0):.2f} %")
    print(f"    MAPE = {s.get('demand_mape', 0):.2f}%  "
          f"R2 = {s.get('demand_r2', 0):.4f}")
    print(f"    Feasible = {s.get('feasible', 0):.1f}%  "
          f"elapsed={elapsed:.1f}s")

    return result


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_seeds(results: List[ModelResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate ModelResult objects from multiple seeds.

    Parameters
    ----------
    results : list of ModelResult from the same (model, dataset) combination
              across different random seeds

    Returns
    -------
    dict { metric_name: {"mean": ..., "std": ..., "ci95": ...} }
    """
    all_summaries = [r.summary() for r in results]
    keys = all_summaries[0].keys()
    agg  = {}
    for k in keys:
        vals = np.array([s[k] for s in all_summaries if k in s])
        agg[k] = {
            "mean":  float(vals.mean()),
            "std":   float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "ci95":  float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
                     if len(vals) > 1 else 0.0,
            "min":   float(vals.min()),
            "max":   float(vals.max()),
        }
    return agg


# --------------------------------------------------------------------------- #
# Table printer (manuscript Table 2)
# --------------------------------------------------------------------------- #
MODEL_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "A1", "A2", "A4"]
METRIC_COLS = ["NR_mean", "NR_std", "demand_mape", "demand_r2", "feasible"]
METRIC_LABELS = {
    "NR_mean":      "NR (%) ?",
    "NR_std":       "+/-",
    "demand_mape":  "MAPE (%) ?",
    "demand_r2":    "R2 ?",
    "feasible":     "Feasible (%) ?",
}


def print_results_table(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
    dataset_name: str = "",
) -> None:
    """Print a markdown-formatted results table (manuscript Table 2 format).

    Parameters
    ----------
    aggregated : {model_name: {metric_name: {"mean":..., "std":...}}}
                 e.g. from {m: aggregate_seeds(results[m]) for m in results}
    dataset_name : label shown in the table header
    """
    header = f"=== Results: {dataset_name} ===" if dataset_name else "=== Results ==="
    print(f"\n{header}")

    # Column widths
    col_w = 12
    hdr = f"{'Model':<8}" + "".join(
        f"{METRIC_LABELS.get(c, c):>{col_w}}" for c in METRIC_COLS)
    print(hdr)
    print("-" * len(hdr))

    for model_name in MODEL_ORDER:
        if model_name not in aggregated:
            continue
        agg = aggregated[model_name]
        row = f"{model_name:<8}"
        for c in METRIC_COLS:
            if c not in agg:
                row += f"{'?':>{col_w}}"
            elif c == "NR_std":
                row += f"{agg[c]['mean']:>{col_w}.2f}"
            else:
                row += f"{agg[c]['mean']:>{col_w}.2f}"
        print(row)
    print()
