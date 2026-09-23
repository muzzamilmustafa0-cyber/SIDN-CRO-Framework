"""
SIDN-CRO -- Feasibility analysis (Step 2g, part 3).

Reports the fraction of test-week SQP solutions that satisfy all constraints
under each model's predicted optimal decisions z*(D?).

A solution z*(D?) is feasible if:
    (a) Advertising budget:  sum_j (b_s_ij + b_m_ij + b_r_ij) <= AdBudget_i  ?i
    (b) Recycled availability: sum_i (u_s + u_m + u_r) ? D_ij <= A_R_j        ?j
    (c) Capacity:             sum_ij D_ij <= Capacity

Infeasibility can occur when demand predictions are wildly optimistic,
causing the solver to push towards high-demand decisions that violate
the recycled-input availability constraint.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from src.eval.metrics import ModelResult, WeekResult


def feasibility_report(
    results: Dict[str, ModelResult],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Summarise constraint feasibility across models.

    Parameters
    ----------
    results : { model_name: ModelResult }
              from evaluate_model_on_dataset() for each model on the same dataset

    Returns
    -------
    { model_name: { "feasible_pct": ..., "n_infeasible": ..., "n_weeks": ... } }
    """
    report: Dict[str, Dict] = {}

    for name, mr in results.items():
        n_weeks      = len(mr.week_results)
        n_feasible   = sum(w.feasible for w in mr.week_results)
        n_infeasible = n_weeks - n_feasible
        pct          = 100.0 * n_feasible / max(n_weeks, 1)

        report[name] = {
            "feasible_pct":  pct,
            "n_feasible":    n_feasible,
            "n_infeasible":  n_infeasible,
            "n_weeks":       n_weeks,
        }

        if verbose:
            flag = "OK" if pct >= 99.9 else ("!" if pct >= 90.0 else "FAIL")
            print(f"  {flag} {name:<18}  feasible={pct:.1f}%  "
                  f"({n_infeasible} infeasible / {n_weeks} weeks)")

    return report


def constraint_violation_details(
    week_results: List[WeekResult],
) -> Dict[str, float]:
    """For infeasible weeks, summarise which constraint was most often violated.

    Since WeekResult only stores the feasibility flag (not the specific
    violated constraint), this function counts infeasible weeks and their
    profit shortfall ? the regret is usually higher for infeasible weeks
    because the solver settles for a suboptimal feasible point.

    Returns
    -------
    dict with keys:
        infeasible_count, mean_regret_infeasible, mean_regret_feasible
    """
    inf_regrets = [w.regret_norm for w in week_results if not w.feasible]
    fea_regrets = [w.regret_norm for w in week_results if w.feasible]

    return {
        "infeasible_count":      len(inf_regrets),
        "mean_regret_infeasible": float(np.mean(inf_regrets)) * 100
                                  if inf_regrets else 0.0,
        "mean_regret_feasible":   float(np.mean(fea_regrets)) * 100
                                  if fea_regrets else 0.0,
    }


def print_feasibility_table(report: Dict[str, Dict[str, float]]) -> None:
    """Print a LaTeX-ready feasibility table."""
    print("\n" + "=" * 55)
    print(f"{'Model':<20} {'Feasible%':>12} {'#Infeasible':>14}")
    print("-" * 55)
    for name, d in sorted(report.items()):
        print(f"{name:<20} {d['feasible_pct']:>11.1f}% "
              f"{d['n_infeasible']:>14}/{d['n_weeks']}")
    print("=" * 55)
