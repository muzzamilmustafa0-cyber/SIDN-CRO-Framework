"""SIDN-CRO optimisation layer."""

from src.optim.sqp_layer import (
    SCMParameters, SCMSolution, AnalyticalDemandModel,
    solve_sqp, params_from_panel_mean,
)
from src.optim.differentiable import (
    profit_torch, dfl_loss, solve_true_optimum,
    evaluate_regret, batch_regret, normalised_regret,
)

__all__ = [
    "SCMParameters", "SCMSolution", "AnalyticalDemandModel",
    "solve_sqp", "params_from_panel_mean",
    "profit_torch", "dfl_loss", "solve_true_optimum",
    "evaluate_regret", "batch_regret", "normalised_regret",
]
