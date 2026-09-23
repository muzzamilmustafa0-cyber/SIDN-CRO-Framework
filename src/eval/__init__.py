"""SIDN-CRO evaluation utilities."""

from src.eval.metrics import (
    ModelResult, EvalConfig,
    evaluate_model_on_dataset,
    aggregate_seeds,
    print_results_table,
)
from src.eval.sensitivity import (
    rho_sensitivity_table,
    window_sensitivity_table,
    dataset_shift_table,
)
from src.eval.feasibility import feasibility_report

__all__ = [
    "ModelResult", "EvalConfig",
    "evaluate_model_on_dataset", "aggregate_seeds", "print_results_table",
    "rho_sensitivity_table", "window_sensitivity_table", "dataset_shift_table",
    "feasibility_report",
]
