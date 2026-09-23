"""SIDN-CRO model registry."""

from src.models.base import DemandModel, PanelSchema, decision_columns
from src.models.baselines import (
    AnalyticalDemand,   # B0
    MLPDemand,          # B1
    XGBoostDemand,      # B2
)
from src.models.recurrent import (
    LSTMDemand,         # B3
    BiLSTMAttnDemand,   # B4
)
from src.models.sidn import SIDNDemand                     # A1 / A2
from src.models.conformal import CRODemand, ConformalCalibrator, build_cro_model  # A4

__all__ = [
    "DemandModel", "PanelSchema", "decision_columns",
    "AnalyticalDemand", "MLPDemand", "XGBoostDemand",
    "LSTMDemand", "BiLSTMAttnDemand",
    "SIDNDemand",
    "CRODemand", "ConformalCalibrator", "build_cro_model",
]
