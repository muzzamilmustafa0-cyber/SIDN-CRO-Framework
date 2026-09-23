"""
SIDN-CRO -- Dataset loading utilities for training scripts.

Loads the pre-processed arrays (produced by src/data_pipeline/make_train_val_test*.py)
and packages them into a ``DatasetBundle`` that every training function accepts.

Directory layout expected
-------------------------
preprocessed_dir/
    schema.json
    x_scaler.joblib
    y_scaler.joblib
    Train/
        train_X_cont.npy    (N_train, n_cont) ? StandardScaler-scaled
        train_X_cat.npy     (N_train, n_cat)  ? one-hot categorical
        train_y.npy         (N_train,)         ? StandardScaler-scaled target
        train_full.parquet  (N_train, *)       ? full panel (incl. week_start)
    Validation/
        val_X_cont.npy
        val_X_cat.npy
        val_y.npy
        val_full.parquet
    Test/
        test_X_cont.npy
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.models.base import PanelSchema


# --------------------------------------------------------------------------- #
# DatasetBundle
# --------------------------------------------------------------------------- #
@dataclass
class SplitArrays:
    """All arrays for one data split (train / val / test)."""
    X_cont:   np.ndarray    # (N, n_cont) ? StandardScaler-scaled
    X_cat:    np.ndarray    # (N, n_cat)  ? one-hot floats
    y:        np.ndarray    # (N,)        ? StandardScaler-scaled target
    X_cont_u: np.ndarray    # (N, n_cont) ? ORIGINAL unscaled continuous features
    y_u:      np.ndarray    # (N,)        ? original unscaled demand
    week_idx: np.ndarray    # (N,) int    ? integer week index (0-based)
    panel_df: pd.DataFrame  # full parquet panel (for context building, params)


@dataclass
class DatasetBundle:
    """Everything a training function needs for one dataset.

    Parameters
    ----------
    name         : human-readable dataset name (e.g., "DataCo", "Olist")
    preprocessed_dir : root of the preprocessed directory
    schema       : PanelSchema ? feature names, products, retailers
    x_scaler     : fitted StandardScaler for continuous features
    y_scaler     : fitted StandardScaler for the demand target
    train        : SplitArrays for the training set
    val          : SplitArrays for the validation set
    test         : SplitArrays for the test set
    """
    name:             str
    preprocessed_dir: Path
    schema:           PanelSchema
    x_scaler:         object
    y_scaler:         object
    train:            SplitArrays
    val:              SplitArrays
    test:             SplitArrays

    @property
    def n_products(self) -> int:
        return len(self.schema.products)

    @property
    def n_retailers(self) -> int:
        return len(self.schema.retailers)


# --------------------------------------------------------------------------- #
# Loading helper
# --------------------------------------------------------------------------- #
_SPLIT_DIR_MAP = {
    "train": ("Train",      "train"),
    "val":   ("Validation", "val"),
    "test":  ("Test",       "test"),
}


def _load_split(preprocessed_dir: Path, split: str,
                x_scaler, y_scaler) -> SplitArrays:
    dir_name, prefix = _SPLIT_DIR_MAP[split]
    split_dir = preprocessed_dir / dir_name

    X_cont = np.load(split_dir / f"{prefix}_X_cont.npy").astype(np.float32)
    X_cat  = np.load(split_dir / f"{prefix}_X_cat.npy").astype(np.float32)
    y      = np.load(split_dir / f"{prefix}_y.npy").astype(np.float32)

    # Unscaled versions (needed by B0 AnalyticalDemand and DFL context building)
    X_cont_u = x_scaler.inverse_transform(X_cont).astype(np.float32)
    y_u      = y_scaler.inverse_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    # Load the full parquet to obtain week indices
    parquet_path = split_dir / f"{prefix}_full.parquet"
    if parquet_path.exists():
        panel_df = pd.read_parquet(parquet_path)
        # Derive integer week index from week_start (0 = earliest in the panel)
        ws = pd.to_datetime(panel_df["week_start"])
        min_week = ws.min()
        week_idx = ((ws - min_week).dt.days // 7).astype(np.int32).values
    else:
        # Fallback: assign sequential week indices (assumes row order = time order)
        panel_df = pd.DataFrame()
        n_pairs  = int(round(len(X_cont) / max(1, len(X_cont))))
        week_idx = np.arange(len(X_cont), dtype=np.int32)

    return SplitArrays(
        X_cont=X_cont, X_cat=X_cat, y=y,
        X_cont_u=X_cont_u, y_u=y_u,
        week_idx=week_idx,
        panel_df=panel_df,
    )


def load_dataset(preprocessed_dir: Path, name: str = "") -> DatasetBundle:
    """Load a complete dataset from a preprocessed directory.

    Parameters
    ----------
    preprocessed_dir : path to the dataset's Preprocessed_* directory
    name             : optional human-readable label (used in log messages)

    Returns
    -------
    DatasetBundle
    """
    preprocessed_dir = Path(preprocessed_dir)
    schema   = PanelSchema.load(preprocessed_dir / "schema.json")
    x_scaler = joblib.load(preprocessed_dir / "x_scaler.joblib")
    y_scaler = joblib.load(preprocessed_dir / "y_scaler.joblib")

    train = _load_split(preprocessed_dir, "train", x_scaler, y_scaler)
    val   = _load_split(preprocessed_dir, "val",   x_scaler, y_scaler)
    test  = _load_split(preprocessed_dir, "test",  x_scaler, y_scaler)

    return DatasetBundle(
        name=name or preprocessed_dir.name,
        preprocessed_dir=preprocessed_dir,
        schema=schema,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train=train,
        val=val,
        test=test,
    )


# --------------------------------------------------------------------------- #
# Evaluation metrics (shared across all training scripts)
# --------------------------------------------------------------------------- #
def compute_demand_metrics(y_pred: np.ndarray,
                           y_true: np.ndarray,
                           eps: float = 1.0) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R2 on demand in **original units**.

    Parameters
    ----------
    y_pred, y_true : (N,) arrays in original (unscaled) demand units
    eps            : small constant added to denominator of MAPE to avoid Inf

    Returns
    -------
    dict with keys: mae, rmse, mape, r2
    """
    err  = y_pred - y_true
    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((np.abs(err) / (np.abs(y_true) + eps)).mean()) * 100.0
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2   = 1.0 - ss_res / (ss_tot + 1e-12)
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


def evaluate_split(model, split: SplitArrays,
                   model_name: str = "") -> Dict[str, float]:
    """Call ``model.raw_predict`` on a split and return demand metrics."""
    y_pred = model.raw_predict(split.X_cont_u, split.X_cat)
    metrics = compute_demand_metrics(y_pred, split.y_u)
    if model_name:
        print(f"  [{model_name}]  MAE={metrics['mae']:.1f}  "
              f"RMSE={metrics['rmse']:.1f}  MAPE={metrics['mape']:.2f}%  "
              f"R2={metrics['r2']:.4f}")
    return metrics


# --------------------------------------------------------------------------- #
# Context dict builder (used by DFL training and evaluation)
# --------------------------------------------------------------------------- #
def build_week_context(split: SplitArrays,
                       week: int,
                       n_products: int,
                       n_retailers: int,
                       dec_idx: Dict[str, int]) -> Optional[Dict]:
    """Build the ``context`` dict for one planning-period (week).

    The context contains the non-decision continuous features for all
    (product, retailer) cells in the given week, ready for the SQP solver.

    Parameters
    ----------
    split        : SplitArrays for the appropriate data split
    week         : integer week index
    n_products, n_retailers : I, J
    dec_idx      : decision-column indices from decision_columns(schema)

    Returns
    -------
    dict with keys ``base_cont``, ``X_cat``, ``shape``, or ``None`` if the
    week is not present or does not have exactly I*J rows.
    """
    mask = split.week_idx == week
    if mask.sum() != n_products * n_retailers:
        return None   # incomplete week ? skip

    base_cont = split.X_cont_u[mask]    # (I*J, n_cont) unscaled
    X_cat     = split.X_cat[mask]       # (I*J, n_cat)

    return {
        "base_cont": base_cont,
        "X_cat":     X_cat,
        "shape":     (n_products, n_retailers),
    }
