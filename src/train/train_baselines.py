"""
SIDN-CRO -- Train baseline demand models (B0 ? B4).

Each baseline is trained on the training split and evaluated on the validation
split.  Results are returned as a dict; models are optionally saved to disk.

Usage (programmatic)
--------------------
    from src.train import load_dataset, train_baseline
    bundle = load_dataset(preprocessed_dir, name="DataCo")
    model, metrics = train_baseline("B1", bundle, seed=42)

Usage (command line)
--------------------
    python -m src.train.train_baselines --dataset dataco --seed 42

Configuration for each baseline
--------------------------------
B0  AnalyticalDemand   -- no learning; only a fit step to derive D0/p0 tables
B1  MLPDemand          -- 2x64 ReLU, 200 epochs, Adam 1e-3
B2  XGBoostDemand      -- 400 trees, max_depth=6, lr=0.05
B3  LSTMDemand         -- window=8, hidden=32, 100 epochs
B4  BiLSTMAttnDemand   -- window=8, hidden=32, 100 epochs (Bi-directional + attention)
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.models.base import PanelSchema
from src.models.baselines import AnalyticalDemand, MLPDemand, XGBoostDemand
from src.models.recurrent import LSTMDemand, BiLSTMAttnDemand
# B5 is handled in run_all.py (requires A1 + DRO calibration, not a simple baseline)
# It is listed here only for reference; train_baseline() will raise for "B5".
from src.train.dataset import DatasetBundle, load_dataset, evaluate_split


# --------------------------------------------------------------------------- #
# Model configuration registry
# --------------------------------------------------------------------------- #
BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "B0": {
        "cls":     AnalyticalDemand,
        "fit_kw":  {},                              # no hyperparams to tune
        "unscaled_fit": True,                       # B0 takes raw demand/price
    },
    "B1": {
        "cls":     MLPDemand,
        "init_kw": {"hidden": 64, "n_layers": 2, "dropout": 0.10},
        "fit_kw":  {"epochs": 200, "batch_size": 64, "lr": 1e-3,
                    "weight_decay": 1e-5},
    },
    "B2": {
        "cls":     XGBoostDemand,
        "init_kw": {"max_depth": 6, "n_estimators": 400,
                    "learning_rate": 0.05, "subsample": 0.8,
                    "colsample_bytree": 0.8},
        "fit_kw":  {},
    },
    "B3": {
        "cls":     LSTMDemand,
        "init_kw": {"window": 8, "hidden": 32, "dropout": 0.10},
        "fit_kw":  {"epochs": 100, "batch_size": 32, "lr": 1e-3,
                    "weight_decay": 1e-5},
        "needs_week_idx": True,
    },
    "B4": {
        "cls":     BiLSTMAttnDemand,
        "init_kw": {"window": 8, "hidden": 32, "dropout": 0.10},
        "fit_kw":  {"epochs": 100, "batch_size": 32, "lr": 1e-3,
                    "weight_decay": 1e-5},
        "needs_week_idx": True,
    },
}


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #
def train_baseline(
    model_name: str,
    bundle: DatasetBundle,
    seed: int = 42,
    verbose: bool = False,
    save_dir: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Dict[str, float]]]:
    """Train a single baseline model.

    Parameters
    ----------
    model_name : one of "B0", "B1", "B2", "B3", "B4"
    bundle     : DatasetBundle from load_dataset()
    seed       : random seed (controls PyTorch, XGBoost, and NumPy RNGs)
    verbose    : print training progress
    save_dir   : if given, save the trained model to ``save_dir/{model_name}.pkl``

    Returns
    -------
    model   : fitted DemandModel instance
    metrics : dict { "train": {...}, "val": {...} } each with MAE/RMSE/MAPE/R2
    """
    if model_name not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown baseline {model_name!r}. "
                         f"Choose from {list(BASELINE_CONFIGS)}")

    cfg = BASELINE_CONFIGS[model_name]
    cls = cfg["cls"]
    init_kw = cfg.get("init_kw", {})
    fit_kw  = cfg.get("fit_kw",  {})

    print(f"\n{'='*60}")
    print(f"  Training {model_name} ({cls.__name__}) on {bundle.name}  seed={seed}")
    print(f"{'='*60}")

    # -- Build model ----------------------------------------------------------
    model = cls(bundle.schema, bundle.x_scaler, bundle.y_scaler, **init_kw)

    tr = bundle.train
    t0 = time.time()

    # -- Fit ------------------------------------------------------------------
    if cfg.get("unscaled_fit"):
        # B0 AnalyticalDemand expects UNSCALED continuous features + demand
        model.fit(tr.X_cont_u, tr.X_cat, tr.y_u, seed=seed, **fit_kw)
    elif cfg.get("needs_week_idx"):
        # B3/B4 recurrent models need week_idx to build sequence buffers
        model.fit(tr.X_cont, tr.X_cat, tr.y,
                  week_idx=tr.week_idx, seed=seed,
                  verbose=verbose, **fit_kw)
    else:
        model.fit(tr.X_cont, tr.X_cat, tr.y,
                  seed=seed, verbose=verbose, **fit_kw)

    elapsed = time.time() - t0
    print(f"  Fit completed in {elapsed:.1f}s")

    # -- Evaluate -------------------------------------------------------------
    print("  Train set:")
    train_metrics = evaluate_split(model, tr, model_name=model_name)
    print("  Val set  :")
    val_metrics   = evaluate_split(model, bundle.val, model_name=model_name)

    # -- Save -----------------------------------------------------------------
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"{model_name}_seed{seed}.pkl"
        model.save(out)
        print(f"  Saved: {out}")

    metrics = {"train": train_metrics, "val": val_metrics, "fit_time_s": elapsed}
    return model, metrics


# --------------------------------------------------------------------------- #
# Convenience: train all baselines
# --------------------------------------------------------------------------- #
def train_all_baselines(
    bundle: DatasetBundle,
    seed: int = 42,
    verbose: bool = False,
    save_dir: Optional[Path] = None,
) -> Dict[str, Tuple[Any, Dict]]:
    """Train B0 through B4 on ``bundle`` and return all results.

    Returns
    -------
    dict { model_name: (fitted_model, metrics) }
    """
    results = {}
    for name in BASELINE_CONFIGS:
        try:
            model, metrics = train_baseline(
                name, bundle, seed=seed,
                verbose=verbose, save_dir=save_dir,
            )
            results[name] = (model, metrics)
        except Exception as exc:
            print(f"  [WARN] {name} failed: {exc}")
            results[name] = (None, {"error": str(exc)})
    return results


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def _cli():
    from src.utils.paths import (
        PREPROCESSED_DIR, OLIST_PREPROCESSED_DIR,
        HM_PREPROCESSED_DIR, SYNTH26_PREPROCESSED_DIR,
    )
    _DATASET_MAP = {
        "dataco": (PREPROCESSED_DIR,         "DataCo"),
        "olist":  (OLIST_PREPROCESSED_DIR,   "Olist"),
        "hm":     (HM_PREPROCESSED_DIR,      "H&M"),
        "synth":  (SYNTH26_PREPROCESSED_DIR, "Synth-2026"),
    }

    parser = argparse.ArgumentParser(description="Train SIDN-CRO baselines")
    parser.add_argument("--dataset", choices=list(_DATASET_MAP), default="dataco")
    parser.add_argument("--model",   default="all",
                        help="B0/B1/B2/B3/B4 or 'all' (default)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    prep_dir, name = _DATASET_MAP[args.dataset]
    bundle = load_dataset(prep_dir, name=name)
    save   = Path(args.save_dir) if args.save_dir else None

    if args.model == "all":
        train_all_baselines(bundle, seed=args.seed,
                            verbose=args.verbose, save_dir=save)
    else:
        train_baseline(args.model, bundle, seed=args.seed,
                       verbose=args.verbose, save_dir=save)


if __name__ == "__main__":
    _cli()
