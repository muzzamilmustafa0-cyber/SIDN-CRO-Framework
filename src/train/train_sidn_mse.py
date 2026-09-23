"""
SIDN-CRO -- Train SIDN with MSE loss (variant A1).

This is the first of two SIDN training stages:

    A1  SIDN + Huber-MSE (log1p-space)   <-- this file
    A2  SIDN + SPO+ DFL loss             (train_sidn_dfl.py)

A1 training is fast (~1 min on CPU) and produces a well-initialised model that
A2 can then fine-tune with the SPO+ loss.

The MSE loss is Huber loss on log1p-transformed demand ? this is the same loss
used internally in SIDN.fit() and is exposed here so the experiment runner can
control all hyperparameters consistently.

Structural guarantees (from SIDN architecture):
    dD?/dp   < 0,   dD?/db > 0,   dD?/du > 0,   D? > 0
These hold at every training step, not only at convergence.

Usage (programmatic)
--------------------
    from src.train import load_dataset, train_sidn_mse
    bundle = load_dataset(preprocessed_dir, name="DataCo")
    model, metrics = train_sidn_mse(bundle, seed=42)

Usage (command line)
--------------------
    python -m src.train.train_sidn_mse --dataset dataco --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.models.sidn import SIDNDemand
from src.train.dataset import DatasetBundle, load_dataset, evaluate_split


# --------------------------------------------------------------------------- #
# Default hyperparameters (tuned by preliminary grid search)
# --------------------------------------------------------------------------- #
SIDN_MSE_DEFAULTS = dict(
    hidden       = 64,     # ContextNet trunk hidden width
    dropout      = 0.10,   # dropout probability
    epochs       = 300,    # maximum training epochs
    batch_size   = 64,
    lr           = 5e-4,   # cosine-annealed from lr to lr/20
    weight_decay = 1e-5,
    patience     = 25,     # early-stopping patience
)


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #
def train_sidn_mse(
    bundle: DatasetBundle,
    seed:    int  = 42,
    verbose: bool = False,
    save_dir: Optional[Path] = None,
    **hyperparams,
) -> Tuple[SIDNDemand, Dict]:
    """Train SIDN with Huber-MSE on log1p-demand (model variant A1).

    Parameters
    ----------
    bundle      : DatasetBundle from load_dataset()
    seed        : RNG seed (controls PyTorch and NumPy)
    verbose     : print per-20-epoch progress
    save_dir    : if given, save checkpoint to ``save_dir/A1_seed{seed}.pkl``
    **hyperparams : override any of SIDN_MSE_DEFAULTS

    Returns
    -------
    model   : fitted SIDNDemand (name set to "A1_SIDNDemand")
    metrics : dict { "train": {...}, "val": {...}, "fit_time_s": float,
                     "structural_params": dict_of_mean_params }
    """
    hp = {**SIDN_MSE_DEFAULTS, **hyperparams}

    print(f"\n{'='*60}")
    print(f"  Training A1 SIDN-MSE on {bundle.name}  seed={seed}")
    print(f"  hidden={hp['hidden']}  lr={hp['lr']}  "
          f"epochs={hp['epochs']}  patience={hp['patience']}")
    print(f"{'='*60}")

    # -- Build model ----------------------------------------------------------
    model = SIDNDemand(
        bundle.schema, bundle.x_scaler, bundle.y_scaler,
        hidden=hp["hidden"], dropout=hp["dropout"],
    )
    model.name = "A1_SIDNDemand"

    # -- Fit ------------------------------------------------------------------
    tr = bundle.train
    t0 = time.time()
    model.fit(
        tr.X_cont, tr.X_cat, tr.y,
        epochs=hp["epochs"], batch_size=hp["batch_size"],
        lr=hp["lr"], weight_decay=hp["weight_decay"],
        patience=hp["patience"],
        verbose=verbose, seed=seed,
    )
    elapsed = time.time() - t0
    print(f"  Fit completed in {elapsed:.1f}s")

    # -- Evaluate demand-forecasting accuracy ------------------------------
    print("  Train set:")
    train_m = evaluate_split(model, tr, model_name="A1")
    print("  Val set  :")
    val_m   = evaluate_split(model, bundle.val, model_name="A1")

    # -- Structural parameter summary (Table 3 in paper) -------------------
    try:
        params_summary = _structural_param_stats(model, bundle.val)
        print("\n  Structural parameters on val set:")
        model.summarise_params(bundle.val.X_cont_u, bundle.val.X_cat)
    except Exception:
        params_summary = {}

    # -- Save -----------------------------------------------------------------
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"A1_seed{seed}.pkl"
        model.save(out)
        print(f"  Saved: {out}")

    metrics = {
        "train":             train_m,
        "val":               val_m,
        "fit_time_s":        elapsed,
        "structural_params": params_summary,
    }
    return model, metrics


def _structural_param_stats(model: SIDNDemand,
                             split) -> Dict[str, Dict[str, float]]:
    """Return mean/std of structural parameters (beta, k1, k2, k3, alpha) on split."""
    params = model.inspect_params(split.X_cont_u, split.X_cat)
    return {
        name: {"mean": float(v.mean()), "std": float(v.std()),
               "min": float(v.min()), "max": float(v.max())}
        for name, v in params.items()
    }


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

    parser = argparse.ArgumentParser(description="Train SIDN-MSE (A1)")
    parser.add_argument("--dataset",  choices=list(_DATASET_MAP), default="dataco")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--epochs",   type=int,   default=SIDN_MSE_DEFAULTS["epochs"])
    parser.add_argument("--hidden",   type=int,   default=SIDN_MSE_DEFAULTS["hidden"])
    parser.add_argument("--lr",       type=float, default=SIDN_MSE_DEFAULTS["lr"])
    parser.add_argument("--patience", type=int,   default=SIDN_MSE_DEFAULTS["patience"])
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    prep_dir, name = _DATASET_MAP[args.dataset]
    bundle = load_dataset(prep_dir, name=name)
    save   = Path(args.save_dir) if args.save_dir else None

    train_sidn_mse(
        bundle, seed=args.seed,
        verbose=args.verbose, save_dir=save,
        epochs=args.epochs, hidden=args.hidden,
        lr=args.lr, patience=args.patience,
    )


if __name__ == "__main__":
    _cli()
