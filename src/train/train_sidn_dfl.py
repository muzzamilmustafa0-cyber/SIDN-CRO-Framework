"""
SIDN-CRO -- Train SIDN with Decision-Focused Learning / SPO+ loss (variant A2).

This is the second of two SIDN training stages:

    A1  SIDN + Huber-MSE         (train_sidn_mse.py)
    A2  SIDN + SPO+ DFL loss     <-- this file

Training procedure
------------------
1.  Start from a pre-trained A1 checkpoint (strongly recommended) or train
    from scratch.
2.  For each epoch, iterate over **weeks** (one SQP solve per week):
        a.  Solve SQP under current SIDN predictions -> z*(D?)          [no grad]
        b.  Re-evaluate D? at z* with autograd -> differentiable profit  [grad ON]
        c.  Loss = true_profit_star ? profit  (SPO+ surrogate)
        d.  Backprop -> update theta
3.  Early-stop on validation set demand MAPE (correlates with regret in
    experiments; avoids expensive per-week SQP solves on val every epoch).

Why SPO+ improves on MSE
------------------------
MSE minimises predictive error uniformly across all (product, retailer, week)
cells.  SPO+ instead concentrates gradient signal on *high-leverage* cells:
those where a demand mis-prediction causes the SQP solver to choose
suboptimal decisions.  The structural gradient (?3.3 of the manuscript) is:

    dL / dD?_ij = ?(p*_ij ? C_var*_ij + gamma*_ij)

i.e., demand predictions for high-margin (product, retailer) pairs receive
larger gradient magnitude and are corrected first.

Computational cost
------------------
For I=5 products, J=3 retailers:
* ~0.5 s per SQP solve (trust-constr, warm-started)
* ~100?150 training weeks -> 50?75 s per epoch
* 10?20 DFL epochs -> 15?25 min total fine-tuning

Usage (programmatic)
--------------------
    from src.train import load_dataset, train_sidn_mse, train_sidn_dfl
    from src.optim import params_from_panel_mean

    bundle    = load_dataset(preprocessed_dir, name="DataCo")
    a1_model, _ = train_sidn_mse(bundle, seed=42)
    sqp_params  = params_from_panel_mean(
        bundle.train.panel_df, bundle.n_products, bundle.n_retailers)
    a2_model, metrics = train_sidn_dfl(bundle, sqp_params,
                                       pretrained_sidn=a1_model, seed=42)

Usage (command line)
--------------------
    python -m src.train.train_sidn_dfl --dataset dataco --seed 42 \\
        --a1_checkpoint results/models/A1_seed42.pkl
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.sidn import SIDNDemand
from src.models.base import DemandModel
from src.optim.sqp_layer import SCMParameters, _pack
from src.optim.differentiable import dfl_loss
from src.train.dataset import (
    DatasetBundle, SplitArrays, load_dataset,
    evaluate_split, build_week_context,
)


# --------------------------------------------------------------------------- #
# Default hyperparameters
# --------------------------------------------------------------------------- #
SIDN_DFL_DEFAULTS = dict(
    dfl_epochs   = 15,     # SPO+ fine-tuning epochs (each epoch = all weeks)
    lr           = 1e-4,   # lower LR than MSE stage to avoid catastrophic forgetting
    weight_decay = 1e-5,
    grad_clip    = 0.5,    # tighter gradient clipping for stability
    patience     = 8,      # early-stop on val MAPE
    sqp_method   = "trust-constr",
    sqp_tol      = 1e-5,
    sqp_max_iter = 300,
)


# --------------------------------------------------------------------------- #
# True-optimal profit cache
# --------------------------------------------------------------------------- #
def _compute_true_profit_star_cache(
    true_demand_fn,
    sqp_params: SCMParameters,
    split: SplitArrays,
    weeks: List[int],
    n_products: int,
    n_retailers: int,
    dec_idx: Dict,
    sqp_method: str,
    sqp_tol: float,
    sqp_max_iter: int,
    verbose: bool = True,
) -> Dict[int, float]:
    """Pre-compute pi*(D_true) for each week.  Cached to avoid re-solving.

    This is run **once** before the DFL training loop.  The cost is the same as
    one full training epoch (one SQP solve per training week).
    """
    from src.optim.differentiable import solve_true_optimum

    cache: Dict[int, float] = {}
    for k, week in enumerate(weeks):
        ctx = build_week_context(split, week, n_products, n_retailers, dec_idx)
        if ctx is None:
            continue
        sol = solve_true_optimum(
            sqp_params, true_demand_fn, ctx,
            sqp_method=sqp_method, sqp_tol=sqp_tol,
            sqp_max_iter=sqp_max_iter,
        )
        cache[week] = sol.profit
        if verbose and (k + 1) % 20 == 0:
            print(f"    pi* cache: week {k+1}/{len(weeks)}  "
                  f"pi*={sol.profit:.0f}  feasible={sol.feasible}")
    return cache


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #
def train_sidn_dfl(
    bundle: DatasetBundle,
    sqp_params: SCMParameters,
    pretrained_sidn: Optional[SIDNDemand] = None,
    true_demand_fn=None,
    seed: int = 42,
    verbose: bool = False,
    save_dir: Optional[Path] = None,
    **hyperparams,
) -> Tuple[SIDNDemand, Dict]:
    """Fine-tune SIDN with SPO+ decision-focused learning loss (model A2).

    Parameters
    ----------
    bundle          : DatasetBundle from load_dataset()
    sqp_params      : SCMParameters for this dataset (one instance shared across
                      all training weeks ? week-specific cost variations are
                      modelled through the demand function, not the parameters)
    pretrained_sidn : A1 checkpoint to start from (recommended).  If None, a
                      fresh SIDN is initialised from scratch.
    true_demand_fn  : demand model used to compute pi*(D_true) for the SPO+ loss
                      upper bound.  If None, the A1 model is used as proxy.
    seed            : RNG seed
    verbose         : print per-epoch diagnostics
    save_dir        : save A2 checkpoint here
    **hyperparams   : override any of SIDN_DFL_DEFAULTS

    Returns
    -------
    model   : fine-tuned SIDNDemand (name = "A2_SIDNDemand")
    metrics : { "train_regret": [...per_epoch...], "val": {...}, "fit_time_s": ... }
    """
    hp  = {**SIDN_DFL_DEFAULTS, **hyperparams}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  Training A2 SIDN-DFL on {bundle.name}  seed={seed}")
    print(f"  dfl_epochs={hp['dfl_epochs']}  lr={hp['lr']}  "
          f"sqp_method={hp['sqp_method']}")
    print(f"{'='*60}")

    # -- Initialise SIDN ------------------------------------------------------
    if pretrained_sidn is not None:
        model = copy.deepcopy(pretrained_sidn)
        print("  Starting from pre-trained A1 checkpoint.")
    else:
        model = SIDNDemand(
            bundle.schema, bundle.x_scaler, bundle.y_scaler,
            hidden=64, dropout=0.10,
        )
        print("  Starting from random initialisation (A1 pre-training recommended).")
    model.name = "A2_SIDNDemand"

    # True demand proxy (used to compute pi* upper bound)
    _true_fn = true_demand_fn if true_demand_fn is not None else model

    # -- Identify valid training weeks ----------------------------------------
    tr       = bundle.train
    dec_idx  = model._dec_idx
    I, J     = bundle.n_products, bundle.n_retailers

    all_weeks = sorted(np.unique(tr.week_idx).tolist())
    valid_weeks = [
        w for w in all_weeks
        if build_week_context(tr, w, I, J, dec_idx) is not None
    ]
    print(f"  Training weeks: {len(valid_weeks)} / {len(all_weeks)} complete")

    # -- Pre-compute pi* cache -------------------------------------------------
    print("  Computing pi*(D_true) cache for training weeks ...")
    t_cache = time.time()
    pi_star_cache = _compute_true_profit_star_cache(
        _true_fn, sqp_params, tr, valid_weeks,
        I, J, dec_idx,
        sqp_method=hp["sqp_method"],
        sqp_tol=hp["sqp_tol"],
        sqp_max_iter=hp["sqp_max_iter"],
        verbose=verbose,
    )
    print(f"  pi* cache built in {time.time() - t_cache:.1f}s  "
          f"(mean pi*={np.mean(list(pi_star_cache.values())):.0f})")

    # -- Optimiser ------------------------------------------------------------
    opt = torch.optim.Adam(
        model.model.parameters(),
        lr=hp["lr"], weight_decay=hp["weight_decay"],
    )

    t0              = time.time()
    best_val_mape   = float("inf")
    best_state      = None
    no_improve      = 0
    train_regrets: List[float] = []

    # -- DFL fine-tuning epochs -----------------------------------------------
    for ep in range(hp["dfl_epochs"]):
        model.model.train()
        week_order = rng.permutation(valid_weeks).tolist()
        ep_regret  = 0.0
        ep_n       = 0
        x0_cache: Dict[int, np.ndarray] = {}  # warm-start per week

        for week in week_order:
            ctx = build_week_context(tr, week, I, J, dec_idx)
            if ctx is None:
                continue
            pi_star = pi_star_cache.get(week, 0.0)

            loss, sol = dfl_loss(
                model, sqp_params, ctx,
                true_profit_star=pi_star,
                solver_x0=x0_cache.get(week),
                sqp_method=hp["sqp_method"],
                sqp_tol=hp["sqp_tol"],
                sqp_max_iter=hp["sqp_max_iter"],
            )
            # Warm-start next epoch from this solution
            x0_cache[week] = _pack(
                sol.p, sol.b_s, sol.b_m, sol.b_r,
                sol.u_s, sol.u_m, sol.u_r,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.model.parameters(), max_norm=hp["grad_clip"])
            opt.step()

            regret_val = loss.item()
            ep_regret += regret_val
            ep_n      += 1

        ep_regret /= max(ep_n, 1)
        train_regrets.append(ep_regret)

        # -- Validation (demand MAPE ? fast proxy for regret) --------------
        model.model.eval()
        val_m = evaluate_split(model, bundle.val)
        val_mape = val_m["mape"]

        if verbose or True:   # always print epoch summary
            print(f"  DFL epoch {ep+1:3d}  "
                  f"train_regret={ep_regret:.1f}  "
                  f"val_MAPE={val_mape:.2f}%")

        if val_mape < best_val_mape - 0.01:
            best_val_mape = val_mape
            best_state    = {k: v.clone() for k, v in
                             model.model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= hp["patience"]:
            print(f"  Early stop at epoch {ep+1}  best_val_MAPE={best_val_mape:.2f}%")
            break

    elapsed = time.time() - t0

    # -- Restore best state ------------------------------------------------
    if best_state is not None:
        model.model.load_state_dict(best_state)

    # -- Final evaluation --------------------------------------------------
    print("  Final val set  :")
    val_metrics = evaluate_split(model, bundle.val, model_name="A2")

    # -- Save -----------------------------------------------------------------
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"A2_seed{seed}.pkl"
        model.save(out)
        print(f"  Saved: {out}")

    metrics = {
        "val":           val_metrics,
        "train_regrets": train_regrets,
        "fit_time_s":    elapsed,
        "best_val_mape": best_val_mape,
    }
    return model, metrics


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def _cli():
    from src.utils.paths import (
        PREPROCESSED_DIR, OLIST_PREPROCESSED_DIR,
        HM_PREPROCESSED_DIR, SYNTH26_PREPROCESSED_DIR,
    )
    from src.optim import params_from_panel_mean
    from src.models import DemandModel

    _DATASET_MAP = {
        "dataco": (PREPROCESSED_DIR,         "DataCo"),
        "olist":  (OLIST_PREPROCESSED_DIR,   "Olist"),
        "hm":     (HM_PREPROCESSED_DIR,      "H&M"),
        "synth":  (SYNTH26_PREPROCESSED_DIR, "Synth-2026"),
    }

    parser = argparse.ArgumentParser(description="Train SIDN-DFL (A2)")
    parser.add_argument("--dataset",       choices=list(_DATASET_MAP), default="dataco")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dfl_epochs",    type=int,   default=SIDN_DFL_DEFAULTS["dfl_epochs"])
    parser.add_argument("--lr",            type=float, default=SIDN_DFL_DEFAULTS["lr"])
    parser.add_argument("--sqp_method",    default=SIDN_DFL_DEFAULTS["sqp_method"])
    parser.add_argument("--a1_checkpoint", default=None,
                        help="Path to A1 model checkpoint (.pkl)")
    parser.add_argument("--verbose",       action="store_true")
    parser.add_argument("--save_dir",      default=None)
    args = parser.parse_args()

    prep_dir, name = _DATASET_MAP[args.dataset]
    bundle = load_dataset(prep_dir, name=name)
    save   = Path(args.save_dir) if args.save_dir else None

    # Load A1 checkpoint or train from scratch
    if args.a1_checkpoint:
        a1_model = DemandModel.load(Path(args.a1_checkpoint))
        print(f"Loaded A1 from {args.a1_checkpoint}")
    else:
        from src.train.train_sidn_mse import train_sidn_mse
        print("No A1 checkpoint provided ? running A1 pre-training first ...")
        a1_model, _ = train_sidn_mse(bundle, seed=args.seed, verbose=args.verbose)

    sqp_params = params_from_panel_mean(
        bundle.train.panel_df,
        bundle.n_products,
        bundle.n_retailers,
        rng_seed=args.seed,
    )

    train_sidn_dfl(
        bundle, sqp_params,
        pretrained_sidn=a1_model,
        seed=args.seed,
        verbose=args.verbose,
        save_dir=save,
        dfl_epochs=args.dfl_epochs,
        lr=args.lr,
        sqp_method=args.sqp_method,
    )


if __name__ == "__main__":
    _cli()
